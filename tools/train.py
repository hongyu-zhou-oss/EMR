import argparse, json, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_single_image_proposer_v1 import (
    read_gray,
    estimate_pseudo_background,
    gt_box_from_mask,
    norm01,
)
from models.v23_context_residual import V23ContextResidual, LABELS


LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}


def clip_box(b, w=256, h=256):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


def expand_box(b, pad, w=256, h=256):
    x1, y1, x2, y2 = clip_box(b, w, h)
    return [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]


def crop_to_box(img, box, size=256, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (size, size), interpolation=interp)


def random_bg_box(mask, min_size=48, max_size=150, tries=50):
    h, w = mask.shape[:2]
    for _ in range(tries):
        bw = random.randint(min_size, max_size)
        bh = random.randint(min_size, max_size)
        x1 = random.randint(0, max(0, w - bw))
        y1 = random.randint(0, max(0, h - bh))
        x2, y2 = x1 + bw, y1 + bh

        crop_m = mask[y1:y2, x1:x2]
        if crop_m.max() == 0:
            return [x1, y1, x2, y2]

    return [0, 0, w, h]


def dark_defect_augment(obs, mask):
    if mask is None or mask.max() == 0:
        return obs

    out = obs.copy().astype(np.float32)
    m = (mask > 0).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    m_edge = cv2.dilate(m, k, iterations=1).astype(np.float32)

    dark_value = random.uniform(3, 38)
    alpha = random.uniform(0.70, 0.96)

    out = out * (1 - alpha * m_edge) + dark_value * (alpha * m_edge)

    # 保持一定硬边，不做强模糊；这点很关键
    if random.random() < 0.35:
        # 轻微不均匀暗显影
        noise = np.random.normal(0, random.uniform(2, 8), out.shape).astype(np.float32)
        out = out + noise * m_edge

    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_aug(obs):
    out = obs.astype(np.float32)
    gain = random.uniform(0.75, 1.25)
    bias = random.uniform(-18, 18)
    out = out * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_or_noise_aug(obs):
    out = obs.copy()
    if random.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=random.uniform(0.3, 0.9))
    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(2, 7), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def make_input(obs):
    bg = estimate_pseudo_background(obs)
    d = norm01(np.abs(obs.astype(np.float32) - bg.astype(np.float32)))

    x = np.stack([
        bg.astype(np.float32) / 255.0,
        obs.astype(np.float32) / 255.0,
        d.astype(np.float32),
    ], axis=0).astype(np.float32)

    return x, d.astype(np.float32)


def collect_split(data_root, split_file):
    root = Path(data_root)
    rels = [x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]
    samples = []

    for rel in rels:
        cls = rel.split("/")[0]
        if cls not in LABEL_TO_ID:
            continue

        sd = root / rel
        img_p = sd / "counterfactual.png"
        mask_p = sd / "mask.png"

        if img_p.exists() and mask_p.exists():
            samples.append({
                "type": "labeled",
                "cls": cls,
                "img": str(img_p),
                "mask": str(mask_p),
                "rel": rel,
            })

    return samples


class V31Dataset(Dataset):
    def __init__(self, samples, neg_per_pos=1, dark_defect_prob=0.55, train=True):
        self.samples = samples
        self.neg_per_pos = neg_per_pos
        self.dark_defect_prob = dark_defect_prob
        self.train = train
        self.group = 1 + neg_per_pos if train else 1

    def __len__(self):
        return len(self.samples) * self.group

    def __getitem__(self, idx):
        base_idx = idx // self.group
        sub_idx = idx % self.group
        s = self.samples[base_idx]

        obs = read_gray(s["img"], 256)
        mask = read_gray(s["mask"], 256)
        cls = s["cls"]
        cls_id = LABEL_TO_ID[cls]

        is_negative = self.train and sub_idx > 0

        if is_negative:
            box = random_bg_box(mask)
            crop = crop_to_box(obs, box, 256)
            mcrop = np.zeros((256, 256), dtype=np.uint8)
            target = np.zeros(3, dtype=np.float32)
            cls_target = -1
            sample_type = "other_bg"
        else:
            gt = gt_box_from_mask(mask)
            pad = random.randint(16, 36) if self.train else 24
            box = expand_box(gt, pad, 256, 256)
            crop = crop_to_box(obs, box, 256)
            mcrop = crop_to_box(mask, box, 256, interp=cv2.INTER_NEAREST)

            if self.train:
                if cls == "defect" and random.random() < self.dark_defect_prob:
                    crop = dark_defect_augment(crop, mcrop)
                    sample_type = "dark_defect"
                else:
                    sample_type = "positive"

                if random.random() < 0.40:
                    crop = brightness_aug(crop)
                if random.random() < 0.35:
                    crop = blur_or_noise_aug(crop)
            else:
                sample_type = "positive"

            target = np.zeros(3, dtype=np.float32)
            target[cls_id] = 1.0
            cls_target = cls_id

        x, d = make_input(crop)

        return {
            "x": torch.from_numpy(x).float(),
            "d": torch.from_numpy(d[None]).float(),
            "target": torch.from_numpy(target).float(),
            "cls": torch.tensor(cls_target).long(),
            "is_pos": torch.tensor(0 if cls_target < 0 else 1).float(),
            "sample_type": sample_type,
            "gt_name": cls if not is_negative else "other",
        }


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch], 0),
        "d": torch.stack([b["d"] for b in batch], 0),
        "target": torch.stack([b["target"] for b in batch], 0),
        "cls": torch.stack([b["cls"] for b in batch], 0),
        "is_pos": torch.stack([b["is_pos"] for b in batch], 0),
        "sample_type": [b["sample_type"] for b in batch],
        "gt_name": [b["gt_name"] for b in batch],
    }


def margin_loss_for_mechanism(logits, cls, sample_types, margin=0.80):
    """
    机制感知 margin：
    - 正样本：目标类 logit 应高于非目标类；
    - dark_defect：尤其要求 defect 高于 oil/artifact，防止纯黑共性证据被 oil 吸收；
    - other 负样本由 BCE target=[0,0,0] 约束。
    """
    losses = []

    for i in range(logits.shape[0]):
        c = int(cls[i].item())
        if c < 0:
            continue

        target_logit = logits[i, c]
        others = [j for j in range(3) if j != c]
        max_other = torch.stack([logits[i, j] for j in others]).max()

        m = margin
        if sample_types[i] == "dark_defect" and c == LABEL_TO_ID["defect"]:
            m = margin + 0.30

        losses.append(F.relu(max_other - target_logit + m))

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device, thr=0.625):
    model.eval()

    y_true = []
    y_pred = []
    other_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        d = batch["d"].to(device)
        cls = batch["cls"].numpy()

        logits, A, d_ctx, d_total = model(x, d)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        for i in range(len(cls)):
            if cls[i] < 0:
                continue
            pred = int(np.argmax(prob[i]))
            y_true.append(int(cls[i]))
            y_pred.append(pred)
            if float(prob[i].max()) < thr:
                other_count += 1

    if not y_true:
        return {"acc": 0.0, "per_class": {}, "cm": [[0]*3 for _ in range(3)]}

    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1

    acc = float(np.trace(cm) / max(1, cm.sum()))
    per_class = {}
    for i, name in enumerate(LABELS):
        per_class[name] = float(cm[i, i] / max(1, cm[i].sum()))

    return {
        "acc": acc,
        "per_class": per_class,
        "cm": cm.tolist(),
        "other_rate_on_positive": float(other_count / max(1, len(y_true))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset_path_v2_500_each_subtype")
    ap.add_argument("--train-split", default="./dataset_path_v2_500_each_subtype/splits/train.txt")
    ap.add_argument("--val-split", default="./dataset_path_v2_500_each_subtype/splits/val.txt")
    ap.add_argument("--init-run", default="./overnight_runs_20260501_235103/runs_v25_path_activation_other")
    ap.add_argument("--outdir", default="./runs_v31_full_classifier_mechanism")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--dark-defect-prob", type=float, default=0.60)
    ap.add_argument("--patience", type=int, default=60)
    args = ap.parse_args()

    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    train_samples = collect_split(args.data_root, args.train_split)
    val_samples = collect_split(args.data_root, args.val_split)

    print(f"[DATA] train={len(train_samples)} val={len(val_samples)}", flush=True)

    train_ds = V31Dataset(
        train_samples,
        neg_per_pos=args.neg_per_pos,
        dark_defect_prob=args.dark_defect_prob,
        train=True,
    )
    val_ds = V31Dataset(val_samples, neg_per_pos=0, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device, flush=True)

    model = V23ContextResidual(num_classes=3, base=48).to(device)

    init_path = Path(args.init_run) / "best.pt"
    if init_path.exists():
        ck = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        init_thr = float(ck.get("best_thr", 0.625))
        print(f"[INIT] loaded {init_path}", flush=True)
    else:
        init_thr = 0.625
        print("[INIT] train from scratch", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = -1
    best = None
    bad_epochs = 0

    train_log = []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        losses = []
        bce_losses = []
        ce_losses = []
        margin_losses = []

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            d = batch["d"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            cls = batch["cls"].to(device, non_blocking=True)
            is_pos = batch["is_pos"].to(device, non_blocking=True)
            sample_types = batch["sample_type"]

            logits, A, d_ctx, d_total = model(x, d)

            # 多路径 BCE：正样本 one-hot，other/bg 全 0
            loss_bce = F.binary_cross_entropy_with_logits(logits, target)

            # 正样本 CE：让 argmax 分类更稳定
            pos_mask = is_pos > 0.5
            if pos_mask.any():
                loss_ce = F.cross_entropy(logits[pos_mask], cls[pos_mask])
            else:
                loss_ce = logits.sum() * 0.0

            loss_margin = margin_loss_for_mechanism(logits, cls, sample_types, margin=0.80)

            # other 负样本压低所有路径
            neg_mask = is_pos < 0.5
            if neg_mask.any():
                loss_other = torch.relu(logits[neg_mask] + 0.35).mean()
            else:
                loss_other = logits.sum() * 0.0

            loss = loss_bce + 0.55 * loss_ce + 0.35 * loss_margin + 0.20 * loss_other

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.item()))
            bce_losses.append(float(loss_bce.item()))
            ce_losses.append(float(loss_ce.item()))
            margin_losses.append(float(loss_margin.item()))

        scheduler.step()

        val = evaluate(model, val_loader, device, thr=init_thr)

        score = val["acc"] + 0.15 * np.mean(list(val["per_class"].values()))

        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "bce": float(np.mean(bce_losses)),
            "ce": float(np.mean(ce_losses)),
            "margin": float(np.mean(margin_losses)),
            "val": val,
            "lr": float(opt.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        train_log.append(row)

        print(
            f"[EP {ep:03d}] "
            f"loss={row['loss']:.4f} bce={row['bce']:.4f} ce={row['ce']:.4f} margin={row['margin']:.4f} "
            f"val_acc={val['acc']:.4f} "
            f"D={val['per_class'].get('defect',0):.4f} "
            f"O={val['per_class'].get('oil',0):.4f} "
            f"A={val['per_class'].get('artifact',0):.4f} "
            f"otherRate={val['other_rate_on_positive']:.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best = {
                "epoch": ep,
                "score": float(score),
                "val": val,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            bad_epochs = 0

            torch.save({
                "model": best["model"],
                "epoch": ep,
                "val": val,
                "best_score": float(score),
                "best_thr": init_thr,
                "labels": LABELS,
                "note": "V31 full classifier retrained from V25 with dark-defect augmentation, other negatives, and mechanism-aware margin loss.",
            }, out / "best.pt")
        else:
            bad_epochs += 1

        (out / "train_log.json").write_text(json.dumps(train_log, ensure_ascii=False, indent=2), encoding="utf-8")

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] patience={args.patience}", flush=True)
            break

    report = {
        "best_epoch": best["epoch"] if best else None,
        "best_score": best_score,
        "best_val": best["val"] if best else None,
        "args": vars(args),
        "labels": LABELS,
    }

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("[DONE]", out, flush=True)


if __name__ == "__main__":
    main()
