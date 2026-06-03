import argparse, json, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_single_image_proposer_v1 import (
    read_gray,
    estimate_pseudo_background,
    gt_box_from_mask,
    norm01,
)
from train_v23_context_residual import V23ContextResidual, LABELS
from train_v30_mechanism_calibrator import (
    MechanismCalibrator,
    mechanism_features,
)

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


def make_input(obs):
    bg = estimate_pseudo_background(obs)
    d = norm01(np.abs(obs.astype(np.float32) - bg.astype(np.float32)))

    x = np.stack([
        bg.astype(np.float32) / 255.0,
        obs.astype(np.float32) / 255.0,
        d.astype(np.float32),
    ], axis=0).astype(np.float32)

    return x, d.astype(np.float32)


def random_bg_box(mask, min_size=48, max_size=150, tries=50):
    h, w = mask.shape[:2]
    for _ in range(tries):
        bw = random.randint(min_size, max_size)
        bh = random.randint(min_size, max_size)
        x1 = random.randint(0, max(0, w - bw))
        y1 = random.randint(0, max(0, h - bh))
        x2, y2 = x1 + bw, y1 + bh
        if mask[y1:y2, x1:x2].max() == 0:
            return [x1, y1, x2, y2]
    return [0, 0, w, h]


def dark_defect_augment(obs, mask):
    if mask is None or mask.max() == 0:
        return obs

    out = obs.copy().astype(np.float32)
    m = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m2 = cv2.dilate(m, k, iterations=1).astype(np.float32)

    dark_value = random.uniform(3, 38)
    alpha = random.uniform(0.70, 0.96)

    out = out * (1 - alpha * m2) + dark_value * (alpha * m2)

    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(2, 8), out.shape).astype(np.float32)
        out = out + noise * m2

    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_aug(obs):
    out = obs.astype(np.float32)
    gain = random.uniform(0.75, 1.25)
    bias = random.uniform(-18, 18)
    return np.clip(out * gain + bias, 0, 255).astype(np.uint8)


def blur_or_noise_aug(obs):
    out = obs.copy()
    if random.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=random.uniform(0.3, 0.9))
    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(2, 7), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


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
                "cls": cls,
                "img": str(img_p),
                "mask": str(mask_p),
                "rel": rel,
            })

    return samples


@torch.no_grad()
def teacher_v31_scores(model, device, obs_u8):
    x, d = make_input(obs_u8)
    xt = torch.from_numpy(x)[None].float().to(device)
    dt = torch.from_numpy(d[None])[None].float().to(device)
    logits, A, d_ctx, d_total = model(xt, dt)
    s = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
    return s


@torch.no_grad()
def teacher_v30_calibrate(calib, device, mean, std, raw_scores, mech_feats):
    x = np.concatenate([raw_scores, mech_feats], axis=0).astype(np.float32)
    xn = (x[None, :] - mean) / std
    xt = torch.from_numpy(xn).float().to(device)
    clog, mlog = calib(xt)
    p = torch.softmax(clog, dim=1)[0].detach().cpu().numpy().astype(np.float32)
    m = torch.sigmoid(mlog)[0].detach().cpu().numpy().astype(np.float32)
    return p, m


def build_precomputed_items(samples, teacher_base, teacher_calib, mean, std, device,
                            neg_per_pos=1, dark_defect_aug=2, train=True, seed=3407):
    random.seed(seed)
    np.random.seed(seed)

    items = []

    for s in samples:
        cls = s["cls"]
        cls_id = LABEL_TO_ID[cls]

        obs = read_gray(s["img"], 256)
        mask = read_gray(s["mask"], 256)

        gt = gt_box_from_mask(mask)
        box = expand_box(gt, 24, 256, 256)

        crop = crop_to_box(obs, box, 256)
        mcrop = crop_to_box(mask, box, 256, interp=cv2.INTER_NEAREST)

        variants = [("orig", crop, mcrop)]

        if train and cls == "defect":
            for k in range(dark_defect_aug):
                variants.append((f"dark_defect_{k}", dark_defect_augment(crop, mcrop), mcrop))

        for tag, im, mm in variants:
            if train:
                if random.random() < 0.35:
                    im = brightness_aug(im)
                if random.random() < 0.30:
                    im = blur_or_noise_aug(im)

            raw = teacher_v31_scores(teacher_base, device, im)
            feats = mechanism_features(im, mm)
            cal_p, mech_p = teacher_v30_calibrate(teacher_calib, device, mean, std, raw, feats)

            target = np.zeros(3, dtype=np.float32)
            target[cls_id] = 1.0

            items.append({
                "crop": im,
                "mask": mm,
                "target": target,
                "cls": cls_id,
                "is_pos": 1.0,
                "teacher_class": cal_p,
                "teacher_mech": mech_p,
                "sample_type": tag,
                "gt_name": cls,
                "rel": s["rel"],
            })

        if train and neg_per_pos > 0:
            for k in range(neg_per_pos):
                nb = random_bg_box(mask)
                bg_crop = crop_to_box(obs, nb, 256)
                if random.random() < 0.35:
                    bg_crop = brightness_aug(bg_crop)

                items.append({
                    "crop": bg_crop,
                    "mask": np.zeros((256, 256), dtype=np.uint8),
                    "target": np.zeros(3, dtype=np.float32),
                    "cls": -1,
                    "is_pos": 0.0,
                    "teacher_class": np.ones(3, dtype=np.float32) / 3.0,
                    "teacher_mech": np.zeros(4, dtype=np.float32),
                    "sample_type": "other_bg",
                    "gt_name": "other",
                    "rel": s["rel"],
                })

    return items


class PrecomputedCropDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        crop = it["crop"]
        x, d = make_input(crop)

        return {
            "x": torch.from_numpy(x).float(),
            "d": torch.from_numpy(d[None]).float(),
            "target": torch.from_numpy(it["target"]).float(),
            "cls": torch.tensor(it["cls"]).long(),
            "is_pos": torch.tensor(it["is_pos"]).float(),
            "teacher_class": torch.from_numpy(it["teacher_class"]).float(),
            "teacher_mech": torch.from_numpy(it["teacher_mech"]).float(),
            "sample_type": it["sample_type"],
            "gt_name": it["gt_name"],
        }


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch], 0),
        "d": torch.stack([b["d"] for b in batch], 0),
        "target": torch.stack([b["target"] for b in batch], 0),
        "cls": torch.stack([b["cls"] for b in batch], 0),
        "is_pos": torch.stack([b["is_pos"] for b in batch], 0),
        "teacher_class": torch.stack([b["teacher_class"] for b in batch], 0),
        "teacher_mech": torch.stack([b["teacher_mech"] for b in batch], 0),
        "sample_type": [b["sample_type"] for b in batch],
        "gt_name": [b["gt_name"] for b in batch],
    }


def pool_tensor(t, B, device):
    if t is None:
        return torch.zeros(B, 2, device=device)

    z = t
    if z.dim() == 2:
        z = z[:, None, :, None]
    elif z.dim() == 3:
        z = z[:, None, :, :]
    elif z.dim() == 4:
        pass
    else:
        return torch.zeros(B, 2, device=device)

    flat = z.reshape(B, -1)
    return torch.stack([flat.mean(dim=1), flat.amax(dim=1)], dim=1)


class V32InternalMechanismClassifier(nn.Module):
    """
    推理时只使用这个单模型：
    base V31/V23 -> internal mechanism head -> mechanism-to-class correction -> final logits

    V30 只在训练期提供 teacher_class / teacher_mech。
    """
    def __init__(self, base=48, num_classes=3, num_mech=4, hidden=64):
        super().__init__()
        self.base_model = V23ContextResidual(num_classes=num_classes, base=base)

        # 输入：raw logits(3) + A/d_ctx/d_total 每个 mean/max，共 6，总计 9
        self.mech_head = nn.Sequential(
            nn.Linear(9, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_mech),
        )

        self.mech_to_class = nn.Linear(num_mech, num_classes, bias=False)

        # 初始化机制到类别的先验投影
        # [shared_dark, structural_break, diffuse_contamination, specular_artifact] -> [defect, oil, artifact]
        prior = torch.tensor([
            [0.08,  1.00, -0.20, -0.20],   # defect
            [0.08, -0.35,  1.00, -0.10],   # oil
            [0.08, -0.25, -0.10,  1.00],   # artifact
        ], dtype=torch.float32)
        with torch.no_grad():
            self.mech_to_class.weight.copy_(prior)

        self.alpha = nn.Parameter(torch.tensor(0.65))

    def load_base_state(self, state):
        self.base_model.load_state_dict(state, strict=True)

    def forward(self, x, d):
        raw_logits, A, d_ctx, d_total = self.base_model(x, d)
        B = x.shape[0]
        device = x.device

        pA = pool_tensor(A, B, device)
        pC = pool_tensor(d_ctx, B, device)
        pT = pool_tensor(d_total, B, device)

        mech_in = torch.cat([raw_logits, pA, pC, pT], dim=1)
        mech_logits = self.mech_head(mech_in)
        mech_prob = torch.sigmoid(mech_logits)

        mech_corr = self.mech_to_class(mech_prob)
        final_logits = raw_logits + torch.clamp(self.alpha, 0.0, 1.5) * mech_corr

        return final_logits, mech_logits, raw_logits, A, d_ctx, d_total


def margin_loss(logits, cls, sample_types, margin=0.80):
    losses = []
    for i in range(logits.shape[0]):
        c = int(cls[i].item())
        if c < 0:
            continue

        target_logit = logits[i, c]
        others = [j for j in range(3) if j != c]
        max_other = torch.stack([logits[i, j] for j in others]).max()

        m = margin
        if sample_types[i].startswith("dark_defect") and c == LABEL_TO_ID["defect"]:
            m = margin + 0.35

        losses.append(F.relu(max_other - target_logit + m))

    if not losses:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def kl_distill_loss(logits, teacher_probs, T=2.0):
    # teacher_probs 是概率分布
    logp = F.log_softmax(logits / T, dim=1)
    q = teacher_probs.clamp(1e-6, 1.0)
    q = q / q.sum(dim=1, keepdim=True)
    return F.kl_div(logp, q, reduction="batchmean") * (T * T)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.625):
    model.eval()
    y_true, y_pred = [], []
    other_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        d = batch["d"].to(device)
        cls = batch["cls"].cpu().numpy()

        final_logits, mech_logits, raw_logits, A, d_ctx, d_total = model(x, d)
        prob = torch.sigmoid(final_logits).detach().cpu().numpy()

        for i in range(len(cls)):
            if cls[i] < 0:
                continue
            pred = int(np.argmax(prob[i]))
            y_true.append(int(cls[i]))
            y_pred.append(pred)
            if float(prob[i].max()) < thr:
                other_count += 1

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
    ap.add_argument("--init-run", default="./runs_v31_full_classifier_mechanism")
    ap.add_argument("--teacher-calib", default="./runs_v30_mechanism_calibrator/v30_mechanism_calibrator.pt")
    ap.add_argument("--outdir", default="./runs_v32_internal_mechanism_classifier")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--dark-defect-aug", type=int, default=2)
    ap.add_argument("--patience", type=int, default=60)
    args = ap.parse_args()

    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device, flush=True)

    # teacher base: V31
    teacher_base = V23ContextResidual(num_classes=3, base=48).to(device)
    init_ck = torch.load(Path(args.init_run) / "best.pt", map_location=device, weights_only=False)
    teacher_base.load_state_dict(init_ck["model"], strict=True)
    teacher_base.eval()
    init_thr = float(init_ck.get("best_thr", 0.625))

    # teacher calibrator: V30
    pack = torch.load(args.teacher_calib, map_location=device, weights_only=False)
    teacher_calib = MechanismCalibrator(in_dim=pack["feature_dim"]).to(device)
    teacher_calib.load_state_dict(pack["model"])
    teacher_calib.eval()
    mean = pack["mean"]
    std = pack["std"]

    train_samples = collect_split(args.data_root, args.train_split)
    val_samples = collect_split(args.data_root, args.val_split)

    print(f"[DATA] train_samples={len(train_samples)} val_samples={len(val_samples)}", flush=True)
    print("[PRECOMPUTE] building teacher targets...", flush=True)

    train_items = build_precomputed_items(
        train_samples, teacher_base, teacher_calib, mean, std, device,
        neg_per_pos=args.neg_per_pos,
        dark_defect_aug=args.dark_defect_aug,
        train=True,
        seed=3407,
    )
    val_items = build_precomputed_items(
        val_samples, teacher_base, teacher_calib, mean, std, device,
        neg_per_pos=0,
        dark_defect_aug=0,
        train=False,
        seed=4407,
    )

    print(f"[ITEMS] train={len(train_items)} val={len(val_items)}", flush=True)

    train_ds = PrecomputedCropDataset(train_items)
    val_ds = PrecomputedCropDataset(val_items)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate
    )

    model = V32InternalMechanismClassifier(base=48).to(device)
    model.load_base_state(init_ck["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = -1.0
    best = None
    bad = 0
    logs = []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_meter = []

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            d = batch["d"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            cls = batch["cls"].to(device, non_blocking=True)
            is_pos = batch["is_pos"].to(device, non_blocking=True)
            teacher_class = batch["teacher_class"].to(device, non_blocking=True)
            teacher_mech = batch["teacher_mech"].to(device, non_blocking=True)
            sample_types = batch["sample_type"]

            final_logits, mech_logits, raw_logits, A, d_ctx, d_total = model(x, d)

            loss_bce = F.binary_cross_entropy_with_logits(final_logits, target)

            pos = is_pos > 0.5
            if pos.any():
                loss_ce = F.cross_entropy(final_logits[pos], cls[pos])
                loss_distill = kl_distill_loss(final_logits[pos], teacher_class[pos], T=2.0)
                loss_mech = F.binary_cross_entropy_with_logits(mech_logits[pos], teacher_mech[pos])
            else:
                loss_ce = final_logits.sum() * 0.0
                loss_distill = final_logits.sum() * 0.0
                loss_mech = final_logits.sum() * 0.0

            loss_margin = margin_loss(final_logits, cls, sample_types, margin=0.80)

            neg = is_pos < 0.5
            if neg.any():
                loss_other = torch.relu(final_logits[neg] + 0.35).mean()
                loss_mech_neg = F.binary_cross_entropy_with_logits(mech_logits[neg], teacher_mech[neg])
            else:
                loss_other = final_logits.sum() * 0.0
                loss_mech_neg = final_logits.sum() * 0.0

            # 机制内化：mech 与 distill 是辅助，但不压过真实类别
            loss = (
                loss_bce
                + 0.55 * loss_ce
                + 0.30 * loss_margin
                + 0.25 * loss_mech
                + 0.10 * loss_mech_neg
                + 0.22 * loss_distill
                + 0.18 * loss_other
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_meter.append(float(loss.item()))

        scheduler.step()

        val = evaluate(model, val_loader, device, thr=init_thr)
        score = val["acc"] + 0.15 * np.mean(list(val["per_class"].values()))

        row = {
            "epoch": ep,
            "loss": float(np.mean(loss_meter)),
            "val": val,
            "score": float(score),
            "alpha": float(torch.clamp(model.alpha, 0.0, 1.5).detach().cpu().item()),
            "lr": float(opt.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        logs.append(row)

        print(
            f"[EP {ep:03d}] loss={row['loss']:.4f} val_acc={val['acc']:.4f} "
            f"D={val['per_class'].get('defect',0):.4f} "
            f"O={val['per_class'].get('oil',0):.4f} "
            f"A={val['per_class'].get('artifact',0):.4f} "
            f"otherRate={val['other_rate_on_positive']:.4f} alpha={row['alpha']:.3f}",
            flush=True
        )

        if score > best_score:
            best_score = score
            best = {
                "epoch": ep,
                "score": float(score),
                "val": val,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            bad = 0

            torch.save({
                "model": best["model"],
                "epoch": ep,
                "val": val,
                "best_score": float(score),
                "best_thr": init_thr,
                "labels": LABELS,
                "type": "V32InternalMechanismClassifier",
                "note": "V32 internalizes V30 mechanism layer into V31. V30 is training-only teacher; inference uses only V32.",
            }, out / "best.pt")
        else:
            bad += 1

        (out / "train_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

        if bad >= args.patience:
            print(f"[EARLY STOP] patience={args.patience}", flush=True)
            break

    report = {
        "best_epoch": best["epoch"] if best else None,
        "best_score": best_score,
        "best_val": best["val"] if best else None,
        "args": vars(args),
        "labels": LABELS,
        "type": "V32InternalMechanismClassifier",
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("[DONE]", out, flush=True)


if __name__ == "__main__":
    main()
