import argparse, json, random, csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def seed_all(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gray(p, size=256):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def norm01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def estimate_pseudo_background(obs_u8):
    h, w = obs_u8.shape[:2]
    k = max(17, int(min(h, w) * 0.09) | 1)
    med = cv2.medianBlur(obs_u8, k)
    gau = cv2.GaussianBlur(obs_u8, (0, 0), sigmaX=max(5, k / 3), sigmaY=max(5, k / 3))
    bg = cv2.addWeighted(med, 0.60, gau, 0.40, 0)
    return bg.astype(np.uint8)


def make_soft_target(mask_u8):
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(np.float32)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    hard = m.astype(np.float32)
    dil1 = cv2.dilate(m, k1, iterations=1).astype(np.float32) * 0.85
    dil2 = cv2.dilate(m, k2, iterations=1).astype(np.float32) * 0.35
    soft = np.maximum.reduce([hard, dil1, dil2])
    soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=1.2, sigmaY=1.2)
    soft = np.clip(soft, 0, 1)
    return soft.astype(np.float32)


def gt_box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(1e-6, aa + bb - inter))


def nms_boxes(boxes, scores, iou_thr=0.20):
    if not boxes:
        return []
    idxs = np.argsort(scores)[::-1].tolist()
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def extract_boxes_from_prob(prob, topk=4, high_thr=0.38, low_thr=0.18, min_area=8, pad=28, max_area_ratio=0.28):
    h, w = prob.shape[:2]
    p = cv2.GaussianBlur(prob.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)

    seeds = (p >= high_thr).astype(np.uint8)
    if seeds.sum() < min_area:
        seeds = (p >= low_thr).astype(np.uint8)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    boxes, scores = [], []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, bw, bh = [int(v) for v in stats[i, :4]]

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        ar = ((x2 - x1) * (y2 - y1)) / float(h * w)
        if ar > max_area_ratio:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            side = int(np.sqrt(max_area_ratio * h * w))
            half = max(40, side // 2)
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

        comp = lab == i
        score = float(p[comp].mean() + 0.5 * p[comp].max())
        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    if not boxes:
        yy, xx = np.unravel_index(np.argmax(p), p.shape)
        half = 64
        boxes = [[max(0, int(xx) - half), max(0, int(yy) - half), min(w, int(xx) + half), min(h, int(yy) + half)]]
        scores = [float(p[yy, xx])]

    keep = nms_boxes(boxes, scores, 0.20)[:topk]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


class ProposerDataset(Dataset):
    def __init__(self, root, split, size=256):
        self.root = Path(root)
        self.size = size
        rels = [x.strip() for x in (self.root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]
        self.items = []
        for rel in rels:
            cls = rel.split("/")[0]
            sd = self.root / rel
            if (sd / "counterfactual.png").exists() and (sd / "mask.png").exists():
                self.items.append((sd, cls, rel))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sd, cls, rel = self.items[idx]
        obs = read_gray(sd / "counterfactual.png", self.size)
        mask = read_gray(sd / "mask.png", self.size)

        bg = estimate_pseudo_background(obs)
        residual = norm01(np.abs(obs.astype(np.float32) - bg.astype(np.float32)))
        target = make_soft_target(mask)

        x = np.stack([
            obs.astype(np.float32) / 255.0,
            bg.astype(np.float32) / 255.0,
            residual.astype(np.float32),
        ], axis=0)

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "hard": torch.from_numpy((mask > 0).astype(np.float32)[None]),
            "rel": rel,
            "cls": cls,
        }


class Conv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class SingleImageProposer(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1 = Conv(3, base)
        self.e2 = Conv(base, base * 2)
        self.e3 = Conv(base * 2, base * 4)
        self.e4 = Conv(base * 4, base * 6)
        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(base * 6, base * 4, 2, 2)
        self.d3 = Conv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = Conv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = Conv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))

        return self.out(d1)


def dice_loss_from_logits(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, device, out_vis=None, vis_n=60):
    model.eval()
    best_ious, hit01, hit03, cover = [], 0, 0, 0
    saved = 0

    if out_vis:
        out_vis = Path(out_vis)
        out_vis.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        x = batch["x"].to(device)
        hard = batch["hard"].numpy()
        rels = batch["rel"]

        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        xs = x.detach().cpu().numpy()

        for i in range(prob.shape[0]):
            p = prob[i]
            gt_mask = hard[i, 0]
            gt = gt_box_from_mask((gt_mask > 0).astype(np.uint8))

            boxes, scores = extract_boxes_from_prob(p, topk=4)
            ious = [box_iou(b, gt) for b in boxes]
            biou = max(ious) if ious else 0.0

            best_ious.append(biou)
            hit01 += int(biou >= 0.1)
            hit03 += int(biou >= 0.3)
            cover += int(biou > 0)

            if out_vis is not None and saved < vis_n:
                obs = (xs[i, 0] * 255).astype(np.uint8)
                bg = (xs[i, 1] * 255).astype(np.uint8)
                res = xs[i, 2]
                heat = cv2.applyColorMap((np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                img = cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR)

                if gt is not None:
                    x1, y1, x2, y2 = gt
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "GT", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                for j, b in enumerate(boxes):
                    x1, y1, x2, y2 = b
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"P{j} {ious[j]:.2f}", (x1, min(250, y2+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

                panels = [
                    img,
                    cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
                    cv2.applyColorMap((np.clip(res,0,1)*255).astype(np.uint8), cv2.COLORMAP_JET),
                    heat,
                    cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                ]
                names = ["obs+boxes", "pseudo-bg", "residual", "pred prob", "GT mask"]
                for im, name in zip(panels, names):
                    cv2.putText(im, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                board = np.hstack(panels)
                safe = rels[i].replace("/", "_")
                cv2.imwrite(str(out_vis / f"{saved:04d}_{safe}_iou{biou:.2f}.png"), board)
                saved += 1

    n = max(1, len(best_ious))
    return {
        "mean_best_iou": float(np.mean(best_ious)),
        "hit_iou_ge_0.1": float(hit01 / n),
        "hit_iou_ge_0.3": float(hit03 / n),
        "cover_hit_rate": float(cover / n),
        "count": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset_path_v2_500_each_subtype")
    ap.add_argument("--outdir", default="runs_single_image_proposer_v1")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_all(3407)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device, flush=True)

    train_ds = ProposerDataset(args.data_root, "train", args.img_size)
    val_ds = ProposerDataset(args.data_root, "val", args.img_size)
    test_ds = ProposerDataset(args.data_root, "test", args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = SingleImageProposer(base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = -1
    best = None
    bad = 0
    patience = 40

    with open(outdir / "train_log.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "loss", "val_mean_iou", "val_hit01", "val_hit03", "val_cover"])

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)

            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            dice = dice_loss_from_logits(logits, target)
            loss = bce + 1.2 * dice

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        val = evaluate(model, val_loader, device)
        score = val["cover_hit_rate"] + val["hit_iou_ge_0.1"] + 0.3 * val["hit_iou_ge_0.3"]

        print(f"epoch={ep:03d} loss={np.mean(losses):.4f} val_iou={val['mean_best_iou']:.4f} hit01={val['hit_iou_ge_0.1']:.4f} hit03={val['hit_iou_ge_0.3']:.4f} cover={val['cover_hit_rate']:.4f}", flush=True)

        with open(outdir / "train_log.csv", "a", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([ep, np.mean(losses), val["mean_best_iou"], val["hit_iou_ge_0.1"], val["hit_iou_ge_0.3"], val["cover_hit_rate"]])

        if score > best_score:
            best_score = score
            bad = 0
            best = {"model": model.state_dict(), "epoch": ep, "val": val, "args": vars(args)}
            torch.save(best, outdir / "best.pt")
        else:
            bad += 1

        if bad >= patience:
            print(f"[EARLY STOP] epoch={ep} best_epoch={best['epoch']} best_score={best_score:.4f}", flush=True)
            break

    model.load_state_dict(torch.load(outdir / "best.pt", map_location=device, weights_only=False)["model"])
    val = evaluate(model, val_loader, device, out_vis=outdir / "val_vis", vis_n=80)
    test = evaluate(model, test_loader, device, out_vis=outdir / "test_vis", vis_n=120)

    metrics = {"best_epoch": best["epoch"], "best_score": best_score, "val": val, "test": test}
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("[DONE]", outdir, flush=True)


if __name__ == "__main__":
    main()
