import argparse, json, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_single_image_proposer_v1 import (
    read_gray,
    estimate_pseudo_background,
    gt_box_from_mask,
    norm01,
)
from train_v23_context_residual import V23ContextResidual, LABELS


LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
MECH_NAMES = [
    "shared_dark",
    "structural_break",
    "diffuse_contamination",
    "specular_artifact",
]


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


def grad_mag(x01):
    gx = cv2.Sobel(x01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x01, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def region_from_mask_or_residual(obs_u8, mask_u8=None):
    x = obs_u8.astype(np.float32) / 255.0
    bg = estimate_pseudo_background(obs_u8)
    res = norm01(np.abs(obs_u8.astype(np.float32) - bg.astype(np.float32)))

    if mask_u8 is not None and mask_u8.max() > 0:
        m = (mask_u8 > 0).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m, res

    thr = max(0.35, float(np.quantile(res.reshape(-1), 0.82)))
    m = (res >= thr).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m, res


def mechanism_features(obs_u8, mask_u8=None):
    """
    机制描述符只作为软证据，不做硬排他。
    重点：
    - darkness 是 shared，不直接代表 oil。
    - sharp/texture break 更接近 structural mechanism。
    - softness/diffusion 更接近 contamination mechanism。
    """
    obs = obs_u8.astype(np.float32) / 255.0
    m, res = region_from_mask_or_residual(obs_u8, mask_u8)

    if m.sum() < 4:
        m = (res >= np.quantile(res.reshape(-1), 0.85)).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=2)
    ero = cv2.erode(m, k, iterations=1)
    boundary = np.clip(dil - ero, 0, 1).astype(np.uint8)
    outside_ring = np.clip(dil - m, 0, 1).astype(np.uint8)

    g = norm01(grad_mag(obs))

    region = m.astype(bool)
    bnd = boundary.astype(bool)
    ring = outside_ring.astype(bool)

    if region.sum() < 1:
        region = np.ones_like(obs, dtype=bool)
    if bnd.sum() < 1:
        bnd = region
    if ring.sum() < 1:
        ring = ~region

    interior_darkness = float(1.0 - obs[region].mean())
    interior_std = float(obs[region].std())
    residual_mean = float(res[region].mean())
    residual_top = float(np.quantile(res[region], 0.85)) if region.sum() > 0 else 0.0

    boundary_grad = float(g[bnd].mean())
    boundary_topgrad = float(np.quantile(g[bnd], 0.85)) if bnd.sum() > 0 else 0.0

    ring_grad = float(g[ring].mean()) if ring.sum() > 0 else 0.0
    edge_contrast = float(np.clip(boundary_grad - ring_grad, 0, 1))

    area = float(m.sum())
    fill_ratio = float(area / (obs.shape[0] * obs.shape[1] + 1e-6))

    num, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if num > 1:
        largest = float(stats[1:, cv2.CC_STAT_AREA].max())
        comp_ratio = largest / (area + 1e-6)
        x, y, ww, hh, aa = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
        elongation = float(max(ww, hh) / (min(ww, hh) + 1e-6))
        bbox_fill = float(aa / (ww * hh + 1e-6))
    else:
        comp_ratio = 0.0
        elongation = 1.0
        bbox_fill = 0.0

    # softness/diffusion: 边界梯度越低、区域越铺开，越像扩散污染机制
    boundary_softness = float(np.clip(1.0 - boundary_grad, 0, 1))
    diffusion_score = float(np.clip(0.50 * boundary_softness + 0.30 * bbox_fill + 0.20 * min(fill_ratio * 8.0, 1.0), 0, 1))

    # structural break: 锋利边界 + 残差强 + 纹理/边缘突变
    structural_break = float(np.clip(0.45 * boundary_topgrad + 0.30 * edge_contrast + 0.25 * residual_top, 0, 1))

    # specular/artifact proxy: 高梯度但 residual/evidence 不一定集中；这里只是弱代理
    specular_proxy = float(np.clip(0.55 * boundary_topgrad + 0.45 * (1.0 - comp_ratio), 0, 1))

    feats = np.array([
        interior_darkness,
        interior_std,
        residual_mean,
        residual_top,
        boundary_grad,
        boundary_topgrad,
        edge_contrast,
        boundary_softness,
        diffusion_score,
        structural_break,
        specular_proxy,
        fill_ratio,
        comp_ratio,
        min(elongation / 8.0, 1.0),
        bbox_fill,
    ], dtype=np.float32)

    return feats


@torch.no_grad()
def v25_scores(model, device, obs_u8):
    bg = estimate_pseudo_background(obs_u8)
    d = norm01(np.abs(obs_u8.astype(np.float32) - bg.astype(np.float32)))

    x = np.stack([
        bg.astype(np.float32) / 255.0,
        obs_u8.astype(np.float32) / 255.0,
        d.astype(np.float32),
    ], axis=0).astype(np.float32)

    xt = torch.from_numpy(x)[None].to(device)
    dt = torch.from_numpy(d[None].astype(np.float32))[None].to(device)

    logits, A, d_ctx, d_total = model(xt, dt)
    s = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
    return s


def mech_target(cls_name):
    # 软标签，不是硬排他。
    # shared_dark 对所有异常均可为 1，表示它不决定具体类别。
    y = np.zeros(4, dtype=np.float32)
    y[0] = 1.0  # shared_dark / shared anomaly support

    if cls_name == "defect":
        y[1] = 1.0
    elif cls_name == "oil":
        y[2] = 1.0
    elif cls_name == "artifact":
        y[3] = 1.0

    return y


def dark_defect_augment(obs, mask):
    out = obs.copy().astype(np.float32)
    if mask is None or mask.max() == 0:
        return obs
    m = (mask > 0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    m2 = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(np.float32)

    # 保持边界相对锋利：让低角度暗显影 defect 出现在训练中
    dark_value = random.uniform(5, 35)
    alpha = random.uniform(0.75, 0.95)
    out = out * (1 - alpha * m2) + dark_value * (alpha * m2)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


class MechanismCalibrator(nn.Module):
    def __init__(self, in_dim, hidden=64, num_classes=3, num_mech=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.class_head = nn.Linear(hidden, num_classes)
        self.mech_head = nn.Linear(hidden, num_mech)

    def forward(self, x):
        h = self.trunk(x)
        return self.class_head(h), self.mech_head(h)


def collect_split(data_root, split_file):
    root = Path(data_root)
    rels = [x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]
    samples = []
    for rel in rels:
        sd = root / rel
        cls = rel.split("/")[0]
        if cls not in LABEL_TO_ID:
            continue
        img_p = sd / "counterfactual.png"
        mask_p = sd / "mask.png"
        if img_p.exists():
            samples.append((cls, img_p, mask_p if mask_p.exists() else None))
    return samples


def build_features(samples, model, device, aug_dark_defect=2, seed=3407):
    random.seed(seed)
    X, yc, ym, meta = [], [], [], []

    for cls, img_p, mask_p in samples:
        obs = read_gray(img_p, 256)
        mask = read_gray(mask_p, 256) if mask_p is not None else np.zeros_like(obs)

        box = gt_box_from_mask(mask) if mask.max() > 0 else [0, 0, 256, 256]
        box = expand_box(box, 24, 256, 256)

        crop = crop_to_box(obs, box, 256)
        mcrop = crop_to_box(mask, box, 256, interp=cv2.INTER_NEAREST)

        variants = [("orig", crop, mcrop)]

        if cls == "defect":
            for k in range(aug_dark_defect):
                variants.append((f"dark_defect_{k}", dark_defect_augment(crop, mcrop), mcrop))

        for tag, im, mm in variants:
            scores = v25_scores(model, device, im)
            feats = mechanism_features(im, mm)
            vec = np.concatenate([scores, feats], axis=0).astype(np.float32)

            X.append(vec)
            yc.append(LABEL_TO_ID[cls])
            ym.append(mech_target(cls))
            meta.append({
                "cls": cls,
                "img": str(img_p),
                "variant": tag,
                "raw_scores": scores.tolist(),
                "mechanism_features": feats.tolist(),
            })

    return np.stack(X), np.array(yc, dtype=np.int64), np.stack(ym), meta


def train_model(Xtr, ytr, mtr, Xva, yva, mva, out, epochs=600, lr=1e-3, seed=3407):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6

    Xtrn = (Xtr - mean) / std
    Xvan = (Xva - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MechanismCalibrator(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)

    xt = torch.from_numpy(Xtrn).float().to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    mt = torch.from_numpy(mtr).float().to(device)

    xv = torch.from_numpy(Xvan).float().to(device)
    yv = torch.from_numpy(yva).long().to(device)
    mv = torch.from_numpy(mva).float().to(device)

    best = {"acc": -1, "epoch": -1}

    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        clog, mlog = net(xt)

        loss_cls = F.cross_entropy(clog, yt)
        loss_mech = F.binary_cross_entropy_with_logits(mlog, mt)

        # 机制层约束：shared_dark 不应单独支配类别，这里通过机制辅助头弱约束。
        loss = loss_cls + 0.35 * loss_mech
        loss.backward()
        opt.step()

        if ep % 25 == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                vc, vm = net(xv)
                pred = vc.argmax(1)
                acc = float((pred == yv).float().mean().item())
                mech_loss = float(F.binary_cross_entropy_with_logits(vm, mv).item())

            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "epoch": ep,
                    "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "mech_loss": mech_loss,
                }

            print(f"[EP {ep:04d}] loss={float(loss.item()):.4f} val_acc={acc:.4f} mech_loss={mech_loss:.4f}", flush=True)

    net.load_state_dict(best["state"])

    torch.save({
        "model": net.state_dict(),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "labels": LABELS,
        "mechanism_names": MECH_NAMES,
        "feature_dim": int(Xtr.shape[1]),
        "best": {k: v for k, v in best.items() if k != "state"},
    }, out / "v30_mechanism_calibrator.pt")

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset_path_v2_500_each_subtype")
    ap.add_argument("--train-split", default="./dataset_path_v2_500_each_subtype/splits/train.txt")
    ap.add_argument("--val-split", default="./dataset_path_v2_500_each_subtype/splits/val.txt")
    ap.add_argument("--v25-run", default="./overnight_runs_20260501_235103/runs_v25_path_activation_other")
    ap.add_argument("--outdir", default="./runs_v30_mechanism_calibrator")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--dark-defect-aug", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v25 = V23ContextResidual(num_classes=3, base=48).to(device)
    ck = torch.load(Path(args.v25_run) / "best.pt", map_location=device, weights_only=False)
    v25.load_state_dict(ck["model"])
    v25.eval()

    train_samples = collect_split(args.data_root, args.train_split)

    if Path(args.val_split).exists():
        val_samples = collect_split(args.data_root, args.val_split)
    else:
        random.shuffle(train_samples)
        n = max(1, int(0.2 * len(train_samples)))
        val_samples = train_samples[:n]
        train_samples = train_samples[n:]

    print(f"[DATA] train={len(train_samples)} val={len(val_samples)}", flush=True)

    Xtr, ytr, mtr, meta_tr = build_features(
        train_samples, v25, device,
        aug_dark_defect=args.dark_defect_aug,
        seed=3407,
    )
    Xva, yva, mva, meta_va = build_features(
        val_samples, v25, device,
        aug_dark_defect=1,
        seed=4407,
    )

    (out / "train_meta.json").write_text(json.dumps(meta_tr, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "val_meta.json").write_text(json.dumps(meta_va, ensure_ascii=False, indent=2), encoding="utf-8")

    best = train_model(Xtr, ytr, mtr, Xva, yva, mva, out, epochs=args.epochs)

    report = {
        "best": {k: v for k, v in best.items() if k != "state"},
        "num_train_features": int(len(Xtr)),
        "num_val_features": int(len(Xva)),
        "labels": LABELS,
        "mechanism_names": MECH_NAMES,
        "note": "V30 is a mechanism-layer calibrator on top of V25 raw path scores and interpretable mechanism descriptors.",
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("[DONE]", out, flush=True)


if __name__ == "__main__":
    main()
