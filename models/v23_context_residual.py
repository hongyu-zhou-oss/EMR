import argparse
import csv
import json
import math
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


LABELS = ["defect", "oil", "artifact"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}


def seed_all(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_gray(path: Path, img_size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def norm01_np(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


class PairDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=256):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size

        split_file = self.data_root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)

        self.rels = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.items = []
        for rel in self.rels:
            cls = rel.split("/")[0]
            if cls not in LABEL_TO_ID:
                continue
            sd = self.data_root / rel
            if not (sd / "factual.png").exists():
                continue
            if not (sd / "counterfactual.png").exists():
                continue
            self.items.append((sd, LABEL_TO_ID[cls], cls, rel))

        if not self.items:
            raise RuntimeError(f"empty split: {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sd, y, cls, rel = self.items[idx]

        f = read_gray(sd / "factual.png", self.img_size)
        c = read_gray(sd / "counterfactual.png", self.img_size)
        d = np.abs(c - f)
        d = norm01_np(d)

        x = np.stack([f, c, d], axis=0).astype(np.float32)
        d_cf = d[None, :, :].astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "d_cf": torch.from_numpy(d_cf),
            "y": torch.tensor(y, dtype=torch.long),
            "rel": rel,
        }


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def compute_ctx_map(feat, k=7):
    local = F.avg_pool2d(feat, kernel_size=k, stride=1, padding=k // 2)
    diff = (feat - local).pow(2).mean(dim=1, keepdim=True)
    diff = diff / (diff.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return diff


class V23ContextResidual(nn.Module):
    """
    稳定版 V23：
    - 不用 hard mask
    - 输入 fact/cf/diff
    - 内部计算 D_ctx: feature 与局部背景的不一致
    - attention 不单独接管分类，而是 residual feature branch
    """
    def __init__(self, num_classes=3, base=48):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBlock(3, base, stride=1),
            ConvBlock(base, base * 2, stride=2),
            ConvBlock(base * 2, base * 3, stride=2),
            ConvBlock(base * 3, base * 4, stride=2),
        )
        c = base * 4
        self.out_channels = c

        self.att_head = nn.Sequential(
            nn.Conv2d(c + 2, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 1, 1)
        )

        # 初始 attention 不要太极端，避免 collapse
        nn.init.zeros_(self.att_head[-1].weight)
        nn.init.constant_(self.att_head[-1].bias, 0.0)

        # global + attention + evidence summary
        self.classifier = nn.Sequential(
            nn.Linear(c * 2 + 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, d_cf):
        feat = self.stem(x)
        b, c, h, w = feat.shape

        d_cf_small = F.interpolate(d_cf, size=(h, w), mode="bilinear", align_corners=False)
        d_cf_small = d_cf_small / (d_cf_small.amax(dim=(2, 3), keepdim=True) + 1e-6)

        d_ctx = compute_ctx_map(feat, k=7).detach()

        att_in = torch.cat([feat, d_cf_small, d_ctx], dim=1)
        att_logits = self.att_head(att_in)
        A = torch.sigmoid(att_logits)

        global_pool = feat.mean(dim=(2, 3))

        att_pool = (feat * A).sum(dim=(2, 3)) / (A.sum(dim=(2, 3)) + 1e-6)

        ev = torch.cat([
            d_cf_small.mean(dim=(2, 3)),
            d_cf_small.amax(dim=(2, 3)),
            d_ctx.mean(dim=(2, 3)),
            d_ctx.amax(dim=(2, 3)),
        ], dim=1)

        z = torch.cat([global_pool, att_pool, ev], dim=1)
        logits = self.classifier(z)

        d_total = 0.65 * d_cf_small + 0.35 * d_ctx
        d_total = d_total / (d_total.amax(dim=(2, 3), keepdim=True) + 1e-6)

        return logits, A, d_ctx, d_total


def attention_losses(A, d_total, stage):
    # stage1 不约束 attention，只让模型先学会分类
    if stage == 1:
        return A.new_tensor(0.0), A.new_tensor(0.0), A.new_tensor(0.0)

    target = d_total.detach()

    A_norm = A / (A.sum(dim=(2, 3), keepdim=True) + 1e-6)
    T_norm = target / (target.sum(dim=(2, 3), keepdim=True) + 1e-6)

    # 用 MSE 而不是 KL，稳定很多
    loss_align = F.mse_loss(A_norm, T_norm)

    loss_sparse = A.mean()

    loss_smooth = (
        (A[:, :, 1:, :] - A[:, :, :-1, :]).abs().mean()
        + (A[:, :, :, 1:] - A[:, :, :, :-1]).abs().mean()
    )

    return loss_align, loss_sparse, loss_smooth


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((3, 3), dtype=np.int64)
    correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        d_cf = batch["d_cf"].to(device)
        y = batch["y"].to(device)

        logits, A, d_ctx, d_total = model(x, d_cf)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(t), int(p)] += 1

    acc = correct / max(1, total)
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    bal_acc = float(recalls.mean())
    return acc, bal_acc, cm


@torch.no_grad()
def save_attention_vis(model, dataset, outdir, device, n=48):
    model.eval()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    idxs = list(range(len(dataset)))
    random.Random(3407).shuffle(idxs)
    idxs = idxs[:n]

    for k, idx in enumerate(idxs):
        item = dataset[idx]
        x = item["x"][None].to(device)
        d_cf = item["d_cf"][None].to(device)
        rel = item["rel"].replace("/", "_")

        logits, A, d_ctx, d_total = model(x, d_cf)
        pred = logits.argmax(dim=1).item()

        arr = item["x"].numpy()
        f = arr[0]
        c = arr[1]
        d = arr[2]

        A_np = A[0, 0].detach().cpu().numpy()
        C_np = d_ctx[0, 0].detach().cpu().numpy()
        T_np = d_total[0, 0].detach().cpu().numpy()

        A_np = cv2.resize(A_np, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
        C_np = cv2.resize(C_np, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
        T_np = cv2.resize(T_np, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)

        def gray3(x):
            return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def heat(x):
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            return cv2.applyColorMap(x, cv2.COLORMAP_JET)

        panels = [
            gray3(f),
            gray3(c),
            heat(d),
            heat(C_np),
            heat(T_np),
            heat(A_np),
        ]
        names = ["factual", "counterfactual", "D_cf", "D_ctx", "D_total", f"A pred={LABELS[pred]}"]

        for im, name in zip(panels, names):
            cv2.putText(im, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        board = np.hstack(panels)
        cv2.imwrite(str(outdir / f"{k:04d}_{rel}.png"), board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset_path_v2_500_each_subtype")
    ap.add_argument("--outdir", default="runs_path_v23_context_residual")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device, flush=True)

    train_ds = PairDataset(args.data_root, "train", args.img_size)
    val_ds = PairDataset(args.data_root, "val", args.img_size)
    test_ds = PairDataset(args.data_root, "test", args.img_size)

    counts = Counter([int(x[1]) for x in train_ds.items])
    weights = []
    for i in range(3):
        weights.append(len(train_ds) / max(1, counts[i]))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.mean()

    print("[INFO] train counts:", counts, flush=True)
    print("[INFO] class weights:", class_weights.detach().cpu().numpy().tolist(), flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = V23ContextResidual(num_classes=3, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    patience = 70
    bad_epochs = 0

    log_csv = outdir / "training_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "stage", "loss", "loss_cls", "loss_align", "loss_sparse", "loss_smooth", "val_acc", "val_bal_acc"])

    for epoch in range(1, args.epochs + 1):
        if epoch <= 50:
            stage = 1
            lam_align, lam_sparse, lam_smooth = 0.0, 0.0, 0.0
        elif epoch <= 140:
            stage = 2
            lam_align, lam_sparse, lam_smooth = 0.02, 0.0005, 0.002
        else:
            stage = 3
            lam_align, lam_sparse, lam_smooth = 0.05, 0.001, 0.004

        model.train()
        losses = []
        cls_losses = []
        align_losses = []
        sparse_losses = []
        smooth_losses = []

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            d_cf = batch["d_cf"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            logits, A, d_ctx, d_total = model(x, d_cf)

            loss_cls = F.cross_entropy(logits, y, weight=class_weights)
            loss_align, loss_sparse, loss_smooth = attention_losses(A, d_total, stage)

            loss = (
                loss_cls
                + lam_align * loss_align
                + lam_sparse * loss_sparse
                + lam_smooth * loss_smooth
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            cls_losses.append(float(loss_cls.detach().cpu()))
            align_losses.append(float(loss_align.detach().cpu()))
            sparse_losses.append(float(loss_sparse.detach().cpu()))
            smooth_losses.append(float(loss_smooth.detach().cpu()))

        val_acc, val_bal, val_cm = evaluate(model, val_loader, device)

        msg = (
            f"epoch={epoch:03d} stage={stage} "
            f"loss={np.mean(losses):.4f} cls={np.mean(cls_losses):.4f} "
            f"align={np.mean(align_losses):.5f} val_acc={val_acc:.4f} val_bal={val_bal:.4f}"
        )
        print(msg, flush=True)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                epoch, stage,
                np.mean(losses), np.mean(cls_losses), np.mean(align_losses),
                np.mean(sparse_losses), np.mean(smooth_losses),
                val_acc, val_bal
            ])

        score = val_bal
        if score > best_val:
            best_val = score
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_bal_acc": val_bal,
                "val_confusion": val_cm.tolist(),
                "args": vars(args),
            }
            torch.save(best_state, outdir / "best.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch} best_val_bal={best_val:.4f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    val_acc, val_bal, val_cm = evaluate(model, val_loader, device)
    test_acc, test_bal, test_cm = evaluate(model, test_loader, device)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_score_balanced_acc": float(best_val),
        "val_acc": float(val_acc),
        "val_balanced_acc": float(val_bal),
        "test_acc": float(test_acc),
        "test_balanced_acc": float(test_bal),
        "val_confusion": val_cm.tolist(),
        "test_confusion": test_cm.tolist(),
        "labels": LABELS,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    save_attention_vis(model, test_ds, outdir / "attention_vis", device, n=64)

    print("[DONE]", outdir, flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
