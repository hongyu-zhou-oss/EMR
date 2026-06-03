import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_single_image_proposer_v1 import (
    SingleImageProposer,
    estimate_pseudo_background,
    extract_boxes_from_prob,
    norm01,
)
from models.v23_context_residual import V23ContextResidual
from models.v30_mechanism_calibrator import (
    MechanismCalibrator,
    mechanism_features,
    v25_scores,
)
from models.v32_internal_mechanism import make_input


CLASSES = ["defect", "oil", "artifact"]
CLS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def read_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def clip_box(b, w=256, h=256):
    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


def area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def inter_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ia = inter_area(a, b)
    return ia / (area(a) + area(b) - ia + 1e-6)


def overlap_small(a, b):
    ia = inter_area(a, b)
    return ia / (min(area(a), area(b)) + 1e-6)


def union_box(a, b):
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def compactness(a, b):
    u = union_box(a, b)
    return (area(a) + area(b)) / (area(u) + 1e-6)


def gap_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    else:
        dx = 0

    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    else:
        dy = 0

    return float((dx * dx + dy * dy) ** 0.5)


def center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalize_list(xs):
    xs = np.array(xs, dtype=np.float32)
    if len(xs) == 0:
        return xs
    lo, hi = float(xs.min()), float(xs.max())
    if hi - lo < 1e-6:
        return np.ones_like(xs) * 0.5
    return (xs - lo) / (hi - lo + 1e-6)


@torch.no_grad()
def make_local_prob(proposer, device, obs):
    bg = estimate_pseudo_background(obs)
    res = norm01(np.abs(obs.astype(np.float32) - bg.astype(np.float32)))

    x = np.stack([
        obs.astype(np.float32) / 255.0,
        bg.astype(np.float32) / 255.0,
        res.astype(np.float32),
    ], axis=0).astype(np.float32)

    xt = torch.from_numpy(x)[None].to(device)
    prob = torch.sigmoid(proposer(xt))[0, 0].detach().cpu().numpy()
    return norm01(prob), bg, res


def make_global_map(obs, bg):
    res = norm01(np.abs(obs.astype(np.float32) - bg.astype(np.float32)))
    gx = cv2.Sobel(obs.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(obs.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    grad = norm01(np.sqrt(gx * gx + gy * gy))
    return norm01(0.65 * res + 0.35 * grad)


def crop_box(img, b):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box(b, w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    return crop


def crop_resize(img, b, size=256):
    crop = crop_box(img, b)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def calibrate(calib, device, mean, std, raw_scores, mech_feats):
    x = np.concatenate([raw_scores, mech_feats], axis=0).astype(np.float32)
    xn = (x[None, :] - mean) / std
    xt = torch.from_numpy(xn).float().to(device)
    clog, mlog = calib(xt)
    probs = torch.softmax(clog, dim=1)[0].detach().cpu().numpy()
    mech = torch.sigmoid(mlog)[0].detach().cpu().numpy()
    return probs, mech


@torch.no_grad()
def support_map_from_classifier(model, device, gray256):
    x, d = make_input(gray256)
    xt = torch.from_numpy(x)[None].float().to(device)
    dt = torch.from_numpy(d[None])[None].float().to(device)

    out = model(xt, dt)
    if not isinstance(out, (tuple, list)):
        return np.zeros((256, 256), dtype=np.float32)

    tensors = []
    for t in out[1:]:
        if t is None:
            continue
        z = t[0].detach().float().cpu().numpy()
        if z.ndim == 3:
            z = np.mean(np.abs(z), axis=0)
        elif z.ndim == 2:
            z = np.abs(z)
        else:
            continue
        z = cv2.resize(z, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensors.append(z)

    if not tensors:
        return np.zeros((256, 256), dtype=np.float32)

    sm = np.mean(np.stack(tensors, axis=0), axis=0)
    sm = sm - sm.min()
    sm = sm / (sm.max() + 1e-6)
    return sm.astype(np.float32)


def refine_box_largest_cc(crop_gray, support, q=0.72, min_refined_w=10, min_refined_h=8):
    H, W = crop_gray.shape[:2]
    support = support.astype(np.float32)
    support = support - support.min()
    support = support / (support.max() + 1e-6)

    thr = float(np.quantile(support.reshape(-1), q))
    mask = (support >= thr).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    min_area = max(8, int(0.0015 * H * W))
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        comp_mask = (cc == i)
        mean_support = float(support[comp_mask].mean())
        score = a * mean_support

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append((score, [x, y, x + w, y + h]))

    if not comps:
        # fallback: center third, not full box
        cx, cy = W // 2, H // 2
        bw, bh = max(min_refined_w, W // 3), max(min_refined_h, H // 3)
        return [
            max(0, cx - bw // 2),
            max(0, cy - bh // 2),
            min(W, cx + bw // 2),
            min(H, cy + bh // 2),
        ]

    comps.sort(key=lambda x: x[0], reverse=True)
    rb = comps[0][1]

    pad = 4
    rb = [
        max(0, rb[0] - pad),
        max(0, rb[1] - pad),
        min(W, rb[2] + pad),
        min(H, rb[3] + pad),
    ]

    x1, y1, x2, y2 = rb
    if x2 - x1 < min_refined_w:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_refined_w // 2)
        x2 = min(W, x1 + min_refined_w)
    if y2 - y1 < min_refined_h:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_refined_h // 2)
        y2 = min(H, y1 + min_refined_h)

    return [int(x1), int(y1), int(x2), int(y2)]


def map_local_box_to_parent(parent_box, local_box, crop_shape):
    px1, py1, px2, py2 = parent_box
    ch, cw = crop_shape[:2]
    lx1, ly1, lx2, ly2 = local_box

    sx = cw / 256.0
    sy = ch / 256.0

    return [
        px1 + lx1 * sx,
        py1 + ly1 * sy,
        px1 + lx2 * sx,
        py1 + ly2 * sy,
    ]


def select_by_locator_score(boxes, loc_scores, select_thr=0.40, max_keep=4, nms_thr=0.45):
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: loc_scores[i], reverse=True)
    candidates = [i for i in order if loc_scores[i] >= select_thr]
    # no-force mode: allow empty prediction on negative images
    if not candidates:
        return []

    keep = []
    for i in candidates:
        if any(iou(boxes[i], boxes[j]) > nms_thr for j in keep):
            continue
        keep.append(i)
        if len(keep) >= max_keep:
            break
    return keep


def bridge_score_between(support, a, b, thickness=7):
    H, W = support.shape[:2]
    ca = center(a)
    cb = center(b)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(ca[0])), int(round(ca[1]))),
        (int(round(cb[0])), int(round(cb[1]))),
        1,
        thickness=thickness,
    )

    k = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)

    ax1, ay1, ax2, ay2 = clip_box(a)
    bx1, by1, bx2, by2 = clip_box(b)
    mask[ay1:ay2, ax1:ax2] = 0
    mask[by1:by2, bx1:bx2] = 0

    vals = support[mask > 0]
    if vals.size < 8:
        u = union_box(a, b)
        ux1, uy1, ux2, uy2 = clip_box(u)
        mask2 = np.zeros((H, W), dtype=np.uint8)
        mask2[uy1:uy2, ux1:ux2] = 1
        mask2[ay1:ay2, ax1:ax2] = 0
        mask2[by1:by2, bx1:bx2] = 0
        vals = support[mask2 > 0]

    if vals.size == 0:
        return 0.0

    mean_v = float(vals.mean())
    q80_v = float(np.quantile(vals, 0.80))
    max_v = float(vals.max())
    return 0.45 * mean_v + 0.45 * q80_v + 0.10 * max_v


def group_label(group, cand_infos):
    best_i = None
    best_s = -1.0
    for idx in group["idxs"]:
        if 0 <= idx < len(cand_infos):
            s = float(cand_infos[idx].get("loc_score", 0.0)) + 0.15 * float(cand_infos[idx].get("cal_score", 0.0))
            if s > best_s:
                best_s = s
                best_i = idx
    if best_i is None:
        return "defect", 0.01
    return cand_infos[best_i]["cal_pred"], float(cand_infos[best_i]["cal_score"])


def group_mech_min(group, cand_infos):
    vals = []
    for idx in group["idxs"]:
        if 0 <= idx < len(cand_infos):
            vals.append(float(cand_infos[idx].get("mechanism_existence", 0.0)))
    return min(vals) if vals else 0.0


def should_group(g1, g2, cand_infos, support,
                 overlap_iou_thr=0.08,
                 overlap_small_thr=0.22,
                 overlap_compact_thr=0.38,
                 bridge_max_gap=22,
                 bridge_score_thr=0.40,
                 bridge_min_mech=0.40,
                 bridge_compact_thr=0.12,
                 defect_artifact_extra_thr=0.08):
    a, b = g1["box"], g2["box"]

    if inter_area(a, b) > 0:
        cp = compactness(a, b)
        if cp >= overlap_compact_thr:
            if iou(a, b) >= overlap_iou_thr or overlap_small(a, b) >= overlap_small_thr:
                return True

    gap = gap_distance(a, b)
    if gap > bridge_max_gap:
        return False

    la, _ = group_label(g1, cand_infos)
    lb, _ = group_label(g2, cand_infos)

    allowed = la == lb
    if {la, lb} == {"defect", "artifact"}:
        allowed = True
    if not allowed:
        return False

    if group_mech_min(g1, cand_infos) < bridge_min_mech or group_mech_min(g2, cand_infos) < bridge_min_mech:
        return False

    if compactness(a, b) < bridge_compact_thr:
        return False

    bs = bridge_score_between(support, a, b, thickness=7)
    need = bridge_score_thr
    if {la, lb} == {"defect", "artifact"}:
        need += defect_artifact_extra_thr

    return bs >= need


def bridge_group(refined_items, cand_infos, support):
    groups = []
    for item in refined_items:
        b = clip_box(item["box"])
        if area(b) <= 0:
            continue
        groups.append({
            "box": b,
            "idxs": [int(item["candidate_index"])],
        })

    changed = True
    while changed:
        changed = False
        used = [False] * len(groups)
        out = []

        for i in range(len(groups)):
            if used[i]:
                continue
            cur = groups[i]
            used[i] = True

            again = True
            while again:
                again = False
                for j in range(len(groups)):
                    if used[j]:
                        continue
                    if should_group(cur, groups[j], cand_infos, support):
                        cur = {
                            "box": union_box(cur["box"], groups[j]["box"]),
                            "idxs": cur["idxs"] + groups[j]["idxs"],
                        }
                        used[j] = True
                        changed = True
                        again = True
            out.append(cur)
        groups = out

    return groups


def map_box_256_to_original_norm(b, ow, oh):
    x1, y1, x2, y2 = b
    x1 = x1 / 256.0 * ow
    x2 = x2 / 256.0 * ow
    y1 = y1 / 256.0 * oh
    y2 = y2 / 256.0 * oh

    x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], ow, oh)
    cx = ((x1 + x2) / 2.0) / ow
    cy = ((y1 + y2) / 2.0) / oh
    bw = (x2 - x1) / ow
    bh = (y2 - y1) / oh
    return cx, cy, bw, bh


def draw_vis(gray256, groups, cand_infos, out_path):
    canvas = cv2.cvtColor(gray256, cv2.COLOR_GRAY2BGR)
    for gi, g in enumerate(groups):
        label, score = group_label(g, cand_infos)
        x1, y1, x2, y2 = clip_box(g["box"])
        color = (0, 255, 0) if label == "defect" else ((255, 120, 0) if label == "oil" else (0, 180, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{label} {score:.2f}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--proposer-run", default="./runs_v24_single_teacher_student_20260428_175456/proposer")
    ap.add_argument("--classifier-run", default="./runs_v31_full_classifier_mechanism")
    ap.add_argument("--calib", default="./runs_v30_mechanism_calibrator/v30_mechanism_calibrator.pt")
    ap.add_argument("--output-mode", choices=["merged", "subboxes"], default="merged")
    ap.add_argument("--select-thr", type=float, default=0.40)
    ap.add_argument("--max-keep", type=int, default=4)
    ap.add_argument("--shrink-q", type=float, default=0.72)
    ap.add_argument("--conf-thr", type=float, default=0.01)

    # absolute gates for negative-image rejection
    # These gates prevent per-image normalization from forcing one weak background candidate to survive.
    ap.add_argument("--abs-loc-thr", type=float, default=0.55)
    ap.add_argument("--min-box-score", type=float, default=0.08)
    ap.add_argument("--min-raw-abnormal", type=float, default=0.20)
    ap.add_argument("--min-mech-existence", type=float, default=0.20)
    args = ap.parse_args()

    image_dir = Path(args.images)
    outdir = Path(args.outdir)
    pred_dir = outdir / "pred_labels"
    vis_dir = outdir / "vis"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        img_paths.extend(sorted(image_dir.glob(ext)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)
    print("[IMAGES]", len(img_paths))

    proposer = SingleImageProposer(base=32).to(device)
    pk = torch.load(Path(args.proposer_run) / "best.pt", map_location=device, weights_only=False)
    proposer.load_state_dict(pk["model"])
    proposer.eval()

    classifier = V23ContextResidual(num_classes=3, base=48).to(device)
    ck = torch.load(Path(args.classifier_run) / "best.pt", map_location=device, weights_only=False)
    classifier.load_state_dict(ck["model"], strict=True)
    classifier.eval()

    pack = torch.load(args.calib, map_location=device, weights_only=False)
    calib = MechanismCalibrator(in_dim=pack["feature_dim"]).to(device)
    calib.load_state_dict(pack["model"])
    calib.eval()
    mean, std = pack["mean"], pack["std"]

    all_records = []

    for idx, img_path in enumerate(img_paths):
        orig = read_gray(img_path)
        if orig is None:
            continue

        oh, ow = orig.shape[:2]
        obs = cv2.resize(orig, (256, 256), interpolation=cv2.INTER_AREA)

        local_map, bg, res = make_local_prob(proposer, device, obs)
        global_map = make_global_map(obs, bg)
        base_map = norm01(0.60 * local_map + 0.40 * global_map)
        base_map = cv2.GaussianBlur(base_map, (0, 0), sigmaX=0.8, sigmaY=0.8)

        boxes, box_scores = extract_boxes_from_prob(
            base_map,
            topk=10,
            high_thr=0.34,
            low_thr=0.16,
            min_area=6,
            pad=20,
            max_area_ratio=0.35,
        )

        boxes = [clip_box(b) for b in boxes]
        box_scores = [float(s) for s in box_scores]

        cand_infos = []
        raw_ab, mech_ex = [], []

        for bi, b in enumerate(boxes):
            crop = crop_resize(obs, b, 256)
            raw = v25_scores(classifier, device, crop)
            feats = mechanism_features(crop, None)
            probs, mech = calibrate(calib, device, mean, std, raw, feats)

            pred_i = int(np.argmax(probs))
            info = {
                "box": b,
                "box_score": box_scores[bi] if bi < len(box_scores) else 0.5,
                "raw_scores": raw.tolist(),
                "raw_abnormal": float(np.max(raw)),
                "mechanism_probs": mech.tolist(),
                "mechanism_existence": float(np.max(mech)),
                "calibrated_probs": probs.tolist(),
                "cal_pred": CLASSES[pred_i],
                "cal_score": float(probs[pred_i]),
            }
            cand_infos.append(info)
            raw_ab.append(info["raw_abnormal"])
            mech_ex.append(info["mechanism_existence"])

        box_norm = normalize_list(box_scores)
        raw_norm = normalize_list(raw_ab)
        mech_norm = normalize_list(mech_ex)

        loc_scores = []
        for i, ci in enumerate(cand_infos):
            # relative score: used only for within-image ranking
            bs = float(box_norm[i]) if i < len(box_norm) else 0.5
            ra = float(raw_norm[i]) if i < len(raw_norm) else 0.5
            me = float(mech_norm[i]) if i < len(mech_norm) else 0.5
            loc_rel = 0.45 * bs + 0.25 * ra + 0.30 * me

            # absolute score: used for negative-image rejection
            box_abs = float(ci.get("box_score", 0.0))
            raw_abs = float(ci.get("raw_abnormal", 0.0))
            mech_abs = float(ci.get("mechanism_existence", 0.0))
            loc_abs = 0.45 * box_abs + 0.25 * raw_abs + 0.30 * mech_abs

            ci["loc_score"] = float(loc_rel)
            ci["loc_abs_score"] = float(loc_abs)
            loc_scores.append(float(loc_rel))

        selected = select_by_locator_score(
            boxes, loc_scores,
            select_thr=args.select_thr,
            max_keep=args.max_keep,
            nms_thr=0.45,
        )

        # absolute abnormality gate:
        # A candidate must be strong in absolute evidence, not merely the best candidate within this image.
        selected = [
            i for i in selected
            if cand_infos[i].get("loc_abs_score", 0.0) >= args.abs_loc_thr
            and cand_infos[i].get("box_score", 0.0) >= args.min_box_score
            and (
                cand_infos[i].get("raw_abnormal", 0.0) >= args.min_raw_abnormal
                or cand_infos[i].get("mechanism_existence", 0.0) >= args.min_mech_existence
            )
        ]

        refined = []
        for si in selected:
            b = boxes[si]
            crop = crop_box(obs, b)
            crop_rs = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            support_local = support_map_from_classifier(classifier, device, crop_rs)
            local_rb = refine_box_largest_cc(crop_rs, support_local, q=args.shrink_q)

            rb = map_local_box_to_parent(b, local_rb, crop.shape)
            rb = clip_box(rb)
            refined.append({"candidate_index": int(si), "box": rb})
            cand_infos[si]["refined_box"] = rb

        support_global = support_map_from_classifier(classifier, device, obs)

        if args.output_mode == "merged":
            groups = bridge_group(refined, cand_infos, support_global)
            outputs = []
            for g in groups:
                label, score = group_label(g, cand_infos)
                conf = float(score)
                # group-level absolute gate
                member_abs = [
                    cand_infos[m].get("loc_abs_score", 0.0)
                    for m in g["idxs"]
                    if 0 <= m < len(cand_infos)
                ]
                if conf < args.conf_thr:
                    continue
                if member_abs and max(member_abs) < args.abs_loc_thr:
                    continue
                outputs.append({
                    "box": clip_box(g["box"]),
                    "label": label,
                    "score": conf,
                    "members": g["idxs"],
                })
        else:
            outputs = []
            for item in refined:
                ci = cand_infos[item["candidate_index"]]
                conf = float(ci["cal_score"])
                if conf < args.conf_thr:
                    continue
                if ci.get("loc_abs_score", 0.0) < args.abs_loc_thr:
                    continue
                outputs.append({
                    "box": clip_box(item["box"]),
                    "label": ci["cal_pred"],
                    "score": conf,
                    "members": [item["candidate_index"]],
                })

        # write YOLO bbox prediction labels normalized to original image size
        lines = []
        for o in outputs:
            cls = CLS_TO_ID.get(o["label"], 0)
            cx, cy, bw, bh = map_box_256_to_original_norm(o["box"], ow, oh)
            score = float(o["score"])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.6f}")

        (pred_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        draw_vis(obs, [{"box": o["box"], "idxs": o["members"]} for o in outputs], cand_infos, vis_dir / f"{img_path.stem}.jpg")

        all_records.append({
            "image": str(img_path),
            "orig_size": [ow, oh],
            "boxes": boxes,
            "selected": selected,
            "refined": refined,
            "cand_infos": cand_infos,
            "outputs": outputs,
            "pred_label": str(pred_dir / f"{img_path.stem}.txt"),
        })

        print(f"[{idx+1}/{len(img_paths)}] {img_path.name} cand={len(boxes)} sel={len(selected)} out={len(outputs)}", flush=True)

    (outdir / "records.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]", outdir)
    print("[PRED_LABELS]", pred_dir)


if __name__ == "__main__":
    main()
