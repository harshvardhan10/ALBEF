import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# Data structure
# ============================================================

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    image_id: str


# ============================================================
# General utilities
# ============================================================

def safe_filename(name: str) -> str:
    name = str(name).replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "label"


def minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def resize_map(arr: np.ndarray, out_hw: Tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=interpolation)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# CSV loading
# ============================================================

def load_annotations(annotations_csv: Path) -> pd.DataFrame:
    ann = pd.read_csv(annotations_csv)
    ann.columns = [c.strip() for c in ann.columns]

    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"annotations_csv missing columns: {sorted(missing)}")

    ann["image_id"] = ann["image_id"].astype(str)
    ann["class_name"] = ann["class_name"].astype(str)

    return ann


def load_meta(meta_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    meta.columns = [c.strip() for c in meta.columns]

    width_col = next((c for c in ["width", "W", "image_width", "dim1"] if c in meta.columns), None)
    height_col = next((c for c in ["height", "H", "image_height", "dim0"] if c in meta.columns), None)

    if width_col is None or height_col is None or "image_id" not in meta.columns:
        raise ValueError(
            "meta_csv must contain image_id and width/height columns. "
            f"Found columns: {list(meta.columns)}"
        )

    meta = meta.rename(columns={width_col: "width", height_col: "height"})
    meta["image_id"] = meta["image_id"].astype(str)

    return meta[["image_id", "width", "height"]].copy()


def load_heatmap_index(heatmaps_dir: Path) -> pd.DataFrame:
    idx_csv = heatmaps_dir / "crossattn_gradcam_index.csv"
    if not idx_csv.exists():
        raise FileNotFoundError(f"Missing heatmap index CSV: {idx_csv}")

    idx = pd.read_csv(idx_csv)
    idx.columns = [c.strip() for c in idx.columns]

    img_col = next((c for c in ["image_id", "dicom_id", "id"] if c in idx.columns), None)
    path_col = next((c for c in ["heatmap_path", "path", "file", "pt_path", "npz_path"] if c in idx.columns), None)

    if img_col is None or path_col is None:
        raise ValueError(
            "crossattn_gradcam_index.csv must contain image_id-like and path-like columns. "
            f"Found columns: {list(idx.columns)}"
        )

    idx = idx.rename(columns={img_col: "image_id", path_col: "heatmap_path"})
    idx["image_id"] = idx["image_id"].astype(str)

    def _resolve(p: str) -> str:
        p = str(p)

        if os.path.isabs(p):
            return p

        p_from_cwd = Path(p)
        if p_from_cwd.exists():
            return str(p_from_cwd.resolve())

        p_from_heatmaps_dir = heatmaps_dir / p
        if p_from_heatmaps_dir.exists():
            return str(p_from_heatmaps_dir.resolve())

        return str(p_from_heatmaps_dir.resolve())

    idx["heatmap_path"] = idx["heatmap_path"].astype(str).map(_resolve)

    return idx[["image_id", "heatmap_path"]].drop_duplicates().copy()


# ============================================================
# Heatmap loading
# ============================================================

def load_heatmap_file(path: str):
    path = str(path)

    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "heatmaps" in data.files:
            obj = data["heatmaps"]
            return obj.item() if obj.dtype == object and obj.size == 1 else obj
        if len(data.files) == 1:
            obj = data[data.files[0]]
            return obj.item() if obj.dtype == object and obj.size == 1 else obj
        return {k: data[k] for k in data.files}

    if path.endswith(".npy"):
        obj = np.load(path, allow_pickle=True)
        return obj.item() if obj.dtype == object and obj.size == 1 else obj

    raise ValueError(f"Unsupported heatmap file extension: {path}")


def extract_cam_from_object(heatmaps_obj, label: str, cam_key: str) -> np.ndarray:
    if label not in heatmaps_obj:
        raise KeyError(
            f"Label '{label}' not found in heatmap object. "
            f"Available keys: {list(heatmaps_obj.keys())}"
        )

    label_obj = heatmaps_obj[label]

    if cam_key not in label_obj:
        raise KeyError(
            f"cam_key '{cam_key}' not found for label '{label}'. "
            f"Available keys: {list(label_obj.keys())}"
        )

    cam = label_obj[cam_key]

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()

    cam = np.asarray(cam, dtype=np.float32)
    cam = np.squeeze(cam)

    if cam.ndim != 2:
        raise ValueError(f"Expected 2D CAM for label={label}, key={cam_key}, got shape={cam.shape}")

    return cam


def load_heatmap_for_image(heatmap_path: str, label: str, cam_key: str) -> np.ndarray:
    obj = load_heatmap_file(heatmap_path)
    return extract_cam_from_object(obj, label=label, cam_key=cam_key)


# ============================================================
# GT boxes
# ============================================================

def get_gt_boxes(ann_df: pd.DataFrame, image_id: str, label: str) -> List[Box]:
    sub = ann_df[(ann_df["image_id"] == image_id) & (ann_df["class_name"] == label)]

    boxes: List[Box] = []
    for _, r in sub.iterrows():
        boxes.append(
            Box(
                x1=float(r.x_min),
                y1=float(r.y_min),
                x2=float(r.x_max),
                y2=float(r.y_max),
                score=1.0,
                label=label,
                image_id=image_id,
            )
        )

    return boxes


def boxes_to_df(boxes: Sequence[Box]) -> pd.DataFrame:
    rows = []
    for b in boxes:
        rows.append(
            {
                "image_id": b.image_id,
                "class_name": b.label,
                "x_min": b.x1,
                "y_min": b.y1,
                "x_max": b.x2,
                "y_max": b.y2,
                "score": b.score,
            }
        )

    columns = ["image_id", "class_name", "x_min", "y_min", "x_max", "y_max", "score"]
    return pd.DataFrame(rows, columns=columns)


# ============================================================
# CAM -> boxes
# ============================================================

def heatmap_to_boxes(
    heatmap: np.ndarray,
    image_id: str,
    label: str,
    threshold: float,
    min_box_area_frac: float,
    score_mode: str = "max",
    connectivity: int = 8,
) -> List[Box]:
    heatmap = minmax_norm(heatmap)
    binary_map = (heatmap >= float(threshold)).astype(np.uint8)

    h, w = binary_map.shape
    min_area = max(1, int(round(float(min_box_area_frac) * h * w)))

    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        binary_map.astype(np.uint8),
        connectivity=connectivity,
    )

    boxes: List[Box] = []

    for comp_id in range(1, num_labels):
        x, y, bw, bh, area = stats[comp_id]

        if area < min_area:
            continue

        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

        comp_scores = heatmap[y1:y2, x1:x2]

        if score_mode == "max":
            score = float(comp_scores.max())
        elif score_mode == "mean":
            score = float(comp_scores.mean())
        elif score_mode == "area_mean":
            score = float(comp_scores.mean() * area)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

        boxes.append(
            Box(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=score,
                label=label,
                image_id=image_id,
            )
        )

    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes


# ============================================================
# Matching
# ============================================================

def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def box_center_xy(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def quadrant_of_point(x: float, y: float, center_x: float, center_y: float) -> int:
    if x < center_x and y < center_y:
        return 0
    if x >= center_x and y < center_y:
        return 1
    if x < center_x and y >= center_y:
        return 2
    return 3


def quadrant_match(
    pred_box: Tuple[float, float, float, float],
    gt_box: Tuple[float, float, float, float],
    image_w: float,
    image_h: float,
) -> bool:
    px, py = box_center_xy(pred_box)
    gx, gy = box_center_xy(gt_box)

    center_x = image_w / 2.0
    center_y = image_h / 2.0

    return quadrant_of_point(px, py, center_x, center_y) == quadrant_of_point(gx, gy, center_x, center_y)


def boxes_match(
    pred_box: Tuple[float, float, float, float],
    gt_box: Tuple[float, float, float, float],
    image_w: float,
    image_h: float,
    match_mode: str,
    iou_threshold: float,
) -> bool:
    if match_mode == "quadrant":
        return quadrant_match(pred_box, gt_box, image_w=image_w, image_h=image_h)

    if match_mode == "iou":
        return box_iou(pred_box, gt_box) >= float(iou_threshold)

    raise ValueError(f"Unknown match_mode: {match_mode}")


# ============================================================
# FROC
# ============================================================

def evaluate_froc_for_label(
    predictions: Sequence[Box],
    gt_boxes: Sequence[Box],
    label: str,
    num_images: int,
    image_size_lookup: Dict[str, Tuple[int, int]],
    match_mode: str,
    iou_threshold: float,
    targets: Sequence[float],
) -> Tuple[pd.DataFrame, Dict[float, float]]:
    preds = [p for p in predictions if p.label == label]
    gts = [g for g in gt_boxes if g.label == label]

    if len(gts) == 0:
        empty_curve = pd.DataFrame(columns=["fp_per_image", "sensitivity", "threshold_score", "tp", "fp"])
        return empty_curve, {float(t): 0.0 for t in targets}

    gt_by_image: Dict[str, List[Box]] = {}
    for g in gts:
        gt_by_image.setdefault(g.image_id, []).append(g)

    preds = sorted(preds, key=lambda x: -x.score)

    matched_gt = set()
    tp = 0
    fp = 0
    rows = []

    for pred in preds:
        img_id = pred.image_id

        if img_id not in image_size_lookup:
            raise KeyError(f"Missing image size for image_id={img_id}")

        image_w, image_h = image_size_lookup[img_id]

        pbox = (pred.x1, pred.y1, pred.x2, pred.y2)

        matched = False

        for g in gt_by_image.get(img_id, []):
            gt_key = (img_id, label, g.x1, g.y1, g.x2, g.y2)

            if gt_key in matched_gt:
                continue

            gbox = (g.x1, g.y1, g.x2, g.y2)

            if boxes_match(
                pred_box=pbox,
                gt_box=gbox,
                image_w=image_w,
                image_h=image_h,
                match_mode=match_mode,
                iou_threshold=iou_threshold,
            ):
                matched_gt.add(gt_key)
                matched = True
                break

        if matched:
            tp += 1
        else:
            fp += 1

        fp_per_image = fp / float(num_images)
        sensitivity = tp / float(len(gts))

        rows.append(
            {
                "fp_per_image": fp_per_image,
                "sensitivity": sensitivity,
                "threshold_score": pred.score,
                "tp": tp,
                "fp": fp,
            }
        )

    curve_df = pd.DataFrame(rows)

    sens_at = {}
    for t in targets:
        valid = curve_df[curve_df["fp_per_image"] <= float(t)]
        sens_at[float(t)] = float(valid["sensitivity"].max()) if len(valid) > 0 else 0.0

    return curve_df, sens_at


def evaluate_froc(
    label: str,
    predictions: Sequence[Box],
    gt_boxes: Sequence[Box],
    num_images: int,
    image_size_lookup: Dict[str, Tuple[int, int]],
    output_dir: Path,
    prefix: str,
    match_mode: str,
    iou_threshold: float,
    targets: Sequence[float],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_df, sens_at = evaluate_froc_for_label(
        predictions=predictions,
        gt_boxes=gt_boxes,
        label=label,
        num_images=num_images,
        image_size_lookup=image_size_lookup,
        match_mode=match_mode,
        iou_threshold=iou_threshold,
        targets=targets,
    )

    curve_path = output_dir / f"{prefix}_froc_curve_{safe_filename(label)}.csv"
    curve_df.to_csv(curve_path, index=False)

    n_gt = sum(1 for g in gt_boxes if g.label == label)
    n_pred = sum(1 for p in predictions if p.label == label)

    row = {
        "label": label,
        "num_images": int(num_images),
        "num_gt_boxes": int(n_gt),
        "num_preds": int(n_pred),
        "match_mode": match_mode,
        "iou_threshold": float(iou_threshold),
        "curve_csv": str(curve_path),
    }

    for t in targets:
        row[f"sens@{t:.2f}"] = sens_at[float(t)]

    summary_df = pd.DataFrame([row])
    summary_path = output_dir / f"{prefix}_froc_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    msg = (
        f"[FROC:{prefix}] {label} | "
        f"N={num_images} GT={n_gt} Pred={n_pred} "
    )
    msg += " ".join([f"S@{t:.2f}={sens_at[float(t)]:.4f}" for t in targets])
    print(msg, flush=True)

    return summary_df


# ============================================================
# Main
# ============================================================

def run_froc_from_heatmaps(
    heatmaps_dir: Path,
    annotations_csv: Path,
    meta_csv: Path,
    output_dir: Path,
    label: str,
    cam_key: str,
    threshold: float,
    min_box_area_frac: float,
    score_mode: str,
    match_mode: str,
    iou_threshold: float,
    targets: Sequence[float],
    max_images: Optional[int],
    prefix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("[Config] FROC from Grad-CAM heatmaps", flush=True)
    print(f"[Config] heatmaps_dir      = {heatmaps_dir}", flush=True)
    print(f"[Config] annotations_csv   = {annotations_csv}", flush=True)
    print(f"[Config] meta_csv          = {meta_csv}", flush=True)
    print(f"[Config] output_dir        = {output_dir}", flush=True)
    print(f"[Config] label             = {label}", flush=True)
    print(f"[Config] cam_key           = {cam_key}", flush=True)
    print(f"[Config] heatmap_threshold = {threshold}", flush=True)
    print(f"[Config] min_area_frac     = {min_box_area_frac}", flush=True)
    print(f"[Config] score_mode        = {score_mode}", flush=True)
    print(f"[Config] match_mode        = {match_mode}", flush=True)
    print(f"[Config] iou_threshold     = {iou_threshold}", flush=True)
    print(f"[Config] targets           = {targets}", flush=True)
    print("=" * 80, flush=True)

    ann = load_annotations(annotations_csv)
    meta = load_meta(meta_csv)
    hm_idx = load_heatmap_index(heatmaps_dir)

    ann_label = ann[ann["class_name"] == label].copy()

    full_df = hm_idx.merge(meta, on="image_id", how="left")
    full_df = full_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    missing_meta = full_df["width"].isna().sum() + full_df["height"].isna().sum()
    if missing_meta > 0:
        bad = full_df[full_df["width"].isna() | full_df["height"].isna()]["image_id"].head(10).tolist()
        raise RuntimeError(f"Missing meta width/height for some images. Examples: {bad}")

    if max_images is not None:
        full_df = full_df.iloc[:max_images].reset_index(drop=True)

    print(f"[Data] Number of evaluated images = {len(full_df)}", flush=True)
    print(f"[Data] Number of GT boxes for {label} = {len(ann_label)}", flush=True)

    all_pred_boxes: List[Box] = []
    all_gt_boxes: List[Box] = []
    diagnostics: List[Dict[str, object]] = []
    image_size_lookup: Dict[str, Tuple[int, int]] = {}

    for i, row in full_df.iterrows():
        image_id = str(row["image_id"])
        heatmap_path = str(row["heatmap_path"])

        width = int(row["width"])
        height = int(row["height"])

        image_size_lookup[image_id] = (width, height)

        cam = load_heatmap_for_image(
            heatmap_path=heatmap_path,
            label=label,
            cam_key=cam_key,
        )

        cam = minmax_norm(cam)
        cam_full = resize_map(cam, (height, width), interpolation=cv2.INTER_LINEAR)
        cam_full = minmax_norm(cam_full)

        pred_boxes = heatmap_to_boxes(
            heatmap=cam_full,
            image_id=image_id,
            label=label,
            threshold=threshold,
            min_box_area_frac=min_box_area_frac,
            score_mode=score_mode,
            connectivity=8,
        )

        gt_boxes = get_gt_boxes(ann_label, image_id=image_id, label=label)

        all_pred_boxes.extend(pred_boxes)
        all_gt_boxes.extend(gt_boxes)

        diagnostics.append(
            {
                "image_id": image_id,
                "heatmap_path": heatmap_path,
                "width": width,
                "height": height,
                "num_gt_boxes": len(gt_boxes),
                "num_pred_boxes": len(pred_boxes),
                "heatmap_max": float(cam_full.max()),
                "heatmap_mean": float(cam_full.mean()),
                "threshold": float(threshold),
            }
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(full_df):
            print(f"[Process] {i + 1}/{len(full_df)} images", flush=True)

    # Deduplicate GT boxes
    gt_df = boxes_to_df(all_gt_boxes).drop_duplicates()
    all_gt_boxes = [
        Box(
            x1=float(r.x_min),
            y1=float(r.y_min),
            x2=float(r.x_max),
            y2=float(r.y_max),
            score=1.0,
            label=str(r.class_name),
            image_id=str(r.image_id),
        )
        for _, r in gt_df.iterrows()
    ]

    predictions_csv = output_dir / f"{prefix}_predictions_{safe_filename(label)}.csv"
    gt_csv = output_dir / f"{prefix}_gt_boxes_{safe_filename(label)}.csv"
    diagnostics_csv = output_dir / f"{prefix}_diagnostics_{safe_filename(label)}.csv"

    boxes_to_df(all_pred_boxes).to_csv(predictions_csv, index=False)
    boxes_to_df(all_gt_boxes).to_csv(gt_csv, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

    print(f"[Output] Saved predictions: {predictions_csv}", flush=True)
    print(f"[Output] Saved GT boxes: {gt_csv}", flush=True)
    print(f"[Output] Saved diagnostics: {diagnostics_csv}", flush=True)

    num_images = len(full_df)

    froc_dir = output_dir / f"{prefix}_froc_{safe_filename(label)}"

    froc_summary = evaluate_froc(
        label=label,
        predictions=all_pred_boxes,
        gt_boxes=all_gt_boxes,
        num_images=num_images,
        image_size_lookup=image_size_lookup,
        output_dir=froc_dir,
        prefix=prefix,
        match_mode=match_mode,
        iou_threshold=iou_threshold,
        targets=targets,
    )

    summary_config = {
        "label": label,
        "num_images": int(num_images),
        "num_gt_boxes": int(len(all_gt_boxes)),
        "num_predictions": int(len(all_pred_boxes)),
        "heatmaps_dir": str(heatmaps_dir),
        "annotations_csv": str(annotations_csv),
        "meta_csv": str(meta_csv),
        "output_dir": str(output_dir),
        "cam_key": cam_key,
        "heatmap_threshold": float(threshold),
        "min_box_area_frac": float(min_box_area_frac),
        "score_mode": score_mode,
        "match_mode": match_mode,
        "iou_threshold": float(iou_threshold),
        "froc_targets": [float(x) for x in targets],
        "predictions_csv": str(predictions_csv),
        "gt_csv": str(gt_csv),
        "diagnostics_csv": str(diagnostics_csv),
    }

    with open(output_dir / f"{prefix}_summary_config.json", "w") as f:
        json.dump(summary_config, f, indent=2)

    print("[Done] FROC computation complete.", flush=True)
    print(froc_summary, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Cardiomegaly FROC from extracted ALBEF Grad-CAM heatmaps."
    )

    parser.add_argument("--heatmaps_dir", type=str, required=True)
    parser.add_argument("--annotations_csv", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--label", type=str, default="Cardiomegaly")

    parser.add_argument(
        "--cam_key",
        type=str,
        default="cam_vis_up",
        help="Which heatmap key to use from the .pt file. Recommended: cam_vis_up.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Heatmap threshold for connected components.",
    )

    parser.add_argument(
        "--min_box_area_frac",
        type=float,
        default=0.002,
        help="Minimum connected-component area as fraction of image area.",
    )

    parser.add_argument(
        "--score_mode",
        type=str,
        default="max",
        choices=["max", "mean", "area_mean"],
        help="Score assigned to each connected component.",
    )

    parser.add_argument(
        "--match_mode",
        type=str,
        default="quadrant",
        choices=["quadrant", "iou"],
        help="Matching rule. Use quadrant to reproduce your earlier evaluation.",
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.1,
        help="IoU threshold if --match_mode iou is used.",
    )

    parser.add_argument(
        "--froc_targets",
        type=str,
        default="0.10,0.25,0.50",
        help="Comma-separated FP/image points.",
    )

    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument(
        "--prefix",
        type=str,
        default="gradcam",
        help="Prefix for output files, e.g. A0, A1, A2, A3.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    targets = parse_float_list(args.froc_targets)

    run_froc_from_heatmaps(
        heatmaps_dir=Path(args.heatmaps_dir),
        annotations_csv=Path(args.annotations_csv),
        meta_csv=Path(args.meta_csv),
        output_dir=Path(args.output_dir),
        label=args.label,
        cam_key=args.cam_key,
        threshold=args.threshold,
        min_box_area_frac=args.min_box_area_frac,
        score_mode=args.score_mode,
        match_mode=args.match_mode,
        iou_threshold=args.iou_threshold,
        targets=targets,
        max_images=args.max_images,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()