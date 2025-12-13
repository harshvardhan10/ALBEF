"""
FROC evaluation for VinDr-CXR CAM heatmaps with quadrant-based matching.

- FP rate = total false positives over entire dataset / number of test images (FP/image)
- Sensitivity reported at FP/image = {0.10, 0.25, 0.50}
- A TP occurs if the CENTER of the predicted box lies in the SAME QUADRANT
  as the CENTER of a GT bounding box (within the same image and label).
- One-to-one matching: each GT box can be matched at most once.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from src import (
    load_meta,
    scale_box_to_256
)

# ---------------------------
# Quadrant matching helpers
# ---------------------------

def quadrant_of_point(x: float, y: float, center: float = 128.0) -> int:
    """
    Quadrants w.r.t. image center (128,128) for 256x256.
    Returns 0=TL, 1=TR, 2=BL, 3=BR.
    """
    if x < center and y < center:
        return 0
    if x >= center and y < center:
        return 1
    if x < center and y >= center:
        return 2
    return 3


def box_center_xy(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box_xyxy
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def quadrant_match(pred_box: Tuple[float, float, float, float],
                   gt_box: Tuple[float, float, float, float]) -> bool:
    px, py = box_center_xy(pred_box)
    gx, gy = box_center_xy(gt_box)
    return quadrant_of_point(px, py) == quadrant_of_point(gx, gy)


# ---------------------------
# CAM -> predicted boxes
# ---------------------------

def normalize_cam_01(cam: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure CAM is within [0,1].
    """
    cam = cam.astype(np.float32)
    mn = float(cam.min())
    mx = float(cam.max())
    if mx - mn < eps:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - mn) / (mx - mn + eps)
    cam = np.clip(cam, 0.0, 1.0)
    return cam


def cam_to_boxes(cam: np.ndarray,
                 thresholds: np.ndarray,
                 min_area: int = 10,
                 connectivity: int = 2,
                 top_k: int = 1) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    Convert one CAM (256x256) into a pool of candidate predicted boxes by:
      - thresholding CAM >= t for multiple thresholds
      - connected components on the binary mask
      - bounding box per component
      - score per box = max CAM inside component
      - deduplicate identical boxes (keep best score)
      - keep top_k boxes per image per label (by score) to control FP explosion

    Returns list of (box_xyxy, score).
    """
    cam = normalize_cam_01(cam)

    # dedupe identical boxes across thresholds, keep max score
    best_by_box: Dict[Tuple[int, int, int, int], float] = {}

    # connectivity: 2 -> 8-neighborhood, 1 -> 4-neighborhood
    struct = ndi.generate_binary_structure(rank=2, connectivity=connectivity)

    for t in thresholds:
        mask = cam >= float(t)
        if not mask.any():
            continue

        labeled, n = ndi.label(mask, structure=struct)
        if n == 0:
            continue

        for comp_id in range(1, n + 1):
            ys, xs = np.where(labeled == comp_id)
            if ys.size == 0:
                continue
            if ys.size < min_area:
                continue

            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            score = float(cam[ys, xs].max())
            box = (x0, y0, x1, y1)

            prev = best_by_box.get(box)
            if prev is None or score > prev:
                best_by_box[box] = score

    out = [(box, score) for box, score in best_by_box.items()]
    out.sort(key=lambda x: -x[1])  # sort by score descending
    return out[:top_k]


# ---------------------------
# Loading GT and predictions
# ---------------------------

def load_image_ids(labels_csv: Path) -> List[str]:
    """
    Use image_labels_test.csv to define the test image set and N for FP/image.
    """
    df = pd.read_csv(labels_csv)
    id_col = df.columns[0]
    return df[id_col].astype(str).tolist()


def load_all_label_names_from_labels_csv(labels_csv: Path) -> List[str]:
    """
    Load all label columns (excluding the first ID column) from image_labels_test.csv.
    These are the labels you likely generated heatmaps for.
    """
    df = pd.read_csv(labels_csv)
    return [c for c in df.columns[1:]]


def load_gt_boxes_scaled(
    ann_csv: Path,
    meta: Dict[str, Tuple[int, int]],
    target_size: int = 256,
) -> List[Dict]:
    """
    Load GT boxes from annotations_test.csv and scale to 256x256.
    Returns list of:
      {"image_id": str, "label": str, "box": (x0,y0,x1,y1)}
    """
    df = pd.read_csv(ann_csv)

    out = []
    for _, r in df.iterrows():
        label = str(r["class_name"])
        if label.lower() == "no finding":
            continue
        image_id = str(r["image_id"])
        if image_id not in meta:
            print(f"[WARN] image_id={image_id} not found in meta CSV; skipping.")
            continue

        orig_h, orig_w = meta[image_id]
        box = scale_box_to_256(r, orig_h, orig_w, target_size=target_size)
        out.append({"image_id": image_id, "label": label, "box": box})
    return out


def build_predictions_from_heatmaps(
    heatmaps_dir: Path,
    image_ids: List[str],
    labels: List[str],
    thresholds: np.ndarray,
    min_area: int = 10,
    connectivity: int = 2,
    top_k: int = 1,
) -> List[Dict]:
    """
    For each image_id, load <heatmaps_dir>/<image_id>.pt and generate candidate
    predictions for each label in 'labels'.

    Returns list of dict:
      {"image_id": str, "label": str, "box": (x0,y0,x1,y1), "score": float}
    """
    preds = []
    missing = 0

    for image_id in image_ids:
        p = heatmaps_dir / f"{image_id}.pt"
        if not p.exists():
            missing += 1
            continue

        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Heatmap file {p} must be a dict label->heatmap")

        # For robustness: iterate only labels that exist in this file
        # but still restrict to the global label list
        for label in labels:
            if label not in obj:
                continue

            cam_t = obj[label]
            if isinstance(cam_t, torch.Tensor):
                cam = cam_t.detach().cpu().numpy()
            else:
                cam = np.asarray(cam_t)

            boxes = cam_to_boxes(
                cam=cam,
                thresholds=thresholds,
                min_area=min_area,
                connectivity=connectivity,
                top_k=top_k
            )

            for box, score in boxes:
                preds.append({
                    "image_id": image_id,
                    "label": label,
                    "box": box,
                    "score": float(score),
                })

    if missing > 0:
        print(f"[WARN] Missing heatmap files: {missing}/{len(image_ids)}")

    return preds


# ---------------------------
# FROC evaluation
# ---------------------------

def evaluate_froc_for_label(
    predictions: List[Dict],
    gt_boxes: List[Dict],
    label: str,
    num_images: int,
) -> Tuple[List[Tuple[float, float]], Dict[float, float]]:
    """
    Standard FROC sweep:
      - Sort predictions by score desc
      - One-to-one matching to GT (quadrant rule)
      - Track FP/image and sensitivity after each prediction

    Returns:
      curve: list of (fp_per_image, sensitivity)
      sens_at: dict {0.10, 0.25, 0.50} -> sensitivity
    """
    preds = [p for p in predictions if p["label"] == label]
    gts = [g for g in gt_boxes if g["label"] == label]

    if len(gts) == 0:
        return [], {0.10: 0.0, 0.25: 0.0, 0.50: 0.0}

    # Index GT by image
    gt_by_image: Dict[str, List[Dict]] = {}
    for g in gts:
        gt_by_image.setdefault(g["image_id"], []).append(g)

    preds = sorted(preds, key=lambda x: -x["score"])

    matched_gt = set()
    tp = 0
    fp = 0
    curve: List[Tuple[float, float]] = []

    for pred in preds:
        img_id = pred["image_id"]
        pbox = pred["box"]

        matched = False
        for g in gt_by_image.get(img_id, []):
            gt_key = (img_id, label, tuple(g["box"]))
            if gt_key in matched_gt:
                continue
            if quadrant_match(pbox, g["box"]):
                matched_gt.add(gt_key)
                matched = True
                break

        if matched:
            tp += 1
        else:
            fp += 1

        fp_per_image = fp / float(num_images)
        sensitivity = tp / float(len(gts))
        curve.append((fp_per_image, sensitivity))

    targets = [0.10, 0.25, 0.50]
    sens_at = {}
    for t in targets:
        valid = [s for f, s in curve if f <= t]
        sens_at[t] = max(valid) if valid else 0.0

    return curve, sens_at


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmaps_dir", type=str, required=True,
                        help="Directory containing <image_id>.pt heatmap dicts")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="image_labels_test.csv (defines test image_ids and label names)")
    parser.add_argument("--ann_csv", type=str, required=True,
                        help="annotations_test.csv (GT boxes in original coords)")
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="test_meta.csv with columns: image_id, dim0, dim1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=256)

    # Threshold policy (20 thresholds by default)
    parser.add_argument("--n_thr_low", type=int, default=6)
    parser.add_argument("--n_thr_high", type=int, default=14)
    parser.add_argument("--thr_low_min", type=float, default=0.10)
    parser.add_argument("--thr_low_max", type=float, default=0.30)
    parser.add_argument("--thr_high_min", type=float, default=0.35)
    parser.add_argument("--thr_high_max", type=float, default=0.95)

    # Component filtering
    parser.add_argument("--min_area", type=int, default=10,
                        help="Minimum component area in pixels to be considered a detection")
    parser.add_argument("--connectivity", type=int, default=2, choices=[1, 2],
                        help="2=8-connected, 1=4-connected")

    # Optional: limit images (debug)
    parser.add_argument("--max_images", type=int, default=-1)

    # Select top_k predicted boxes to keep (per image per label)
    parser.add_argument("--top_k", type=int, default=1)

    args = parser.parse_args()

    heatmaps_dir = Path(args.heatmaps_dir)
    labels_csv = Path(args.labels_csv)
    ann_csv = Path(args.ann_csv)
    meta_csv = Path(args.meta_csv)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get ALL label names from image_labels_test.csv
    labels = load_all_label_names_from_labels_csv(labels_csv)
    if len(labels) == 0:
        raise RuntimeError("No label columns found in labels_csv.")
    print(f"[Labels] Evaluating {len(labels)} labels from: {labels_csv}")

    # Thresholds (default 20)
    thresholds = np.concatenate([
        np.linspace(args.thr_low_min, args.thr_low_max, args.n_thr_low),
        np.linspace(args.thr_high_min, args.thr_high_max, args.n_thr_high),
    ]).astype(np.float32)

    # Define test set (N images) from image_labels_test.csv
    image_ids = load_image_ids(labels_csv)
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    num_images = len(image_ids)
    print(f"[Data] N images for FP/image normalization: {num_images}")

    # Load meta and GT boxes
    meta = load_meta(meta_csv)
    gt_boxes = load_gt_boxes_scaled(
        ann_csv=ann_csv,
        meta=meta,
        target_size=args.target_size,
    )
    print(f"[GT] Loaded {len(gt_boxes)} GT boxes total (all labels present in annotations).")

    # Build predictions from CAM heatmaps for ALL labels
    predictions = build_predictions_from_heatmaps(
        heatmaps_dir=heatmaps_dir,
        image_ids=image_ids,
        labels=labels,
        thresholds=thresholds,
        min_area=args.min_area,
        connectivity=args.connectivity,
        top_k=args.top_k
    )
    print(f"[Pred] Generated {len(predictions)} candidate predictions total.")

    # Evaluate per label
    rows = []
    for label in labels:
        curve, sens_at = evaluate_froc_for_label(
            predictions=predictions,
            gt_boxes=gt_boxes,
            label=label,
            num_images=num_images,
        )

        n_gt = sum(1 for g in gt_boxes if g["label"] == label)
        n_pred = sum(1 for p in predictions if p["label"] == label)

        # Save per-label curve for plotting
        curve_path = output_dir / f"froc_curve_{label.replace(' ', '_')}.csv"
        pd.DataFrame(curve, columns=["fp_per_image", "sensitivity"]).to_csv(curve_path, index=False)

        rows.append({
            "label": label,
            "sens@0.10": sens_at[0.10],
            "sens@0.25": sens_at[0.25],
            "sens@0.50": sens_at[0.50],
            "num_gt_boxes": n_gt,
            "num_preds": n_pred,
            "curve_csv": str(curve_path),
        })

        print(f"[FROC] {label:30s}  GT={n_gt:4d}  Pred={n_pred:6d}  "
              f"S@0.10={sens_at[0.10]:.4f}  S@0.25={sens_at[0.25]:.4f}  S@0.50={sens_at[0.50]:.4f}")

    # Save summary
    out_df = pd.DataFrame(rows).sort_values(by="label")
    out_path = output_dir / "froc_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[Saved] Summary: {out_path}")


if __name__ == "__main__":
    main()
