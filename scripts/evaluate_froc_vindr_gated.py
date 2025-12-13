"""
FROC evaluation for VinDr-CXR CAM heatmaps with quadrant-based matching,
WITH zero-shot gating (per-image, per-label classification scores).

Key changes vs your existing pipeline
- Uses CAM_VIS/CAM_NORM stored in heatmap dicts (no extra min-max normalization by default).
- Generates predictions ONLY for labels that pass a classification gate per image:
    (A) keep top_k_labels per image, OR
    (B) keep labels with cls_score >= cls_thr, OR both.
- Prediction score = cls_score * cam_score (cam_score = max CAM inside component),
  so ranking across dataset is driven by both classification and localization.
- Still computes per-label FROC with your quadrant matching + one-to-one GT matching.

Inputs expected
- heatmaps_dir: contains <image_id>.pt with dict[label] -> either:
      - a 256x256 tensor (assumed cam_vis), OR
      - a dict with keys {"cam_vis", "cam_raw"} (we will pick cam_vis by default)
- labels_csv: image_labels_test.csv (defines image_ids + label list)
- ann_csv: annotations_test.csv (GT boxes)
- meta_csv: test_meta.csv (original dims for scaling)
- cls_scores_csv: per-image, per-label classification scores (created by your updated zero-shot script)
    columns: image_id, <label1>, <label2>, ...
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from src import load_meta
from evaluate_froc_vindr import (
    quadrant_match,
    safe_filename,
    load_image_ids,
    load_all_label_names_from_labels_csv,
    load_gt_boxes_scaled,
)


# ---------------------------
# Classification scores (per image, per label)
# ---------------------------

def load_cls_scores(cls_scores_csv: Path, labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Returns: cls_scores[image_id][label] = float
    Expects a CSV with columns: image_id, <label1>, <label2>, ...
    """
    df = pd.read_csv(cls_scores_csv)
    id_col = df.columns[0]
    missing = [lb for lb in labels if lb not in df.columns]
    if missing:
        raise ValueError(
            f"cls_scores_csv is missing {len(missing)} label columns. Example: {missing[:5]}"
        )

    scores: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        img_id = str(r[id_col])
        scores[img_id] = {lb: float(r[lb]) for lb in labels}
    return scores


def select_labels_for_image(
    cls_scores_img: Dict[str, float],
    labels: List[str],
    top_k_labels: int = 0,
    cls_thr: Optional[float] = None,
) -> List[str]:
    """
    Apply gating:
      - if top_k_labels > 0: keep only the top-K labels by cls score
      - if cls_thr is not None: additionally require score >= cls_thr
    """
    items = [(lb, cls_scores_img.get(lb, float("-inf"))) for lb in labels]
    items.sort(key=lambda x: -x[1])

    if top_k_labels and top_k_labels > 0:
        items = items[:top_k_labels]

    if cls_thr is not None:
        items = [(lb, s) for lb, s in items if s >= cls_thr]

    return [lb for lb, _ in items]


# ---------------------------
# CAM -> predicted boxes
# ---------------------------

def extract_cam_from_obj(obj_for_label, cam_key: str = "cam_vis") -> np.ndarray:
    """
    Handles two formats:
      A) obj[label] is a Tensor/ndarray shaped (256,256): assume this is cam_vis already
      B) obj[label] is a dict with keys like {"cam_raw", "cam_vis"}: pick cam_key
    """
    if isinstance(obj_for_label, dict):
        if cam_key not in obj_for_label:
            raise KeyError(f"Heatmap dict missing key '{cam_key}'. Available keys: {list(obj_for_label.keys())}")
        x = obj_for_label[cam_key]
    else:
        x = obj_for_label

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.astype(np.float32)
    return x


def cam_to_boxes(
    cam: np.ndarray,
    thresholds: np.ndarray,
    min_area: int = 10,
    connectivity: int = 2,
    top_k_boxes: int = 1,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    Convert CAM (assumed already roughly in [0,1]) into boxes via:
      - threshold sweep
      - connected components
      - score per component = max CAM in component
      - dedupe identical boxes (keep best score)
      - keep top_k_boxes
    """

    best_by_box: Dict[Tuple[int, int, int, int], float] = {}
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
    out.sort(key=lambda x: -x[1])
    return out[:top_k_boxes]


def build_predictions_from_heatmaps_gated(
    heatmaps_dir: Path,
    image_ids: List[str],
    labels: List[str],
    cls_scores: Dict[str, Dict[str, float]],
    thresholds: np.ndarray,
    cam_key: str = "cam_vis",
    top_k_labels: int = 0,
    cls_thr: Optional[float] = None,
    min_area: int = 10,
    connectivity: int = 2,
    top_k_boxes: int = 1,
    score_fusion: str = "mul",  # "mul" or "cam" or "cls"
) -> List[Dict]:
    """
    Gated prediction building.
    For each image:
      - pick candidate labels based on cls scores
      - for each selected label: generate up to top_k_boxes
      - final prediction score controls FROC ranking:
          "mul": score = cls_score * cam_score
          "cam": score = cam_score
          "cls": score = cls_score
    """
    preds: List[Dict] = []
    missing = 0
    missing_cls = 0

    for image_id in image_ids:
        p = heatmaps_dir / f"{image_id}.pt"
        if not p.exists():
            missing += 1
            continue

        if image_id not in cls_scores:
            missing_cls += 1
            continue

        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Heatmap file {p} must be a dict label->heatmap or label->dict")

        # Label gating
        selected = select_labels_for_image(
            cls_scores_img=cls_scores[image_id],
            labels=labels,
            top_k_labels=top_k_labels,
            cls_thr=cls_thr,
        )
        if len(selected) == 0:
            continue

        for label in selected:
            if label not in obj:
                continue

            cam = extract_cam_from_obj(obj[label], cam_key=cam_key)
            if cam.shape != (256, 256):
                raise ValueError(f"Expected CAM 256x256, got {cam.shape} for image_id={image_id}, label={label}")

            boxes = cam_to_boxes(
                cam=cam,
                thresholds=thresholds,
                min_area=min_area,
                connectivity=connectivity,
                top_k_boxes=top_k_boxes,
            )

            cls_s = float(cls_scores[image_id][label])
            for box, cam_s in boxes:
                if score_fusion == "mul":
                    score = cls_s * float(cam_s)
                elif score_fusion == "cam":
                    score = float(cam_s)
                elif score_fusion == "cls":
                    score = cls_s
                else:
                    raise ValueError(f"Unknown score_fusion={score_fusion}")

                preds.append({
                    "image_id": image_id,
                    "label": label,
                    "box": box,
                    "score": float(score),
                    "cls_score": cls_s,
                    "cam_score": float(cam_s),
                })

    if missing > 0:
        print(f"[WARN] Missing heatmap files: {missing}/{len(image_ids)}")
    if missing_cls > 0:
        print(f"[WARN] Missing cls scores for images: {missing_cls}/{len(image_ids)}")

    return preds


# ---------------------------
# FROC evaluation (per label)
# ---------------------------

def evaluate_froc_for_label(
    predictions: List[Dict],
    gt_boxes: List[Dict],
    label: str,
    num_images: int,
) -> Tuple[List[Tuple[float, float]], Dict[float, float]]:
    preds = [p for p in predictions if p["label"] == label]
    gts = [g for g in gt_boxes if g["label"] == label]

    if len(gts) == 0:
        return [], {0.10: 0.0, 0.25: 0.0, 0.50: 0.0}

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
    sens_at: Dict[float, float] = {}
    for t in targets:
        valid = [s for f, s in curve if f <= t]
        sens_at[t] = max(valid) if valid else 0.0

    return curve, sens_at


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--heatmaps_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--ann_csv", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, required=True)

    parser.add_argument("--cls_scores_csv", type=str, required=True,
                        help="CSV with per-image per-label scores: image_id + label columns")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=256)

    # CAM selection
    parser.add_argument("--cam_key", type=str, default="cam_vis",
                        help="If heatmap objects store dicts, which key to use (cam_vis/cam_norm/etc)")

    # Thresholds for CAM->boxes
    parser.add_argument("--n_thr", type=int, default=14)
    parser.add_argument("--thr_min", type=float, default=0.35)
    parser.add_argument("--thr_max", type=float, default=0.95)

    # Component filtering
    parser.add_argument("--min_area", type=int, default=10)
    parser.add_argument("--connectivity", type=int, default=2, choices=[1, 2])

    # Debug limit
    parser.add_argument("--max_images", type=int, default=-1)

    # NEW: gating knobs
    parser.add_argument("--top_k_labels", type=int, default=0,
                        help="Keep top-K labels per image by cls score. 0 disables.")
    parser.add_argument("--cls_thr", type=float, default=None,
                        help="Keep labels with cls_score >= cls_thr. None disables.")
    parser.add_argument("--top_k_boxes", type=int, default=1,
                        help="Keep top-K boxes per image per selected label (post-dedupe).")

    # NEW: score fusion
    parser.add_argument("--score_fusion", type=str, default="mul",
                        choices=["mul", "cam", "cls"],
                        help="How to score predictions for sorting in FROC.")

    args = parser.parse_args()

    heatmaps_dir = Path(args.heatmaps_dir)
    labels_csv = Path(args.labels_csv)
    ann_csv = Path(args.ann_csv)
    meta_csv = Path(args.meta_csv)
    cls_scores_csv = Path(args.cls_scores_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Labels + image ids
    labels = load_all_label_names_from_labels_csv(labels_csv)
    print(f"[Labels] {len(labels)} labels from: {labels_csv}")

    image_ids = load_image_ids(labels_csv)
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    num_images = len(image_ids)
    print(f"[Data] N images for FP/image normalization: {num_images}")

    # Thresholds
    thresholds = np.linspace(args.thr_min, args.thr_max, args.n_thr).astype(np.float32)

    # Load classification scores
    cls_scores = load_cls_scores(cls_scores_csv, labels)

    # Load meta and GT
    meta = load_meta(meta_csv)
    gt_boxes = load_gt_boxes_scaled(ann_csv, meta, target_size=args.target_size)
    print(f"[GT] Loaded {len(gt_boxes)} GT boxes total.")

    # Predictions (gated)
    predictions = build_predictions_from_heatmaps_gated(
        heatmaps_dir=heatmaps_dir,
        image_ids=image_ids,
        labels=labels,
        cls_scores=cls_scores,
        thresholds=thresholds,
        cam_key=args.cam_key,
        top_k_labels=args.top_k_labels,
        cls_thr=args.cls_thr,
        min_area=args.min_area,
        connectivity=args.connectivity,
        top_k_boxes=args.top_k_boxes,
        score_fusion=args.score_fusion,
    )
    print(f"[Pred] Generated {len(predictions)} candidate predictions total "
          f"(top_k_labels={args.top_k_labels}, cls_thr={args.cls_thr}, top_k_boxes={args.top_k_boxes}, fusion={args.score_fusion}).")

    # Evaluate per-label
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

        label_stem = safe_filename(label)
        curve_path = output_dir / f"froc_curve_{label_stem}.csv"
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

    out_df = pd.DataFrame(rows).sort_values(by="label")
    out_path = output_dir / "froc_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[Saved] Summary: {out_path}")


if __name__ == "__main__":
    main()
