"""
FROC evaluation for VinDr-CXR CAM heatmaps with quadrant-based matching,
WITH zero-shot gating (per-image, per-label classification scores from NPZ).

Key changes vs a CAM-only pipeline
- Uses cam_vis stored in heatmap dicts (no extra min-max normalization by default).
- Generates predictions ONLY for labels that pass a classification gate per image:
    (A) keep top_k_labels per image, OR
    (B) keep labels with cls_score >= cls_thr, OR both.
- Prediction ranking score is configurable:
    "mul": score = cls_score * cam_score
    "cam": score = cam_score
    "cls": score = cls_score
    "sum": score = alpha*cam_score + (1-alpha)*cls_score
  where cam_score = max CAM value inside a connected component.
- Computes per-label FROC with your quadrant matching + one-to-one GT matching.

Inputs expected
- heatmaps_dir: contains <image_id>.pt with dict[label] -> either:
      - a 256x256 tensor (assumed cam_vis), OR
      - a dict with keys {"cam_vis", "cam_raw"} (we will pick cam_vis by default)
- labels_csv: image_labels_test.csv (defines test image_ids + label list)
- ann_csv: annotations_test.csv (GT boxes)
- meta_csv: test_meta.csv (original dims for scaling)
- scores_npz: npz with per-image, per-label classification scores
    fields: image_ids, label_names, scores (N,L) and optional y_true
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src import (
    load_meta
)

from evaluate_froc_vindr import (
    safe_filename,
    load_image_ids,
    load_all_label_names_from_labels_csv,
    load_gt_boxes_scaled,
    cam_to_boxes,
    evaluate_froc_for_label,
    evaluate_froc
)


# ---------------------------
# Zero-shot classification scores (NPZ)
# ---------------------------

def load_scores_npz(scores_npz: Path):
    """
    Load NPZ written by save_scores_npz():
      - image_ids (N,) object
      - label_names (L,) object
      - scores (N,L) float32
      - y_true optional
    Returns:
      zs_image_ids: list[str]
      zs_label_names: list[str]
      zs_scores: np.ndarray float32 (N,L)
    """
    z = np.load(scores_npz, allow_pickle=True)
    zs_image_ids = [str(x) for x in z["image_ids"].tolist()]
    zs_label_names = [str(x) for x in z["label_names"].tolist()]
    zs_scores = z["scores"].astype(np.float32)
    if zs_scores.ndim != 2:
        raise ValueError(f"Expected scores shape (N,L), got {zs_scores.shape}")
    return zs_image_ids, zs_label_names, zs_scores


def build_score_lookup(zs_image_ids: List[str], zs_label_names: List[str], zs_scores: np.ndarray):
    """
    Returns:
      score_by_image: dict[image_id] -> np.ndarray (L,)
      label_to_index: dict[label] -> j
    """
    label_to_index = {lb: j for j, lb in enumerate(zs_label_names)}
    score_by_image: Dict[str, np.ndarray] = {}
    for i, img_id in enumerate(zs_image_ids):
        score_by_image[img_id] = zs_scores[i]
    return score_by_image, label_to_index


def select_labels_for_image(
    image_id: str,
    labels: List[str],
    score_by_image: Dict[str, np.ndarray],
    label_to_index: Dict[str, int],
    top_k_labels: int = 0,
    cls_thr: Optional[float] = None,
) -> List[str]:
    """
    Apply gating:
      - if top_k_labels > 0: keep only the top-K labels by cls score
      - if cls_thr is not None: additionally require score >= cls_thr
    Notes:
      - If a label is missing in NPZ label space, it is treated as -inf (i.e., filtered out).
      - If image_id missing, returns [].
    """
    vec = score_by_image.get(image_id, None)
    if vec is None:
        return []

    items = []
    for lb in labels:
        j = label_to_index.get(lb, None)
        s = float(vec[j]) if j is not None else float("-inf")
        items.append((lb, s))

    items.sort(key=lambda x: -x[1])

    if top_k_labels and top_k_labels > 0:
        items = items[:top_k_labels]

    if cls_thr is not None:
        items = [(lb, s) for lb, s in items if s >= float(cls_thr)]

    return [lb for lb, _ in items]


def get_cls_score(
    image_id: str,
    label: str,
    score_by_image: Dict[str, np.ndarray],
    label_to_index: Dict[str, int],
) -> Optional[float]:
    vec = score_by_image.get(image_id, None)
    if vec is None:
        return None
    j = label_to_index.get(label, None)
    if j is None:
        return None
    return float(vec[j])


# ---------------------------
# CAM -> predicted boxes
# ---------------------------

def extract_cam_from_obj(obj_for_label, cam_key: str = "cam_vis") -> np.ndarray:
    x = obj_for_label[cam_key]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = x.astype(np.float32)
    return x



def fuse_score(cam_score: float,
               cls_score: float,
               mode: str = "mul",
               alpha: float = 0.5) -> float:
    """
    mode:
      - mul: cam*cls
      - cam: cam
      - cls: cls
      - sum: alpha*cam + (1-alpha)*cls
    """
    cam_s = float(cam_score)
    cls_s = float(cls_score)
    if mode == "mul":
        return cam_s * cls_s
    if mode == "cam":
        return cam_s
    if mode == "cls":
        return cls_s
    if mode == "sum":
        a = float(alpha)
        return a * cam_s + (1.0 - a) * cls_s
    raise ValueError(f"Unknown score_fusion='{mode}'")


def build_predictions_from_heatmaps_gated(
    heatmaps_dir: Path,
    image_ids: List[str],
    labels: List[str],
    score_by_image: Dict[str, np.ndarray],
    label_to_index: Dict[str, int],
    thresholds: np.ndarray,
    cam_key: str = "cam_vis_up",
    top_k_labels: int = 0,
    cls_thr: Optional[float] = None,
    min_area: int = 10,
    connectivity: int = 2,
    top_k_boxes: int = 1,
    score_fusion: str = "mul",  # "mul" | "cam" | "cls" | "sum"
    alpha: float = 0.5,
) -> List[Dict]:
    """
    Gated prediction building.
    For each image:
      - pick candidate labels based on cls scores (NPZ)
      - for each selected label: generate up to top_k_boxes
      - final prediction score controls FROC ranking
    """
    preds: List[Dict] = []
    missing_hm = 0
    missing_cls = 0

    for image_id in image_ids:
        p = heatmaps_dir / f"{image_id}.pt"
        if not p.exists():
            missing_hm += 1
            continue

        if image_id not in score_by_image:
            missing_cls += 1
            continue

        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Heatmap file {p} must be a dict label->heatmap or label->dict")

        # label gating
        selected = select_labels_for_image(
            image_id=image_id,
            labels=labels,
            score_by_image=score_by_image,
            label_to_index=label_to_index,
            top_k_labels=top_k_labels,
            cls_thr=cls_thr,
        )
        if len(selected) == 0:
            continue

        for label in selected:
            if label not in obj:
                continue

            cls_s = get_cls_score(image_id, label, score_by_image, label_to_index)
            if cls_s is None:
                continue

            cam = extract_cam_from_obj(obj[label], cam_key=cam_key)
            if cam.shape != (256, 256):
                raise ValueError(
                    f"Expected CAM 256x256, got {cam.shape} for image_id={image_id}, label={label}"
                )

            boxes = cam_to_boxes(
                cam=cam,
                thresholds=thresholds,
                min_area=min_area,
                connectivity=connectivity,
                top_k_boxes=top_k_boxes,
            )

            for box, cam_s in boxes:
                score = fuse_score(cam_s, cls_s, mode=score_fusion, alpha=alpha)
                preds.append({
                    "image_id": image_id,
                    "label": label,
                    "box": box,
                    "score": float(score),
                    "cls_score": float(cls_s),
                    "cam_score": float(cam_s),
                })

    if missing_hm > 0:
        print(f"[WARN] Missing heatmap files: {missing_hm}/{len(image_ids)}")
    if missing_cls > 0:
        print(f"[WARN] Missing cls scores for images: {missing_cls}/{len(image_ids)}")

    return preds


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--heatmaps_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--ann_csv", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--scores_npz", type=str, required=True,
                        help="NPZ with per-image per-label scores (image_ids, label_names, scores).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=256)

    # CAM selection
    parser.add_argument("--cam_key", type=str, default="cam_vis_up",
                        help="If heatmap objects store dicts, which key to use (cam_vis/cam_raw/etc).")

    # Thresholds for CAM->boxes
    parser.add_argument("--n_thr", type=int, default=14)
    parser.add_argument("--thr_min", type=float, default=0.35)
    parser.add_argument("--thr_max", type=float, default=0.95)

    # Component filtering
    parser.add_argument("--min_area", type=int, default=10)
    parser.add_argument("--connectivity", type=int, default=2, choices=[1, 2])

    # Debug limit
    parser.add_argument("--max_images", type=int, default=-1)

    parser.add_argument("--top_k_labels", type=int, default=0,
                        help="Keep top-K labels per image by cls score. 0 disables.")
    parser.add_argument("--cls_thr", type=float, default=None,
                        help="Keep labels with cls_score >= cls_thr. None disables.")
    parser.add_argument("--top_k_boxes", type=int, default=1,
                        help="Keep top-K boxes per image per selected label (post-dedupe).")

    parser.add_argument("--score_fusion", type=str, default="mul",
                        choices=["mul", "cam", "cls", "sum"],
                        help="How to score predictions for sorting in FROC.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Only used if score_fusion='sum'.")

    args = parser.parse_args()

    heatmaps_dir = Path(args.heatmaps_dir)
    labels_csv = Path(args.labels_csv)
    ann_csv = Path(args.ann_csv)
    meta_csv = Path(args.meta_csv)
    scores_npz = Path(args.scores_npz)

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

    # Load NPZ scores
    zs_image_ids, zs_label_names, zs_scores = load_scores_npz(scores_npz)
    score_by_image, label_to_index = build_score_lookup(zs_image_ids, zs_label_names, zs_scores)
    print(f"[ZeroShot] Loaded NPZ scores: {scores_npz}")
    print(f"[ZeroShot] scores shape: {zs_scores.shape}")

    # Load meta and GT
    meta = load_meta(meta_csv)
    gt_boxes = load_gt_boxes_scaled(ann_csv=ann_csv, meta=meta, target_size=args.target_size)
    print(f"[GT] Loaded {len(gt_boxes)} GT boxes total.")

    # Predictions (gated)
    predictions = build_predictions_from_heatmaps_gated(
        heatmaps_dir=heatmaps_dir,
        image_ids=image_ids,
        labels=labels,
        score_by_image=score_by_image,
        label_to_index=label_to_index,
        thresholds=thresholds,
        cam_key=args.cam_key,
        top_k_labels=args.top_k_labels,
        cls_thr=args.cls_thr,
        min_area=args.min_area,
        connectivity=args.connectivity,
        top_k_boxes=args.top_k_boxes,
        score_fusion=args.score_fusion,
        alpha=args.alpha,
    )
    print(
        f"[Pred] Generated {len(predictions)} candidate predictions total "
        f"(top_k_labels={args.top_k_labels}, cls_thr={args.cls_thr}, "
        f"top_k_boxes={args.top_k_boxes}, fusion={args.score_fusion})."
    )

    evaluate_froc(labels, predictions, gt_boxes, num_images, output_dir)


if __name__ == "__main__":
    main()
