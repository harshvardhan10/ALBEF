#!/usr/bin/env python
"""
Evaluate ALBEF heatmaps on VinDr-CXR against bounding boxes.

For each GT box (image_id, class_name, x_min, y_min, x_max, y_max):

  1. Load heatmap for that image + label from {image_id}.pt
     (saved by your Grad-CAM / attention-Grad-CAM extractor).
  2. Scale GT box from original resolution to 256x256.
  3. Compute:

     - mean_in_box      : mean heat in the GT box
     - max_in_box       : max  heat in the GT box
     - mean_global      : mean heat over entire 256x256 image
     - top_p_coverage   : for each percentile p in PERC_LIST,
                          fraction of the top-p% mask that lies inside the box
     - top_p_recall     : fraction of the GT box pixels that are covered by
                          the top-p% mask

  4. Aggregate per label and print summary.

This does NOT depend on any threshold for detection metrics yet,
it just answers: "is the heat concentrated on the bounding boxes or not?"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ------------------ Helpers ------------------


def load_meta(meta_csv):
    """
    Load VinDr meta CSV with columns: image_id, dim0, dim1
    Returns dict: image_id -> (orig_height, orig_width)
    """
    df = pd.read_csv(meta_csv)
    return {r["image_id"]: (int(r["dim0"]), int(r["dim1"])) for _, r in df.iterrows()}


def scale_box_to_256(r, orig_h, orig_w, target_size=256):
    """
    Scale a GT box from original resolution to target_size x target_size space.

    r: DataFrame row with x_min, y_min, x_max, y_max in original px.
    Returns (x_min_256, y_min_256, x_max_256, y_max_256) as ints, clamped to [0, target_size-1].
    """
    scale_x = float(target_size) / float(orig_w)
    scale_y = float(target_size) / float(orig_h)

    x_min = r["x_min"] * scale_x
    y_min = r["y_min"] * scale_y
    x_max = r["x_max"] * scale_x
    y_max = r["y_max"] * scale_y

    # convert to int indices for slicing heatmaps
    x_min_i = int(np.floor(x_min))
    y_min_i = int(np.floor(y_min))
    x_max_i = int(np.ceil(x_max))
    y_max_i = int(np.ceil(y_max))

    # clamp
    x_min_i = max(0, min(target_size - 1, x_min_i))
    y_min_i = max(0, min(target_size - 1, y_min_i))
    x_max_i = max(0, min(target_size, x_max_i))
    y_max_i = max(0, min(target_size, y_max_i))

    # ensure at least 1 pixel in each dimension
    if x_max_i <= x_min_i:
        x_max_i = min(target_size, x_min_i + 1)
    if y_max_i <= y_min_i:
        y_max_i = min(target_size, y_min_i + 1)

    return x_min_i, y_min_i, x_max_i, y_max_i


def compute_box_stats_for_heatmap(
    heatmap: np.ndarray,
    box_coords,
    percentiles,
):
    """
    heatmap: (H, W) in [0,1]
    box_coords: (x_min, y_min, x_max, y_max) in 256x256 indices
    percentiles: list of percentiles to evaluate (e.g. [90,95,99])

    Returns a dict with:
      - mean_in_box
      - max_in_box
      - mean_global
      - For each p in percentiles:
          top{p}_coverage
          top{p}_recall
    """
    H, W = heatmap.shape
    x_min, y_min, x_max, y_max = box_coords

    # region inside box
    box_region = heatmap[y_min:y_max, x_min:x_max]
    mean_in_box = float(box_region.mean())
    max_in_box = float(box_region.max())

    mean_global = float(heatmap.mean())

    results = {
        "mean_in_box": mean_in_box,
        "max_in_box": max_in_box,
        "mean_global": mean_global,
    }

    box_area = box_region.size  # number of pixels in box

    # Flatten for percentile computation
    flat = heatmap.reshape(-1)

    for p in percentiles:
        thr = np.percentile(flat, p)
        mask = (heatmap >= thr).astype(np.float32)  # 1 on top-p% pixels

        mask_area = mask.sum()

        # intersection: pixels in mask AND in box
        mask_box = mask[y_min:y_max, x_min:x_max]
        inter = mask_box.sum()

        if mask_area > 0:
            coverage = float(inter / mask_area)       # how much of top-p% area lies inside box
        else:
            coverage = 0.0

        recall = float(inter / box_area) if box_area > 0 else 0.0  # how much of box is "hot"

        results[f"top{p}_coverage"] = coverage
        results[f"top{p}_recall"] = recall

    return results


# ------------------ Main evaluation ------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALBEF heatmaps vs VinDr GT boxes.")
    parser.add_argument("--annotations_csv", type=str, required=True,
                        help="VinDr annotations_test.csv with GT boxes.")
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="VinDr test_meta.csv with original image sizes.")
    parser.add_argument("--heatmaps_root", type=str, required=True,
                        help="Directory with {image_id}.pt heatmaps (label -> (H,W) tensor).")
    parser.add_argument("--only_labels", type=str, nargs="*", default=None,
                        help="Optional: evaluate only a subset of class_name labels.")
    parser.add_argument("--percentiles", type=float, nargs="*", default=[90.0, 95.0, 99.0],
                        help="Percentiles to evaluate for top-p% coverage/recall.")
    parser.add_argument("--max_boxes", type=int, default=None,
                        help="Optional: limit total number of GT boxes (debug).")

    args = parser.parse_args()

    annotations_csv = Path(args.annotations_csv)
    meta_csv = Path(args.meta_csv)
    heatmaps_root = Path(args.heatmaps_root)

    # ------------------ Load data ------------------
    df_ann = pd.read_csv(annotations_csv)
    meta = load_meta(meta_csv)

    if args.only_labels is not None:
        df_ann = df_ann[df_ann["class_name"].isin(args.only_labels)].reset_index(drop=True)
        print(f"[Filter] Restricting to labels: {args.only_labels}")

    if args.max_boxes is not None:
        df_ann = df_ann.iloc[: args.max_boxes].reset_index(drop=True)
        print(f"[Filter] Restricting to first {len(df_ann)} GT boxes (max_boxes={args.max_boxes}).")

    print(f"[Data] Number of GT boxes to evaluate: {len(df_ann)}")

    # ------------------ Loop over GT boxes ------------------
    records = []
    missing_heatmap_images = set()

    for idx, r in df_ann.iterrows():
        image_id = r["image_id"]
        label = r["class_name"]

        if image_id not in meta:
            print(f"[WARN] image_id={image_id} not found in meta CSV; skipping.")
            continue

        orig_h, orig_w = meta[image_id]

        hm_path = heatmaps_root / f"{image_id}.pt"
        if not hm_path.exists():
            if image_id not in missing_heatmap_images:
                print(f"[WARN] Missing heatmap file for {image_id}: {hm_path}")
                missing_heatmap_images.add(image_id)
            continue

        heatmaps = torch.load(hm_path, map_location="cpu")
        if label not in heatmaps:
            print(f"[WARN] No heatmap for label '{label}' in {hm_path.name}; skipping this GT box.")
            continue

        hm_tensor = heatmaps[label].float()
        # Normalization (defensive; should already be ~[0,1])
        if hm_tensor.max() > hm_tensor.min():
            hm_tensor = (hm_tensor - hm_tensor.min()) / (hm_tensor.max() - hm_tensor.min())
        else:
            hm_tensor = torch.zeros_like(hm_tensor)

        heatmap = hm_tensor.detach().cpu().numpy()
        H, W = heatmap.shape
        if H != 256 or W != 256:
            raise ValueError(f"Expected heatmap 256x256, got {H}x{W} for {image_id}, label={label}")

        # Scale GT box to 256x256
        box_256 = scale_box_to_256(r, orig_h=orig_h, orig_w=orig_w, target_size=256)

        stats = compute_box_stats_for_heatmap(
            heatmap=heatmap,
            box_coords=box_256,
            percentiles=args.percentiles,
        )

        row_record = {
            "image_id": image_id,
            "label": label,
            "x_min_256": box_256[0],
            "y_min_256": box_256[1],
            "x_max_256": box_256[2],
            "y_max_256": box_256[3],
        }
        row_record.update(stats)
        records.append(row_record)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(df_ann):
            print(f"[Eval] Processed {idx + 1}/{len(df_ann)} GT boxes")

    if not records:
        print("[Result] No records to evaluate. Check inputs / filters.")
        return

    df_res = pd.DataFrame(records)
    print("\n[Result] Overall statistics:")
    print(df_res.describe())

    # ------------------ Per-label summaries ------------------
    print("\n[Result] Per-label statistics (mean over boxes):")
    group_cols = ["mean_in_box", "max_in_box", "mean_global"]
    for p in args.percentiles:
        group_cols.append(f"top{p}_coverage")
        group_cols.append(f"top{p}_recall")

    df_label = df_res.groupby("label")[group_cols].mean().sort_values("mean_in_box", ascending=False)
    print(df_label)

    # Optionally, save CSVs for deeper analysis
    out_full = annotations_csv.with_name("heatmap_eval_per_box.csv")
    out_label = annotations_csv.with_name("heatmap_eval_per_label.csv")
    df_res.to_csv(out_full, index=False)
    df_label.to_csv(out_label)
    print(f"\n[Output] Saved per-box results to: {out_full}")
    print(f"[Output] Saved per-label summary to: {out_label}")


if __name__ == "__main__":
    main()
