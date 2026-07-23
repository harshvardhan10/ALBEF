"""
Build a fixed VinDr-train validation subset for ALBEF checkpoint selection.

The script samples Cardiomegaly-positive and negative images with a fixed seed,
requires the original PNG and all requested mask-cache files to exist, and saves:
  1. image-level validation labels CSV
  2. validation bounding-box annotations CSV

The same CSV should be reused for original, lung-only, heart-only, and later
bone-suppressed checkpoint selection.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument(
        "--required_mask_roots",
        nargs="*",
        default=[],
        help="Optional sharded mask roots that every selected image must have.",
    )
    parser.add_argument("--output_labels_csv", required=True)
    parser.add_argument("--output_annotations_csv", required=True)
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument("--num_positive", type=int, default=200)
    parser.add_argument("--num_negative", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    labels_path = Path(args.labels_csv)
    annotations_path = Path(args.annotations_csv)
    images_root = Path(args.images_root)
    mask_roots = [Path(path) for path in args.required_mask_roots]

    labels = pd.read_csv(labels_path)
    annotations = pd.read_csv(annotations_path)
    labels.columns = [column.strip() for column in labels.columns]
    annotations.columns = [column.strip() for column in annotations.columns]

    id_col = labels.columns[0]
    if args.label not in labels.columns:
        raise ValueError(
            f"Label {args.label!r} not present in {labels_path}. "
            f"Columns: {list(labels.columns)}"
        )

    required_box_columns = {
        "image_id",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    missing = required_box_columns - set(annotations.columns)
    if missing:
        raise ValueError(f"annotations_csv missing columns: {sorted(missing)}")

    labels[id_col] = labels[id_col].astype(str)
    annotations["image_id"] = annotations["image_id"].astype(str)
    annotations["class_name"] = annotations["class_name"].astype(str)

    values = labels[args.label].to_numpy()
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError(
            f"{args.label} must be binary 0/1. Found: {np.unique(values).tolist()}"
        )

    box_positive_ids = set(
        annotations.loc[
            annotations["class_name"] == args.label,
            "image_id",
        ].astype(str)
    )

    def has_all_files(image_id: str) -> bool:
        image_path = images_root / f"{image_id}.png"
        if not image_path.exists():
            return False
        return all(
            (mask_root / image_id[:2] / f"{image_id}.png").exists()
            for mask_root in mask_roots
        )

    valid_file_mask = labels[id_col].map(has_all_files)
    candidates = labels[valid_file_mask].copy()

    positive = candidates[
        (candidates[args.label] == 1)
        & (candidates[id_col].isin(box_positive_ids))
    ].copy()
    negative = candidates[candidates[args.label] == 0].copy()

    if len(positive) < args.num_positive:
        raise ValueError(
            f"Requested {args.num_positive} positives but only {len(positive)} "
            "have labels, boxes, original images, and all required masks."
        )
    if len(negative) < args.num_negative:
        raise ValueError(
            f"Requested {args.num_negative} negatives but only {len(negative)} "
            "have original images and all required masks."
        )

    rng = np.random.default_rng(args.seed)
    positive_indices = rng.choice(
        positive.index.to_numpy(), size=args.num_positive, replace=False
    )
    negative_indices = rng.choice(
        negative.index.to_numpy(), size=args.num_negative, replace=False
    )

    selected = pd.concat(
        [labels.loc[positive_indices], labels.loc[negative_indices]],
        axis=0,
    )
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    selected_ids = set(selected[id_col].astype(str))

    selected_annotations = annotations[
        annotations["image_id"].isin(selected_ids)
    ].copy()

    output_labels = Path(args.output_labels_csv)
    output_annotations = Path(args.output_annotations_csv)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    output_annotations.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(output_labels, index=False)
    selected_annotations.to_csv(output_annotations, index=False)

    print(f"Saved labels: {output_labels}")
    print(f"Saved annotations: {output_annotations}")
    print(f"Images: {len(selected)}")
    print(f"{args.label} positives: {int(selected[args.label].sum())}")
    print(f"{args.label} negatives: {int((selected[args.label] == 0).sum())}")
    print(
        f"{args.label} boxes: "
        f"{int((selected_annotations['class_name'] == args.label).sum())}"
    )


if __name__ == "__main__":
    main()
