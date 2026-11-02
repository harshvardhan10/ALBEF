"""Build a fixed VinDr-train validation subset for checkpoint selection.

The split can optionally be restricted to image IDs present in a CheXmask VinDr
CSV, without requiring decoded PNG masks to exist yet. This allows the split to
be frozen first and the small validation-only mask cache to be decoded next.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.strip().lower())
        if match is not None:
            return str(match)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_labels_csv", required=True)
    parser.add_argument("--output_annotations_csv", required=True)
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument("--num_positive", type=int, default=200)
    parser.add_argument("--num_negative", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--chexmask_csv",
        default=None,
        help=(
            "Optional VinDr CheXmask CSV. When supplied, only IDs present in "
            "this CSV are eligible; decoded PNG masks are not required yet."
        ),
    )
    parser.add_argument(
        "--chexmask_id_col",
        default=None,
        help="Optional CheXmask image-ID column. Defaults to the first column.",
    )
    parser.add_argument(
        "--min_dice_rca_mean",
        type=float,
        default=0.7,
        help=(
            "Minimum CheXmask Dice RCA (Mean) when the column exists. "
            "Set below 0 to disable quality filtering."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    labels_path = Path(args.labels_csv)
    annotations_path = Path(args.annotations_csv)
    images_root = Path(args.images_root)

    labels = pd.read_csv(labels_path)
    annotations = pd.read_csv(annotations_path)
    labels.columns = [str(column).strip() for column in labels.columns]
    annotations.columns = [str(column).strip() for column in annotations.columns]

    id_col = str(labels.columns[0])
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

    eligible_mask = labels[id_col].map(
        lambda image_id: (images_root / f"{image_id}.png").exists()
    )

    if args.chexmask_csv:
        chexmask_path = Path(args.chexmask_csv)
        chexmask = pd.read_csv(chexmask_path)
        chexmask.columns = [str(column).strip() for column in chexmask.columns]
        chexmask_id_col = args.chexmask_id_col or str(chexmask.columns[0])
        if chexmask_id_col not in chexmask.columns:
            raise ValueError(
                f"CheXmask ID column {chexmask_id_col!r} not found. "
                f"Columns: {list(chexmask.columns)}"
            )
        chexmask[chexmask_id_col] = chexmask[chexmask_id_col].astype(str)

        quality_col = find_column(
            chexmask,
            ["Dice RCA (Mean)", "dice_rca_mean", "dice rca mean"],
        )
        if quality_col is not None and args.min_dice_rca_mean >= 0:
            before = len(chexmask)
            quality = pd.to_numeric(chexmask[quality_col], errors="coerce")
            chexmask = chexmask[quality >= float(args.min_dice_rca_mean)].copy()
            print(
                f"CheXmask quality filter {quality_col}>="
                f"{args.min_dice_rca_mean}: {before} -> {len(chexmask)}",
                flush=True,
            )

        chexmask_ids = set(chexmask[chexmask_id_col].astype(str))
        eligible_mask &= labels[id_col].isin(chexmask_ids)
        print(
            f"Eligible CheXmask IDs after filtering: {len(chexmask_ids)}",
            flush=True,
        )

    candidates = labels[eligible_mask].copy()
    box_positive_ids = set(
        annotations.loc[
            annotations["class_name"] == args.label,
            "image_id",
        ].astype(str)
    )

    positive = candidates[
        (candidates[args.label] == 1)
        & (candidates[id_col].isin(box_positive_ids))
    ].copy()
    negative = candidates[candidates[args.label] == 0].copy()

    if len(positive) < args.num_positive:
        raise ValueError(
            f"Requested {args.num_positive} positives but only {len(positive)} "
            "eligible positives have labels, boxes, images, and optional "
            "CheXmask records."
        )
    if len(negative) < args.num_negative:
        raise ValueError(
            f"Requested {args.num_negative} negatives but only {len(negative)} "
            "eligible negatives have images and optional CheXmask records."
        )

    rng = np.random.default_rng(args.seed)
    positive_indices = rng.choice(
        positive.index.to_numpy(), size=args.num_positive, replace=False
    )
    negative_indices = rng.choice(
        negative.index.to_numpy(), size=args.num_negative, replace=False
    )

    selected = pd.concat(
        [labels.loc[positive_indices], labels.loc[negative_indices]], axis=0
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
