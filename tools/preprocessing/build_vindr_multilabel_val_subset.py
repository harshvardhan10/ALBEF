#!/usr/bin/env python3
"""
Build a fixed 2,000-image VinDr-CXR training validation subset that preserves
the multi-label distribution of the eligible VinDr training population.

The script:
  1. Reads a one-row-per-image binary label CSV.
  2. Optionally restricts eligibility to existing PNGs and CheXmask coverage.
  3. Generates several iterative multi-label stratified candidate subsets.
  4. Chooses the candidate with the smallest label/co-occurrence distribution error.
  5. Writes:
       - selected multi-label CSV
       - selected image-ID CSV
       - filtered bounding-box annotation CSV
       - per-label distribution report
       - JSON summary

It uses the `iterative-stratification` package when available and otherwise
falls back to a self-contained greedy iterative stratifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ID_CANDIDATES = (
    "image_id",
    "ImageId",
    "imageId",
    "image",
    "study_id",
    "StudyInstanceUID",
    "SOPInstanceUID",
)

NON_LABEL_COLUMNS = {
    "image_id",
    "ImageId",
    "imageId",
    "image",
    "study_id",
    "StudyInstanceUID",
    "SOPInstanceUID",
    "rad_id",
    "radiologist_id",
    "split",
    "width",
    "height",
    "W",
    "H",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-label-stratified VinDr validation subset."
    )
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--annotations_csv", default="")
    parser.add_argument("--images_root", default="")
    parser.add_argument("--chexmask_csv", default="")

    parser.add_argument("--output_labels_csv", required=True)
    parser.add_argument("--output_ids_csv", required=True)
    parser.add_argument("--output_annotations_csv", default="")
    parser.add_argument("--output_report_csv", required=True)
    parser.add_argument("--output_summary_json", required=True)

    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--annotation_id_col", default="")
    parser.add_argument("--chexmask_id_col", default="")
    parser.add_argument("--image_extension", default=".png")

    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_trials",
        type=int,
        default=50,
        help="Generate this many deterministic candidate splits and keep the best.",
    )
    parser.add_argument(
        "--min_positive_warning",
        type=int,
        default=20,
        help="Warn when a selected label has fewer positives than this.",
    )
    parser.add_argument(
        "--strict_chexmask_coverage",
        action="store_true",
        help="Fail rather than filter when eligible labels lack CheXmask rows.",
    )
    parser.add_argument(
        "--strict_image_coverage",
        action="store_true",
        help="Fail rather than filter when eligible labels lack PNG files.",
    )
    return parser.parse_args()


def normalize_id(value: object) -> str:
    value = str(value).strip()
    # CheXmask or manifest CSVs sometimes store a filename rather than a bare ID.
    return Path(value).stem


def infer_id_column(columns: Iterable[str], preferred: str = "") -> str:
    columns = list(columns)
    if preferred:
        if preferred not in columns:
            raise ValueError(
                f"Requested ID column {preferred!r} not found. Columns: {columns}"
            )
        return preferred

    for candidate in ID_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError(
        "Could not infer an image-ID column. "
        f"Tried {ID_CANDIDATES}; columns are {columns}"
    )


def find_binary_label_columns(
    df: pd.DataFrame,
    id_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    converted = df.copy()
    label_cols: list[str] = []

    for col in converted.columns:
        if col == id_col or col in NON_LABEL_COLUMNS:
            continue

        numeric = pd.to_numeric(converted[col], errors="coerce")
        if numeric.isna().any():
            continue

        values = set(numeric.unique().tolist())
        if values.issubset({0, 1, 0.0, 1.0, False, True}):
            converted[col] = numeric.astype(np.uint8)
            label_cols.append(col)

    if not label_cols:
        raise RuntimeError(
            "No complete binary label columns were found in the labels CSV."
        )

    return converted, label_cols


def image_exists(root: Path, image_id: str, extension: str) -> bool:
    candidates = (
        root / f"{image_id}{extension}",
        root / image_id[:2] / f"{image_id}{extension}",
        root / image_id,
        root / image_id[:2] / image_id,
    )
    return any(path.is_file() for path in candidates)


def load_chexmask_ids(path: Path, preferred_col: str = "") -> set[str]:
    header = pd.read_csv(path, nrows=0)
    id_col = infer_id_column(header.columns, preferred_col)
    values = pd.read_csv(path, usecols=[id_col])[id_col]
    return {normalize_id(value) for value in values.dropna()}


def greedy_iterative_split(
    y: np.ndarray,
    n_validation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Self-contained two-way iterative multi-label stratification fallback.

    Subset 0 is validation and subset 1 is the remainder. Rare labels are
    assigned first, following the core iterative-stratification principle.
    """
    n_samples, n_labels = y.shape
    capacities = np.array(
        [n_validation, n_samples - n_validation],
        dtype=np.int64,
    )

    label_totals = y.sum(axis=0).astype(np.float64)
    proportions = capacities.astype(np.float64) / float(n_samples)
    desired = proportions[:, None] * label_totals[None, :]

    label_sets = [
        set(np.flatnonzero(y[:, label_idx]).tolist())
        for label_idx in range(n_labels)
    ]
    remaining_label_counts = np.array(
        [len(values) for values in label_sets],
        dtype=np.int64,
    )

    assignment = np.full(n_samples, -1, dtype=np.int8)
    unassigned = np.ones(n_samples, dtype=bool)

    while np.any(remaining_label_counts > 0):
        active = np.flatnonzero(remaining_label_counts > 0)
        rarest_count = remaining_label_counts[active].min()
        rarest_labels = active[remaining_label_counts[active] == rarest_count]
        label_idx = int(rng.choice(rarest_labels))

        candidates = np.array(list(label_sets[label_idx]), dtype=np.int64)
        rng.shuffle(candidates)

        for sample_idx in candidates:
            if not unassigned[sample_idx]:
                continue

            available_subsets = np.flatnonzero(capacities > 0)
            if len(available_subsets) == 0:
                raise RuntimeError("No remaining subset capacity.")

            label_need = desired[available_subsets, label_idx]
            best_need = label_need.max()
            choices = available_subsets[np.isclose(label_need, best_need)]

            if len(choices) > 1:
                remaining_caps = capacities[choices]
                choices = choices[remaining_caps == remaining_caps.max()]

            subset_idx = int(rng.choice(choices))

            assignment[sample_idx] = subset_idx
            unassigned[sample_idx] = False
            capacities[subset_idx] -= 1

            positive_labels = np.flatnonzero(y[sample_idx])
            desired[subset_idx, positive_labels] -= 1.0

            for positive_label in positive_labels:
                if sample_idx in label_sets[positive_label]:
                    label_sets[positive_label].remove(int(sample_idx))
                    remaining_label_counts[positive_label] -= 1

    # Assign all-zero-label images according to the exact remaining capacities.
    remaining_samples = np.flatnonzero(unassigned)
    rng.shuffle(remaining_samples)

    cursor = 0
    for subset_idx in range(2):
        count = int(capacities[subset_idx])
        selected = remaining_samples[cursor : cursor + count]
        assignment[selected] = subset_idx
        cursor += count

    if np.any(assignment < 0):
        raise RuntimeError("Some samples were not assigned.")
    if int((assignment == 0).sum()) != n_validation:
        raise RuntimeError("Validation subset has the wrong size.")

    return np.flatnonzero(assignment == 0)


def split_with_iterstrat(
    y: np.ndarray,
    n_validation: int,
    seed: int,
) -> np.ndarray:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=n_validation,
        random_state=seed,
    )
    dummy_x = np.zeros((len(y), 1), dtype=np.uint8)
    _, validation_indices = next(splitter.split(dummy_x, y))
    return np.asarray(validation_indices, dtype=np.int64)


def distribution_score(
    y_all: np.ndarray,
    y_subset: np.ndarray,
) -> dict[str, float]:
    full_prev = y_all.mean(axis=0)
    subset_prev = y_subset.mean(axis=0)
    label_abs = np.abs(subset_prev - full_prev)

    full_pair = (y_all.T @ y_all).astype(np.float64) / len(y_all)
    subset_pair = (y_subset.T @ y_subset).astype(np.float64) / len(y_subset)
    pair_mask = np.triu(np.ones_like(full_pair, dtype=bool), k=1)

    if pair_mask.any():
        pair_mean_abs = float(
            np.abs(subset_pair[pair_mask] - full_pair[pair_mask]).mean()
        )
    else:
        pair_mean_abs = 0.0

    cardinality_error = abs(
        float(y_subset.sum(axis=1).mean())
        - float(y_all.sum(axis=1).mean())
    ) / max(1, y_all.shape[1])

    mean_label_abs = float(label_abs.mean())
    max_label_abs = float(label_abs.max())

    # Used only to rank candidate splits.
    objective = (
        mean_label_abs
        + 0.25 * max_label_abs
        + 0.25 * pair_mean_abs
        + 0.10 * cardinality_error
    )

    return {
        "objective": float(objective),
        "mean_label_abs_error": mean_label_abs,
        "max_label_abs_error": max_label_abs,
        "mean_pair_abs_error": pair_mean_abs,
        "normalized_cardinality_error": float(cardinality_error),
    }


def create_distribution_report(
    y_all: np.ndarray,
    y_subset: np.ndarray,
    labels: Sequence[str],
) -> pd.DataFrame:
    n_all = len(y_all)
    n_subset = len(y_subset)

    full_positive = y_all.sum(axis=0).astype(int)
    subset_positive = y_subset.sum(axis=0).astype(int)

    report = pd.DataFrame(
        {
            "label": list(labels),
            "full_positive": full_positive,
            "full_negative": n_all - full_positive,
            "full_prevalence": full_positive / n_all,
            "expected_subset_positive": full_positive * n_subset / n_all,
            "subset_positive": subset_positive,
            "subset_negative": n_subset - subset_positive,
            "subset_prevalence": subset_positive / n_subset,
        }
    )
    report["prevalence_difference"] = (
        report["subset_prevalence"] - report["full_prevalence"]
    )
    report["absolute_prevalence_difference"] = report[
        "prevalence_difference"
    ].abs()

    return report.sort_values(
        "absolute_prevalence_difference",
        ascending=False,
    ).reset_index(drop=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels_csv)
    labels_df = pd.read_csv(labels_path)

    if args.id_col not in labels_df.columns:
        raise ValueError(
            f"Missing ID column {args.id_col!r}. "
            f"Columns: {list(labels_df.columns)}"
        )

    labels_df[args.id_col] = labels_df[args.id_col].map(normalize_id)

    if labels_df[args.id_col].duplicated().any():
        examples = labels_df.loc[
            labels_df[args.id_col].duplicated(keep=False),
            args.id_col,
        ].head(20).tolist()
        raise ValueError(
            "The labels CSV must contain one row per image. "
            f"Duplicate examples: {examples}"
        )

    labels_df, label_cols = find_binary_label_columns(
        labels_df,
        args.id_col,
    )

    eligible = np.ones(len(labels_df), dtype=bool)

    missing_images: list[str] = []
    if args.images_root:
        root = Path(args.images_root)
        if not root.is_dir():
            raise FileNotFoundError(root)

        exists = labels_df[args.id_col].map(
            lambda image_id: image_exists(
                root,
                image_id,
                args.image_extension,
            )
        )
        missing_images = labels_df.loc[~exists, args.id_col].tolist()

        if missing_images and args.strict_image_coverage:
            raise RuntimeError(
                f"{len(missing_images)} label rows lack image files. "
                f"Examples: {missing_images[:10]}"
            )
        eligible &= exists.to_numpy()

    missing_chexmask: list[str] = []
    if args.chexmask_csv:
        chexmask_ids = load_chexmask_ids(
            Path(args.chexmask_csv),
            args.chexmask_id_col,
        )
        covered = labels_df[args.id_col].isin(chexmask_ids)
        missing_chexmask = labels_df.loc[~covered, args.id_col].tolist()

        if missing_chexmask and args.strict_chexmask_coverage:
            raise RuntimeError(
                f"{len(missing_chexmask)} label rows lack CheXmask coverage. "
                f"Examples: {missing_chexmask[:10]}"
            )
        eligible &= covered.to_numpy()

    eligible_df = labels_df.loc[eligible].reset_index(drop=True)

    if args.subset_size <= 0:
        raise ValueError("--subset_size must be positive.")
    if args.subset_size >= len(eligible_df):
        raise ValueError(
            f"subset_size={args.subset_size} must be smaller than "
            f"eligible population={len(eligible_df)}."
        )
    if args.num_trials <= 0:
        raise ValueError("--num_trials must be positive.")

    y_all = eligible_df[label_cols].to_numpy(dtype=np.uint8)

    try:
        import iterstrat  # noqa: F401

        method = "iterative-stratification.MultilabelStratifiedShuffleSplit"
        split_function = lambda trial_seed: split_with_iterstrat(
            y_all,
            args.subset_size,
            trial_seed,
        )
    except ImportError:
        method = "self-contained greedy iterative stratification"
        split_function = lambda trial_seed: greedy_iterative_split(
            y_all,
            args.subset_size,
            np.random.default_rng(trial_seed),
        )

    best_indices: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    best_trial_seed: int | None = None

    print(f"[Data] Total label rows: {len(labels_df)}")
    print(f"[Data] Eligible rows: {len(eligible_df)}")
    print(f"[Data] Binary labels: {len(label_cols)}")
    print(f"[Split] Method: {method}")
    print(f"[Split] Trials: {args.num_trials}")

    for trial in range(args.num_trials):
        trial_seed = args.seed + trial
        validation_indices = split_function(trial_seed)

        if len(validation_indices) != args.subset_size:
            raise RuntimeError(
                f"Candidate split has {len(validation_indices)} rows, "
                f"expected {args.subset_size}."
            )

        metrics = distribution_score(
            y_all,
            y_all[validation_indices],
        )

        if (
            best_metrics is None
            or metrics["objective"] < best_metrics["objective"]
        ):
            best_indices = validation_indices
            best_metrics = metrics
            best_trial_seed = trial_seed

    assert best_indices is not None
    assert best_metrics is not None
    assert best_trial_seed is not None

    selected_df = eligible_df.iloc[
        np.sort(best_indices)
    ].copy().reset_index(drop=True)

    if len(selected_df) != args.subset_size:
        raise RuntimeError("Final subset has the wrong number of rows.")
    if selected_df[args.id_col].duplicated().any():
        raise RuntimeError("Final subset contains duplicate image IDs.")

    selected_ids = set(selected_df[args.id_col])
    selected_ids_df = selected_df[[args.id_col]].copy()

    report_df = create_distribution_report(
        y_all,
        selected_df[label_cols].to_numpy(dtype=np.uint8),
        label_cols,
    )

    output_annotations_rows = None
    if args.annotations_csv:
        if not args.output_annotations_csv:
            raise ValueError(
                "--output_annotations_csv is required when "
                "--annotations_csv is provided."
            )

        annotations_df = pd.read_csv(args.annotations_csv)
        annotation_id_col = infer_id_column(
            annotations_df.columns,
            args.annotation_id_col,
        )
        annotations_df[annotation_id_col] = annotations_df[
            annotation_id_col
        ].map(normalize_id)

        selected_annotations = annotations_df[
            annotations_df[annotation_id_col].isin(selected_ids)
        ].copy()

        atomic_write_csv(
            selected_annotations,
            Path(args.output_annotations_csv),
        )
        output_annotations_rows = int(len(selected_annotations))

    atomic_write_csv(selected_df, Path(args.output_labels_csv))
    atomic_write_csv(selected_ids_df, Path(args.output_ids_csv))
    atomic_write_csv(report_df, Path(args.output_report_csv))

    selected_y = selected_df[label_cols].to_numpy(dtype=np.uint8)
    selected_positive = selected_y.sum(axis=0).astype(int)

    warnings = []
    for label, positive_count in zip(label_cols, selected_positive):
        negative_count = len(selected_df) - int(positive_count)
        if (
            positive_count < args.min_positive_warning
            or negative_count < args.min_positive_warning
        ):
            warnings.append(
                {
                    "label": label,
                    "positive": int(positive_count),
                    "negative": int(negative_count),
                }
            )

    summary = {
        "labels_csv": str(labels_path),
        "annotations_csv": args.annotations_csv or None,
        "images_root": args.images_root or None,
        "chexmask_csv": args.chexmask_csv or None,
        "id_column": args.id_col,
        "label_columns": label_cols,
        "total_label_rows": int(len(labels_df)),
        "eligible_rows": int(len(eligible_df)),
        "subset_rows": int(len(selected_df)),
        "seed": int(args.seed),
        "selected_trial_seed": int(best_trial_seed),
        "num_trials": int(args.num_trials),
        "stratification_method": method,
        "distribution_metrics": best_metrics,
        "full_mean_labels_per_image": float(y_all.sum(axis=1).mean()),
        "subset_mean_labels_per_image": float(
            selected_y.sum(axis=1).mean()
        ),
        "missing_image_count": int(len(missing_images)),
        "missing_chexmask_count": int(len(missing_chexmask)),
        "output_annotations_rows": output_annotations_rows,
        "low_count_label_warnings": warnings,
        "output_labels_csv": args.output_labels_csv,
        "output_ids_csv": args.output_ids_csv,
        "output_annotations_csv": args.output_annotations_csv or None,
        "output_report_csv": args.output_report_csv,
    }
    atomic_write_json(summary, Path(args.output_summary_json))

    print()
    print("=" * 76)
    print("SELECTED VALIDATION SUBSET")
    print("=" * 76)
    print(f"Rows: {len(selected_df)}")
    print(f"Unique IDs: {selected_df[args.id_col].nunique()}")
    print(f"Selected trial seed: {best_trial_seed}")
    for key, value in best_metrics.items():
        print(f"{key}: {value:.8f}")

    print()
    print("Largest prevalence differences:")
    display_cols = [
        "label",
        "full_positive",
        "full_prevalence",
        "subset_positive",
        "subset_prevalence",
        "prevalence_difference",
    ]
    print(report_df[display_cols].head(20).to_string(index=False))

    if warnings:
        print()
        print(
            "[WARNING] Labels with too few positives or negatives for stable "
            "validation AUC:"
        )
        for row in warnings:
            print(
                f"  {row['label']}: "
                f"positive={row['positive']}, negative={row['negative']}"
            )

    print()
    print(f"Labels: {args.output_labels_csv}")
    print(f"IDs: {args.output_ids_csv}")
    if args.output_annotations_csv:
        print(f"Annotations: {args.output_annotations_csv}")
    print(f"Report: {args.output_report_csv}")
    print(f"Summary: {args.output_summary_json}")


if __name__ == "__main__":
    main()
