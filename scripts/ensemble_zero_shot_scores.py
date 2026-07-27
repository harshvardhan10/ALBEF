#!/usr/bin/env python3
"""
Ensemble saved VinDr zero-shot classification scores.

The input files are the ``.npz`` files written by zero_shot_eval_vindr.py (or
its mask-cache variant). Rows are aligned by image ID and columns are aligned
by label name before averaging. Array position is never assumed to carry the
same meaning across models.

Example
-------
python scripts/ensemble_zero_shot_scores.py \
    --score_files \
        outputs/original/vindr_zero_shot_scores_checkpoint_best_macro_auc_stable.npz \
        outputs/lung/vindr_zero_shot_scores_checkpoint_best_macro_auc_stable.npz \
        outputs/heart/vindr_zero_shot_scores_checkpoint_best_macro_auc_stable.npz \
    --model_names original lung heart \
    --output_dir outputs/ensemble/stable \
    --ensemble_name stable \
    --threshold 0.3948 \
    --save_scores_csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


REQUIRED_KEYS = {"image_ids", "label_names", "scores", "y_true"}


def _string_list(values: np.ndarray, field: str, path: Path) -> List[str]:
    result = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    if any(value == "" for value in result):
        raise ValueError(f"{path}: {field} contains an empty value")
    duplicates = sorted(
        value for value, count in Counter(result).items() if count > 1
    )
    if duplicates:
        preview = duplicates[:10]
        raise ValueError(
            f"{path}: duplicate {field} values found: {preview}"
            + (" ..." if len(duplicates) > 10 else "")
        )
    return result


def load_score_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Score file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing required arrays: {sorted(missing)}")

        image_ids = _string_list(data["image_ids"], "image_ids", path)
        label_names = _string_list(data["label_names"], "label_names", path)
        scores = np.asarray(data["scores"], dtype=np.float32)
        y_true = np.asarray(data["y_true"], dtype=np.float32)

    expected_shape = (len(image_ids), len(label_names))
    if scores.shape != expected_shape:
        raise ValueError(
            f"{path}: scores shape {scores.shape} != expected {expected_shape}"
        )
    if y_true.shape != expected_shape:
        raise ValueError(
            f"{path}: y_true shape {y_true.shape} != expected {expected_shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: scores contain NaN or infinite values")
    if not np.isfinite(y_true).all():
        raise ValueError(f"{path}: y_true contains NaN or infinite values")
    if not np.isin(y_true, [0.0, 1.0]).all():
        invalid = np.unique(y_true[~np.isin(y_true, [0.0, 1.0])])
        raise ValueError(f"{path}: y_true is not binary; invalid values={invalid}")

    return {
        "path": str(path),
        "image_ids": image_ids,
        "label_names": label_names,
        "scores": scores,
        "y_true": y_true.astype(np.int64),
    }


def align_to_canonical(
    item: Dict[str, Any],
    canonical_image_ids: Sequence[str],
    canonical_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    item_ids = item["image_ids"]
    item_labels = item["label_names"]
    path = item["path"]

    canonical_id_set = set(canonical_image_ids)
    item_id_set = set(item_ids)
    if item_id_set != canonical_id_set:
        missing = sorted(canonical_id_set - item_id_set)
        extra = sorted(item_id_set - canonical_id_set)
        raise ValueError(
            f"{path}: image-ID set differs from the canonical file; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    canonical_label_set = set(canonical_labels)
    item_label_set = set(item_labels)
    if item_label_set != canonical_label_set:
        missing = sorted(canonical_label_set - item_label_set)
        extra = sorted(item_label_set - canonical_label_set)
        raise ValueError(
            f"{path}: label set differs from the canonical file; "
            f"missing={missing}, extra={extra}"
        )

    row_lookup = {image_id: index for index, image_id in enumerate(item_ids)}
    col_lookup = {label: index for index, label in enumerate(item_labels)}
    row_order = np.asarray([row_lookup[x] for x in canonical_image_ids])
    col_order = np.asarray([col_lookup[x] for x in canonical_labels])

    aligned_scores = item["scores"][row_order][:, col_order]
    aligned_y_true = item["y_true"][row_order][:, col_order]
    return aligned_scores, aligned_y_true


def compute_map_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 10):
    ap_values = []
    for y, score in zip(y_true, scores):
        positive_indices = np.where(y == 1)[0]
        if len(positive_indices) == 0:
            continue

        top_k = np.argsort(-score)[:k]
        hits = 0
        precisions = []
        for rank, index in enumerate(top_k, start=1):
            if y[index] == 1:
                hits += 1
                precisions.append(hits / rank)

        denominator = min(len(positive_indices), k)
        ap_values.append(
            float(np.sum(precisions) / denominator) if precisions else 0.0
        )

    return float(np.mean(ap_values)) if ap_values else None


def compute_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    """Match the metrics produced by the supplied zero-shot evaluator."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    if y_true.shape != scores.shape:
        raise ValueError(
            f"Metric input shape mismatch: y_true={y_true.shape}, scores={scores.shape}"
        )

    metrics: Dict[str, Any] = {}
    per_label_auc: Dict[str, Optional[float]] = {}
    auc_values = []

    for column, label in enumerate(label_names):
        y = y_true[:, column]
        if len(np.unique(y)) < 2:
            per_label_auc[label] = None
            continue
        auc = float(roc_auc_score(y, scores[:, column]))
        per_label_auc[label] = auc
        auc_values.append(auc)

    metrics["per_label_auc"] = per_label_auc
    metrics["macro_auc"] = float(np.mean(auc_values)) if auc_values else None

    try:
        metrics["micro_auc"] = float(
            roc_auc_score(y_true.ravel(), scores.ravel())
        )
    except ValueError:
        metrics["micro_auc"] = None

    y_pred = (scores >= threshold).astype(int)
    per_label_f1: Dict[str, Optional[float]] = {}
    per_label_support: Dict[str, int] = {}
    per_label_pred_pos: Dict[str, int] = {}
    f1_values = []

    for column, label in enumerate(label_names):
        y = y_true[:, column]
        y_hat = y_pred[:, column]
        per_label_support[label] = int(y.sum())
        per_label_pred_pos[label] = int(y_hat.sum())
        if len(np.unique(y)) < 2:
            per_label_f1[label] = None
            continue
        value = float(f1_score(y, y_hat, zero_division=0))
        per_label_f1[label] = value
        f1_values.append(value)

    metrics["threshold_fixed"] = float(threshold)
    metrics["per_label_f1"] = per_label_f1
    metrics["per_label_support"] = per_label_support
    metrics["per_label_pred_pos"] = per_label_pred_pos
    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else None
    metrics["micro_f1"] = float(
        f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    )

    thresholds = np.linspace(float(scores.min()), float(scores.max()), 20)
    per_label_best_f1: Dict[str, Optional[float]] = {}
    per_label_best_threshold: Dict[str, Optional[float]] = {}
    best_f1_values = []

    for column, label in enumerate(label_names):
        y = y_true[:, column]
        score = scores[:, column]
        if len(np.unique(y)) < 2:
            per_label_best_f1[label] = None
            per_label_best_threshold[label] = None
            continue

        candidates = [
            (float(f1_score(y, score >= value, zero_division=0)), float(value))
            for value in thresholds
        ]
        best_f1, best_threshold = max(candidates, key=lambda item: item[0])
        per_label_best_f1[label] = best_f1
        per_label_best_threshold[label] = best_threshold
        best_f1_values.append(best_f1)

    metrics["threshold_grid"] = [float(value) for value in thresholds]
    metrics["per_label_best_f1"] = per_label_best_f1
    metrics["per_label_best_threshold"] = per_label_best_threshold
    metrics["macro_best_f1"] = (
        float(np.mean(best_f1_values)) if best_f1_values else None
    )

    global_candidates = [
        (
            float(
                f1_score(
                    y_true.ravel(),
                    (scores >= value).astype(int).ravel(),
                    zero_division=0,
                )
            ),
            float(value),
        )
        for value in thresholds
    ]
    best_micro_f1, best_global_threshold = max(
        global_candidates, key=lambda item: item[0]
    )
    metrics["micro_best_f1"] = best_micro_f1
    metrics["best_global_threshold"] = best_global_threshold
    metrics["map_at_10"] = compute_map_at_k(y_true, scores, k=10)
    return metrics


def selected_macro_auc(
    per_label_auc: Dict[str, Optional[float]],
    requested_labels: Optional[Sequence[str]],
) -> Optional[Dict[str, Any]]:
    if not requested_labels:
        return None
    duplicates = sorted(
        label
        for label, count in Counter(requested_labels).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"--macro_auc_labels contains duplicates: {duplicates}")

    missing = [label for label in requested_labels if label not in per_label_auc]
    if missing:
        raise ValueError(f"Configured macro-AUC labels are missing: {missing}")

    undefined = [
        label for label in requested_labels if per_label_auc[label] is None
    ]
    if undefined:
        raise ValueError(
            f"Configured macro-AUC labels have undefined test AUC: {undefined}"
        )

    values = [float(per_label_auc[label]) for label in requested_labels]
    return {
        "labels": list(requested_labels),
        "per_label_auc": {
            label: float(per_label_auc[label]) for label in requested_labels
        },
        "macro_auc": float(np.mean(values)),
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    if not cleaned:
        raise ValueError("ensemble_name must contain at least one safe character")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align VinDr zero-shot scores by image ID and label name, then "
            "compute a weighted score-level ensemble."
        )
    )
    parser.add_argument(
        "--score_files",
        nargs="+",
        required=True,
        help="Two or more score .npz files produced with --save_scores.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Optional display names in the same order as --score_files.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional non-negative weights; defaults to equal averaging.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ensemble_name", default="ensemble")
    parser.add_argument("--threshold", type=float, default=0.3948)
    parser.add_argument(
        "--macro_auc_labels",
        nargs="+",
        default=None,
        help=(
            "Optional fixed labels for an additional matched macro AUC. "
            "Quote labels containing spaces."
        ),
    )
    parser.add_argument(
        "--save_scores_csv",
        action="store_true",
        help="Also save a human-readable wide CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = [Path(path) for path in args.score_files]
    if len(score_paths) < 2:
        raise ValueError("At least two --score_files are required")

    if args.model_names is None:
        model_names = [path.stem for path in score_paths]
    else:
        model_names = list(args.model_names)
        if len(model_names) != len(score_paths):
            raise ValueError(
                "--model_names must have the same length as --score_files"
            )
        if len(set(model_names)) != len(model_names):
            raise ValueError("--model_names must be unique")

    if args.weights is None:
        weights = np.ones(len(score_paths), dtype=np.float64)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if len(weights) != len(score_paths):
            raise ValueError("--weights must have the same length as --score_files")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("--weights must be finite and non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one ensemble weight must be positive")
    weights = weights / weights.sum()

    loaded = [load_score_file(path) for path in score_paths]
    canonical_image_ids = loaded[0]["image_ids"]
    canonical_labels = loaded[0]["label_names"]
    canonical_y_true = loaded[0]["y_true"]

    aligned_scores = []
    for model_name, item in zip(model_names, loaded):
        scores, y_true = align_to_canonical(
            item, canonical_image_ids, canonical_labels
        )
        if not np.array_equal(y_true, canonical_y_true):
            mismatch_count = int(np.count_nonzero(y_true != canonical_y_true))
            raise ValueError(
                f"{item['path']}: aligned ground truth disagrees with the "
                f"canonical file at {mismatch_count} cells ({model_name})"
            )
        aligned_scores.append(scores.astype(np.float64))

    ensemble_scores = np.zeros_like(aligned_scores[0], dtype=np.float64)
    for weight, scores in zip(weights, aligned_scores):
        ensemble_scores += float(weight) * scores
    ensemble_scores = ensemble_scores.astype(np.float32)

    metrics = compute_classification_metrics(
        y_true=canonical_y_true,
        scores=ensemble_scores,
        label_names=canonical_labels,
        threshold=float(args.threshold),
    )
    matched = selected_macro_auc(
        metrics["per_label_auc"], args.macro_auc_labels
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(args.ensemble_name)

    scores_path = output_dir / f"vindr_zero_shot_scores_{name}.npz"
    np.savez_compressed(
        scores_path,
        image_ids=np.asarray(canonical_image_ids, dtype=object),
        label_names=np.asarray(canonical_labels, dtype=object),
        scores=ensemble_scores,
        y_true=canonical_y_true.astype(np.float32),
    )

    csv_path = None
    if args.save_scores_csv:
        score_df = pd.DataFrame(
            ensemble_scores,
            columns=[f"score::{label}" for label in canonical_labels],
        )
        score_df.insert(0, "image_id", canonical_image_ids)
        truth_df = pd.DataFrame(
            canonical_y_true,
            columns=[f"y::{label}" for label in canonical_labels],
        )
        csv_path = output_dir / f"vindr_zero_shot_scores_{name}.csv"
        pd.concat([score_df, truth_df], axis=1).to_csv(csv_path, index=False)

    result = {
        "ensemble_name": args.ensemble_name,
        "method": "weighted_arithmetic_mean_of_raw_scores",
        "model_names": model_names,
        "score_files": [str(path) for path in score_paths],
        "normalized_weights": {
            model: float(weight) for model, weight in zip(model_names, weights)
        },
        "alignment": {
            "rows": "image_id",
            "columns": "label_name",
            "canonical_file": str(score_paths[0]),
        },
        "num_images": len(canonical_image_ids),
        "label_names": canonical_labels,
        "threshold": float(args.threshold),
        "classification": metrics,
        "matched_macro_auc": matched,
        "scores_file_npz": str(scores_path),
        "scores_file_csv": str(csv_path) if csv_path is not None else None,
    }

    json_path = output_dir / f"vindr_zero_shot_{name}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("[Alignment] Rows aligned explicitly by image_id")
    print("[Alignment] Columns aligned explicitly by label name")
    print(f"[Ensemble] models={model_names}")
    print(f"[Ensemble] weights={weights.tolist()}")
    print(f"[Metrics] macro_auc={metrics['macro_auc']}")
    if matched is not None:
        print(f"[Metrics] matched_macro_auc={matched['macro_auc']}")
    print(f"[Output] Scores:  {scores_path}")
    print(f"[Output] Metrics: {json_path}")


if __name__ == "__main__":
    main()
