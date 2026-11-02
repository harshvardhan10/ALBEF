#!/usr/bin/env python3
"""
Validation-calibrated ensemble of saved VinDr zero-shot scores.

Rows are aligned by image ID and columns by the literal class name. For the
default ``validation_zscore`` method, each model/class is standardized using
only that model's 2,000-image validation scores:

    z = (score - validation_mean) / validation_std

The aligned standardized scores are averaged. Per-class F1 thresholds are
selected on the ensembled validation scores and applied unchanged to the
ensembled VinDr test scores. Test labels are never used for normalization,
threshold selection, or model weighting.
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
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


REQUIRED_KEYS = {"image_ids", "label_names", "scores", "y_true"}


def _unique_strings(values: np.ndarray, field: str, path: Path) -> List[str]:
    result = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    duplicates = sorted(
        value for value, count in Counter(result).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{path}: duplicate {field}: {duplicates[:10]}")
    if any(not value for value in result):
        raise ValueError(f"{path}: {field} contains an empty value")
    return result


def load_scores(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing arrays {sorted(missing)}")
        image_ids = _unique_strings(data["image_ids"], "image_ids", path)
        label_names = _unique_strings(data["label_names"], "label_names", path)
        scores = np.asarray(data["scores"], dtype=np.float64)
        y_true = np.asarray(data["y_true"], dtype=np.int64)

    expected = (len(image_ids), len(label_names))
    if scores.shape != expected or y_true.shape != expected:
        raise ValueError(
            f"{path}: expected arrays of shape {expected}; "
            f"scores={scores.shape}, y_true={y_true.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: scores contain NaN or infinity")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(f"{path}: y_true must be binary")
    return {
        "path": str(path),
        "image_ids": image_ids,
        "label_names": label_names,
        "scores": scores,
        "y_true": y_true,
    }


def align_to_canonical(
    item: Dict[str, Any],
    canonical_image_ids: Sequence[str],
    canonical_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if set(item["image_ids"]) != set(canonical_image_ids):
        missing = sorted(set(canonical_image_ids) - set(item["image_ids"]))
        extra = sorted(set(item["image_ids"]) - set(canonical_image_ids))
        raise ValueError(
            f"{item['path']}: image-ID set differs; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if set(item["label_names"]) != set(canonical_labels):
        missing = sorted(set(canonical_labels) - set(item["label_names"]))
        extra = sorted(set(item["label_names"]) - set(canonical_labels))
        raise ValueError(
            f"{item['path']}: label set differs; missing={missing}, extra={extra}"
        )

    row_lookup = {
        image_id: index for index, image_id in enumerate(item["image_ids"])
    }
    column_lookup = {
        label: index for index, label in enumerate(item["label_names"])
    }
    row_order = np.asarray([row_lookup[value] for value in canonical_image_ids])
    column_order = np.asarray([column_lookup[value] for value in canonical_labels])
    return (
        item["scores"][row_order][:, column_order],
        item["y_true"][row_order][:, column_order],
    )


def align_split(
    items: Sequence[Dict[str, Any]], model_names: Sequence[str], split_name: str
) -> Tuple[List[str], List[str], np.ndarray, List[np.ndarray]]:
    canonical_ids = items[0]["image_ids"]
    canonical_labels = items[0]["label_names"]
    canonical_y = items[0]["y_true"]
    aligned_scores: List[np.ndarray] = []
    for item, model_name in zip(items, model_names):
        scores, y_true = align_to_canonical(
            item, canonical_ids, canonical_labels
        )
        if not np.array_equal(y_true, canonical_y):
            mismatches = int(np.count_nonzero(y_true != canonical_y))
            raise ValueError(
                f"{split_name}/{model_name}: ground truth differs at "
                f"{mismatches} cells after explicit alignment"
            )
        aligned_scores.append(scores)
    return canonical_ids, canonical_labels, canonical_y, aligned_scores


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return None
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0.0, atol=1e-12))
    return float(np.max(thresholds[best]))


def select_validation_thresholds(
    y_true: np.ndarray, scores: np.ndarray, label_names: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, Optional[float]], Optional[float]]:
    thresholds = np.full(len(label_names), np.nan, dtype=np.float64)
    mapping: Dict[str, Optional[float]] = {}
    for column, label in enumerate(label_names):
        threshold = best_f1_threshold(y_true[:, column], scores[:, column])
        mapping[label] = threshold
        if threshold is not None:
            thresholds[column] = threshold
    return (
        thresholds,
        mapping,
        best_f1_threshold(y_true.ravel(), scores.ravel()),
    )


def _mean_defined(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def compute_metrics(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    label_names: Sequence[str],
    thresholds: np.ndarray,
    global_threshold: Optional[float],
    stable_labels: Optional[Sequence[str]],
    target_label: str,
) -> Dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    if target_label not in label_to_index:
        raise ValueError(f"Target label {target_label!r} is absent")
    stable = list(stable_labels or [])
    missing_stable = [label for label in stable if label not in label_to_index]
    if missing_stable:
        raise ValueError(f"Stable labels are absent: {missing_stable}")

    per_label_auc: Dict[str, Optional[float]] = {}
    per_label_f1: Dict[str, Optional[float]] = {}
    per_label_support: Dict[str, int] = {}
    per_label_predicted_positive: Dict[str, Optional[int]] = {}
    for column, label in enumerate(label_names):
        target = y_true[:, column]
        per_label_support[label] = int(target.sum())
        if np.unique(target).size < 2:
            per_label_auc[label] = None
            per_label_f1[label] = None
            per_label_predicted_positive[label] = None
            continue
        per_label_auc[label] = float(roc_auc_score(target, scores[:, column]))
        if np.isnan(thresholds[column]):
            per_label_f1[label] = None
            per_label_predicted_positive[label] = None
        else:
            prediction = (scores[:, column] >= thresholds[column]).astype(int)
            per_label_f1[label] = float(
                f1_score(target, prediction, zero_division=0)
            )
            per_label_predicted_positive[label] = int(prediction.sum())

    if stable:
        undefined_auc = [label for label in stable if per_label_auc[label] is None]
        undefined_f1 = [label for label in stable if per_label_f1[label] is None]
        if undefined_auc or undefined_f1:
            raise ValueError(
                "A stable-label metric is undefined; "
                f"AUC={undefined_auc}, F1={undefined_f1}"
            )

    metrics: Dict[str, Any] = {
        "num_images": int(y_true.shape[0]),
        "num_labels": int(y_true.shape[1]),
        "per_label_auc": per_label_auc,
        "macro_auc_all_evaluable": _mean_defined(list(per_label_auc.values())),
        "per_label_f1_validation_threshold": per_label_f1,
        "macro_f1_all_evaluable_validation_threshold": _mean_defined(
            list(per_label_f1.values())
        ),
        "per_label_support": per_label_support,
        "per_label_predicted_positive": per_label_predicted_positive,
        "cardiomegaly_auc": per_label_auc[target_label],
        "cardiomegaly_f1_validation_threshold": per_label_f1[target_label],
    }
    if stable:
        metrics["stable_labels"] = stable
        metrics["macro_auc_stable"] = float(
            np.mean([per_label_auc[label] for label in stable])
        )
        metrics["macro_f1_stable_validation_threshold"] = float(
            np.mean([per_label_f1[label] for label in stable])
        )

    try:
        metrics["micro_auc"] = float(
            roc_auc_score(y_true.ravel(), scores.ravel())
        )
    except ValueError:
        metrics["micro_auc"] = None
    metrics["global_threshold_selected_on_validation"] = global_threshold
    metrics["micro_f1_validation_threshold"] = (
        None
        if global_threshold is None
        else float(
            f1_score(
                y_true.ravel(),
                (scores >= global_threshold).astype(int).ravel(),
                zero_division=0,
            )
        )
    )
    return metrics


def load_stable_labels(
    config_path: Optional[str], cli_labels: Optional[Sequence[str]]
) -> Optional[List[str]]:
    if cli_labels is not None:
        labels = list(cli_labels)
    elif config_path is not None:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required when --config is used") from error
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        labels = list(config.get("vindr_validation", {}).get("macro_auc_labels", []))
    else:
        return None
    if not labels:
        raise ValueError("The stable-label list is empty")
    duplicates = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Stable-label list contains duplicates: {duplicates}")
    return [str(label) for label in labels]


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    if not result:
        raise ValueError("--ensemble_name is empty after sanitization")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_score_files",
        nargs="+",
        required=True,
        help="One saved validation NPZ per model, in model-name order.",
    )
    parser.add_argument(
        "--test_score_files",
        nargs="+",
        required=True,
        help="One saved test NPZ per model, in the same order.",
    )
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=["validation_zscore", "raw_mean"],
        default="validation_zscore",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ensemble_name", required=True)
    parser.add_argument("--target_label", default="Cardiomegaly")
    parser.add_argument("--config", default=None)
    parser.add_argument("--macro_auc_labels", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--save_scores_csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_paths = [Path(value) for value in args.validation_score_files]
    test_paths = [Path(value) for value in args.test_score_files]
    model_names = list(args.model_names)
    count = len(model_names)
    if count < 2:
        raise ValueError("An ensemble requires at least two models")
    if len(set(model_names)) != count:
        raise ValueError("--model_names must be unique")
    if len(validation_paths) != count or len(test_paths) != count:
        raise ValueError(
            "--validation_score_files, --test_score_files and --model_names "
            "must have the same length"
        )

    if args.weights is None:
        weights = np.ones(count, dtype=np.float64)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if weights.shape != (count,):
            raise ValueError("--weights must have one value per model")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("--weights must be finite and non-negative")
    if float(weights.sum()) <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    weights /= weights.sum()

    validation_items = [load_scores(path) for path in validation_paths]
    test_items = [load_scores(path) for path in test_paths]
    val_ids, labels, val_y, val_model_scores = align_split(
        validation_items, model_names, "validation"
    )
    test_ids, test_labels, test_y, test_model_scores = align_split(
        test_items, model_names, "test"
    )
    if set(test_labels) != set(labels):
        raise ValueError("Validation and test label sets differ")
    if test_labels != labels:
        lookup = {label: index for index, label in enumerate(test_labels)}
        order = np.asarray([lookup[label] for label in labels])
        test_y = test_y[:, order]
        test_model_scores = [scores[:, order] for scores in test_model_scores]

    normalization: Dict[str, Any] = {}
    transformed_val: List[np.ndarray] = []
    transformed_test: List[np.ndarray] = []
    for model_name, val_scores, test_scores in zip(
        model_names, val_model_scores, test_model_scores
    ):
        if args.method == "validation_zscore":
            mean = val_scores.mean(axis=0)
            observed_std = val_scores.std(axis=0, ddof=0)
            constant = observed_std < float(args.epsilon)
            scale = np.where(constant, 1.0, observed_std)
            val_scores = (val_scores - mean[None, :]) / scale[None, :]
            test_scores = (test_scores - mean[None, :]) / scale[None, :]
            normalization[model_name] = {
                "fitted_on": "validation",
                "per_label_mean": {
                    label: float(mean[index])
                    for index, label in enumerate(labels)
                },
                "per_label_std": {
                    label: float(observed_std[index])
                    for index, label in enumerate(labels)
                },
                "constant_score_labels_scale_set_to_one": [
                    label
                    for index, label in enumerate(labels)
                    if constant[index]
                ],
            }
        transformed_val.append(val_scores)
        transformed_test.append(test_scores)

    ensemble_val = np.average(
        np.stack(transformed_val, axis=0), axis=0, weights=weights
    )
    ensemble_test = np.average(
        np.stack(transformed_test, axis=0), axis=0, weights=weights
    )
    thresholds, threshold_map, global_threshold = select_validation_thresholds(
        val_y, ensemble_val, labels
    )
    stable_labels = load_stable_labels(args.config, args.macro_auc_labels)
    validation_metrics = compute_metrics(
        y_true=val_y,
        scores=ensemble_val,
        label_names=labels,
        thresholds=thresholds,
        global_threshold=global_threshold,
        stable_labels=stable_labels,
        target_label=args.target_label,
    )
    test_metrics = compute_metrics(
        y_true=test_y,
        scores=ensemble_test,
        label_names=labels,
        thresholds=thresholds,
        global_threshold=global_threshold,
        stable_labels=stable_labels,
        target_label=args.target_label,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(args.ensemble_name)
    validation_output = output_dir / f"vindr_validation_scores_{name}.npz"
    test_output = output_dir / f"vindr_test_scores_{name}.npz"
    np.savez_compressed(
        validation_output,
        image_ids=np.asarray(val_ids, dtype=object),
        label_names=np.asarray(labels, dtype=object),
        scores=ensemble_val.astype(np.float32),
        y_true=val_y.astype(np.int8),
        per_label_thresholds=thresholds.astype(np.float32),
        global_threshold=np.asarray(global_threshold, dtype=np.float32),
    )
    np.savez_compressed(
        test_output,
        image_ids=np.asarray(test_ids, dtype=object),
        label_names=np.asarray(labels, dtype=object),
        scores=ensemble_test.astype(np.float32),
        y_true=test_y.astype(np.int8),
        per_label_thresholds=thresholds.astype(np.float32),
        global_threshold=np.asarray(global_threshold, dtype=np.float32),
    )

    csv_output = None
    if args.save_scores_csv:
        score_frame = pd.DataFrame(
            ensemble_test,
            columns=[f"score::{label}" for label in labels],
        )
        score_frame.insert(0, "image_id", test_ids)
        truth_frame = pd.DataFrame(
            test_y, columns=[f"y::{label}" for label in labels]
        )
        csv_output = output_dir / f"vindr_test_scores_{name}.csv"
        pd.concat([score_frame, truth_frame], axis=1).to_csv(
            csv_output, index=False
        )

    result = {
        "ensemble_name": args.ensemble_name,
        "method": args.method,
        "model_names": model_names,
        "normalized_weights": {
            model: float(weight)
            for model, weight in zip(model_names, weights)
        },
        "validation_score_files": [str(path) for path in validation_paths],
        "test_score_files": [str(path) for path in test_paths],
        "alignment": {
            "rows": "explicit image_id lookup within each split",
            "columns": "explicit literal label-name lookup",
        },
        "normalization": normalization,
        "threshold_source": "ensembled_2000_image_validation_scores",
        "per_label_thresholds": threshold_map,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_ensemble_scores_file": str(validation_output),
        "test_ensemble_scores_file": str(test_output),
        "test_ensemble_scores_csv": (
            None if csv_output is None else str(csv_output)
        ),
        "note": (
            "No normalization statistic, threshold, or weight was selected "
            "using VinDr test labels. Oracle/test-optimized F1 is not reported."
        ),
    }
    metrics_output = output_dir / f"vindr_ensemble_metrics_{name}.json"
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("[Alignment] rows=image_id, columns=literal label name")
    print(f"[Ensemble] method={args.method}, weights={weights.tolist()}")
    print(f"[Test] macro AUC all = {test_metrics['macro_auc_all_evaluable']}")
    if "macro_auc_stable" in test_metrics:
        print(f"[Test] macro AUC stable = {test_metrics['macro_auc_stable']}")
        print(
            "[Test] macro F1 stable (validation thresholds) = "
            f"{test_metrics['macro_f1_stable_validation_threshold']}"
        )
    print(f"[Test] {args.target_label} AUC = {test_metrics['cardiomegaly_auc']}")
    print(
        f"[Test] {args.target_label} F1 (validation threshold) = "
        f"{test_metrics['cardiomegaly_f1_validation_threshold']}"
    )
    print(f"[Output] {metrics_output}")


if __name__ == "__main__":
    main()
