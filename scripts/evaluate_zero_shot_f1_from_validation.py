#!/usr/bin/env python3
"""
Evaluate one saved VinDr zero-shot score file without tuning on test labels.

The F1 threshold for every class is selected on the checkpoint's saved
2,000-image VinDr validation scores. The thresholds are then applied unchanged
to the VinDr test scores. ROC-AUC is always computed on the test scores.

Both inputs must be NPZ files containing:
    image_ids, label_names, scores, y_true
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def align_label_columns(
    item: Dict[str, Any], canonical_labels: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    labels = item["label_names"]
    if set(labels) != set(canonical_labels):
        missing = sorted(set(canonical_labels) - set(labels))
        extra = sorted(set(labels) - set(canonical_labels))
        raise ValueError(
            f"{item['path']}: label set differs; missing={missing}, extra={extra}"
        )
    lookup = {label: index for index, label in enumerate(labels)}
    order = np.asarray([lookup[label] for label in canonical_labels])
    return item["scores"][:, order], item["y_true"][:, order]


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Find an exact validation-set F1 optimum; break ties conservatively."""
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
    # A higher threshold predicts fewer positives when validation F1 is tied.
    return float(np.max(thresholds[best]))


def select_validation_thresholds(
    y_true: np.ndarray, scores: np.ndarray, label_names: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, Optional[float]], Optional[float]]:
    thresholds = np.full(len(label_names), np.nan, dtype=np.float64)
    by_label: Dict[str, Optional[float]] = {}
    for column, label in enumerate(label_names):
        threshold = best_f1_threshold(y_true[:, column], scores[:, column])
        by_label[label] = threshold
        if threshold is not None:
            thresholds[column] = threshold
    global_threshold = best_f1_threshold(y_true.ravel(), scores.ravel())
    return thresholds, by_label, global_threshold


def _mean_defined(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def compute_test_metrics(
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
                "A stable-label metric is undefined on validation/test; "
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
        raise ValueError("--model_name is empty after sanitization")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_scores", required=True)
    parser.add_argument("--test_scores", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--target_label", default="Cardiomegaly")
    parser.add_argument(
        "--config",
        default=None,
        help="Training YAML; reads vindr_validation.macro_auc_labels.",
    )
    parser.add_argument(
        "--macro_auc_labels",
        nargs="+",
        default=None,
        help="Overrides the stable-label list read from --config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = load_scores(Path(args.validation_scores))
    test = load_scores(Path(args.test_scores))
    labels = validation["label_names"]
    test_scores, test_y_true = align_label_columns(test, labels)
    stable_labels = load_stable_labels(args.config, args.macro_auc_labels)

    thresholds, threshold_map, global_threshold = select_validation_thresholds(
        validation["y_true"], validation["scores"], labels
    )
    test_metrics = compute_test_metrics(
        y_true=test_y_true,
        scores=test_scores,
        label_names=labels,
        thresholds=thresholds,
        global_threshold=global_threshold,
        stable_labels=stable_labels,
        target_label=args.target_label,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(args.model_name)
    output_path = output_dir / f"vindr_final_metrics_{name}.json"
    result = {
        "model_name": args.model_name,
        "method": "individual_raw_scores_with_validation_selected_thresholds",
        "validation_scores_file": str(Path(args.validation_scores)),
        "test_scores_file": str(Path(args.test_scores)),
        "threshold_source": "saved_2000_image_validation_scores",
        "per_label_thresholds": threshold_map,
        "test_metrics": test_metrics,
        "note": (
            "No threshold or normalization parameter was estimated from test "
            "labels. Test-optimized/oracle F1 is intentionally not reported."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"[Model] {args.model_name}")
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
    print(f"[Output] {output_path}")


if __name__ == "__main__":
    main()
