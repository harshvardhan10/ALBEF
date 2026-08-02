#!/usr/bin/env python3
"""Learn two validation-only multi-view ensembles from saved score matrices.

This script fits one non-negative, sum-to-one weight vector for each checkpoint
family:

1. ``best_macro``: maximize validation macro ROC-AUC over a predefined label set.
2. ``best_cardiomegaly``: maximize validation Cardiomegaly ROC-AUC.

Each model/label is optionally z-score normalized with statistics fitted only on
that checkpoint family's validation predictions. The learned transformation and
weights are then applied unchanged to optional test predictions.

Expected NPZ arrays: image_ids (N,), label_names (L,), scores (N,L), y_true
(N,L). Rows and columns are explicitly aligned before fitting or ensembling.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


REQUIRED_KEYS = {"image_ids", "label_names", "scores", "y_true"}


def unique_strings(values: np.ndarray, field: str, path: Path) -> List[str]:
    result = [str(v) for v in np.asarray(values).reshape(-1).tolist()]
    duplicates = sorted(v for v, n in Counter(result).items() if n > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate {field}: {duplicates[:10]}")
    if any(not v for v in result):
        raise ValueError(f"{path}: {field} contains an empty value")
    return result


def load_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing arrays {sorted(missing)}")
        image_ids = unique_strings(data["image_ids"], "image_ids", path)
        labels = unique_strings(data["label_names"], "label_names", path)
        scores = np.asarray(data["scores"], dtype=np.float64)
        y_true = np.asarray(data["y_true"], dtype=np.int64)
    expected = (len(image_ids), len(labels))
    if scores.shape != expected or y_true.shape != expected:
        raise ValueError(
            f"{path}: expected {expected}; scores={scores.shape}, y_true={y_true.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: scores contain NaN or infinity")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(f"{path}: y_true must be binary")
    return {"path": str(path), "image_ids": image_ids, "label_names": labels,
            "scores": scores, "y_true": y_true}


def align_item(
    item: Dict[str, Any], canonical_ids: Sequence[str], canonical_labels: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if set(item["image_ids"]) != set(canonical_ids):
        raise ValueError(f"{item['path']}: image-ID set differs from the first file")
    if set(item["label_names"]) != set(canonical_labels):
        raise ValueError(f"{item['path']}: label set differs from the first file")
    rows = {v: i for i, v in enumerate(item["image_ids"])}
    cols = {v: i for i, v in enumerate(item["label_names"])}
    row_order = np.asarray([rows[v] for v in canonical_ids])
    col_order = np.asarray([cols[v] for v in canonical_labels])
    return item["scores"][row_order][:, col_order], item["y_true"][row_order][:, col_order]


def align_files(
    paths: Sequence[Path], model_names: Sequence[str], split: str
) -> Tuple[List[str], List[str], np.ndarray, List[np.ndarray]]:
    items = [load_npz(path) for path in paths]
    ids, labels, y_true = items[0]["image_ids"], items[0]["label_names"], items[0]["y_true"]
    scores: List[np.ndarray] = []
    for item, model in zip(items, model_names):
        aligned_scores, aligned_y = align_item(item, ids, labels)
        if not np.array_equal(aligned_y, y_true):
            count = int(np.count_nonzero(aligned_y != y_true))
            raise ValueError(f"{split}/{model}: ground truth differs at {count} cells")
        scores.append(aligned_scores)
    return ids, labels, y_true, scores


def load_macro_labels(config: Optional[str], cli_labels: Optional[Sequence[str]]) -> List[str]:
    if cli_labels:
        labels = [str(x) for x in cli_labels]
    elif config:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install PyYAML to use --config") from exc
        with open(config, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        labels = [str(x) for x in cfg.get("vindr_validation", {}).get("macro_auc_labels", [])]
    else:
        raise ValueError("Provide --macro_auc_labels or --config")
    if not labels:
        raise ValueError("macro_auc_labels is empty")
    if len(set(labels)) != len(labels):
        raise ValueError("macro_auc_labels contains duplicates")
    return labels


def fit_zscore(
    val_scores: Sequence[np.ndarray], epsilon: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    means, scales, transformed = [], [], []
    for scores in val_scores:
        mean = scores.mean(axis=0)
        std = scores.std(axis=0, ddof=0)
        scale = np.where(std < epsilon, 1.0, std)
        means.append(mean)
        scales.append(scale)
        transformed.append((scores - mean[None, :]) / scale[None, :])
    return transformed, means, scales


def transform_scores(
    scores: Sequence[np.ndarray], means: Sequence[np.ndarray], scales: Sequence[np.ndarray]
) -> List[np.ndarray]:
    return [(s - m[None, :]) / z[None, :] for s, m, z in zip(scores, means, scales)]


def simplex_weights(count: int, step: float):
    units_float = 1.0 / step
    units = int(round(units_float))
    if not np.isclose(units_float, units, atol=1e-10):
        raise ValueError("--weight_step must divide 1 exactly (for example 0.1, 0.05, 0.02, 0.01)")

    def compositions(total: int, parts: int, prefix: Tuple[int, ...] = ()):
        if parts == 1:
            yield prefix + (total,)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, prefix + (value,))

    for values in compositions(units, count):
        yield np.asarray(values, dtype=np.float64) / units


def auc_for_label(y: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y).size != 2:
        raise ValueError("ROC-AUC is undefined because a selected label has one class")
    return float(roc_auc_score(y, scores))


def objective(
    y_true: np.ndarray, scores: np.ndarray, columns: Sequence[int]
) -> float:
    return float(np.mean([auc_for_label(y_true[:, c], scores[:, c]) for c in columns]))


def weighted_sum(model_scores: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, np.stack(model_scores, axis=0), axes=(0, 0))


def learn_weights(
    model_scores: Sequence[np.ndarray], y_true: np.ndarray,
    columns: Sequence[int], step: float,
) -> Tuple[np.ndarray, float, int]:
    best_weights: Optional[np.ndarray] = None
    best_value = -np.inf
    best_distance = np.inf
    equal = np.full(len(model_scores), 1.0 / len(model_scores))
    evaluated = 0
    for weights in simplex_weights(len(model_scores), step):
        value = objective(y_true, weighted_sum(model_scores, weights), columns)
        distance = float(np.sum((weights - equal) ** 2))
        # Deterministic tie-break: closest to equal weights, then lexicographic.
        if (value > best_value + 1e-12 or
                (np.isclose(value, best_value, atol=1e-12, rtol=0.0) and distance < best_distance - 1e-12)):
            best_weights, best_value, best_distance = weights.copy(), value, distance
        evaluated += 1
    assert best_weights is not None
    return best_weights, float(best_value), evaluated


def per_label_auc(y_true: np.ndarray, scores: np.ndarray, labels: Sequence[str]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for i, label in enumerate(labels):
        result[label] = None if np.unique(y_true[:, i]).size != 2 else auc_for_label(y_true[:, i], scores[:, i])
    return result


def family_run(
    *, name: str, val_paths: Sequence[Path], test_paths: Optional[Sequence[Path]],
    model_names: Sequence[str], objective_labels: Sequence[str], objective_name: str,
    method: str, step: float, epsilon: float, output_dir: Path,
) -> Dict[str, Any]:
    val_ids, labels, val_y, raw_val = align_files(val_paths, model_names, f"{name}/validation")
    missing = [label for label in objective_labels if label not in labels]
    if missing:
        raise ValueError(f"{name}: objective labels absent from score files: {missing}")
    columns = [labels.index(label) for label in objective_labels]

    if method == "validation_zscore":
        val_scores, means, scales = fit_zscore(raw_val, epsilon)
    else:
        val_scores = raw_val
        means = [np.zeros(len(labels)) for _ in raw_val]
        scales = [np.ones(len(labels)) for _ in raw_val]

    weights, best_auc, evaluated = learn_weights(val_scores, val_y, columns, step)
    equal_weights = np.full(len(model_names), 1.0 / len(model_names))
    ensemble_val = weighted_sum(val_scores, weights)
    equal_val = weighted_sum(val_scores, equal_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_output = output_dir / f"{name}_validation_ensemble_scores.npz"
    np.savez_compressed(val_output, image_ids=np.asarray(val_ids, dtype=object),
                        label_names=np.asarray(labels, dtype=object),
                        scores=ensemble_val.astype(np.float32), y_true=val_y.astype(np.int8),
                        weights=weights.astype(np.float64))

    result: Dict[str, Any] = {
        "checkpoint_family": name,
        "objective": objective_name,
        "objective_labels": list(objective_labels),
        "normalization": method,
        "weight_step": step,
        "candidate_weight_vectors_evaluated": evaluated,
        "learned_weights": {m: float(w) for m, w in zip(model_names, weights)},
        "validation_objective_auc": best_auc,
        "equal_weight_validation_objective_auc": objective(val_y, equal_val, columns),
        "validation_per_label_auc": per_label_auc(val_y, ensemble_val, labels),
        "validation_score_files": [str(p) for p in val_paths],
        "validation_ensemble_scores_file": str(val_output),
        "zscore_parameters": ({
            model: {
                "per_label_mean": {label: float(means[i][j]) for j, label in enumerate(labels)},
                "per_label_scale": {label: float(scales[i][j]) for j, label in enumerate(labels)},
            } for i, model in enumerate(model_names)
        } if method == "validation_zscore" else {}),
    }

    if test_paths:
        test_ids, test_labels, test_y, raw_test = align_files(test_paths, model_names, f"{name}/test")
        if set(test_labels) != set(labels):
            raise ValueError(f"{name}: validation and test label sets differ")
        if test_labels != labels:
            order = np.asarray([test_labels.index(label) for label in labels])
            test_y = test_y[:, order]
            raw_test = [scores[:, order] for scores in raw_test]
        test_scores = transform_scores(raw_test, means, scales)
        ensemble_test = weighted_sum(test_scores, weights)
        test_output = output_dir / f"{name}_test_ensemble_scores.npz"
        np.savez_compressed(test_output, image_ids=np.asarray(test_ids, dtype=object),
                            label_names=np.asarray(labels, dtype=object),
                            scores=ensemble_test.astype(np.float32), y_true=test_y.astype(np.int8),
                            weights=weights.astype(np.float64))
        result.update({
            "test_score_files": [str(p) for p in test_paths],
            "test_objective_auc": objective(test_y, ensemble_test, columns),
            "test_per_label_auc": per_label_auc(test_y, ensemble_test, labels),
            "test_ensemble_scores_file": str(test_output),
        })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_names", nargs="+", default=["original", "lung", "heart"])
    parser.add_argument("--best_macro_val_files", nargs="+", required=True)
    parser.add_argument("--best_cardio_val_files", nargs="+", required=True)
    parser.add_argument("--best_macro_test_files", nargs="+", default=None)
    parser.add_argument("--best_cardio_test_files", nargs="+", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--macro_auc_labels", nargs="+", default=None)
    parser.add_argument("--target_label", default="Cardiomegaly")
    parser.add_argument("--method", choices=["validation_zscore", "raw"], default="validation_zscore")
    parser.add_argument("--weight_step", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(args.model_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("--model_names must contain at least two unique names")
    families = [args.best_macro_val_files, args.best_cardio_val_files]
    optional = [args.best_macro_test_files, args.best_cardio_test_files]
    for paths in families + [x for x in optional if x is not None]:
        if len(paths) != len(names):
            raise ValueError("Every supplied file list must have one file per model")
    if not (0.0 < args.weight_step <= 1.0):
        raise ValueError("--weight_step must be in (0, 1]")

    macro_labels = load_macro_labels(args.config, args.macro_auc_labels)
    output_dir = Path(args.output_dir)
    macro_result = family_run(
        name="best_macro_auc", val_paths=[Path(x) for x in args.best_macro_val_files],
        test_paths=None if args.best_macro_test_files is None else [Path(x) for x in args.best_macro_test_files],
        model_names=names, objective_labels=macro_labels,
        objective_name="macro ROC-AUC over macro_auc_labels", method=args.method,
        step=args.weight_step, epsilon=args.epsilon, output_dir=output_dir)
    cardio_result = family_run(
        name="best_cardiomegaly_auc", val_paths=[Path(x) for x in args.best_cardio_val_files],
        test_paths=None if args.best_cardio_test_files is None else [Path(x) for x in args.best_cardio_test_files],
        model_names=names, objective_labels=[args.target_label],
        objective_name=f"{args.target_label} ROC-AUC", method=args.method,
        step=args.weight_step, epsilon=args.epsilon, output_dir=output_dir)

    report = {
        "model_names": names,
        "fit_policy": "non-negative weights summing to one; selected using validation labels only",
        "normalization": args.method,
        "best_macro_auc_family": macro_result,
        "best_cardiomegaly_auc_family": cardio_result,
        "note": "Test labels are used only for final reporting when test files are supplied, never for fitting.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "learned_ensemble_weights_and_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for result in (macro_result, cardio_result):
        print(f"[{result['checkpoint_family']}] {result['objective']}")
        print(f"  weights: {result['learned_weights']}")
        print(f"  learned validation AUC: {result['validation_objective_auc']:.6f}")
        print(f"  equal-weight validation AUC: {result['equal_weight_validation_objective_auc']:.6f}")
    print(f"[Output] {report_path}")


if __name__ == "__main__":
    main()
