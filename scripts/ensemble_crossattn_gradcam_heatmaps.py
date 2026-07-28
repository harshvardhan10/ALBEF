#!/usr/bin/env python3
"""
Ensemble aligned ALBEF cross-attention Grad-CAM heatmaps.

Each input directory must contain ``crossattn_gradcam_index.csv`` and the
per-image ``.pt`` files produced by extract_crossattn_gradcam_heatmaps.py.
Images are matched by image ID and the CAM is selected by its literal label
name (for example, ``Cardiomegaly``), never by class position.

Example
-------
python scripts/ensemble_crossattn_gradcam_heatmaps.py \
    --heatmap_dirs \
        outputs/gradcam/stable/original \
        outputs/gradcam/stable/lung \
        outputs/gradcam/stable/heart \
    --model_names original lung heart \
    --output_dir outputs/gradcam/stable/ensemble \
    --label Cardiomegaly
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _resolve_heatmap_path(value: str, heatmaps_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (heatmaps_dir / path).resolve()


def load_index(heatmaps_dir: Path) -> pd.DataFrame:
    index_path = heatmaps_dir / "crossattn_gradcam_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing heatmap index: {index_path}")

    frame = pd.read_csv(index_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    image_column = next(
        (column for column in ("image_id", "dicom_id", "id") if column in frame),
        None,
    )
    path_column = next(
        (
            column
            for column in ("heatmap_path", "path", "file", "pt_path", "npz_path")
            if column in frame
        ),
        None,
    )
    if image_column is None or path_column is None:
        raise ValueError(
            f"{index_path}: expected image-ID and heatmap-path columns; "
            f"found {list(frame.columns)}"
        )

    frame = frame.rename(
        columns={image_column: "image_id", path_column: "heatmap_path"}
    )
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].eq("").any():
        raise ValueError(f"{index_path}: empty image ID found")
    if frame["image_id"].duplicated().any():
        duplicates = (
            frame.loc[frame["image_id"].duplicated(False), "image_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"{index_path}: duplicate image IDs: {duplicates[:10]}")

    frame["heatmap_path"] = frame["heatmap_path"].map(
        lambda value: str(_resolve_heatmap_path(value, heatmaps_dir))
    )
    missing_files = [
        path for path in frame["heatmap_path"].tolist() if not Path(path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{index_path}: {len(missing_files)} indexed heatmap files are missing; "
            f"examples={missing_files[:5]}"
        )
    return frame


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_heatmap_object(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch_load_cpu(path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "heatmaps" in data.files:
                value = data["heatmaps"]
                return (
                    value.item()
                    if value.dtype == object and value.size == 1
                    else value
                )
            if len(data.files) == 1:
                value = data[data.files[0]]
                return (
                    value.item()
                    if value.dtype == object and value.size == 1
                    else value
                )
            return {key: data[key] for key in data.files}
    if suffix == ".npy":
        value = np.load(path, allow_pickle=True)
        return value.item() if value.dtype == object and value.size == 1 else value
    raise ValueError(f"Unsupported heatmap file extension: {path}")


def extract_label_object(
    heatmaps_object: Any, label: str, path: Path
) -> Dict[str, Any]:
    if not isinstance(heatmaps_object, dict):
        raise TypeError(f"{path}: expected a dictionary, got {type(heatmaps_object)}")
    if label not in heatmaps_object:
        raise KeyError(
            f"{path}: label {label!r} is missing; "
            f"available labels={list(heatmaps_object.keys())}"
        )
    label_object = heatmaps_object[label]
    if not isinstance(label_object, dict):
        raise TypeError(
            f"{path}: object for label {label!r} must be a dictionary"
        )
    return label_object


def as_2d_float_array(value: Any, context: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.squeeze(np.asarray(value, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"{context}: expected a 2D CAM, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: CAM contains NaN or infinite values")
    return array


def minmax_normalize(array: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    scale = maximum - minimum
    if scale <= epsilon:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / scale).astype(np.float32)


def ensemble_arrays(
    arrays: Sequence[np.ndarray],
    weights: np.ndarray,
    normalization: str,
) -> np.ndarray:
    shapes = [array.shape for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"CAM shapes do not match across models: {shapes}")

    if normalization == "per_model_minmax":
        prepared = [minmax_normalize(array) for array in arrays]
    elif normalization == "none":
        prepared = [array.astype(np.float32, copy=False) for array in arrays]
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    result = np.zeros_like(prepared[0], dtype=np.float64)
    for weight, array in zip(weights, prepared):
        result += float(weight) * array

    # The FROC evaluator expects a non-negative visual CAM. Re-normalizing also
    # keeps ensemble outputs comparable to the individual cam_vis(_up) files.
    return minmax_normalize(result.astype(np.float32))


def common_layers(
    label_objects: Sequence[Dict[str, Any]],
    paths: Sequence[Path],
) -> Optional[List[int]]:
    values = []
    for label_object, path in zip(label_objects, paths):
        raw = label_object.get("layers_to_use")
        if raw is None:
            values.append(None)
        elif isinstance(raw, (list, tuple)):
            values.append([int(value) for value in raw])
        else:
            values.append([int(raw)])

    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError(
            "layers_to_use metadata is present in only some model heatmaps: "
            + ", ".join(f"{path}={value}" for path, value in zip(paths, values))
        )
    if any(value != present[0] for value in present[1:]):
        raise ValueError(
            "layers_to_use differs across model heatmaps: "
            + ", ".join(f"{path}={value}" for path, value in zip(paths, values))
        )
    return present[0]


def verify_index_label_values(
    frames: Sequence[pd.DataFrame],
    row_indices: Sequence[int],
    label: str,
    image_id: str,
) -> Optional[float]:
    column = f"y::{label}"
    values = []
    for frame, row_index in zip(frames, row_indices):
        if column not in frame.columns or pd.isna(frame.iloc[row_index][column]):
            continue
        values.append(float(frame.iloc[row_index][column]))
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"Index ground truth disagrees for image_id={image_id}, "
            f"label={label}: {values}"
        )
    return values[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align cross-attention Grad-CAM files by image ID and label name, "
            "then create a pixelwise weighted ensemble."
        )
    )
    parser.add_argument(
        "--heatmap_dirs",
        nargs="+",
        required=True,
        help="Two or more individual heatmap directories.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Optional model names in the same order as --heatmap_dirs.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional non-negative weights; defaults to equal averaging.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument(
        "--cam_keys",
        nargs="+",
        default=["cam_raw", "cam_vis", "cam_vis_up"],
        help="CAM arrays to ensemble and save.",
    )
    parser.add_argument(
        "--normalization",
        choices=["per_model_minmax", "none"],
        default="per_model_minmax",
        help=(
            "Normalize each model CAM before averaging. "
            "per_model_minmax is recommended for multiview Grad-CAM."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ensemble .pt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heatmap_dirs = [Path(path) for path in args.heatmap_dirs]
    if len(heatmap_dirs) < 2:
        raise ValueError("At least two --heatmap_dirs are required")

    if args.model_names is None:
        model_names = [path.name for path in heatmap_dirs]
    else:
        model_names = list(args.model_names)
        if len(model_names) != len(heatmap_dirs):
            raise ValueError(
                "--model_names must have the same length as --heatmap_dirs"
            )
        if len(set(model_names)) != len(model_names):
            raise ValueError("--model_names must be unique")

    if args.weights is None:
        weights = np.ones(len(heatmap_dirs), dtype=np.float64)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if len(weights) != len(heatmap_dirs):
            raise ValueError(
                "--weights must have the same length as --heatmap_dirs"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("--weights must be finite and non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one ensemble weight must be positive")
    weights = weights / weights.sum()

    if not args.cam_keys or len(set(args.cam_keys)) != len(args.cam_keys):
        raise ValueError("--cam_keys must be a non-empty list without duplicates")

    frames = [load_index(path) for path in heatmap_dirs]
    canonical_ids = frames[0]["image_id"].tolist()
    canonical_set = set(canonical_ids)
    lookups = []

    for model_name, directory, frame in zip(model_names, heatmap_dirs, frames):
        image_ids = frame["image_id"].tolist()
        image_set = set(image_ids)
        if image_set != canonical_set:
            missing = sorted(canonical_set - image_set)
            extra = sorted(image_set - canonical_set)
            raise ValueError(
                f"{directory}: image-ID set differs for model={model_name}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        lookups.append({image_id: index for index, image_id in enumerate(image_ids)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_records = []

    for image_id in tqdm(canonical_ids, desc="Ensembling heatmaps"):
        row_indices = [lookup[image_id] for lookup in lookups]
        input_paths = [
            Path(frame.iloc[row_index]["heatmap_path"])
            for frame, row_index in zip(frames, row_indices)
        ]
        output_path = output_dir / f"{image_id}.pt"

        truth_value = verify_index_label_values(
            frames, row_indices, args.label, image_id
        )

        if output_path.exists() and not args.overwrite:
            record = {
                "image_id": image_id,
                "heatmap_path": str(output_path.resolve()),
                "status": "exists_skipped",
            }
            if truth_value is not None:
                record[f"y::{args.label}"] = truth_value
            index_records.append(record)
            continue

        heatmap_objects = [load_heatmap_object(path) for path in input_paths]
        label_objects = [
            extract_label_object(obj, args.label, path)
            for obj, path in zip(heatmap_objects, input_paths)
        ]
        layers = common_layers(label_objects, input_paths)

        output_label_object: Dict[str, Any] = {}
        for cam_key in args.cam_keys:
            arrays = []
            for label_object, path in zip(label_objects, input_paths):
                if cam_key not in label_object:
                    raise KeyError(
                        f"{path}: CAM key {cam_key!r} is missing for "
                        f"label {args.label!r}; available={list(label_object.keys())}"
                    )
                arrays.append(
                    as_2d_float_array(
                        label_object[cam_key],
                        f"{path}:{args.label}:{cam_key}",
                    )
                )

            ensemble = ensemble_arrays(
                arrays=arrays,
                weights=weights,
                normalization=args.normalization,
            )
            output_label_object[cam_key] = torch.from_numpy(ensemble)

        if layers is not None:
            output_label_object["layers_to_use"] = layers
        output_label_object["ensemble_model_names"] = model_names
        output_label_object["ensemble_weights"] = [
            float(weight) for weight in weights
        ]
        output_label_object["ensemble_normalization"] = args.normalization

        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save({args.label: output_label_object}, temporary_path)
        os.replace(temporary_path, output_path)

        record = {
            "image_id": image_id,
            "heatmap_path": str(output_path.resolve()),
            "status": "saved",
        }
        if truth_value is not None:
            record[f"y::{args.label}"] = truth_value
        index_records.append(record)

    index_path = output_dir / "crossattn_gradcam_index.csv"
    pd.DataFrame(index_records).to_csv(index_path, index=False)

    manifest = {
        "label": args.label,
        "cam_keys": list(args.cam_keys),
        "method": "pixelwise_weighted_arithmetic_mean",
        "normalization": args.normalization,
        "model_names": model_names,
        "heatmap_dirs": [str(path) for path in heatmap_dirs],
        "normalized_weights": {
            model: float(weight) for model, weight in zip(model_names, weights)
        },
        "alignment": {
            "images": "image_id",
            "class": "literal_label_name",
            "canonical_index": str(
                heatmap_dirs[0] / "crossattn_gradcam_index.csv"
            ),
        },
        "num_images": len(canonical_ids),
        "output_index": str(index_path),
    }
    manifest_path = output_dir / "crossattn_gradcam_ensemble_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("[Alignment] Images aligned explicitly by image_id")
    print(f"[Alignment] CAM selected by literal label name: {args.label}")
    print(f"[Ensemble] models={model_names}")
    print(f"[Ensemble] weights={weights.tolist()}")
    print(f"[Output] Index:    {index_path}")
    print(f"[Output] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
