#!/usr/bin/env python3
"""
Convert saved ALBEF cross-attention Grad-CAMs to boxes and compute FROC.

Geometry assumed by this experiment:
  * model/evaluation image: 256 x 256 pixels
  * saved cam_raw:          16 x 16 patches
  * one patch footprint:    16 x 16 = 256 image pixels

Validation mode searches (threshold, minimum component area) and selects the
configuration maximizing mean sensitivity at 0.10, 0.25 and 0.50 FP/image.
Test mode evaluates one frozen configuration.
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


EPS = 1e-12
MODEL_SIZE = 256
PATCH_GRID = 16
PATCH_SIDE = MODEL_SIZE // PATCH_GRID
PATCH_AREA_PX = PATCH_SIDE * PATCH_SIDE  # exactly 256 pixels


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    image_id: str


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return value.strip("_") or "label"


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["image_id"] = df["image_id"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    return df


def load_meta(path: Path) -> pd.DataFrame:
    """
    Load native VinDr image dimensions.

    The canonical train_meta.csv columns are image_id, width and height. A few
    common aliases are accepted so column naming differences fail informatively.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    id_col = "image_id"
    width_col = "dim1"
    height_col = "dim0"

    meta = df[[id_col, width_col, height_col]].copy()
    meta.columns = ["image_id", "original_width", "original_height"]
    meta["image_id"] = meta["image_id"].astype(str)

    duplicated = meta.loc[
        meta["image_id"].duplicated(keep=False), "image_id"
    ].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Duplicate image IDs in meta CSV. Examples: {duplicated[:10]}"
        )

    meta["original_width"] = pd.to_numeric(meta["original_width"], errors="coerce")
    meta["original_height"] = pd.to_numeric(meta["original_height"], errors="coerce")
    invalid = meta[
        ~np.isfinite(meta["original_width"])
        | ~np.isfinite(meta["original_height"])
        | (meta["original_width"] <= 0)
        | (meta["original_height"] <= 0)
    ]
    if not invalid.empty:
        raise ValueError(
            "Invalid native dimensions in meta CSV. Examples: "
            f"{invalid.head(10).to_dict(orient='records')}"
        )
    return meta


def load_evaluation_ids(path: Path, max_images: Optional[int]) -> List[str]:
    """The evaluation CSV defines the denominator, including negative images."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    id_col = next((c for c in ["image_id", "dicom_id", "id"] if c in df.columns), df.columns[0])
    ids = df[id_col].astype(str).tolist()
    duplicates = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate image IDs in evaluation CSV. Examples: {duplicates[:10]}")
    if max_images is not None:
        ids = ids[:max_images]
    if not ids:
        raise ValueError("Evaluation set is empty.")
    return ids


def load_heatmap_index(heatmaps_dir: Path) -> Dict[str, str]:
    path = heatmaps_dir / "crossattn_gradcam_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing heatmap index: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    id_col = next((c for c in ["image_id", "dicom_id", "id"] if c in df.columns), None)
    file_col = next(
        (c for c in ["heatmap_path", "path", "file", "pt_path", "npz_path"] if c in df.columns),
        None,
    )
    if id_col is None or file_col is None:
        raise ValueError(f"Invalid heatmap index columns: {list(df.columns)}")
    df[id_col] = df[id_col].astype(str)
    duplicated = df.loc[df[id_col].duplicated(keep=False), id_col].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicate heatmap index IDs. Examples: {duplicated[:10]}")

    result: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw_path = Path(str(row[file_col]))
        if not raw_path.is_absolute():
            candidate = heatmaps_dir / raw_path
            raw_path = candidate if candidate.exists() else raw_path
        result[str(row[id_col])] = str(raw_path.resolve())
    return result


def load_heatmap_object(path: str):
    if path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu")
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if path.endswith(".npy"):
        value = np.load(path, allow_pickle=True)
        return value.item() if value.dtype == object and value.size == 1 else value
    raise ValueError(f"Unsupported heatmap extension: {path}")


def validate_and_extract_cam(
    obj,
    image_id: str,
    label: str,
    expected_view: Optional[str],
    expected_layer: int,
) -> np.ndarray:
    metadata = obj.get("__metadata__", {}) if isinstance(obj, dict) else {}
    if metadata.get("image_id") not in (None, image_id):
        raise ValueError(
            f"{image_id}: metadata image_id={metadata.get('image_id')} does not match."
        )
    if expected_view is not None and metadata.get("view") != expected_view:
        raise ValueError(
            f"{image_id}: expected view={expected_view}, found {metadata.get('view')}."
        )
    if label not in obj or "cam_raw" not in obj[label]:
        raise KeyError(f"{image_id}: missing {label!r}/cam_raw.")
    layers = obj[label].get("layers_to_use")
    if layers is not None and list(layers) != [expected_layer]:
        raise ValueError(f"{image_id}: expected layer [{expected_layer}], found {layers}.")

    cam = obj[label]["cam_raw"]
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()
    cam = np.squeeze(np.asarray(cam, dtype=np.float32))
    if cam.shape != (PATCH_GRID, PATCH_GRID):
        raise ValueError(
            f"{image_id}: expected cam_raw shape {(PATCH_GRID, PATCH_GRID)}, got {cam.shape}."
        )
    if not np.isfinite(cam).all():
        raise ValueError(f"{image_id}: cam_raw contains NaN or infinity.")
    return np.maximum(cam, 0.0)


def preflight(
    evaluation_ids: Sequence[str],
    heatmap_index: Dict[str, str],
) -> None:
    expected = set(evaluation_ids)
    indexed = set(heatmap_index)
    missing_ids = sorted(expected - indexed)
    extra_ids = sorted(indexed - expected)
    missing_files = [
        image_id for image_id in evaluation_ids
        if image_id in heatmap_index and not Path(heatmap_index[image_id]).is_file()
    ]
    if missing_ids or missing_files:
        messages = []
        if missing_ids:
            messages.append(f"missing heatmaps ({len(missing_ids)}): {missing_ids[:10]}")
        if missing_files:
            messages.append(f"missing files ({len(missing_files)}): {missing_files[:10]}")
        raise RuntimeError("Heatmap coverage check failed: " + "; ".join(messages))
    if extra_ids:
        print(
            f"[Preflight] Ignoring {len(extra_ids)} indexed heatmaps outside "
            "the evaluation CSV.",
            flush=True,
        )


def prepare_maps(raw_patch_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample the unnormalised positive CAM once. Use its normalized copy only
    for spatial thresholding; retain raw_256 for cross-image candidate scores.
    """
    raw_256 = cv2.resize(
        raw_patch_cam,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    raw_256 = np.maximum(raw_256, 0.0)
    maximum = float(raw_256.max())
    norm_256 = raw_256 / (maximum + EPS) if maximum > 0.0 else np.zeros_like(raw_256)
    return raw_256, norm_256


def heatmap_to_boxes(
    raw_256: np.ndarray,
    norm_256: np.ndarray,
    image_id: str,
    label: str,
    threshold: float,
    min_area_patches: int,
    score_mode: str,
) -> List[Box]:
    if raw_256.shape != (MODEL_SIZE, MODEL_SIZE):
        raise ValueError(f"raw map shape must be 256x256, got {raw_256.shape}")
    min_area_px = int(min_area_patches) * PATCH_AREA_PX
    binary = (norm_256 >= float(threshold)).astype(np.uint8)
    n_components, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes: List[Box] = []
    for component_id in range(1, n_components):
        x, y, width, height, area = [int(v) for v in stats[component_id]]
        if area < min_area_px:
            continue

        component_pixels = raw_256[component_map == component_id]
        if score_mode == "sum_sqrt_area":
            score = float(component_pixels.sum(dtype=np.float64) / np.sqrt(area))
        elif score_mode == "max":
            score = float(component_pixels.max())
        elif score_mode == "sum":
            score = float(component_pixels.sum(dtype=np.float64))
        else:
            raise ValueError(f"Unknown score mode: {score_mode}")

        # Half-open [x1, x2), [y1, y2) coordinates in the same 256-space as GT.
        boxes.append(
            Box(
                x1=float(x),
                y1=float(y),
                x2=float(x + width),
                y2=float(y + height),
                score=score,
                label=label,
                image_id=image_id,
            )
        )
    return boxes


def get_gt_boxes(
    annotations: pd.DataFrame,
    meta: pd.DataFrame,
    evaluation_ids: Sequence[str],
    label: str,
) -> List[Box]:
    """
    Filter the full VinDr training annotations to the evaluation split and map
    native-coordinate GT boxes into the model/evaluation 256x256 space.
    """
    evaluation_set = set(evaluation_ids)
    subset = annotations[
        annotations["image_id"].isin(evaluation_set)
        & (annotations["class_name"] == label)
    ].drop_duplicates(
        subset=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"]
    ).copy()

    meta_subset = meta[meta["image_id"].isin(evaluation_set)].copy()
    missing_meta_ids = sorted(evaluation_set - set(meta_subset["image_id"]))
    if missing_meta_ids:
        raise ValueError(
            f"Meta CSV is missing {len(missing_meta_ids)} evaluation image IDs. "
            f"Examples: {missing_meta_ids[:10]}"
        )

    subset = subset.merge(
        meta_subset,
        on="image_id",
        how="left",
        validate="many_to_one",
    )

    boxes = []
    for row in subset.itertuples(index=False):
        native = [
            float(row.x_min),
            float(row.y_min),
            float(row.x_max),
            float(row.y_max),
        ]
        original_width = float(row.original_width)
        original_height = float(row.original_height)
        if not (0 <= native[0] < native[2] <= original_width):
            raise ValueError(
                f"{row.image_id}: invalid native GT x coordinates {native} "
                f"for width={original_width}."
            )
        if not (0 <= native[1] < native[3] <= original_height):
            raise ValueError(
                f"{row.image_id}: invalid native GT y coordinates {native} "
                f"for height={original_height}."
            )

        coords = [
            native[0] * MODEL_SIZE / original_width,
            native[1] * MODEL_SIZE / original_height,
            native[2] * MODEL_SIZE / original_width,
            native[3] * MODEL_SIZE / original_height,
        ]
        boxes.append(
            Box(*coords, score=1.0, label=label, image_id=str(row.image_id))
        )
    return boxes


def quadrant(box: Box) -> int:
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    return (2 if cy >= MODEL_SIZE / 2 else 0) + (1 if cx >= MODEL_SIZE / 2 else 0)


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (a.x2 - a.x1) * (a.y2 - a.y1)
        + (b.x2 - b.x1) * (b.y2 - b.y1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def is_match(prediction: Box, ground_truth: Box, mode: str, iou_threshold: float) -> bool:
    if mode == "quadrant":
        return quadrant(prediction) == quadrant(ground_truth)
    if mode == "iou":
        return iou(prediction, ground_truth) >= iou_threshold
    raise ValueError(f"Unknown match mode: {mode}")


def evaluate_froc(
    predictions: Sequence[Box],
    ground_truth: Sequence[Box],
    num_images: int,
    targets: Sequence[float],
    match_mode: str,
    iou_threshold: float,
) -> Tuple[pd.DataFrame, Dict[float, float]]:
    """
    Predictions with equal scores are consumed as one threshold group.
    A deterministic within-group order is used only for one-to-one matching.
    Curve points are recorded only after the complete tie group.
    """
    gt_by_image: Dict[str, List[Tuple[int, Box]]] = {}
    for gt_id, box in enumerate(ground_truth):
        gt_by_image.setdefault(box.image_id, []).append((gt_id, box))

    ordered = sorted(
        predictions,
        key=lambda b: (-b.score, b.image_id, b.y1, b.x1, b.y2, b.x2),
    )
    matched_gt = set()
    tp = fp = 0
    rows = [
        {
            "fp_per_image": 0.0,
            "sensitivity": 0.0,
            "threshold_score": np.inf,
            "tp": 0,
            "fp": 0,
        }
    ]

    start = 0
    while start < len(ordered):
        score = ordered[start].score
        end = start + 1
        while end < len(ordered) and ordered[end].score == score:
            end += 1

        for prediction in ordered[start:end]:
            match_id = None
            for gt_id, gt in gt_by_image.get(prediction.image_id, []):
                if gt_id not in matched_gt and is_match(
                    prediction, gt, match_mode, iou_threshold
                ):
                    match_id = gt_id
                    break
            if match_id is None:
                fp += 1
            else:
                matched_gt.add(match_id)
                tp += 1

        rows.append(
            {
                "fp_per_image": fp / float(num_images),
                "sensitivity": tp / float(len(ground_truth)) if ground_truth else 0.0,
                "threshold_score": score,
                "tp": tp,
                "fp": fp,
            }
        )
        start = end

    curve = pd.DataFrame(rows)
    sensitivity_at = {}
    for target in targets:
        eligible = curve[curve["fp_per_image"] <= float(target)]
        sensitivity_at[float(target)] = float(eligible["sensitivity"].max())
    return curve, sensitivity_at


def boxes_to_dataframe(boxes: Sequence[Box]) -> pd.DataFrame:
    columns = ["image_id", "class_name", "x_min", "y_min", "x_max", "y_max", "score"]
    rows = [
        {
            "image_id": b.image_id,
            "class_name": b.label,
            "x_min": b.x1,
            "y_min": b.y1,
            "x_max": b.x2,
            "y_max": b.y2,
            "score": b.score,
        }
        for b in boxes
    ]
    return pd.DataFrame(rows, columns=columns)


def load_all_maps(
    evaluation_ids: Sequence[str],
    heatmap_index: Dict[str, str],
    label: str,
    expected_view: Optional[str],
    expected_layer: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    maps = {}
    for index, image_id in enumerate(evaluation_ids, start=1):
        obj = load_heatmap_object(heatmap_index[image_id])
        raw_patch = validate_and_extract_cam(
            obj, image_id, label, expected_view, expected_layer
        )
        maps[image_id] = prepare_maps(raw_patch)
        if index % 100 == 0 or index == len(evaluation_ids):
            print(f"[Load] {index}/{len(evaluation_ids)} heatmaps", flush=True)
    return maps


def build_predictions(
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    label: str,
    threshold: float,
    min_area_patches: int,
    score_mode: str,
) -> List[Box]:
    predictions: List[Box] = []
    for image_id, (raw_256, norm_256) in maps.items():
        predictions.extend(
            heatmap_to_boxes(
                raw_256,
                norm_256,
                image_id,
                label,
                threshold,
                min_area_patches,
                score_mode,
            )
        )
    return predictions


def save_selected_run(
    output_dir: Path,
    prefix: str,
    label: str,
    predictions: Sequence[Box],
    ground_truth: Sequence[Box],
    curve: pd.DataFrame,
    summary: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_filename(prefix)}_{safe_filename(label)}"
    boxes_to_dataframe(predictions).to_csv(
        output_dir / f"{stem}_predictions.csv", index=False
    )
    boxes_to_dataframe(ground_truth).to_csv(
        output_dir / f"{stem}_gt_boxes.csv", index=False
    )
    curve.to_csv(output_dir / f"{stem}_froc_curve.csv", index=False)
    with open(output_dir / f"{stem}_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_ids = load_evaluation_ids(Path(args.evaluation_csv), args.max_images)
    if args.expected_num_images is not None and len(evaluation_ids) != args.expected_num_images:
        raise ValueError(
            f"Expected {args.expected_num_images} images, found {len(evaluation_ids)}."
        )

    heatmap_index = load_heatmap_index(Path(args.heatmaps_dir))
    preflight(evaluation_ids, heatmap_index)
    annotations = load_annotations(Path(args.annotations_csv))
    meta = load_meta(Path(args.meta_csv))
    ground_truth = get_gt_boxes(
        annotations, meta, evaluation_ids, args.label
    )
    maps = load_all_maps(
        evaluation_ids,
        heatmap_index,
        args.label,
        args.expected_view,
        args.expected_layer,
    )
    targets = parse_float_list(args.froc_targets)
    thresholds = (
        parse_float_list(args.thresholds)
        if args.mode == "validation"
        else [float(args.threshold)]
    )
    area_patches = (
        parse_int_list(args.min_area_patches_grid)
        if args.mode == "validation"
        else [int(args.min_area_patches)]
    )

    print(
        f"[Geometry] CAM={PATCH_GRID}x{PATCH_GRID}; image={MODEL_SIZE}x{MODEL_SIZE}; "
        f"patch={PATCH_SIDE}x{PATCH_SIDE}; patch area={PATCH_AREA_PX}px",
        flush=True,
    )
    print(
        f"[Data] images={len(evaluation_ids)} GT={len(ground_truth)} "
        f"thresholds={thresholds} min_area_patches={area_patches}",
        flush=True,
    )

    grid_rows = []
    run_cache = {}
    for threshold in thresholds:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"Threshold must be in (0, 1], got {threshold}.")
        for min_patches in area_patches:
            if min_patches < 0:
                raise ValueError(f"Minimum patch area cannot be negative: {min_patches}")
            predictions = build_predictions(
                maps, args.label, threshold, min_patches, args.score_mode
            )
            curve, sensitivity_at = evaluate_froc(
                predictions,
                ground_truth,
                len(evaluation_ids),
                targets,
                args.match_mode,
                args.iou_threshold,
            )
            objective = float(np.mean([sensitivity_at[t] for t in targets]))
            row = {
                "threshold": threshold,
                "min_area_patches": min_patches,
                "min_area_pixels_256": min_patches * PATCH_AREA_PX,
                "num_predictions": len(predictions),
                "objective_mean_sensitivity": objective,
            }
            row.update({f"sens@{target:.2f}": sensitivity_at[target] for target in targets})
            grid_rows.append(row)
            run_cache[(threshold, min_patches)] = (predictions, curve, sensitivity_at)
            print(
                f"[Run] tau={threshold:.4f} area={min_patches} patches "
                f"({min_patches * PATCH_AREA_PX}px) J={objective:.6f}",
                flush=True,
            )

    # Primary: largest J. Deterministic ties: smaller area, then lower threshold.
    ranked = sorted(
        grid_rows,
        key=lambda r: (
            -r["objective_mean_sensitivity"],
            r["min_area_patches"],
            r["threshold"],
        ),
    )
    selected = ranked[0]
    selected_key = (selected["threshold"], selected["min_area_patches"])
    predictions, curve, sensitivity_at = run_cache[selected_key]

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["objective_mean_sensitivity", "min_area_patches", "threshold"],
        ascending=[False, True, True],
    )
    grid_df.to_csv(output_dir / f"{safe_filename(args.prefix)}_grid_search.csv", index=False)

    summary = {
        "mode": args.mode,
        "label": args.label,
        "num_images": len(evaluation_ids),
        "num_gt_boxes": len(ground_truth),
        "num_predictions": len(predictions),
        "cam_key": "cam_raw",
        "cam_shape": [PATCH_GRID, PATCH_GRID],
        "model_image_shape": [MODEL_SIZE, MODEL_SIZE],
        "patch_side_pixels": PATCH_SIDE,
        "patch_area_pixels": PATCH_AREA_PX,
        "threshold": selected["threshold"],
        "min_area_patches": selected["min_area_patches"],
        "min_area_pixels_256": selected["min_area_pixels_256"],
        "score_mode": args.score_mode,
        "connectivity": 8,
        "match_mode": args.match_mode,
        "iou_threshold": args.iou_threshold,
        "froc_targets": targets,
        "sensitivity_at": {str(k): v for k, v in sensitivity_at.items()},
        "objective_mean_sensitivity": selected["objective_mean_sensitivity"],
        "expected_view": args.expected_view,
        "expected_layer": args.expected_layer,
        "heatmaps_dir": str(Path(args.heatmaps_dir)),
        "evaluation_csv": str(Path(args.evaluation_csv)),
        "annotations_csv": str(Path(args.annotations_csv)),
        "meta_csv": str(Path(args.meta_csv)),
        "gt_input_coordinate_space": "native",
        "gt_evaluation_coordinate_space": [MODEL_SIZE, MODEL_SIZE],
    }
    save_selected_run(
        output_dir,
        args.prefix,
        args.label,
        predictions,
        ground_truth,
        curve,
        summary,
    )
    print("[Selected]", json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FROC from 16x16 ALBEF cam_raw maps in 256x256 space."
    )
    parser.add_argument("--mode", choices=["validation", "test"], required=True)
    parser.add_argument("--heatmaps_dir", required=True)
    parser.add_argument(
        "--evaluation_csv",
        required=True,
        help="CSV defining every evaluated image, including negatives; first column may be image_id.",
    )
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument(
        "--meta_csv",
        required=True,
        help=(
            "VinDr train_meta.csv containing image_id and each image's native "
            "width and height."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument("--prefix", default="gradcam")
    parser.add_argument("--expected_view", choices=["original", "lung_only", "heart_only"])
    parser.add_argument("--expected_layer", type=int, default=8)
    parser.add_argument("--expected_num_images", type=int)
    parser.add_argument("--max_images", type=int)

    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Validation-only relative CAM thresholds.",
    )
    parser.add_argument(
        "--min_area_patches_grid",
        default="0,1,2,4",
        help="Validation-only minimum component areas in patch-equivalent units.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Test-only threshold frozen from validation.",
    )
    parser.add_argument(
        "--min_area_patches",
        type=int,
        help="Test-only minimum area frozen from validation.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["sum_sqrt_area", "max", "sum"],
        default="sum_sqrt_area",
    )
    parser.add_argument("--match_mode", choices=["quadrant", "iou"], default="quadrant")
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--froc_targets", default="0.10,0.25,0.50")
    args = parser.parse_args()

    if args.mode == "test" and (
        args.threshold is None or args.min_area_patches is None
    ):
        parser.error("Test mode requires --threshold and --min_area_patches.")
    if args.max_images is not None and args.expected_num_images is not None:
        parser.error("Do not combine --max_images with --expected_num_images.")
    return args


if __name__ == "__main__":
    run(parse_args())
