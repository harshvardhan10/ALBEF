"""
Online VinDr validation for ALBEF checkpoint selection.

The current in-memory model is evaluated after each pretraining epoch using:
1. Cardiomegaly ROC-AUC as the primary checkpoint-selection metric.
2. Mean FROC sensitivity across configured FP/image targets as a tie-breaker.

No per-epoch checkpoint or Grad-CAM file is written. Only the selected model and
its compact validation score vector are retained by the pretraining script.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src import (
        get_image_embeddings,
        get_image_transform,
        get_label_text_embeddings,
        get_label_text_inputs,
    )
    from albef_crossattn_gradcam import (
        enable_crossattn_attention_saving,
        generate_albef_crossattn_gradcam,
        register_albef_crossattn_gradcam_hooks,
        remove_albef_crossattn_gradcam_hooks,
    )
except ModuleNotFoundError:
    from scripts.src import (
        get_image_embeddings,
        get_image_transform,
        get_label_text_embeddings,
        get_label_text_inputs,
    )
    from scripts.albef_crossattn_gradcam import (
        enable_crossattn_attention_saving,
        generate_albef_crossattn_gradcam,
        register_albef_crossattn_gradcam_hooks,
        remove_albef_crossattn_gradcam_hooks,
    )


class VinDrOnlineValidationDataset(Dataset):
    """Fixed VinDr validation dataset for original, lung, or heart view."""

    def __init__(
        self,
        labels_csv: str | Path,
        images_root: str | Path,
        view_type: str,
        transform,
        label_name: str = "Cardiomegaly",
        mask_root: Optional[str | Path] = None,
        max_images: Optional[int] = None,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.images_root = Path(images_root)
        self.view_type = str(view_type).lower().strip()
        self.transform = transform
        self.label_name = str(label_name)
        self.mask_root = None if mask_root in (None, "") else Path(mask_root)

        if self.view_type not in {"original", "lung", "heart"}:
            raise ValueError(
                f"view_type must be original, lung, or heart; got {self.view_type!r}"
            )
        if self.view_type != "original" and self.mask_root is None:
            raise ValueError(
                f"mask_root is required for view_type={self.view_type!r}. "
                "Generate the fixed VinDr validation masks before training."
            )

        self.df = pd.read_csv(self.labels_csv)
        if self.df.empty:
            raise ValueError(f"Validation labels CSV is empty: {self.labels_csv}")

        self.id_col = self.df.columns[0]
        self.label_cols = list(self.df.columns[1:])
        if self.label_name not in self.label_cols:
            raise ValueError(
                f"Label {self.label_name!r} is not present in {self.labels_csv}. "
                f"Available labels: {self.label_cols}"
            )

        if max_images is not None:
            self.df = self.df.iloc[: int(max_images)].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        image_ids = self.df[self.id_col].astype(str).tolist()
        image_paths = [self.images_root / f"{image_id}.png" for image_id in image_ids]
        missing_images = [path for path in image_paths if not path.exists()]

        missing_masks: List[Path] = []
        if self.view_type != "original":
            assert self.mask_root is not None
            mask_paths = [
                self.mask_root / image_id[:2] / f"{image_id}.png"
                for image_id in image_ids
            ]
            missing_masks = [path for path in mask_paths if not path.exists()]

        if missing_images or missing_masks:
            lines = [
                "Missing files in the fixed VinDr validation subset.",
                f"labels_csv={self.labels_csv}",
                f"view_type={self.view_type}",
                f"selected_rows={len(self.df)}",
            ]
            if missing_images:
                lines.append(
                    f"Missing original images: {len(missing_images)}. First paths:"
                )
                lines.extend(f"  {path}" for path in missing_images[:10])
            if missing_masks:
                lines.append(
                    f"Missing {self.view_type} masks: {len(missing_masks)}. First paths:"
                )
                lines.extend(f"  {path}" for path in missing_masks[:10])
                lines.append(
                    "Decode/generate masks for the fixed validation IDs before starting "
                    "the lung/heart pretraining jobs."
                )
            raise FileNotFoundError("\n".join(lines))

        labels = self.df[self.label_name].to_numpy(dtype=np.int64)
        unique = np.unique(labels)
        if not set(unique).issubset({0, 1}):
            raise ValueError(
                f"{self.label_name} must be binary 0/1. Found: {unique.tolist()}"
            )
        if len(unique) < 2:
            raise ValueError(
                f"Validation subset must contain positive and negative "
                f"{self.label_name} examples. Found: {unique.tolist()}"
            )

        print(
            f"[VinDrOnlineValidationDataset] view={self.view_type} "
            f"images={len(self.df)} positives={int(labels.sum())} "
            f"negatives={int((labels == 0).sum())}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row[self.id_col])
        image_path = self.images_root / f"{image_id}.png"

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        native_width, native_height = image.size

        if self.view_type != "original":
            assert self.mask_root is not None
            mask_path = self.mask_root / image_id[:2] / f"{image_id}.png"
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L")

            if image.size != mask.size:
                raise ValueError(
                    "Image/mask native dimension mismatch for "
                    f"image_id={image_id}: image={image.size} at {image_path}, "
                    f"mask={mask.size} at {mask_path}."
                )

            black = Image.new("RGB", image.size, (0, 0, 0))
            image = Image.composite(image, black, mask)

        image_tensor = self.transform(image)
        label = np.float32(row[self.label_name])
        return image_tensor, label, image_id, native_width, native_height


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    image_id: str


def minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if xmax - xmin < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def heatmap_to_boxes(
    heatmap: np.ndarray,
    image_id: str,
    label: str,
    threshold: float,
    min_box_area_frac: float,
    score_mode: str,
    connectivity: int = 8,
) -> List[Box]:
    heatmap = minmax_norm(heatmap)
    binary_map = (heatmap >= float(threshold)).astype(np.uint8)
    height, width = binary_map.shape
    min_area = max(1, int(round(float(min_box_area_frac) * height * width)))

    num_components, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary_map, connectivity=connectivity
    )

    boxes: List[Box] = []
    for component_id in range(1, num_components):
        x, y, box_width, box_height, area = stats[component_id]
        if area < min_area:
            continue

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + box_width), int(y + box_height)
        component_mask = component_map[y1:y2, x1:x2] == component_id
        values = heatmap[y1:y2, x1:x2][component_mask]
        if values.size == 0:
            continue

        if score_mode == "max":
            score = float(values.max())
        elif score_mode == "mean":
            score = float(values.mean())
        elif score_mode == "area_mean":
            score = float(values.mean() * area)
        else:
            raise ValueError(
                f"score_mode must be max, mean, or area_mean; got {score_mode!r}"
            )

        boxes.append(
            Box(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=score,
                label=label,
                image_id=image_id,
            )
        )

    boxes.sort(key=lambda item: item.score, reverse=True)
    return boxes


def box_iou(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def quadrant_match(
    pred_box: Tuple[float, float, float, float],
    gt_box: Tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> bool:
    def center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def quadrant(x, y):
        return int(y >= image_height / 2.0) * 2 + int(x >= image_width / 2.0)

    return quadrant(*center(pred_box)) == quadrant(*center(gt_box))


def boxes_match(
    pred_box: Tuple[float, float, float, float],
    gt_box: Tuple[float, float, float, float],
    image_width: float,
    image_height: float,
    match_mode: str,
    iou_threshold: float,
) -> bool:
    if match_mode == "quadrant":
        return quadrant_match(
            pred_box,
            gt_box,
            image_width=image_width,
            image_height=image_height,
        )
    if match_mode == "iou":
        return box_iou(pred_box, gt_box) >= float(iou_threshold)
    raise ValueError(f"match_mode must be quadrant or iou; got {match_mode!r}")


def evaluate_froc_for_label(
    predictions: Sequence[Box],
    gt_boxes: Sequence[Box],
    label: str,
    num_images: int,
    image_size_lookup: Dict[str, Tuple[int, int]],
    match_mode: str,
    iou_threshold: float,
    targets: Sequence[float],
) -> Dict[float, float]:
    preds = sorted(
        [prediction for prediction in predictions if prediction.label == label],
        key=lambda item: -item.score,
    )
    gts = [gt for gt in gt_boxes if gt.label == label]
    if not gts:
        raise ValueError(f"No validation GT boxes found for label={label!r}.")

    gt_by_image: Dict[str, List[Box]] = {}
    for gt in gts:
        gt_by_image.setdefault(gt.image_id, []).append(gt)

    matched_gt = set()
    true_positives = 0
    false_positives = 0
    curve_rows: List[Tuple[float, float]] = []

    for pred in preds:
        if pred.image_id not in image_size_lookup:
            raise KeyError(f"Missing image size for image_id={pred.image_id}")

        image_width, image_height = image_size_lookup[pred.image_id]
        pred_xyxy = (pred.x1, pred.y1, pred.x2, pred.y2)
        matched = False

        for gt_index, gt in enumerate(gt_by_image.get(pred.image_id, [])):
            gt_key = (pred.image_id, label, gt_index)
            if gt_key in matched_gt:
                continue
            gt_xyxy = (gt.x1, gt.y1, gt.x2, gt.y2)
            if boxes_match(
                pred_box=pred_xyxy,
                gt_box=gt_xyxy,
                image_width=image_width,
                image_height=image_height,
                match_mode=match_mode,
                iou_threshold=iou_threshold,
            ):
                matched_gt.add(gt_key)
                matched = True
                break

        if matched:
            true_positives += 1
        else:
            false_positives += 1

        curve_rows.append(
            (
                false_positives / float(num_images),
                true_positives / float(len(gts)),
            )
        )

    sensitivities: Dict[float, float] = {}
    for target in targets:
        valid = [
            sensitivity
            for fp_per_image, sensitivity in curve_rows
            if fp_per_image <= float(target)
        ]
        sensitivities[float(target)] = max(valid) if valid else 0.0
    return sensitivities


@dataclass
class VinDrValidationMetrics:
    epoch: int
    view_type: str
    num_classification_images: int
    num_localization_images: int
    cardiomegaly_auc: float
    cardiomegaly_best_f1: float
    cardiomegaly_best_threshold: float
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    localization_score: float
    froc_sensitivities: Dict[str, float]
    num_gt_boxes: int
    num_pred_boxes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationSelectionState:
    max_auc_seen: float = float("-inf")
    selected_epoch: int = -1
    selected_auc: float = float("-inf")
    selected_localization_score: float = float("-inf")

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> "ValidationSelectionState":
        if not value:
            return cls()
        return cls(
            max_auc_seen=float(value.get("max_auc_seen", float("-inf"))),
            selected_epoch=int(value.get("selected_epoch", -1)),
            selected_auc=float(value.get("selected_auc", float("-inf"))),
            selected_localization_score=float(
                value.get("selected_localization_score", float("-inf"))
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def consider(
        self,
        metrics: VinDrValidationMetrics,
        auc_tolerance: float = 0.002,
        localization_min_delta: float = 0.0,
    ) -> Tuple[bool, str]:
        auc = float(metrics.cardiomegaly_auc)
        localization = float(metrics.localization_score)
        if not np.isfinite(auc) or not np.isfinite(localization):
            return False, "non-finite validation metric"

        if auc > self.max_auc_seen:
            self.max_auc_seen = auc

        if self.selected_epoch < 0:
            should_select = True
            reason = "first validated checkpoint"
        else:
            selected_outside_band = (
                self.selected_auc < self.max_auc_seen - float(auc_tolerance)
            )
            candidate_inside_band = (
                auc >= self.max_auc_seen - float(auc_tolerance)
            )
            localization_improved = (
                localization
                > self.selected_localization_score + float(localization_min_delta)
            )

            if selected_outside_band and candidate_inside_band:
                should_select = True
                reason = "new AUC maximum moved previous selection outside AUC band"
            elif candidate_inside_band and localization_improved:
                should_select = True
                reason = "AUC within tolerance and localization improved"
            else:
                should_select = False
                reason = "hierarchical AUC/localization criterion not improved"

        if should_select:
            self.selected_epoch = int(metrics.epoch)
            self.selected_auc = auc
            self.selected_localization_score = localization

        return should_select, reason


def atomic_torch_save(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def atomic_save_validation_scores_npz(
    output_path: str | Path,
    *,
    outputs: dict,
    metrics: VinDrValidationMetrics,
) -> None:
    """Atomically save the selected checkpoint's compact validation scores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    required = {"image_ids", "labels", "scores"}
    missing = required - set(outputs)
    if missing:
        raise ValueError(
            f"classification outputs missing required keys: {sorted(missing)}"
        )

    with open(temporary_path, "wb") as handle:
        np.savez_compressed(
            handle,
            image_ids=np.asarray(outputs["image_ids"], dtype=object),
            y_true=np.asarray(outputs["labels"], dtype=np.int64),
            scores=np.asarray(outputs["scores"], dtype=np.float32),
            epoch=np.asarray(metrics.epoch, dtype=np.int64),
            view_type=np.asarray(metrics.view_type),
            cardiomegaly_auc=np.asarray(
                metrics.cardiomegaly_auc, dtype=np.float32
            ),
            best_f1=np.asarray(metrics.cardiomegaly_best_f1, dtype=np.float32),
            best_threshold=np.asarray(
                metrics.cardiomegaly_best_threshold, dtype=np.float32
            ),
            score_min=np.asarray(metrics.score_min, dtype=np.float32),
            score_max=np.asarray(metrics.score_max, dtype=np.float32),
            score_mean=np.asarray(metrics.score_mean, dtype=np.float32),
            score_std=np.asarray(metrics.score_std, dtype=np.float32),
        )
    os.replace(temporary_path, output_path)


class VinDrOnlineValidationRunner:
    def __init__(
        self,
        *,
        classification_labels_csv: str | Path,
        localization_labels_csv: Optional[str | Path],
        localization_annotations_csv: str | Path,
        images_root: str | Path,
        view_type: str,
        image_res: int,
        mask_root: Optional[str | Path] = None,
        label_name: str = "Cardiomegaly",
        classification_batch_size: int = 64,
        classification_num_workers: int = 4,
        max_classification_images: Optional[int] = None,
        max_localization_images: Optional[int] = None,
        layers_to_use: Sequence[int] = (8,),
        max_text_length: int = 32,
        cam_key: str = "cam_vis",
        heatmap_threshold: float = 0.50,
        min_box_area_frac: float = 0.002,
        score_mode: str = "max",
        match_mode: str = "quadrant",
        iou_threshold: float = 0.1,
        froc_targets: Sequence[float] = (0.10, 0.25, 0.50),
        threshold_steps: int = 200,
    ) -> None:
        self.label_name = str(label_name)
        self.view_type = str(view_type).lower().strip()
        self.layers_to_use = [int(layer) for layer in layers_to_use]
        self.max_text_length = int(max_text_length)
        self.cam_key = str(cam_key)
        self.heatmap_threshold = float(heatmap_threshold)
        self.min_box_area_frac = float(min_box_area_frac)
        self.score_mode = str(score_mode)
        self.match_mode = str(match_mode)
        self.iou_threshold = float(iou_threshold)
        self.froc_targets = [float(target) for target in froc_targets]
        self.threshold_steps = max(2, int(threshold_steps))

        transform = get_image_transform(int(image_res))
        self.classification_dataset = VinDrOnlineValidationDataset(
            labels_csv=classification_labels_csv,
            images_root=images_root,
            mask_root=mask_root,
            view_type=self.view_type,
            transform=transform,
            label_name=label_name,
            max_images=max_classification_images,
        )

        localization_csv = (
            classification_labels_csv
            if localization_labels_csv is None
            else localization_labels_csv
        )
        self.localization_dataset = VinDrOnlineValidationDataset(
            labels_csv=localization_csv,
            images_root=images_root,
            mask_root=mask_root,
            view_type=self.view_type,
            transform=transform,
            label_name=label_name,
            max_images=max_localization_images,
        )

        self.classification_loader = DataLoader(
            self.classification_dataset,
            batch_size=int(classification_batch_size),
            shuffle=False,
            num_workers=int(classification_num_workers),
            pin_memory=True,
            drop_last=False,
        )
        self.localization_loader = DataLoader(
            self.localization_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        annotations = pd.read_csv(localization_annotations_csv)
        annotations.columns = [column.strip() for column in annotations.columns]
        required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
        missing = required - set(annotations.columns)
        if missing:
            raise ValueError(
                f"localization_annotations_csv missing columns: {sorted(missing)}"
            )

        annotations["image_id"] = annotations["image_id"].astype(str)
        annotations["class_name"] = annotations["class_name"].astype(str)
        selected_ids = set(
            self.localization_dataset.df[
                self.localization_dataset.id_col
            ].astype(str)
        )
        self.annotations = annotations[
            (annotations["class_name"] == self.label_name)
            & (annotations["image_id"].isin(selected_ids))
        ].copy()
        if self.annotations.empty:
            raise ValueError(
                f"No {self.label_name} boxes remain in the localization subset."
            )

        print(
            f"[VinDrOnlineValidationRunner] view={self.view_type} "
            f"localization_images={len(self.localization_dataset)} "
            f"gt_boxes={len(self.annotations)} layers={self.layers_to_use} "
            f"targets={self.froc_targets}",
            flush=True,
        )

    def _classification_metrics(self, model, tokenizer, device) -> dict:
        label_embeddings = get_label_text_embeddings(
            model=model,
            tokenizer=tokenizer,
            labels=[self.label_name],
            device=device,
            max_length=self.max_text_length,
        ).to(device)

        all_scores: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_ids: List[str] = []

        with torch.no_grad():
            for images, labels, image_ids, _, _ in self.classification_loader:
                images = images.to(device, non_blocking=True)
                image_embeddings = get_image_embeddings(model, images)
                similarities = image_embeddings @ label_embeddings.t()
                all_scores.append(
                    similarities[:, 0].detach().cpu().numpy().astype(np.float32)
                )
                all_labels.append(labels.numpy().astype(np.int64))
                all_ids.extend(map(str, image_ids))

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"Cannot compute {self.label_name} AUC: only one class present."
            )

        auc = float(roc_auc_score(labels, scores))
        thresholds = np.linspace(
            float(scores.min()), float(scores.max()), self.threshold_steps
        )
        best_f1 = -1.0
        best_threshold = float(thresholds[0])
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(np.int64)
            value = float(f1_score(labels, predictions, zero_division=0))
            if value > best_f1:
                best_f1 = value
                best_threshold = float(threshold)

        return {
            "auc": auc,
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "scores": scores,
            "labels": labels,
            "image_ids": all_ids,
        }

    def _ground_truth_boxes(self) -> List[Box]:
        boxes: List[Box] = []
        deduplicated = self.annotations.drop_duplicates(
            subset=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"]
        )
        for row in deduplicated.itertuples(index=False):
            boxes.append(
                Box(
                    x1=float(row.x_min),
                    y1=float(row.y_min),
                    x2=float(row.x_max),
                    y2=float(row.y_max),
                    score=1.0,
                    label=str(row.class_name),
                    image_id=str(row.image_id),
                )
            )
        return boxes

    def _localization_metrics(self, model, tokenizer, device) -> dict:
        input_ids_dict, attention_mask_dict, token_mask_dict = get_label_text_inputs(
            tokenizer=tokenizer,
            labels=[self.label_name],
            max_length=self.max_text_length,
        )

        enable_crossattn_attention_saving(model, layers=self.layers_to_use)
        handles = register_albef_crossattn_gradcam_hooks(model)
        predictions: List[Box] = []
        image_size_lookup: Dict[str, Tuple[int, int]] = {}

        try:
            for index, batch in enumerate(self.localization_loader, start=1):
                images, _, image_ids, native_widths, native_heights = batch
                image_id = str(image_ids[0])
                native_width = int(native_widths[0])
                native_height = int(native_heights[0])
                image_size_lookup[image_id] = (native_width, native_height)

                model.zero_grad(set_to_none=True)
                image_tensor = images.to(device, non_blocking=True)
                with torch.enable_grad():
                    cams = generate_albef_crossattn_gradcam(
                        model=model,
                        img_tensor=image_tensor,
                        input_ids=input_ids_dict[self.label_name],
                        attention_mask=attention_mask_dict[self.label_name],
                        device=device,
                        text_token_mask=token_mask_dict[self.label_name],
                        layers_to_use=self.layers_to_use,
                        prefer_getters=True,
                    )

                if self.cam_key not in cams:
                    raise KeyError(
                        f"cam_key={self.cam_key!r} not returned. "
                        f"Available keys: {list(cams.keys())}"
                    )

                cam = cams[self.cam_key]
                if torch.is_tensor(cam):
                    cam = cam.detach().float().cpu().numpy()
                cam = np.squeeze(np.asarray(cam, dtype=np.float32))
                if cam.ndim != 2:
                    raise ValueError(
                        f"Expected 2D CAM for image_id={image_id}, got {cam.shape}"
                    )

                cam_native = cv2.resize(
                    minmax_norm(cam),
                    (native_width, native_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                cam_native = minmax_norm(cam_native)
                predictions.extend(
                    heatmap_to_boxes(
                        heatmap=cam_native,
                        image_id=image_id,
                        label=self.label_name,
                        threshold=self.heatmap_threshold,
                        min_box_area_frac=self.min_box_area_frac,
                        score_mode=self.score_mode,
                    )
                )
                model.zero_grad(set_to_none=True)

                if index % 50 == 0 or index == len(self.localization_loader):
                    print(
                        f"[VinDr validation:localization] {index}/"
                        f"{len(self.localization_loader)} images",
                        flush=True,
                    )
        finally:
            remove_albef_crossattn_gradcam_hooks(handles)
            model.zero_grad(set_to_none=True)

        gt_boxes = self._ground_truth_boxes()
        sensitivities = evaluate_froc_for_label(
            predictions=predictions,
            gt_boxes=gt_boxes,
            label=self.label_name,
            num_images=len(self.localization_dataset),
            image_size_lookup=image_size_lookup,
            match_mode=self.match_mode,
            iou_threshold=self.iou_threshold,
            targets=self.froc_targets,
        )
        return {
            "localization_score": float(np.mean(list(sensitivities.values()))),
            "sensitivities": sensitivities,
            "num_gt_boxes": len(gt_boxes),
            "num_pred_boxes": len(predictions),
        }

    def evaluate(
        self,
        model,
        tokenizer,
        device,
        epoch: int,
        return_classification_outputs: bool = False,
    ):
        was_training = bool(model.training)
        model.eval()
        try:
            classification = self._classification_metrics(model, tokenizer, device)
            localization = self._localization_metrics(model, tokenizer, device)
        finally:
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()

        froc_dict = {
            f"sens@{target:.2f}": float(localization["sensitivities"][target])
            for target in self.froc_targets
        }
        metrics = VinDrValidationMetrics(
            epoch=int(epoch),
            view_type=self.view_type,
            num_classification_images=len(self.classification_dataset),
            num_localization_images=len(self.localization_dataset),
            cardiomegaly_auc=float(classification["auc"]),
            cardiomegaly_best_f1=float(classification["best_f1"]),
            cardiomegaly_best_threshold=float(classification["best_threshold"]),
            score_min=float(classification["score_min"]),
            score_max=float(classification["score_max"]),
            score_mean=float(classification["score_mean"]),
            score_std=float(classification["score_std"]),
            localization_score=float(localization["localization_score"]),
            froc_sensitivities=froc_dict,
            num_gt_boxes=int(localization["num_gt_boxes"]),
            num_pred_boxes=int(localization["num_pred_boxes"]),
        )

        print(
            f"[VinDr validation] epoch={metrics.epoch} view={metrics.view_type} "
            f"AUC={metrics.cardiomegaly_auc:.6f} "
            f"localization={metrics.localization_score:.6f} "
            + " ".join(
                f"{name}={value:.6f}"
                for name, value in metrics.froc_sensitivities.items()
            ),
            flush=True,
        )

        if return_classification_outputs:
            outputs = {
                "image_ids": classification["image_ids"],
                "labels": classification["labels"],
                "scores": classification["scores"],
            }
            return metrics, outputs
        return metrics
