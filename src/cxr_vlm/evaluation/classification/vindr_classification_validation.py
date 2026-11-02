"""Classification-only VinDr validation for ALBEF checkpoint selection.

This module evaluates every image-level VinDr label with the current in-memory
model. It does not compute Grad-CAM or localization metrics.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from src import get_image_embeddings, get_label_text_embeddings
except ModuleNotFoundError:
    from scripts.src import get_image_embeddings, get_label_text_embeddings


ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD = (0.26862954, 0.26130258, 0.27577711)


def build_validation_transform(image_res: int):
    """Deterministic transform used for checkpoint validation."""
    return transforms.Compose(
        [
            transforms.Resize(
                (int(image_res), int(image_res)),
                interpolation=Image.Resampling.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(ALBEF_MEAN, ALBEF_STD),
        ]
    )


class VinDrClassificationDataset(Dataset):
    """Fixed multi-label VinDr validation dataset for one visual view."""

    def __init__(
        self,
        *,
        labels_csv: str | Path,
        images_root: str | Path,
        view_type: str,
        image_res: int,
        mask_root: Optional[str | Path] = None,
        max_images: Optional[int] = None,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.images_root = Path(images_root)
        self.view_type = str(view_type).lower().strip()
        self.mask_root = None if mask_root in (None, "") else Path(mask_root)
        self.transform = build_validation_transform(image_res)

        if self.view_type not in {"original", "lung", "heart"}:
            raise ValueError(
                f"view_type must be original, lung, or heart; got {self.view_type!r}"
            )
        if self.view_type != "original" and self.mask_root is None:
            raise ValueError(f"mask_root is required for view_type={self.view_type}")

        self.df = pd.read_csv(self.labels_csv)
        if self.df.empty:
            raise ValueError(f"Validation CSV is empty: {self.labels_csv}")

        self.id_col = "image_id" if "image_id" in self.df.columns else self.df.columns[0]
        self.label_cols = [column for column in self.df.columns if column != self.id_col]
        if not self.label_cols:
            raise ValueError(f"No label columns found in {self.labels_csv}")

        self.df[self.id_col] = self.df[self.id_col].astype(str)
        if self.df[self.id_col].duplicated().any():
            examples = self.df.loc[
                self.df[self.id_col].duplicated(), self.id_col
            ].head(10).tolist()
            raise ValueError(
                f"Validation CSV contains duplicate image IDs. Examples: {examples}"
            )

        for label in self.label_cols:
            values = pd.to_numeric(self.df[label], errors="raise")
            unique = set(values.dropna().unique().tolist())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"Label {label!r} must be binary 0/1; found {sorted(unique)[:10]}"
                )
            self.df[label] = values.astype(np.int64)

        if max_images is not None:
            self.df = self.df.iloc[: int(max_images)].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        image_ids = self.df[self.id_col].tolist()
        missing_images = [
            self.images_root / f"{image_id}.png"
            for image_id in image_ids
            if not (self.images_root / f"{image_id}.png").exists()
        ]

        missing_masks: List[Path] = []
        if self.view_type != "original":
            assert self.mask_root is not None
            missing_masks = [
                self.mask_root / image_id[:2] / f"{image_id}.png"
                for image_id in image_ids
                if not (
                    self.mask_root / image_id[:2] / f"{image_id}.png"
                ).exists()
            ]

        if missing_images or missing_masks:
            lines = [
                "Missing files in VinDr classification validation.",
                f"rows={len(self.df)} view={self.view_type}",
            ]
            if missing_images:
                lines.append(f"Missing images: {len(missing_images)}")
                lines.extend(f"  {path}" for path in missing_images[:10])
            if missing_masks:
                lines.append(f"Missing masks: {len(missing_masks)}")
                lines.extend(f"  {path}" for path in missing_masks[:10])
            raise FileNotFoundError("\n".join(lines))

        print(
            f"[VinDrClassificationDataset] view={self.view_type} "
            f"images={len(self.df)} labels={len(self.label_cols)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_id = str(row[self.id_col])
        image_path = self.images_root / f"{image_id}.png"

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        if self.view_type != "original":
            assert self.mask_root is not None
            mask_path = self.mask_root / image_id[:2] / f"{image_id}.png"
            with Image.open(mask_path) as handle:
                mask = handle.convert("L")

            if mask.size != image.size:
                raise ValueError(
                    f"Image/mask mismatch for {image_id}: "
                    f"image={image.size}, mask={mask.size}"
                )
            image = Image.composite(
                image,
                Image.new("RGB", image.size, (0, 0, 0)),
                mask,
            )

        image = self.transform(image)
        labels = row[self.label_cols].to_numpy(dtype=np.float32)
        return image, labels, image_id


@dataclass
class VinDrClassificationMetrics:
    epoch: int
    view_type: str
    num_images: int
    cardiomegaly_auc: float
    macro_auc_all_evaluable: float
    macro_auc_stable: float
    macro_auc: float
    micro_auc: Optional[float]
    num_all_evaluable_auc_labels: int
    num_stable_auc_labels: int
    num_valid_auc_labels: int
    macro_auc_all_evaluable_labels: List[str]
    macro_auc_stable_labels: List[str]
    per_label_auc: Dict[str, Optional[float]]
    skipped_auc_labels: Dict[str, str]
    cardiomegaly_best_f1: float
    cardiomegaly_best_threshold: float
    cardiomegaly_score_min: float
    cardiomegaly_score_max: float
    cardiomegaly_score_mean: float
    cardiomegaly_score_std: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DualAUCSelectionState:
    best_cardiomegaly_auc: float = float("-inf")
    best_cardiomegaly_epoch: int = -1
    best_macro_auc: float = float("-inf")
    best_macro_epoch: int = -1
    validations_without_improvement: int = 0

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> "DualAUCSelectionState":
        if not value:
            return cls()
        return cls(
            best_cardiomegaly_auc=float(
                value.get("best_cardiomegaly_auc", float("-inf"))
            ),
            best_cardiomegaly_epoch=int(value.get("best_cardiomegaly_epoch", -1)),
            best_macro_auc=float(value.get("best_macro_auc", float("-inf"))),
            best_macro_epoch=int(value.get("best_macro_epoch", -1)),
            validations_without_improvement=int(
                value.get("validations_without_improvement", 0)
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def consider(
        self,
        metrics: VinDrClassificationMetrics,
        *,
        cardiomegaly_min_delta: float,
        macro_min_delta: float,
    ) -> Tuple[bool, bool]:
        improved_cardio = bool(
            np.isfinite(metrics.cardiomegaly_auc)
            and metrics.cardiomegaly_auc
            > self.best_cardiomegaly_auc + float(cardiomegaly_min_delta)
        )
        improved_macro = bool(
            np.isfinite(metrics.macro_auc_stable)
            and metrics.macro_auc_stable
            > self.best_macro_auc + float(macro_min_delta)
        )

        if improved_cardio:
            self.best_cardiomegaly_auc = float(metrics.cardiomegaly_auc)
            self.best_cardiomegaly_epoch = int(metrics.epoch)
        if improved_macro:
            self.best_macro_auc = float(metrics.macro_auc_stable)
            self.best_macro_epoch = int(metrics.epoch)

        if improved_cardio or improved_macro:
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1

        return improved_cardio, improved_macro


def atomic_torch_save(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def atomic_save_scores_npz(
    output_path: str | Path,
    *,
    outputs: dict,
    metrics: VinDrClassificationMetrics,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with open(temporary_path, "wb") as handle:
        np.savez_compressed(
            handle,
            image_ids=np.asarray(outputs["image_ids"], dtype=object),
            label_names=np.asarray(outputs["label_names"], dtype=object),
            y_true=np.asarray(outputs["y_true"], dtype=np.int8),
            scores=np.asarray(outputs["scores"], dtype=np.float32),
            epoch=np.asarray(metrics.epoch, dtype=np.int64),
            view_type=np.asarray(metrics.view_type),
            cardiomegaly_auc=np.asarray(metrics.cardiomegaly_auc, dtype=np.float32),
            macro_auc_all_evaluable=np.asarray(
                metrics.macro_auc_all_evaluable, dtype=np.float32
            ),
            macro_auc_stable=np.asarray(
                metrics.macro_auc_stable, dtype=np.float32
            ),
            macro_auc=np.asarray(metrics.macro_auc, dtype=np.float32),
            cardiomegaly_best_threshold=np.asarray(
                metrics.cardiomegaly_best_threshold, dtype=np.float32
            ),
        )
    os.replace(temporary_path, output_path)


class VinDrClassificationValidationRunner:
    def __init__(
        self,
        *,
        labels_csv: str | Path,
        images_root: str | Path,
        view_type: str,
        image_res: int,
        mask_root: Optional[str | Path] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        label_name: str = "Cardiomegaly",
        max_images: Optional[int] = None,
        max_text_length: int = 32,
        threshold_steps: int = 200,
        min_positive_per_label: int = 5,
        min_negative_per_label: int = 5,
        macro_auc_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.view_type = str(view_type).lower().strip()
        self.label_name = str(label_name)
        self.max_text_length = int(max_text_length)
        self.threshold_steps = max(2, int(threshold_steps))
        self.min_positive_per_label = int(min_positive_per_label)
        self.min_negative_per_label = int(min_negative_per_label)

        self.dataset = VinDrClassificationDataset(
            labels_csv=labels_csv,
            images_root=images_root,
            view_type=self.view_type,
            image_res=image_res,
            mask_root=mask_root,
            max_images=max_images,
        )
        if self.label_name not in self.dataset.label_cols:
            raise ValueError(
                f"{self.label_name!r} not found in validation labels: "
                f"{self.dataset.label_cols}"
            )

        if macro_auc_labels is None:
            self.macro_auc_labels = None
        else:
            self.macro_auc_labels = [str(label) for label in macro_auc_labels]
            if not self.macro_auc_labels:
                raise ValueError("macro_auc_labels is configured but empty")
            duplicates = sorted(
                {
                    label
                    for label in self.macro_auc_labels
                    if self.macro_auc_labels.count(label) > 1
                }
            )
            if duplicates:
                raise ValueError(
                    f"macro_auc_labels contains duplicates: {duplicates}"
                )
            unknown = [
                label
                for label in self.macro_auc_labels
                if label not in self.dataset.label_cols
            ]
            if unknown:
                raise ValueError(
                    "macro_auc_labels contains labels absent from the "
                    f"validation CSV: {unknown}"
                )

        self.loader = DataLoader(
            self.dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
        )

    def evaluate(
        self,
        *,
        model,
        tokenizer,
        device,
        epoch: int,
        return_outputs: bool = False,
    ):
        was_training = bool(model.training)
        model.eval()

        try:
            label_embeddings = get_label_text_embeddings(
                model=model,
                tokenizer=tokenizer,
                labels=self.dataset.label_cols,
                device=device,
                max_length=self.max_text_length,
            ).to(device)

            all_scores: List[np.ndarray] = []
            all_labels: List[np.ndarray] = []
            all_ids: List[str] = []

            with torch.no_grad():
                for images, labels, image_ids in self.loader:
                    images = images.to(device, non_blocking=True)
                    image_embeddings = get_image_embeddings(model, images)
                    similarities = image_embeddings @ label_embeddings.t()
                    all_scores.append(
                        similarities.detach().cpu().numpy().astype(np.float32)
                    )
                    all_labels.append(labels.numpy().astype(np.int64))
                    all_ids.extend(map(str, image_ids))

            scores = np.concatenate(all_scores, axis=0)
            y_true = np.concatenate(all_labels, axis=0)

            per_label_auc: Dict[str, Optional[float]] = {}
            skipped: Dict[str, str] = {}
            all_evaluable_labels: List[str] = []
            positive_counts: Dict[str, int] = {}
            negative_counts: Dict[str, int] = {}

            for index, label in enumerate(self.dataset.label_cols):
                target = y_true[:, index]
                positives = int(target.sum())
                negatives = int(len(target) - positives)
                positive_counts[label] = positives
                negative_counts[label] = negatives

                if positives == 0:
                    per_label_auc[label] = None
                    skipped[label] = "0 positives; ROC-AUC is undefined"
                    continue
                if negatives == 0:
                    per_label_auc[label] = None
                    skipped[label] = "0 negatives; ROC-AUC is undefined"
                    continue

                auc = float(roc_auc_score(target, scores[:, index]))
                per_label_auc[label] = auc
                all_evaluable_labels.append(label)

            if not all_evaluable_labels:
                raise ValueError("No labels are evaluable for macro ROC-AUC")

            macro_auc_all_evaluable = float(
                np.mean([per_label_auc[label] for label in all_evaluable_labels])
            )

            if self.macro_auc_labels is None:
                stable_labels = [
                    label
                    for label in all_evaluable_labels
                    if positive_counts[label] >= self.min_positive_per_label
                    and negative_counts[label] >= self.min_negative_per_label
                ]
            else:
                stable_labels = list(self.macro_auc_labels)
                insufficient = [
                    (label, positive_counts[label], negative_counts[label])
                    for label in stable_labels
                    if positive_counts[label] < self.min_positive_per_label
                    or negative_counts[label] < self.min_negative_per_label
                ]
                if insufficient:
                    details = ", ".join(
                        f"{label} ({positive} positive, {negative} negative)"
                        for label, positive, negative in insufficient
                    )
                    raise ValueError(
                        "Configured macro_auc_labels do not meet the support "
                        f"thresholds: {details}"
                    )

            if not stable_labels:
                raise ValueError(
                    "No labels are available for stable macro ROC-AUC"
                )
            macro_auc_stable = float(
                np.mean([per_label_auc[label] for label in stable_labels])
            )
            try:
                micro_auc = float(roc_auc_score(y_true.ravel(), scores.ravel()))
            except ValueError:
                micro_auc = None

            cardio_index = self.dataset.label_cols.index(self.label_name)
            cardio_auc = per_label_auc.get(self.label_name)
            if cardio_auc is None:
                raise ValueError(
                    f"{self.label_name} lacks sufficient validation support: "
                    f"{skipped.get(self.label_name)}"
                )

            cardio_scores = scores[:, cardio_index]
            cardio_labels = y_true[:, cardio_index]
            thresholds = np.linspace(
                float(cardio_scores.min()),
                float(cardio_scores.max()),
                self.threshold_steps,
            )
            best_f1 = -1.0
            best_threshold = float(thresholds[0])
            for threshold in thresholds:
                predictions = (cardio_scores >= threshold).astype(np.int64)
                f1 = float(
                    f1_score(cardio_labels, predictions, zero_division=0)
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)

            metrics = VinDrClassificationMetrics(
                epoch=int(epoch),
                view_type=self.view_type,
                num_images=int(len(y_true)),
                cardiomegaly_auc=float(cardio_auc),
                macro_auc_all_evaluable=macro_auc_all_evaluable,
                macro_auc_stable=macro_auc_stable,
                # Backwards-compatible alias for the previous support-filtered
                # macro metric.
                macro_auc=macro_auc_stable,
                micro_auc=micro_auc,
                num_all_evaluable_auc_labels=len(all_evaluable_labels),
                num_stable_auc_labels=len(stable_labels),
                num_valid_auc_labels=len(stable_labels),
                macro_auc_all_evaluable_labels=all_evaluable_labels,
                macro_auc_stable_labels=stable_labels,
                per_label_auc=per_label_auc,
                skipped_auc_labels=skipped,
                cardiomegaly_best_f1=best_f1,
                cardiomegaly_best_threshold=best_threshold,
                cardiomegaly_score_min=float(cardio_scores.min()),
                cardiomegaly_score_max=float(cardio_scores.max()),
                cardiomegaly_score_mean=float(cardio_scores.mean()),
                cardiomegaly_score_std=float(cardio_scores.std()),
            )

            print(
                f"[VinDr validation] epoch={epoch} view={self.view_type} "
                f"cardio_auc={metrics.cardiomegaly_auc:.6f} "
                f"macro_all={metrics.macro_auc_all_evaluable:.6f} "
                f"macro_stable={metrics.macro_auc_stable:.6f} "
                f"all_labels={metrics.num_all_evaluable_auc_labels} "
                f"stable_labels={metrics.num_stable_auc_labels}",
                flush=True,
            )

            if return_outputs:
                return metrics, {
                    "image_ids": all_ids,
                    "label_names": list(self.dataset.label_cols),
                    "y_true": y_true,
                    "scores": scores,
                }
            return metrics
        finally:
            if was_training:
                model.train()
