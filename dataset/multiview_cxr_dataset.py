"""Synchronized three-view dataset extracted unchanged from the validated training source."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from training.text import pre_caption

ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD = (0.26862954, 0.26130258, 0.27577711)

def _as_path_list(value: Any, field: str) -> list[Path]:
    if isinstance(value, (str, Path)):
        result = [Path(value)]
    elif isinstance(value, Sequence):
        result = [Path(item) for item in value]
    else:
        raise TypeError(f"{field} must be a path or list of paths")

    if not result:
        raise ValueError(f"{field} is empty")

    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"{field}: {path}")

    return result


def load_json_records(paths: Any, field: str) -> list[dict]:
    records: list[dict] = []
    for path in _as_path_list(paths, field):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON list in {path}")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}: record {index} is not a JSON object"
                )
        records.extend(payload)

    if not records:
        raise ValueError(f"No records loaded from {field}")
    return records


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value}")
    return path


class SynchronizedCXRTransform:
    """Apply exactly the same geometric/photometric augmentation to all views."""

    def __init__(self, config: dict):
        self.image_res = int(config["image_res"])
        aug_cfg = config.get("cxr_augmentation", {})
        self.enabled = bool(aug_cfg.get("enabled", True))
        self.degrees = float(aug_cfg.get("rotation_degrees", 5.0))
        self.translate = float(aug_cfg.get("translate_fraction", 0.02))
        self.scale_min = float(aug_cfg.get("scale_min", 0.98))
        self.scale_max = float(aug_cfg.get("scale_max", 1.02))
        self.brightness = float(aug_cfg.get("brightness", 0.10))
        self.contrast = float(aug_cfg.get("contrast", 0.10))
        self.photometric_probability = float(
            aug_cfg.get("photometric_probability", 0.5)
        )

        if self.image_res <= 0:
            raise ValueError("image_res must be positive")
        if self.degrees < 0:
            raise ValueError("rotation_degrees must be >= 0")
        if not 0 <= self.translate <= 1:
            raise ValueError("translate_fraction must be in [0,1]")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("Invalid scale range")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness/contrast must be >= 0")
        if not 0 <= self.photometric_probability <= 1:
            raise ValueError("photometric_probability must be in [0,1]")

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if np.isclose(low, high):
            return float(low)
        return float(torch.empty(1).uniform_(low, high).item())

    def _resize(self, image: Image.Image) -> Image.Image:
        return TF.resize(
            image,
            [self.image_res, self.image_res],
            interpolation=InterpolationMode.BICUBIC,
        )

    def __call__(
        self,
        original: Image.Image,
        lung: Image.Image,
        heart: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = [
            self._resize(original),
            self._resize(lung),
            self._resize(heart),
        ]

        if self.enabled:
            angle = self._uniform(-self.degrees, self.degrees)
            max_dx = self.translate * self.image_res
            max_dy = self.translate * self.image_res
            translate_xy = [
                int(round(self._uniform(-max_dx, max_dx))),
                int(round(self._uniform(-max_dy, max_dy))),
            ]
            scale = self._uniform(self.scale_min, self.scale_max)

            views = [
                TF.affine(
                    image,
                    angle=angle,
                    translate=translate_xy,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                for image in views
            ]

            if (
                (self.brightness > 0 or self.contrast > 0)
                and float(torch.rand(1).item())
                < self.photometric_probability
            ):
                brightness_factor = (
                    self._uniform(
                        max(0.0, 1.0 - self.brightness),
                        1.0 + self.brightness,
                    )
                    if self.brightness > 0
                    else 1.0
                )
                contrast_factor = (
                    self._uniform(
                        max(0.0, 1.0 - self.contrast),
                        1.0 + self.contrast,
                    )
                    if self.contrast > 0
                    else 1.0
                )

                # Use one sampled operation order for all three views.
                if bool(torch.randint(0, 2, (1,)).item()):
                    views = [
                        TF.adjust_contrast(
                            TF.adjust_brightness(image, brightness_factor),
                            contrast_factor,
                        )
                        for image in views
                    ]
                else:
                    views = [
                        TF.adjust_brightness(
                            TF.adjust_contrast(image, contrast_factor),
                            brightness_factor,
                        )
                        for image in views
                    ]

        tensors = []
        for image in views:
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, ALBEF_MEAN, ALBEF_STD)
            tensors.append(tensor)

        return tensors[0], tensors[1], tensors[2]


class MultiViewCXRPretrainDataset(Dataset):
    """Paired original/lung/heart MIMIC-CXR pretraining dataset.

    The existing CheXmask finalization creates separate lung and heart
    manifests. They must describe the same source images in the same order.
    Each record supplies its own `mask_relpath`, allowing lung and heart mask
    relative paths to differ safely.
    """

    def __init__(
        self,
        *,
        lung_ann_files,
        heart_ann_files,
        lung_mask_root: str | Path,
        heart_mask_root: str | Path,
        transform: SynchronizedCXRTransform,
        max_words: int,
        image_key: str = "image",
        caption_key: str = "caption",
        mask_key: str = "mask_relpath",
        verify_files_at_start: bool = False,
    ):
        self.lung_records = load_json_records(
            lung_ann_files,
            "lung_train_file",
        )
        self.heart_records = load_json_records(
            heart_ann_files,
            "heart_train_file",
        )

        if len(self.lung_records) != len(self.heart_records):
            raise ValueError(
                "Lung and heart manifests have different lengths: "
                f"{len(self.lung_records)} vs {len(self.heart_records)}"
            )

        self.lung_mask_root = Path(lung_mask_root)
        self.heart_mask_root = Path(heart_mask_root)
        if not self.lung_mask_root.is_dir():
            raise FileNotFoundError(
                f"lung_mask_root not found: {self.lung_mask_root}"
            )
        if not self.heart_mask_root.is_dir():
            raise FileNotFoundError(
                f"heart_mask_root not found: {self.heart_mask_root}"
            )

        self.transform = transform
        self.max_words = int(max_words)
        self.image_key = str(image_key)
        self.caption_key = str(caption_key)
        self.mask_key = str(mask_key)

        required = {self.image_key, self.caption_key, self.mask_key}
        for index, (lung_item, heart_item) in enumerate(
            zip(self.lung_records, self.heart_records)
        ):
            missing_lung = required - lung_item.keys()
            missing_heart = required - heart_item.keys()
            if missing_lung or missing_heart:
                raise KeyError(
                    f"Manifest record {index}: "
                    f"missing lung={sorted(missing_lung)}, "
                    f"missing heart={sorted(missing_heart)}"
                )

            lung_image = str(lung_item[self.image_key])
            heart_image = str(heart_item[self.image_key])
            if lung_image != heart_image:
                raise ValueError(
                    f"Manifest misalignment at index {index}: "
                    f"lung image={lung_image}, heart image={heart_image}"
                )

            if lung_item[self.caption_key] != heart_item[self.caption_key]:
                raise ValueError(
                    f"Caption mismatch between manifests at index {index}"
                )

        if verify_files_at_start:
            self._verify_all_files()

        print(
            "[MultiViewCXRPretrainDataset] "
            f"paired records={len(self.lung_records):,}",
            flush=True,
        )

    def _paths_for_index(
        self,
        index: int,
    ) -> tuple[Path, Path, Path]:
        lung_item = self.lung_records[index]
        heart_item = self.heart_records[index]

        image_path = Path(str(lung_item[self.image_key])).expanduser()
        lung_rel = _safe_relative_path(
            str(lung_item[self.mask_key]),
            "lung mask_relpath",
        )
        heart_rel = _safe_relative_path(
            str(heart_item[self.mask_key]),
            "heart mask_relpath",
        )
        return (
            image_path,
            self.lung_mask_root / lung_rel,
            self.heart_mask_root / heart_rel,
        )

    def _verify_all_files(self) -> None:
        for index in range(len(self)):
            image_path, lung_path, heart_path = self._paths_for_index(index)
            for path, name in (
                (image_path, "image"),
                (lung_path, "lung mask"),
                (heart_path, "heart mask"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {name} at record {index}: {path}"
                    )
            if index == 0 or (index + 1) % 10000 == 0:
                print(
                    f"[Dataset verify] {index + 1:,}/{len(self):,}",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.lung_records)

    def __getitem__(self, index: int):
        lung_item = self.lung_records[index]
        image_path, lung_path, heart_path = self._paths_for_index(index)

        with Image.open(image_path) as handle:
            original = handle.convert("RGB")

        with Image.open(lung_path) as handle:
            lung_mask = handle.convert("L")

        with Image.open(heart_path) as handle:
            heart_mask = handle.convert("L")

        if lung_mask.size != original.size:
            raise ValueError(
                f"Lung mask size mismatch at index={index}: "
                f"image={original.size}, mask={lung_mask.size}"
            )
        if heart_mask.size != original.size:
            raise ValueError(
                f"Heart mask size mismatch at index={index}: "
                f"image={original.size}, mask={heart_mask.size}"
            )

        black = Image.new("RGB", original.size, (0, 0, 0))
        lung = Image.composite(original, black, lung_mask)
        heart = Image.composite(original, black, heart_mask)

        original_t, lung_t, heart_t = self.transform(
            original,
            lung,
            heart,
        )

        caption_value = lung_item[self.caption_key]
        if isinstance(caption_value, list):
            if not caption_value:
                raise ValueError(f"Empty caption list at index={index}")
            caption_value = random.choice(caption_value)

        caption = pre_caption(
            str(caption_value),
            self.max_words,
        )

        return original_t, lung_t, heart_t, caption


def build_dataset(config: dict) -> MultiViewCXRPretrainDataset:
    required = [
        "lung_train_file",
        "heart_train_file",
        "lung_mask_root",
        "heart_mask_root",
    ]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(
            "Multi-view fusion config is missing: " + ", ".join(missing)
        )

    return MultiViewCXRPretrainDataset(
        lung_ann_files=config["lung_train_file"],
        heart_ann_files=config["heart_train_file"],
        lung_mask_root=config["lung_mask_root"],
        heart_mask_root=config["heart_mask_root"],
        transform=SynchronizedCXRTransform(config),
        max_words=int(config.get("max_words", 30)),
        image_key=config.get("image_key", "image"),
        caption_key=config.get("caption_key", "caption"),
        mask_key=config.get("mask_key", "mask_relpath"),
        verify_files_at_start=bool(
            config.get("verify_training_files_at_start", False)
        ),
    )
