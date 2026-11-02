"""
ALBEF pretraining dataset using compact precomputed CheXmask binary masks.

The original MIMIC image remains unchanged on disk. A one-bit PNG mask is
loaded and applied with PIL.Image.composite before the unchanged ALBEF
RandomResizedCrop / flip / RandAugment pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class CheXmaskCachedMaskPretrainDataset(Dataset):
    def __init__(
        self,
        ann_files: Sequence[str],
        transform,
        mask_root: str,
        max_words: int = 30,
        image_key: str = "image",
        caption_key: str = "caption",
        mask_key: str = "mask_relpath",
    ):
        self.ann = []
        for ann_file in ann_files:
            with open(ann_file, "r") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise TypeError(f"Expected JSON list in {ann_file}")
            self.ann.extend(records)

        self.transform = transform
        self.mask_root = Path(mask_root).expanduser().resolve()
        self.max_words = int(max_words)
        self.image_key = image_key
        self.caption_key = caption_key
        self.mask_key = mask_key

        if not self.mask_root.is_dir():
            raise FileNotFoundError(
                f"Cached-mask directory not found: {self.mask_root}"
            )

    def __len__(self) -> int:
        return len(self.ann)

    def _mask_path(self, ann: Dict[str, object]) -> Path:
        raw = Path(str(ann[self.mask_key])).expanduser()
        return raw if raw.is_absolute() else self.mask_root / raw

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = Path(str(ann[self.image_key])).expanduser()
        mask_path = self._mask_path(ann)

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")

        if mask.size != image.size:
            mask = mask.resize(
                image.size,
                resample=Image.Resampling.NEAREST,
            )

        # C-level PIL operation; equivalent to hard binary multiplication.
        black = Image.new("RGB", image.size, color=(0, 0, 0))
        image = Image.composite(image, black, mask)
        image = self.transform(image)

        raw_caption = ann[self.caption_key]
        if isinstance(raw_caption, list):
            raw_caption = random.choice(raw_caption)
        caption = pre_caption(raw_caption, self.max_words)

        return image, caption
