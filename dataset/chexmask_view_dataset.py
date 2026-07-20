"""
CheXmask-backed anatomical views for ALBEF pretraining and VinDr inference.

The CheXmask CSV is first converted to a compact SQLite index using
scripts/build_chexmask_sqlite_index.py. Each DataLoader worker then performs
read-only point lookups by image ID.

Mask convention
---------------
CheXmask RLE uses row-major flattening with one-indexed run starts. This
matches the official CheXmask DataPostprocessing/utils.py implementation.

Views
-----
original: unchanged image
lung:     original image multiplied by union(left lung, right lung)
heart:    original image multiplied by heart mask
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageFile
from torch.utils.data import Dataset

from utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def normalize_image_id(value: object) -> str:
    """Normalize a CheXmask/MIMIC/VinDr identifier for database lookup."""
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    suffix = Path(value).suffix
    if suffix:
        value = Path(value).stem
    return value


def decode_chexmask_rle(rle: str, height: int, width: int) -> np.ndarray:
    """
    Decode CheXmask RLE to a uint8 binary mask in row-major order.

    CheXmask stores pairs:
        one-indexed_start run_length
    """
    if rle is None:
        raise ValueError("Cannot decode a null RLE string.")

    rle = str(rle).strip()
    if not rle or rle.lower() == "nan":
        raise ValueError("Cannot decode an empty CheXmask RLE string.")

    runs = np.fromstring(rle, sep=" ", dtype=np.int64)
    if runs.size == 0 or runs.size % 2 != 0:
        raise ValueError(
            f"Malformed CheXmask RLE: expected start/length pairs, got {runs.size} values."
        )

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths
    total = int(height) * int(width)

    if np.any(starts < 0) or np.any(ends > total):
        raise ValueError(
            f"RLE runs fall outside mask size {height}x{width}: "
            f"start_min={starts.min()}, end_max={ends.max()}, total={total}"
        )

    # Difference-array decoding avoids a Python loop over every run.
    diff = np.zeros(total + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    flat = np.cumsum(diff[:-1]) > 0
    return flat.reshape((int(height), int(width))).astype(np.uint8)


def _decompress_rle(blob: Optional[bytes]) -> Optional[str]:
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    return zlib.decompress(blob).decode("utf-8")


class CheXmaskMaskStore:
    """Process-safe, lazy, read-only SQLite access for DataLoader workers."""

    def __init__(self, sqlite_path: str | Path):
        self.sqlite_path = str(Path(sqlite_path).expanduser().resolve())
        if not Path(self.sqlite_path).exists():
            raise FileNotFoundError(f"CheXmask SQLite index not found: {self.sqlite_path}")
        self._conn = None
        self._pid = None

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._conn is None or self._pid != pid:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            uri = f"file:{self.sqlite_path}?mode=ro"
            self._conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=60.0,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA query_only=ON")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._pid = pid
        return self._conn

    def get(self, image_id: object) -> Optional[Dict[str, object]]:
        image_id = normalize_image_id(image_id)
        row = self._connection().execute(
            """
            SELECT image_id, dice_rca_mean, height, width,
                   left_lung_rle, right_lung_rle, heart_rle
            FROM masks
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        if row is None:
            return None

        return {
            "image_id": row[0],
            "dice_rca_mean": None if row[1] is None else float(row[1]),
            "height": int(row[2]),
            "width": int(row[3]),
            "left_lung": _decompress_rle(row[4]),
            "right_lung": _decompress_rle(row[5]),
            "heart": _decompress_rle(row[6]),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_pid"] = None
        return state


def _resize_binary_mask(mask: np.ndarray, image_size_wh: tuple[int, int]) -> np.ndarray:
    target_w, target_h = image_size_wh
    if mask.shape == (target_h, target_w):
        return mask.astype(np.uint8)

    mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    mask_pil = mask_pil.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    return (np.asarray(mask_pil) > 0).astype(np.uint8)


def _postprocess_mask(
    mask: np.ndarray,
    dilation_px: int = 0,
    feather_px: float = 0.0,
) -> np.ndarray:
    """Return a float mask in [0,1]."""
    mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")

    dilation_px = int(dilation_px)
    if dilation_px > 0:
        kernel_size = 2 * dilation_px + 1
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(kernel_size))

    feather_px = float(feather_px)
    if feather_px > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_px))

    return np.asarray(mask_pil, dtype=np.float32) / 255.0


class CheXmaskViewApplier:
    """
    Apply a CheXmask anatomical view before the normal ALBEF image transform.

    failure_policy:
      error    -> fail loudly for missing/low-quality masks
      original -> return the original image
      black    -> return a black image
    """

    VALID_VIEWS = {"original", "lung", "heart"}
    VALID_FAILURE_POLICIES = {"error", "original", "black"}

    def __init__(
        self,
        view_type: str,
        sqlite_path: Optional[str | Path] = None,
        min_rca_mean: float = 0.0,
        failure_policy: str = "error",
        dilation_px: int = 0,
        feather_px: float = 0.0,
    ):
        self.view_type = str(view_type).lower().strip()
        if self.view_type not in self.VALID_VIEWS:
            raise ValueError(f"view_type must be one of {sorted(self.VALID_VIEWS)}")

        self.failure_policy = str(failure_policy).lower().strip()
        if self.failure_policy not in self.VALID_FAILURE_POLICIES:
            raise ValueError(
                f"failure_policy must be one of {sorted(self.VALID_FAILURE_POLICIES)}"
            )

        self.min_rca_mean = float(min_rca_mean)
        self.dilation_px = int(dilation_px)
        self.feather_px = float(feather_px)

        self.store = None
        if self.view_type != "original":
            if sqlite_path is None:
                raise ValueError(f"sqlite_path is required for view_type={self.view_type}")
            self.store = CheXmaskMaskStore(sqlite_path)

    def _failure(self, image: Image.Image, image_id: str, reason: str) -> Image.Image:
        if self.failure_policy == "error":
            raise RuntimeError(f"CheXmask failure for image_id={image_id}: {reason}")
        if self.failure_policy == "original":
            return image
        return Image.new("RGB", image.size, color=(0, 0, 0))

    def __call__(self, image: Image.Image, image_id: object) -> Image.Image:
        image = image.convert("RGB")
        image_id = normalize_image_id(image_id)

        if self.view_type == "original":
            return image

        record = self.store.get(image_id)
        if record is None:
            return self._failure(image, image_id, "mask record not found")

        quality = record["dice_rca_mean"]
        if (
            self.min_rca_mean > 0
            and quality is not None
            and float(quality) < self.min_rca_mean
        ):
            return self._failure(
                image,
                image_id,
                f"Dice RCA mean {quality:.4f} < {self.min_rca_mean:.4f}",
            )

        h, w = int(record["height"]), int(record["width"])

        try:
            if self.view_type == "lung":
                left = decode_chexmask_rle(record["left_lung"], h, w)
                right = decode_chexmask_rle(record["right_lung"], h, w)
                mask = np.logical_or(left, right).astype(np.uint8)
            else:
                mask = decode_chexmask_rle(record["heart"], h, w)
        except Exception as exc:
            return self._failure(image, image_id, f"RLE decoding failed: {exc}")

        mask = _resize_binary_mask(mask, image.size)
        mask = _postprocess_mask(
            mask,
            dilation_px=self.dilation_px,
            feather_px=self.feather_px,
        )

        image_arr = np.asarray(image, dtype=np.float32)
        masked = image_arr * mask[..., None]
        masked = np.clip(np.rint(masked), 0, 255).astype(np.uint8)
        return Image.fromarray(masked, mode="RGB")


class CheXmaskPretrainDataset(Dataset):
    """
    ALBEF pretraining dataset using the same JSON format as pretrain_dataset.

    Each annotation record must contain:
      image: absolute image path
      caption: string or list of strings

    Optional:
      image_id / dicom_id: explicit ID used for CheXmask lookup
    """

    def __init__(
        self,
        ann_files: Sequence[str],
        transform,
        view_type: str,
        chexmask_db: Optional[str] = None,
        max_words: int = 30,
        image_key: str = "image",
        caption_key: str = "caption",
        mask_id_key: Optional[str] = None,
        min_rca_mean: float = 0.0,
        failure_policy: str = "error",
        dilation_px: int = 0,
        feather_px: float = 0.0,
    ):
        self.ann = []
        for ann_file in ann_files:
            with open(ann_file, "r") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise TypeError(f"Expected JSON list in {ann_file}")
            self.ann.extend(records)

        self.transform = transform
        self.max_words = int(max_words)
        self.image_key = image_key
        self.caption_key = caption_key
        self.mask_id_key = mask_id_key
        self.view_applier = CheXmaskViewApplier(
            view_type=view_type,
            sqlite_path=chexmask_db,
            min_rca_mean=min_rca_mean,
            failure_policy=failure_policy,
            dilation_px=dilation_px,
            feather_px=feather_px,
        )

    def __len__(self):
        return len(self.ann)

    def _image_id(self, ann: Dict[str, object], image_path: str) -> str:
        if self.mask_id_key and self.mask_id_key in ann:
            return normalize_image_id(ann[self.mask_id_key])
        for candidate in ("dicom_id", "image_id"):
            if candidate in ann:
                return normalize_image_id(ann[candidate])
        return normalize_image_id(image_path)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = str(ann[self.image_key])
        image_id = self._image_id(ann, image_path)

        image = Image.open(image_path).convert("RGB")
        image = self.view_applier(image, image_id)
        image = self.transform(image)

        raw_caption = ann[self.caption_key]
        if isinstance(raw_caption, list):
            raw_caption = random.choice(raw_caption)
        caption = pre_caption(raw_caption, self.max_words)

        return image, caption
