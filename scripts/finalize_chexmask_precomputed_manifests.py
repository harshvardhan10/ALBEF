#!/usr/bin/env python3
"""Merge shard outputs, verify completeness, and write ALBEF manifests."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def normalize_image_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def record_image_id(record: Dict[str, Any], image_key: str) -> str:
    for key in ("dicom_id", "image_id"):
        if record.get(key) not in (None, ""):
            return normalize_image_id(record[key])
    return normalize_image_id(record[image_key])


def decompress(blob: Optional[bytes]) -> str:
    if blob is None:
        raise ValueError("NULL RLE")
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    return zlib.decompress(blob).decode("utf-8").strip()


def paint(flat: np.ndarray, rle: str) -> None:
    runs = np.fromstring(rle, sep=" ", dtype=np.int64)
    starts = runs[0::2] - 1
    ends = starts + runs[1::2]
    for start, end in zip(starts.tolist(), ends.tolist()):
        flat[start:end] = 255


def fresh_views(
    source_path: Path,
    row: Tuple[Any, ...],
) -> Tuple[Image.Image, Image.Image]:
    height, width, left_blob, right_blob, heart_blob = row
    total = int(height) * int(width)

    lung_flat = np.zeros(total, dtype=np.uint8)
    paint(lung_flat, decompress(left_blob))
    paint(lung_flat, decompress(right_blob))

    heart_flat = np.zeros(total, dtype=np.uint8)
    paint(heart_flat, decompress(heart_blob))

    with Image.open(source_path) as opened:
        source = opened.convert("RGB")

    lung_mask = Image.fromarray(
        lung_flat.reshape((int(height), int(width))), mode="L"
    )
    heart_mask = Image.fromarray(
        heart_flat.reshape((int(height), int(width))), mode="L"
    )

    if lung_mask.size != source.size:
        lung_mask = lung_mask.resize(source.size, Image.Resampling.NEAREST)
    if heart_mask.size != source.size:
        heart_mask = heart_mask.resize(source.size, Image.Resampling.NEAREST)

    black = Image.new("RGB", source.size, (0, 0, 0))
    return (
        Image.composite(source, black, lung_mask),
        Image.composite(source, black, heart_mask),
    )


def images_equal(left: Image.Image, right: Image.Image) -> bool:
    if left.mode != right.mode or left.size != right.size:
        return False
    difference = ImageChops.difference(left, right)
    return difference.getbbox() is None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--lung_manifest", required=True)
    parser.add_argument("--heart_manifest", required=True)
    parser.add_argument("--chexmask_db", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--verify_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    lung_manifest = Path(args.lung_manifest).expanduser().resolve()
    heart_manifest = Path(args.heart_manifest).expanduser().resolve()
    db_path = Path(args.chexmask_db).expanduser().resolve()

    source_records = json.loads(source_manifest.read_text())
    expected_count = len(source_records)

    by_index: Dict[int, Dict[str, Any]] = {}
    all_errors: List[Dict[str, Any]] = []

    for shard_index in range(args.num_shards):
        prefix = f"shard_{shard_index:03d}_of_{args.num_shards:03d}"
        result_path = staging_dir / f"{prefix}.json"
        error_path = staging_dir / f"{prefix}.errors.json"

        if not result_path.is_file():
            raise FileNotFoundError(f"Missing shard result: {result_path}")
        if not error_path.is_file():
            raise FileNotFoundError(f"Missing shard error report: {error_path}")

        all_errors.extend(json.loads(error_path.read_text()))
        for group in json.loads(result_path.read_text()):
            for index, record in zip(group["manifest_indices"], group["records"]):
                index = int(index)
                if index in by_index:
                    raise RuntimeError(f"Duplicate manifest index {index}")
                by_index[index] = {
                    "record": record,
                    "image_id": group["image_id"],
                    "source_path": group["source_path"],
                    "lung_path": group["lung_path"],
                    "heart_path": group["heart_path"],
                }

    if all_errors:
        raise RuntimeError(f"Shard reports contain {len(all_errors)} errors")

    missing_indices = sorted(set(range(expected_count)) - set(by_index))
    if missing_indices:
        raise RuntimeError(
            f"Missing {len(missing_indices)} manifest indices; first: "
            f"{missing_indices[:20]}"
        )

    lung_records: List[Dict[str, Any]] = []
    heart_records: List[Dict[str, Any]] = []

    for index in range(expected_count):
        item = by_index[index]
        source_record = source_records[index]
        expected_id = record_image_id(source_record, args.image_key)
        if item["image_id"] != expected_id:
            raise RuntimeError(
                f"ID mismatch at index {index}: "
                f"{item['image_id']} != {expected_id}"
            )

        lung_path = Path(item["lung_path"])
        heart_path = Path(item["heart_path"])
        for path in (lung_path, heart_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing precomputed image: {path}")
            with Image.open(path) as image:
                image.verify()

        lung_record = dict(source_record)
        heart_record = dict(source_record)
        lung_record[args.image_key] = str(lung_path)
        heart_record[args.image_key] = str(heart_path)
        lung_record.setdefault("source_image", str(item["source_path"]))
        heart_record.setdefault("source_image", str(item["source_path"]))
        lung_records.append(lung_record)
        heart_records.append(heart_record)

    verify_count = min(int(args.verify_samples), expected_count)
    rng = random.Random(int(args.seed))
    sample_indices = rng.sample(range(expected_count), verify_count)

    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=120.0
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        for position, index in enumerate(sample_indices, start=1):
            item = by_index[index]
            row = connection.execute(
                """
                SELECT height, width,
                       left_lung_rle, right_lung_rle, heart_rle
                FROM masks
                WHERE image_id = ?
                """,
                (item["image_id"],),
            ).fetchone()
            if row is None:
                raise KeyError(f"Missing DB row for {item['image_id']}")

            expected_lung, expected_heart = fresh_views(
                Path(item["source_path"]), row
            )
            with Image.open(item["lung_path"]) as saved_lung:
                saved_lung = saved_lung.convert("RGB")
                if not images_equal(expected_lung, saved_lung):
                    raise RuntimeError(
                        f"Pixel mismatch in lung view: {item['image_id']}"
                    )
            with Image.open(item["heart_path"]) as saved_heart:
                saved_heart = saved_heart.convert("RGB")
                if not images_equal(expected_heart, saved_heart):
                    raise RuntimeError(
                        f"Pixel mismatch in heart view: {item['image_id']}"
                    )
            print(
                f"[Verify] {position}/{verify_count}: {item['image_id']} OK",
                flush=True,
            )
    finally:
        connection.close()

    lung_manifest.parent.mkdir(parents=True, exist_ok=True)
    heart_manifest.parent.mkdir(parents=True, exist_ok=True)
    lung_manifest.write_text(json.dumps(lung_records, indent=2) + "\n")
    heart_manifest.write_text(json.dumps(heart_records, indent=2) + "\n")

    print("=" * 80)
    print(f"Common records:  {expected_count:,}")
    print(f"Lung manifest:   {lung_manifest}")
    print(f"Heart manifest:  {heart_manifest}")
    print(f"Verified images: {verify_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
