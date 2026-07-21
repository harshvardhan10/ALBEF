#!/usr/bin/env python3
"""
Build a compact, lossless, original-resolution CheXmask cache.

Only binary masks are saved:
    lung/<prefix>/<image_id>.png
    heart/<prefix>/<image_id>.png

The PNGs are one-bit images, so they are dramatically smaller than saving
full-resolution masked CXRs. During training, the original MIMIC JPEG is loaded
normally and combined with the cached mask using PIL.Image.composite.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_WORKER_DB: Optional[sqlite3.Connection] = None
_WORKER_DB_PATH: Optional[str] = None
_WORKER_OUTPUT_ROOT: Optional[Path] = None
_WORKER_IMAGE_KEY: str = "image"
_WORKER_PROJECT_ROOT: Optional[Path] = None
_WORKER_COMPRESSION_LEVEL: int = 6


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


def resolve_image_path(
    value: object,
    project_root: Path,
    manifest_parent: Path,
) -> Path:
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        project_root / raw,
        manifest_parent / raw,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve source image: {value}")


def decompress_rle(blob: Optional[bytes]) -> str:
    if blob is None:
        raise ValueError("RLE blob is NULL")
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    text = zlib.decompress(blob).decode("utf-8").strip()
    if not text or text.lower() == "nan":
        raise ValueError("RLE is empty")
    return text


def add_rle_to_difference(
    difference: np.ndarray,
    rle: str,
    total_pixels: int,
) -> None:
    runs = np.fromstring(rle, sep=" ", dtype=np.int64)
    if runs.size == 0 or runs.size % 2 != 0:
        raise ValueError(
            f"Malformed RLE: expected start/length pairs, got {runs.size} values"
        )

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths

    if np.any(lengths <= 0):
        raise ValueError("RLE contains a non-positive run length")
    if np.any(starts < 0) or np.any(ends > total_pixels):
        raise ValueError(
            f"RLE falls outside the image: total={total_pixels}, "
            f"start_min={starts.min()}, end_max={ends.max()}"
        )

    np.add.at(difference, starts, 1)
    np.add.at(difference, ends, -1)


def decode_union_rles(
    rles: Iterable[str],
    height: int,
    width: int,
) -> np.ndarray:
    total = int(height) * int(width)
    difference = np.zeros(total + 1, dtype=np.int32)
    for rle in rles:
        add_rle_to_difference(difference, rle, total)
    flat = np.cumsum(difference[:-1]) > 0
    return flat.reshape((int(height), int(width))).astype(np.uint8)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if mask.shape == (target_h, target_w):
        return mask
    image = Image.fromarray(mask * 255, mode="L")
    image = image.resize(
        (target_w, target_h),
        resample=Image.Resampling.NEAREST,
    )
    return (np.asarray(image) > 0).astype(np.uint8)


def save_one_bit_png(
    mask: np.ndarray,
    output_path: Path,
    compression_level: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )

    image = Image.fromarray(mask * 255, mode="L").convert("1")
    image.save(
        temporary_path,
        format="PNG",
        compress_level=int(compression_level),
        optimize=False,
    )
    os.replace(temporary_path, output_path)


def existing_mask_is_valid(path: Path, expected_size: Tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size
    except Exception:
        return False


def initialize_worker(
    database_path: str,
    output_root: str,
    image_key: str,
    project_root: str,
    compression_level: int,
) -> None:
    global _WORKER_DB
    global _WORKER_DB_PATH
    global _WORKER_OUTPUT_ROOT
    global _WORKER_IMAGE_KEY
    global _WORKER_PROJECT_ROOT
    global _WORKER_COMPRESSION_LEVEL

    _WORKER_DB_PATH = database_path
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_IMAGE_KEY = image_key
    _WORKER_PROJECT_ROOT = Path(project_root)
    _WORKER_COMPRESSION_LEVEL = int(compression_level)

    uri = f"file:{Path(database_path).resolve()}?mode=ro"
    _WORKER_DB = sqlite3.connect(
        uri,
        uri=True,
        timeout=60.0,
        check_same_thread=False,
    )
    _WORKER_DB.execute("PRAGMA query_only=ON")
    _WORKER_DB.execute("PRAGMA temp_store=MEMORY")


def process_record(task: Tuple[int, Dict[str, Any], str]) -> Dict[str, Any]:
    if _WORKER_DB is None or _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("Worker was not initialized")

    manifest_index, record, manifest_parent_raw = task
    manifest_parent = Path(manifest_parent_raw)

    image_id = record_image_id(record, _WORKER_IMAGE_KEY)
    source_path = resolve_image_path(
        record[_WORKER_IMAGE_KEY],
        project_root=_WORKER_PROJECT_ROOT,
        manifest_parent=manifest_parent,
    )

    with Image.open(source_path) as source:
        source_size = source.size

    prefix = image_id[:2] if len(image_id) >= 2 else "__"
    lung_relpath = Path(prefix) / f"{image_id}.png"
    heart_relpath = Path(prefix) / f"{image_id}.png"

    lung_path = _WORKER_OUTPUT_ROOT / "lung" / lung_relpath
    heart_path = _WORKER_OUTPUT_ROOT / "heart" / heart_relpath

    lung_ok = existing_mask_is_valid(lung_path, source_size)
    heart_ok = existing_mask_is_valid(heart_path, source_size)

    if not (lung_ok and heart_ok):
        row = _WORKER_DB.execute(
            """
            SELECT height, width, left_lung_rle, right_lung_rle, heart_rle
            FROM masks
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        if row is None:
            raise RuntimeError(f"CheXmask row not found for image_id={image_id}")

        height, width, left_blob, right_blob, heart_blob = row
        height = int(height)
        width = int(width)

        left_rle = decompress_rle(left_blob)
        right_rle = decompress_rle(right_blob)
        heart_rle = decompress_rle(heart_blob)

        if not lung_ok:
            lung_mask = decode_union_rles(
                [left_rle, right_rle],
                height,
                width,
            )
            lung_mask = resize_mask(lung_mask, source_size)
            save_one_bit_png(
                lung_mask,
                lung_path,
                _WORKER_COMPRESSION_LEVEL,
            )

        if not heart_ok:
            heart_mask = decode_union_rles(
                [heart_rle],
                height,
                width,
            )
            heart_mask = resize_mask(heart_mask, source_size)
            save_one_bit_png(
                heart_mask,
                heart_path,
                _WORKER_COMPRESSION_LEVEL,
            )

    return {
        "manifest_index": manifest_index,
        "image_id": image_id,
        "source_image": str(source_path),
        "source_width": int(source_size[0]),
        "source_height": int(source_size[1]),
        "lung_mask_relpath": lung_relpath.as_posix(),
        "heart_mask_relpath": heart_relpath.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chexmask_db", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression_level", type=int, default=6)
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")

    manifest_path = Path(args.manifest).expanduser().resolve()
    database_path = Path(args.chexmask_db).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {manifest_path}")

    selected = [
        (index, record, str(manifest_path.parent))
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Compact CheXmask mask-cache generation", flush=True)
    print(f"Manifest records: {len(records):,}", flush=True)
    print(f"Shard:            {args.shard_index}/{args.num_shards}", flush=True)
    print(f"Shard records:    {len(selected):,}", flush=True)
    print(f"Workers:          {args.workers}", flush=True)
    print(f"Output root:      {output_root}", flush=True)
    print("=" * 80, flush=True)

    context = mp.get_context("spawn")
    results = []

    with context.Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(database_path),
            str(output_root),
            args.image_key,
            str(project_root),
            args.compression_level,
        ),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(process_record, selected, chunksize=1),
            start=1,
        ):
            results.append(result)
            if (
                completed == 1
                or completed % args.progress_every == 0
                or completed == len(selected)
            ):
                print(
                    f"[Progress] shard={args.shard_index} "
                    f"{completed:,}/{len(selected):,}",
                    flush=True,
                )

    results.sort(key=lambda item: int(item["manifest_index"]))
    shard_path = staging_dir / (
        f"mask_cache_shard_{args.shard_index:03d}_of_"
        f"{args.num_shards:03d}.json"
    )
    temporary_path = shard_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n")
    os.replace(temporary_path, shard_path)

    print(f"Wrote shard manifest: {shard_path}", flush=True)
    print(f"Completed records:    {len(results):,}", flush=True)


if __name__ == "__main__":
    main()
