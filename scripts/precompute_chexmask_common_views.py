#!/usr/bin/env python3
"""Precompute lossless lung-only and heart-only MIMIC-CXR views.

The script processes one deterministic shard of a common-valid ALBEF manifest.
Each unique image is opened once, its CheXmask row is queried once, and both
views are produced in the same worker invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_DB_PATH: Optional[str] = None
_LUNG_ROOT: Optional[Path] = None
_HEART_ROOT: Optional[Path] = None
_PROJECT_ROOT: Optional[Path] = None
_OVERWRITE = False
_PNG_COMPRESS_LEVEL = 1
_CONN: Optional[sqlite3.Connection] = None


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


def stable_shard(image_id: str, num_shards: int) -> int:
    digest = hashlib.sha1(image_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % num_shards


def resolve_image_path(raw_value: object, manifest_path: Path) -> Path:
    raw = Path(str(raw_value)).expanduser()
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                manifest_path.parent / raw,
                manifest_path.parent.parent / raw,
                Path.cwd() / raw,
            ]
        )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path {raw_value!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def decompress_rle(blob: Optional[bytes]) -> str:
    if blob is None:
        raise ValueError("RLE blob is NULL")
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    text = zlib.decompress(blob).decode("utf-8").strip()
    if not text or text.lower() == "nan":
        raise ValueError("RLE is empty")
    return text


def paint_rle_into(flat: np.ndarray, rle: str) -> None:
    runs = np.fromstring(rle, sep=" ", dtype=np.int64)
    if runs.size == 0 or runs.size % 2 != 0:
        raise ValueError(f"Malformed RLE with {runs.size} values")

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths

    if np.any(starts < 0) or np.any(lengths <= 0) or np.any(ends > flat.size):
        raise ValueError(
            f"RLE outside bounds: start_min={starts.min()}, "
            f"end_max={ends.max()}, total={flat.size}"
        )

    # RLEs contain non-overlapping runs. Direct slice assignment avoids the
    # large int32 difference/cumsum buffers used by the online loader.
    for start, end in zip(starts.tolist(), ends.tolist()):
        flat[start:end] = 255


def decode_heart_and_lungs(
    left_rle: str,
    right_rle: str,
    heart_rle: str,
    height: int,
    width: int,
) -> Tuple[Image.Image, Image.Image]:
    total = int(height) * int(width)

    lung_flat = np.zeros(total, dtype=np.uint8)
    paint_rle_into(lung_flat, left_rle)
    paint_rle_into(lung_flat, right_rle)

    heart_flat = np.zeros(total, dtype=np.uint8)
    paint_rle_into(heart_flat, heart_rle)

    lung_mask = Image.fromarray(
        lung_flat.reshape((int(height), int(width))), mode="L"
    )
    heart_mask = Image.fromarray(
        heart_flat.reshape((int(height), int(width))), mode="L"
    )
    return lung_mask, heart_mask


def output_path(root: Path, image_id: str) -> Path:
    prefix = image_id[:2] if len(image_id) >= 2 else "__"
    return root / prefix / f"{image_id}.png"


def is_valid_existing_image(path: Path, expected_size: Tuple[int, int]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            return image.size == expected_size
    except Exception:
        return False


def init_worker(
    db_path: str,
    lung_root: str,
    heart_root: str,
    project_root: str,
    overwrite: bool,
    png_compress_level: int,
) -> None:
    global _DB_PATH, _LUNG_ROOT, _HEART_ROOT, _PROJECT_ROOT
    global _OVERWRITE, _PNG_COMPRESS_LEVEL, _CONN

    _DB_PATH = db_path
    _LUNG_ROOT = Path(lung_root)
    _HEART_ROOT = Path(heart_root)
    _PROJECT_ROOT = Path(project_root)
    _OVERWRITE = bool(overwrite)
    _PNG_COMPRESS_LEVEL = int(png_compress_level)

    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    _CONN = sqlite3.connect(uri, uri=True, timeout=120.0)
    _CONN.execute("PRAGMA query_only=ON")
    _CONN.execute("PRAGMA temp_store=MEMORY")
    _CONN.execute("PRAGMA mmap_size=1073741824")


def process_one(task: Dict[str, Any]) -> Dict[str, Any]:
    assert _CONN is not None
    assert _LUNG_ROOT is not None
    assert _HEART_ROOT is not None

    image_id = task["image_id"]
    source_path = Path(task["source_path"])
    lung_path = output_path(_LUNG_ROOT, image_id)
    heart_path = output_path(_HEART_ROOT, image_id)

    try:
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
        image_size = image.size

        lung_ok = (
            not _OVERWRITE
            and is_valid_existing_image(lung_path, image_size)
        )
        heart_ok = (
            not _OVERWRITE
            and is_valid_existing_image(heart_path, image_size)
        )

        if not (lung_ok and heart_ok):
            row = _CONN.execute(
                """
                SELECT height, width,
                       left_lung_rle, right_lung_rle, heart_rle
                FROM masks
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()

            if row is None:
                raise KeyError("CheXmask row not found")

            height, width, left_blob, right_blob, heart_blob = row
            lung_mask, heart_mask = decode_heart_and_lungs(
                decompress_rle(left_blob),
                decompress_rle(right_blob),
                decompress_rle(heart_blob),
                int(height),
                int(width),
            )

            if lung_mask.size != image_size:
                lung_mask = lung_mask.resize(
                    image_size, resample=Image.Resampling.NEAREST
                )
            if heart_mask.size != image_size:
                heart_mask = heart_mask.resize(
                    image_size, resample=Image.Resampling.NEAREST
                )

            black = Image.new("RGB", image_size, (0, 0, 0))

            if not lung_ok:
                lung_path.parent.mkdir(parents=True, exist_ok=True)
                lung_view = Image.composite(image, black, lung_mask)
                temporary = lung_path.with_suffix(".png.tmp")
                with temporary.open("wb") as handle:
                    lung_view.save(
                        handle,
                        format="PNG",
                        compress_level=_PNG_COMPRESS_LEVEL,
                        optimize=False,
                    )
                os.replace(temporary, lung_path)

            if not heart_ok:
                heart_path.parent.mkdir(parents=True, exist_ok=True)
                heart_view = Image.composite(image, black, heart_mask)
                temporary = heart_path.with_suffix(".png.tmp")
                with temporary.open("wb") as handle:
                    heart_view.save(
                        handle,
                        format="PNG",
                        compress_level=_PNG_COMPRESS_LEVEL,
                        optimize=False,
                    )
                os.replace(temporary, heart_path)

        return {
            "status": "ok",
            "image_id": image_id,
            "source_path": str(source_path),
            "lung_path": str(lung_path),
            "heart_path": str(heart_path),
            "manifest_indices": task["manifest_indices"],
            "records": task["records"],
        }
    except Exception as exc:
        return {
            "status": "error",
            "image_id": image_id,
            "source_path": str(source_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "manifest_indices": task["manifest_indices"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chexmask_db", required=True)
    parser.add_argument("--lung_output_root", required=True)
    parser.add_argument("--heart_output_root", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--png_compress_level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    db_path = Path(args.chexmask_db).expanduser().resolve()
    lung_root = Path(args.lung_output_root).expanduser().resolve()
    heart_root = Path(args.heart_output_root).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected JSON list in {manifest_path}")

    grouped: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        image_id = record_image_id(record, args.image_key)
        if stable_shard(image_id, args.num_shards) != args.shard_index:
            continue

        group = grouped.setdefault(
            image_id,
            {
                "image_id": image_id,
                "source_path": str(
                    resolve_image_path(record[args.image_key], manifest_path)
                ),
                "manifest_indices": [],
                "records": [],
            },
        )
        group["manifest_indices"].append(index)
        group["records"].append(record)

    tasks = list(grouped.values())
    lung_root.mkdir(parents=True, exist_ok=True)
    heart_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"Manifest:       {manifest_path}", flush=True)
    print(f"Manifest rows:  {len(records):,}", flush=True)
    print(f"Shard:          {args.shard_index}/{args.num_shards}", flush=True)
    print(f"Unique images:  {len(tasks):,}", flush=True)
    print(f"Workers:        {args.num_workers}", flush=True)
    print(f"DB:             {db_path}", flush=True)
    print(f"Lung root:      {lung_root}", flush=True)
    print(f"Heart root:     {heart_root}", flush=True)
    print("=" * 80, flush=True)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    context = mp.get_context("spawn")
    with context.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(
            str(db_path),
            str(lung_root),
            str(heart_root),
            str(manifest_path.parent.parent),
            args.overwrite,
            args.png_compress_level,
        ),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(process_one, tasks, chunksize=args.chunksize),
            start=1,
        ):
            if result["status"] == "ok":
                results.append(result)
            else:
                errors.append(result)

            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[Shard {args.shard_index}] {completed:,}/{len(tasks):,} "
                    f"ok={len(results):,} errors={len(errors):,}",
                    flush=True,
                )

    results.sort(key=lambda item: min(item["manifest_indices"]))
    errors.sort(key=lambda item: min(item["manifest_indices"]))

    shard_prefix = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    result_path = staging_dir / f"{shard_prefix}.json"
    error_path = staging_dir / f"{shard_prefix}.errors.json"

    result_path.write_text(json.dumps(results, indent=2) + "\n")
    error_path.write_text(json.dumps(errors, indent=2) + "\n")

    print(f"Result manifest: {result_path}", flush=True)
    print(f"Error report:    {error_path}", flush=True)

    if errors:
        raise SystemExit(f"Shard completed with {len(errors)} errors")


if __name__ == "__main__":
    main()
