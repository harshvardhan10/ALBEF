#!/usr/bin/env python3
"""Validate compact mask-cache shards and write lung/heart training manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--verify_all", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {manifest_path}")

    by_index: Dict[int, Dict[str, Any]] = {}
    for shard_index in range(args.num_shards):
        shard_path = staging_dir / (
            f"mask_cache_shard_{shard_index:03d}_of_"
            f"{args.num_shards:03d}.json"
        )
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard result: {shard_path}")

        shard_records = json.loads(shard_path.read_text())
        for item in shard_records:
            index = int(item["manifest_index"])
            if index in by_index:
                raise RuntimeError(f"Duplicate manifest index: {index}")
            by_index[index] = item

    expected_indices = set(range(len(records)))
    actual_indices = set(by_index)
    missing = sorted(expected_indices - actual_indices)
    unexpected = sorted(actual_indices - expected_indices)

    if missing or unexpected:
        raise RuntimeError(
            f"Shard coverage failure: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )

    lung_records = []
    heart_records = []
    checked = 0

    for index, source_record in enumerate(records):
        item = by_index[index]

        source_image = Path(str(source_record[args.image_key])).expanduser()
        lung_path = mask_root / "lung" / item["lung_mask_relpath"]
        heart_path = mask_root / "heart" / item["heart_mask_relpath"]

        for view, path in (("lung", lung_path), ("heart", heart_path)):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing {view} mask: {path}")

        if args.verify_all:
            with Image.open(source_image) as source:
                source_size = source.size
            for view, path in (("lung", lung_path), ("heart", heart_path)):
                with Image.open(path) as mask:
                    mask.verify()
                with Image.open(path) as mask:
                    if mask.size != source_size:
                        raise RuntimeError(
                            f"{view} mask size mismatch for "
                            f"{item['image_id']}: mask={mask.size}, "
                            f"source={source_size}"
                        )
            checked += 1
            if checked == 1 or checked % 1000 == 0:
                print(
                    f"[Verify] {checked:,}/{len(records):,}",
                    flush=True,
                )

        lung_record = dict(source_record)
        lung_record["mask_relpath"] = item["lung_mask_relpath"]
        lung_record["chexmask_view"] = "lung"
        lung_records.append(lung_record)

        heart_record = dict(source_record)
        heart_record["mask_relpath"] = item["heart_mask_relpath"]
        heart_record["chexmask_view"] = "heart"
        heart_records.append(heart_record)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem

    lung_manifest = output_dir / f"{stem}_lung_maskcache.json"
    heart_manifest = output_dir / f"{stem}_heart_maskcache.json"
    stats_path = output_dir / f"{stem}_maskcache_stats.json"

    atomic_write_json(lung_manifest, lung_records)
    atomic_write_json(heart_manifest, heart_records)

    lung_bytes = sum(
        path.stat().st_size
        for path in (mask_root / "lung").rglob("*.png")
    )
    heart_bytes = sum(
        path.stat().st_size
        for path in (mask_root / "heart").rglob("*.png")
    )

    stats = {
        "source_manifest": str(manifest_path),
        "records": len(records),
        "mask_root": str(mask_root),
        "lung_manifest": str(lung_manifest),
        "heart_manifest": str(heart_manifest),
        "lung_mask_bytes": lung_bytes,
        "heart_mask_bytes": heart_bytes,
        "total_mask_bytes": lung_bytes + heart_bytes,
        "verified_records": checked,
    }
    atomic_write_json(stats_path, stats)

    print("=" * 80)
    print(f"Records:        {len(records):,}")
    print(f"Lung manifest:  {lung_manifest}")
    print(f"Heart manifest: {heart_manifest}")
    print(f"Lung masks:     {lung_bytes / 1024**3:.3f} GiB")
    print(f"Heart masks:    {heart_bytes / 1024**3:.3f} GiB")
    print(
        f"Total cache:    {(lung_bytes + heart_bytes) / 1024**3:.3f} GiB"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
