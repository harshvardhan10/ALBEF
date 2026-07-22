#!/usr/bin/env python3
"""Build a strict VinDr test-image manifest for CheXmask mask generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def normalize_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def find_image(images_root: Path, image_id: str, extension: str) -> Path:
    extension = extension if extension.startswith(".") else f".{extension}"
    direct = images_root / f"{image_id}{extension}"
    if direct.is_file():
        return direct.resolve()

    # Fallback for a nested image tree. This is only used when the direct path
    # is absent; VinDr test images are normally flat.
    matches = list(images_root.rglob(f"{image_id}{extension}"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for image_id={image_id}: "
            f"{[str(path) for path in matches[:10]]}"
        )
    raise FileNotFoundError(
        f"No image found for image_id={image_id} below {images_root}"
    )


def atomic_json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--extension", default=".png")
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()

    if not labels_csv.is_file():
        raise FileNotFoundError(labels_csv)
    if not images_root.is_dir():
        raise NotADirectoryError(images_root)

    frame = pd.read_csv(labels_csv, usecols=[args.id_col])
    ids = [normalize_id(value) for value in frame[args.id_col].tolist()]

    if not ids:
        raise RuntimeError("The labels CSV contains no image IDs")

    duplicates = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
    if duplicates:
        raise RuntimeError(
            f"Duplicate image IDs in {labels_csv}: {duplicates[:20]}"
        )

    records = []
    for index, image_id in enumerate(ids, start=1):
        image_path = find_image(
            images_root=images_root,
            image_id=image_id,
            extension=args.extension,
        )
        records.append(
            {
                "image_id": image_id,
                "image": str(image_path),
            }
        )
        if index == 1 or index % 500 == 0 or index == len(ids):
            print(
                f"[Manifest] resolved {index:,}/{len(ids):,} images",
                flush=True,
            )

    atomic_json_dump(output_manifest, records)

    print("=" * 80)
    print(f"Labels CSV:      {labels_csv}")
    print(f"Images root:     {images_root}")
    print(f"Records:         {len(records):,}")
    print(f"Output manifest: {output_manifest}")
    print("=" * 80)


if __name__ == "__main__":
    main()
