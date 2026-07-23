"""Decode lung and heart CheXmask RLEs for a fixed VinDr validation subset.

Outputs compact sharded 1-bit PNG masks:
  <output_root>/lung/<image_id[:2]>/<image_id>.png
  <output_root>/heart/<image_id[:2]>/<image_id>.png

The script requires CheXmask masks restored to the original VinDr image size.
It intentionally refuses to resize a mismatched mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.strip().lower())
        if match is not None:
            return match
    raise ValueError(
        f"None of the expected columns {candidates} were found. "
        f"Available columns: {list(df.columns)}"
    )


def decode_rle(rle_value, height: int, width: int) -> np.ndarray:
    if pd.isna(rle_value) or str(rle_value).strip() == "":
        return np.zeros((height, width), dtype=np.uint8)

    runs = np.asarray([int(value) for value in str(rle_value).split()], dtype=np.int64)
    if runs.size % 2 != 0:
        raise ValueError(f"Malformed RLE with odd number of values: {runs.size}")

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        if start < 0 or length < 0 or start + length > mask.size:
            raise ValueError(
                f"Invalid RLE run start={start + 1}, length={length}, "
                f"mask_size={mask.size}"
            )
        mask[start : start + length] = 255
    return mask.reshape((height, width))


def save_one_bit(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").convert("1").save(output_path, format="PNG")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_labels_csv", required=True)
    parser.add_argument("--chexmask_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--chexmask_id_col", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    selected = pd.read_csv(args.selected_labels_csv)
    selected.columns = [str(column).strip() for column in selected.columns]
    selected_id_col = str(selected.columns[0])
    selected_ids = selected[selected_id_col].astype(str).tolist()

    chexmask = pd.read_csv(args.chexmask_csv)
    chexmask.columns = [str(column).strip() for column in chexmask.columns]
    id_col = args.chexmask_id_col or str(chexmask.columns[0])
    if id_col not in chexmask.columns:
        raise ValueError(
            f"CheXmask ID column {id_col!r} not found. "
            f"Columns: {list(chexmask.columns)}"
        )

    left_col = find_column(chexmask, ["Left Lung", "left_lung", "left lung"])
    right_col = find_column(chexmask, ["Right Lung", "right_lung", "right lung"])
    heart_col = find_column(chexmask, ["Heart", "heart"])
    height_col = find_column(chexmask, ["Height", "height"])
    width_col = find_column(chexmask, ["Width", "width"])

    chexmask[id_col] = chexmask[id_col].astype(str)
    chexmask = chexmask.drop_duplicates(subset=[id_col], keep="first").set_index(id_col)

    missing_records = [image_id for image_id in selected_ids if image_id not in chexmask.index]
    if missing_records:
        raise ValueError(
            f"CheXmask CSV is missing {len(missing_records)} selected IDs. "
            f"First IDs: {missing_records[:10]}"
        )

    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    written = 0
    skipped = 0

    for index, image_id in enumerate(selected_ids, start=1):
        row = chexmask.loc[image_id]
        height = int(row[height_col])
        width = int(row[width_col])

        image_path = images_root / f"{image_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing VinDr PNG: {image_path}")
        with Image.open(image_path) as image:
            image_size = image.size
        if image_size != (width, height):
            raise ValueError(
                f"CheXmask dimensions do not match original image for {image_id}: "
                f"CheXmask={(width, height)}, image={image_size}. Use the "
                "original-resolution/restored VinDr CheXmask CSV, not the "
                "1024x1024 preprocessed masks."
            )

        lung_path = output_root / "lung" / image_id[:2] / f"{image_id}.png"
        heart_path = output_root / "heart" / image_id[:2] / f"{image_id}.png"
        if (
            not args.overwrite
            and lung_path.exists()
            and heart_path.exists()
        ):
            skipped += 1
            continue

        left = decode_rle(row[left_col], height, width)
        right = decode_rle(row[right_col], height, width)
        heart = decode_rle(row[heart_col], height, width)
        lung = np.maximum(left, right)

        save_one_bit(lung, lung_path)
        save_one_bit(heart, heart_path)
        written += 1

        if index % 100 == 0 or index == len(selected_ids):
            print(
                f"Decoded {index}/{len(selected_ids)} | "
                f"written={written} skipped={skipped}",
                flush=True,
            )

    print(f"Lung masks:  {output_root / 'lung'}")
    print(f"Heart masks: {output_root / 'heart'}")
    print(f"Written: {written}; skipped existing: {skipped}")


if __name__ == "__main__":
    main()
