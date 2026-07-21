#!/usr/bin/env python3
"""
Validate CheXmask SQLite records against an ALBEF JSON manifest and write
view-specific and common-valid manifests.

This checks more than row coverage:
  * database row exists
  * required compressed RLE blobs are non-null
  * blobs decompress successfully
  * RLE contains valid start/length pairs
  * runs stay within height * width
  * optional Dice RCA threshold is satisfied
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def normalize_image_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def record_image_id(
    record: Dict[str, Any],
    image_key: str,
    explicit_id_key: Optional[str],
) -> str:
    if explicit_id_key and explicit_id_key in record:
        return normalize_image_id(record[explicit_id_key])
    for key in ("dicom_id", "image_id"):
        if key in record:
            return normalize_image_id(record[key])
    return normalize_image_id(record[image_key])


def validate_rle_blob(
    blob: Optional[bytes],
    height: int,
    width: int,
) -> Tuple[bool, str]:
    if blob is None:
        return False, "null_rle"

    if isinstance(blob, memoryview):
        blob = blob.tobytes()

    try:
        text = zlib.decompress(blob).decode("utf-8").strip()
    except Exception as exc:
        return False, f"decompression_error:{type(exc).__name__}"

    if not text or text.lower() == "nan":
        return False, "empty_rle"

    runs = np.fromstring(text, sep=" ", dtype=np.int64)
    if runs.size == 0:
        return False, "empty_parsed_rle"
    if runs.size % 2 != 0:
        return False, "odd_rle_value_count"

    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths
    total = int(height) * int(width)

    if np.any(lengths <= 0):
        return False, "nonpositive_run_length"
    if np.any(starts < 0):
        return False, "negative_start"
    if np.any(ends > total):
        return False, "run_out_of_bounds"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chexmask_db", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--manifest_id_key", default=None)
    parser.add_argument("--min_rca_mean", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    database_path = Path(args.chexmask_db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {manifest_path}")

    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )

    heart_valid_records = []
    lung_valid_records = []
    common_valid_records = []
    excluded_rows = []
    reason_counts: Counter[str] = Counter()

    query = """
        SELECT dice_rca_mean, height, width,
               left_lung_rle, right_lung_rle, heart_rle
        FROM masks
        WHERE image_id = ?
    """

    try:
        for index, record in enumerate(records):
            image_id = record_image_id(
                record,
                image_key=args.image_key,
                explicit_id_key=args.manifest_id_key,
            )

            row = connection.execute(query, (image_id,)).fetchone()

            reasons = []
            if row is None:
                heart_valid = False
                lung_valid = False
                reasons.append("record_missing")
                dice = None
            else:
                dice, height, width, left_blob, right_blob, heart_blob = row

                quality_valid = True
                if args.min_rca_mean > 0:
                    if dice is None:
                        quality_valid = False
                        reasons.append("rca_missing")
                    elif float(dice) < args.min_rca_mean:
                        quality_valid = False
                        reasons.append("rca_below_threshold")

                left_valid, left_reason = validate_rle_blob(
                    left_blob, height, width
                )
                right_valid, right_reason = validate_rle_blob(
                    right_blob, height, width
                )
                heart_rle_valid, heart_reason = validate_rle_blob(
                    heart_blob, height, width
                )

                if not left_valid:
                    reasons.append(f"left_lung_{left_reason}")
                if not right_valid:
                    reasons.append(f"right_lung_{right_reason}")
                if not heart_rle_valid:
                    reasons.append(f"heart_{heart_reason}")

                lung_valid = quality_valid and left_valid and right_valid
                heart_valid = quality_valid and heart_rle_valid

            if heart_valid:
                heart_valid_records.append(record)
            if lung_valid:
                lung_valid_records.append(record)
            if heart_valid and lung_valid:
                common_valid_records.append(record)

            if not (heart_valid and lung_valid):
                for reason in reasons:
                    reason_counts[reason] += 1

                excluded_rows.append(
                    {
                        "manifest_index": index,
                        "image_id": image_id,
                        "dice_rca_mean": "" if dice is None else dice,
                        "heart_valid": int(heart_valid),
                        "lung_valid": int(lung_valid),
                        "reasons": ";".join(reasons),
                    }
                )
    finally:
        connection.close()

    stem = manifest_path.stem
    output_paths = {
        "heart": output_dir / f"{stem}_chexmask_heart_valid.json",
        "lung": output_dir / f"{stem}_chexmask_lung_valid.json",
        "common": output_dir / f"{stem}_chexmask_common_valid.json",
        "excluded": output_dir / f"{stem}_chexmask_excluded.tsv",
        "stats": output_dir / f"{stem}_chexmask_validation_stats.json",
    }

    output_paths["heart"].write_text(
        json.dumps(heart_valid_records, indent=2) + "\n"
    )
    output_paths["lung"].write_text(
        json.dumps(lung_valid_records, indent=2) + "\n"
    )
    output_paths["common"].write_text(
        json.dumps(common_valid_records, indent=2) + "\n"
    )

    with output_paths["excluded"].open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "manifest_index",
                "image_id",
                "dice_rca_mean",
                "heart_valid",
                "lung_valid",
                "reasons",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(excluded_rows)

    total = len(records)
    stats = {
        "source_manifest": str(manifest_path.resolve()),
        "chexmask_db": str(database_path.resolve()),
        "min_rca_mean": args.min_rca_mean,
        "total_manifest_records": total,
        "heart_valid_records": len(heart_valid_records),
        "lung_valid_records": len(lung_valid_records),
        "common_valid_records": len(common_valid_records),
        "heart_excluded_records": total - len(heart_valid_records),
        "lung_excluded_records": total - len(lung_valid_records),
        "common_excluded_records": total - len(common_valid_records),
        "reason_counts": dict(sorted(reason_counts.items())),
        "outputs": {key: str(value.resolve()) for key, value in output_paths.items()},
    }
    output_paths["stats"].write_text(json.dumps(stats, indent=2) + "\n")

    print("=" * 80)
    print(f"Total manifest records: {total:,}")
    print(
        f"Heart-valid records:    {len(heart_valid_records):,} "
        f"({len(heart_valid_records) / max(total, 1):.2%})"
    )
    print(
        f"Lung-valid records:     {len(lung_valid_records):,} "
        f"({len(lung_valid_records) / max(total, 1):.2%})"
    )
    print(
        f"Common-valid records:   {len(common_valid_records):,} "
        f"({len(common_valid_records) / max(total, 1):.2%})"
    )
    print("-" * 80)
    print("Exclusion reasons:")
    for reason, count in sorted(
        reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {reason}: {count:,}")
    print("-" * 80)
    for name, output_path in output_paths.items():
        print(f"{name:>8}: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
