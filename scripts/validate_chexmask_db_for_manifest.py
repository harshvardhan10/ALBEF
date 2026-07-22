#!/usr/bin/env python3
"""Validate that a CheXmask SQLite database fully covers a JSON manifest."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def normalize_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    database_path = Path(args.database).expanduser().resolve()

    records = json.loads(manifest_path.read_text())
    requested = []
    for record in records:
        value = (
            record.get("image_id")
            or record.get("dicom_id")
            or record["image"]
        )
        requested.append(normalize_id(value))

    if len(requested) != len(set(requested)):
        raise RuntimeError("Manifest contains duplicate normalized image IDs")

    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
        timeout=60.0,
    )
    try:
        database_ids = {
            row[0]
            for row in connection.execute("SELECT image_id FROM masks")
        }
        missing = sorted(set(requested) - database_ids)

        null_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM masks
            WHERE image_id IN (
                SELECT image_id FROM masks
            )
              AND (
                left_lung_rle IS NULL
                OR right_lung_rle IS NULL
                OR heart_rle IS NULL
                OR height IS NULL
                OR width IS NULL
              )
            """
        ).fetchone()[0]

        requested_null_rows = 0
        for start in range(0, len(requested), 900):
            batch = requested[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            requested_null_rows += connection.execute(
                f"""
                SELECT COUNT(*)
                FROM masks
                WHERE image_id IN ({placeholders})
                  AND (
                    left_lung_rle IS NULL
                    OR right_lung_rle IS NULL
                    OR heart_rle IS NULL
                    OR height IS NULL
                    OR width IS NULL
                  )
                """,
                batch,
            ).fetchone()[0]

        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        connection.close()

    print("=" * 80)
    print(f"Manifest records:          {len(requested):,}")
    print(f"Database rows:             {len(database_ids):,}")
    print(f"Missing requested IDs:     {len(missing):,}")
    print(f"Requested rows with NULLs: {requested_null_rows:,}")
    print(f"All DB rows with NULLs:    {null_rows:,}")
    print(f"SQLite quick_check:        {quick_check}")
    print("=" * 80)

    if missing:
        print("First missing IDs:")
        for value in missing[:30]:
            print(value)
        raise SystemExit(2)

    if requested_null_rows:
        raise SystemExit(
            f"{requested_null_rows} requested VinDr test rows have incomplete masks"
        )

    if quick_check != "ok":
        raise SystemExit(f"SQLite integrity check failed: {quick_check}")

    print("VinDr test CheXmask coverage is complete.")


if __name__ == "__main__":
    main()
