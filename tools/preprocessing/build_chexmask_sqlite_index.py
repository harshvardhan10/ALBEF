"""
Build a compact CheXmask SQLite index from the official OriginalResolution CSV.

The official CSVs are very large. This script:
  1. streams the CSV row-by-row;
  2. keeps only IDs required by your MIMIC manifest or VinDr labels CSV;
  3. zlib-compresses the three RLE strings;
  4. stores them behind a primary-key SQLite lookup.

Examples
--------
MIMIC:
python scripts/build_chexmask_sqlite_index.py \
  --chexmask_csv /data/CheXmask/OriginalResolution/MIMIC-CXR-JPG.csv \
  --output_db /data/CheXmask/index/mimic_chexmask.sqlite \
  --manifest data/mimic_cxr.json \
  --image_key image \
  --strict_coverage

VinDr:
python scripts/build_chexmask_sqlite_index.py \
  --chexmask_csv /data/CheXmask/OriginalResolution/VinDr-CXR.csv \
  --output_db /data/CheXmask/index/vindr_chexmask.sqlite \
  --ids_csv /data/vindr/image_labels_test.csv \
  --ids_csv_col image_id \
  --strict_coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Set


def normalize_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def load_ids_from_manifests(
    manifest_paths,
    image_key: str,
    explicit_id_key: Optional[str],
) -> Set[str]:
    ids: Set[str] = set()
    for path in manifest_paths or []:
        with open(path, "r") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise TypeError(f"Expected JSON list in manifest: {path}")

        for row in records:
            if explicit_id_key and explicit_id_key in row:
                ids.add(normalize_id(row[explicit_id_key]))
            elif "dicom_id" in row:
                ids.add(normalize_id(row["dicom_id"]))
            elif "image_id" in row:
                ids.add(normalize_id(row["image_id"]))
            else:
                ids.add(normalize_id(row[image_key]))
    return ids


def load_ids_from_csv(path: Optional[str], id_col: Optional[str]) -> Set[str]:
    if path is None:
        return set()
    import pandas as pd
    df = pd.read_csv(path, usecols=[id_col] if id_col else None)
    column = id_col or df.columns[0]
    return {normalize_id(x) for x in df[column].astype(str).tolist()}


def canonical(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def resolve_columns(header):
    lookup = {canonical(name): idx for idx, name in enumerate(header)}

    def find(*candidates, default=None):
        for candidate in candidates:
            key = canonical(candidate)
            if key in lookup:
                return lookup[key]
        if default is not None:
            return default
        raise KeyError(
            f"Could not find any of {candidates} in CheXmask header: {header}"
        )

    return {
        "id": 0,
        "dice": find("Dice RCA (Mean)", "Dice RCA Mean"),
        "left": find("Left Lung"),
        "right": find("Right Lung"),
        "heart": find("Heart"),
        "height": find("Height"),
        "width": find("Width"),
    }


def nullable_float(value):
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return float(value)


def integer(value):
    return int(float(str(value).strip()))


def compress_text(value: str, level: int):
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return sqlite3.Binary(zlib.compress(value.encode("utf-8"), level=level))


def create_database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute(
        """
        CREATE TABLE masks (
            image_id TEXT PRIMARY KEY,
            dice_rca_mean REAL,
            height INTEGER NOT NULL,
            width INTEGER NOT NULL,
            left_lung_rle BLOB,
            right_lung_rle BLOB,
            heart_rle BLOB
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        ) WITHOUT ROWID
        """
    )
    return conn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexmask_csv", required=True)
    parser.add_argument("--output_db", required=True)
    parser.add_argument("--manifest", nargs="*", default=[])
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--manifest_id_key", default=None)
    parser.add_argument("--ids_csv", default=None)
    parser.add_argument("--ids_csv_col", default=None)
    parser.add_argument("--compression_level", type=int, default=1, choices=range(0, 10))
    parser.add_argument("--commit_every", type=int, default=1000)
    parser.add_argument("--strict_coverage", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    requested = load_ids_from_manifests(
        args.manifest,
        image_key=args.image_key,
        explicit_id_key=args.manifest_id_key,
    )
    requested |= load_ids_from_csv(args.ids_csv, args.ids_csv_col)

    if not requested:
        print("[Index] No requested-ID filter supplied; all CSV rows will be stored.")

    output_db = Path(args.output_db)
    conn = create_database(output_db)

    csv.field_size_limit(sys.maxsize)

    csv_path = Path(args.chexmask_csv)
    source = open(csv_path, "r", newline="", encoding="utf-8")
    reader = csv.reader(source)
    header = next(reader)
    cols = resolve_columns(header)

    insert_sql = """
        INSERT OR REPLACE INTO masks (
            image_id, dice_rca_mean, height, width,
            left_lung_rle, right_lung_rle, heart_rle
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    found = set()
    scanned = 0
    kept = 0
    started = time.time()

    try:
        for row in reader:
            scanned += 1
            if args.max_rows is not None and scanned > args.max_rows:
                break

            image_id = normalize_id(row[cols["id"]])
            if requested and image_id not in requested:
                continue

            batch.append(
                (
                    image_id,
                    nullable_float(row[cols["dice"]]),
                    integer(row[cols["height"]]),
                    integer(row[cols["width"]]),
                    compress_text(row[cols["left"]], args.compression_level),
                    compress_text(row[cols["right"]], args.compression_level),
                    compress_text(row[cols["heart"]], args.compression_level),
                )
            )
            found.add(image_id)
            kept += 1

            if len(batch) >= args.commit_every:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

            if scanned % 10000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[Index] scanned={scanned:,} kept={kept:,} "
                    f"rate={scanned/elapsed:,.1f} rows/s",
                    flush=True,
                )
    finally:
        source.close()

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    metadata = {
        "source_csv": str(csv_path.resolve()),
        "scanned_rows": str(scanned),
        "stored_rows": str(kept),
        "compression": "zlib",
        "compression_level": str(args.compression_level),
    }
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        list(metadata.items()),
    )
    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.close()

    missing = sorted(requested - found)
    missing_path = output_db.with_suffix(output_db.suffix + ".missing_ids.txt")
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    print("=" * 80)
    print(f"Scanned rows:  {scanned:,}")
    print(f"Stored rows:   {kept:,}")
    print(f"Requested IDs: {len(requested):,}")
    print(f"Missing IDs:   {len(missing):,}")
    print(f"Database:      {output_db}")
    print(f"DB size:       {output_db.stat().st_size / (1024**3):.2f} GiB")
    print(f"Missing list:  {missing_path}")
    print("=" * 80)

    if args.strict_coverage and missing:
        raise SystemExit(
            f"Strict coverage failed: {len(missing)} requested IDs are absent."
        )


if __name__ == "__main__":
    main()
