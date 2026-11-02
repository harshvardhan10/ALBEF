#!/usr/bin/env python3
"""Create labeled visual previews for CheXmask-excluded MIMIC-CXR records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = None


def normalize_image_id(value: object) -> str:
    value = str(value).strip()
    if "/" in value or "\\" in value:
        value = Path(value).name
    if Path(value).suffix:
        value = Path(value).stem
    return value


def candidate_ids(record: Dict[str, Any], image_key: str) -> Iterable[str]:
    for key in ("dicom_id", "image_id", image_key):
        value = record.get(key)
        if value not in (None, ""):
            yield normalize_image_id(value)


def resolve_image_path(
    value: object,
    manifest_path: Path,
    project_root: Path,
    image_root: Optional[Path],
) -> Optional[Path]:
    raw = Path(str(value)).expanduser()

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if image_root is not None:
            candidates.append(image_root / raw)
        candidates.extend(
            [
                project_root / raw,
                manifest_path.parent / raw,
                Path.cwd() / raw,
            ]
        )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    return None


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("L", size, 0)
    fitted = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""

    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)
    return lines or [""]


def create_preview(
    source_path: Path,
    image_id: str,
    reason: str,
    caption: str,
    output_path: Path,
    panel_size: int,
) -> None:
    with Image.open(source_path) as opened:
        image = opened.convert("L")

    raw_panel = fit_image(image, (panel_size, panel_size))
    contrast_panel = fit_image(ImageOps.autocontrast(image), (panel_size, panel_size))

    margin = 20
    text_height = 150
    width = panel_size * 2 + margin * 3
    height = panel_size + text_height + margin * 2

    preview = Image.new("RGB", (width, height), "white")
    preview.paste(raw_panel.convert("RGB"), (margin, margin + text_height))
    preview.paste(
        contrast_panel.convert("RGB"),
        (panel_size + margin * 2, margin + text_height),
    )

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    header = [
        f"image_id: {image_id}",
        f"reason: {reason}",
        f"source: {source_path}",
        f"caption: {caption}",
        "left: raw grayscale | right: autocontrast (inspection only)",
    ]

    y = margin
    for entry in header:
        for line in wrapped_lines(draw, entry, font, width - 2 * margin):
            draw.text((margin, y), line, fill="black", font=font)
            y += 15

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, quality=95, subsampling=0)


def create_contact_sheets(
    preview_paths: list[Path],
    output_dir: Path,
    columns: int,
    rows: int,
    thumb_width: int,
) -> list[Path]:
    sheets: list[Path] = []
    per_sheet = columns * rows

    for sheet_index in range(math.ceil(len(preview_paths) / per_sheet)):
        subset = preview_paths[
            sheet_index * per_sheet : (sheet_index + 1) * per_sheet
        ]
        if not subset:
            continue

        thumbnails = []
        for path in subset:
            with Image.open(path) as opened:
                thumb = opened.convert("RGB")
                ratio = thumb_width / thumb.width
                thumb = thumb.resize(
                    (thumb_width, max(1, int(thumb.height * ratio))),
                    Image.Resampling.LANCZOS,
                )
                thumbnails.append(thumb)

        cell_height = max(image.height for image in thumbnails)
        sheet = Image.new(
            "RGB",
            (columns * thumb_width, rows * cell_height),
            "white",
        )

        for index, image in enumerate(thumbnails):
            x = (index % columns) * thumb_width
            y = (index // columns) * cell_height
            sheet.paste(image, (x, y))

        output_path = output_dir / f"contact_sheet_{sheet_index + 1:03d}.jpg"
        sheet.save(output_path, quality=92, subsampling=0)
        sheets.append(output_path)

    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--excluded_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--caption_key", default="caption")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--panel_size", type=int, default=768)
    parser.add_argument("--sheet_columns", type=int, default=2)
    parser.add_argument("--sheet_rows", type=int, default=3)
    parser.add_argument("--sheet_thumb_width", type=int, default=900)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    excluded_path = Path(args.excluded_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    project_root = manifest_path.parent.parent
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else None
    )

    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected JSON list: {manifest_path}")

    record_by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for image_id in candidate_ids(record, args.image_key):
            record_by_id.setdefault(image_id, record)

    with excluded_path.open(newline="", encoding="utf-8") as handle:
        excluded_rows = list(csv.DictReader(handle, delimiter="\t"))

    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    inspection_rows = []
    preview_paths: list[Path] = []

    print(f"Manifest records: {len(records):,}", flush=True)
    print(f"Excluded rows:   {len(excluded_rows):,}", flush=True)

    for index, excluded in enumerate(excluded_rows, start=1):
        image_id = normalize_image_id(excluded.get("image_id", ""))
        reason = excluded.get("reasons", "")
        record = record_by_id.get(image_id)

        status = "ok"
        source_path: Optional[Path] = None
        caption = ""

        if record is None:
            status = "manifest_record_not_found"
        else:
            caption = str(record.get(args.caption_key, ""))
            source_path = resolve_image_path(
                record.get(args.image_key, ""),
                manifest_path=manifest_path,
                project_root=project_root,
                image_root=image_root,
            )
            if source_path is None:
                status = "image_file_not_found"

        preview_path = individual_dir / f"{index:04d}_{image_id}.jpg"

        if status == "ok" and source_path is not None:
            try:
                create_preview(
                    source_path=source_path,
                    image_id=image_id,
                    reason=reason,
                    caption=caption,
                    output_path=preview_path,
                    panel_size=args.panel_size,
                )
                preview_paths.append(preview_path)
            except Exception as exc:
                status = f"preview_error:{type(exc).__name__}:{exc}"

        inspection_rows.append(
            {
                "index": index,
                "image_id": image_id,
                "reasons": reason,
                "status": status,
                "source_path": "" if source_path is None else str(source_path),
                "preview_path": (
                    str(preview_path) if preview_path.exists() else ""
                ),
                "caption": caption,
            }
        )

        print(
            f"[{index:03d}/{len(excluded_rows):03d}] "
            f"{image_id} -> {status}",
            flush=True,
        )

    index_path = output_dir / "inspection_index.tsv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "image_id",
                "reasons",
                "status",
                "source_path",
                "preview_path",
                "caption",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(inspection_rows)

    sheets = create_contact_sheets(
        preview_paths=preview_paths,
        output_dir=output_dir,
        columns=args.sheet_columns,
        rows=args.sheet_rows,
        thumb_width=args.sheet_thumb_width,
    )

    successful = sum(row["status"] == "ok" for row in inspection_rows)
    failed = len(inspection_rows) - successful

    print("=" * 80)
    print(f"Successful previews: {successful}")
    print(f"Failed previews:     {failed}")
    print(f"Inspection index:    {index_path}")
    print(f"Individual panels:   {individual_dir}")
    print(f"Contact sheets:      {len(sheets)}")
    for sheet in sheets:
        print(f"  {sheet}")
    print("=" * 80)

    if failed:
        raise SystemExit(
            f"{failed} excluded images could not be rendered; "
            f"inspect {index_path}"
        )


if __name__ == "__main__":
    main()
