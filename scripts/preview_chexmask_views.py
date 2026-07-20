"""Save original | lung-only | heart-only preview panels for QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from dataset.chexmask_view_dataset import CheXmaskViewApplier, normalize_image_id


def add_label(image, label):
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chexmask_db", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--min_rca_mean", type=float, default=0.0)
    args = parser.parse_args()

    records = json.load(open(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lung = CheXmaskViewApplier(
        "lung", args.chexmask_db, args.min_rca_mean, "error"
    )
    heart = CheXmaskViewApplier(
        "heart", args.chexmask_db, args.min_rca_mean, "error"
    )

    saved = 0
    seen = set()
    for row in records:
        path = Path(row[args.image_key])
        image_id = normalize_image_id(row.get("dicom_id", row.get("image_id", path)))
        if image_id in seen:
            continue
        seen.add(image_id)

        original = Image.open(path).convert("RGB")
        lung_img = lung(original, image_id)
        heart_img = heart(original, image_id)

        target_h = 512
        def fit(img):
            copy = img.copy()
            copy.thumbnail((512, target_h), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (512, target_h), "black")
            canvas.paste(copy, ((512-copy.width)//2, (target_h-copy.height)//2))
            return canvas

        panels = [
            add_label(fit(original), "Original"),
            add_label(fit(lung_img), "Lung-only"),
            add_label(fit(heart_img), "Heart-only"),
        ]
        combined = Image.new("RGB", (sum(x.width for x in panels), panels[0].height), "white")
        x = 0
        for panel in panels:
            combined.paste(panel, (x, 0))
            x += panel.width

        combined.save(output_dir / f"{image_id}.jpg", quality=92)
        saved += 1
        if saved >= args.max_images:
            break

    print(f"Saved {saved} preview panels to {output_dir}")


if __name__ == "__main__":
    main()
