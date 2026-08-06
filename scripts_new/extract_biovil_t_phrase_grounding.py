#!/usr/bin/env python3
"""Extract native BioViL-T phrase-grounding maps for VinDr-CXR.

This is deliberately not Grad-CAM. BioViL-T embeds every spatial image
location and a text phrase in the same space; the saved map contains their
cosine similarity. By default, the script deterministically samples 50
ground-truth-positive images for each requested pathology.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


DEFAULT_PROMPTS = {
    "Cardiomegaly": "cardiomegaly",
    "Pleural effusion": "pleural effusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=["Cardiomegaly", "Pleural effusion"],
    )
    parser.add_argument(
        "--prompts_json",
        default=None,
        help="Optional JSON object mapping each target label to one phrase.",
    )
    parser.add_argument(
        "--cases_per_label",
        type=int,
        default=50,
        help="Number of GT-positive images sampled independently per label.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--selection_csv",
        default=None,
        help=(
            "Optional existing CSV with columns image_id,label,prompt. "
            "When supplied, no sampling is performed."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--interpolation",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> dict[str, str]:
    prompts = dict(DEFAULT_PROMPTS)
    if args.prompts_json:
        supplied = json.loads(args.prompts_json)
        if not isinstance(supplied, dict):
            raise ValueError("--prompts_json must decode to a JSON object")
        prompts.update({str(key): str(value) for key, value in supplied.items()})

    missing = [label for label in args.target_labels if label not in prompts]
    if missing:
        raise ValueError(
            "No phrase supplied for: " + ", ".join(missing) +
            ". Add them with --prompts_json."
        )
    return {label: prompts[label] for label in args.target_labels}


def find_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("image_id", "imageId", "image_name", "id"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def select_cases(
    args: argparse.Namespace,
    labels_frame: pd.DataFrame,
    id_column: str,
    prompts: dict[str, str],
) -> pd.DataFrame:
    if args.selection_csv:
        selected = pd.read_csv(args.selection_csv)
        required = {"image_id", "label", "prompt"}
        missing = required.difference(selected.columns)
        if missing:
            raise ValueError(
                f"Selection CSV is missing columns: {sorted(missing)}"
            )
        selected = selected[["image_id", "label", "prompt"]].copy()
        selected["image_id"] = selected["image_id"].astype(str)
        return selected

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for label in args.target_labels:
        if label not in labels_frame.columns:
            raise KeyError(
                f"Label {label!r} is absent from {args.labels_csv}. "
                f"Available labels: {list(labels_frame.columns[1:])}"
            )
        positive = labels_frame.loc[
            pd.to_numeric(labels_frame[label], errors="coerce").fillna(0) == 1,
            id_column,
        ].astype(str).tolist()
        if len(positive) < args.cases_per_label:
            raise ValueError(
                f"{label}: requested {args.cases_per_label} positives but only "
                f"{len(positive)} are available"
            )
        chosen = rng.sample(sorted(positive), args.cases_per_label)
        for image_id in chosen:
            records.append(
                {"image_id": image_id, "label": label, "prompt": prompts[label]}
            )

    selected = pd.DataFrame(records)
    if selected.duplicated(["image_id", "label"]).any():
        raise RuntimeError("Duplicate image-label pairs were selected")
    return selected


def resolve_image_path(images_root: Path, image_id: str) -> Path:
    direct = images_root / f"{image_id}.png"
    if direct.exists():
        return direct
    for suffix in (".jpg", ".jpeg", ".dcm"):
        candidate = images_root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {image_id} under {images_root}")


def as_2d_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    tensor = tensor.squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D similarity map, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Similarity map contains NaN or infinity")
    return tensor


def minmax(tensor: torch.Tensor) -> torch.Tensor:
    low = tensor.min()
    high = tensor.max()
    if float(high - low) <= 1e-12:
        return torch.zeros_like(tensor)
    return (tensor - low) / (high - low)


def load_biovil_t(device: torch.device):
    try:
        from health_multimodal.image.utils import get_image_inference
        from health_multimodal.image.utils import ImageModelType
        from health_multimodal.text.utils import get_bert_inference
        from health_multimodal.text.utils import BertEncoderType
        from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
    except ImportError as error:
        raise RuntimeError(
            "BioViL-T dependencies are missing. Install them with: "
            "pip install --upgrade hi-ml-multimodal"
        ) from error

    text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
    image_inference = get_image_inference(ImageModelType.BIOVIL_T)
    engine = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )
    engine.to(device)
    return engine


def main() -> None:
    args = parse_args()
    if args.cases_per_label <= 0:
        raise ValueError("--cases_per_label must be positive")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    labels_frame = pd.read_csv(args.labels_csv)
    id_column = find_id_column(labels_frame)
    labels_frame[id_column] = labels_frame[id_column].astype(str)
    prompts = load_prompts(args)
    selected = select_cases(args, labels_frame, id_column, prompts)

    selection_path = output_dir / "selected_cases.csv"
    selected.to_csv(selection_path, index=False)
    print(f"[Selection] {len(selected)} image-label pairs -> {selection_path}")
    print(selected.groupby("label").size().to_string())

    images_root = Path(args.images_root)
    engine = load_biovil_t(device)
    manifest_records: list[dict[str, Any]] = []

    for position, row in enumerate(selected.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        label = str(row.label)
        prompt = str(row.prompt)
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        output_path = maps_dir / f"{image_id}__{safe_label}.pt"
        image_path = resolve_image_path(images_root, image_id)

        if output_path.exists() and not args.overwrite:
            print(f"[{position:03d}/{len(selected):03d}] skip {output_path.name}")
            saved = torch.load(output_path, map_location="cpu")
            raw = as_2d_float_tensor(saved["similarity_map_raw"])
        else:
            with torch.inference_mode():
                value = engine.get_similarity_map_from_raw_data(
                    image_path=image_path,
                    query_text=prompt,
                    interpolation=args.interpolation,
                )

            if isinstance(value, torch.Tensor):
                diagnostic = value.detach().float().cpu()
            else:
                diagnostic = torch.as_tensor(value, dtype=torch.float32)

            print(
                f"[Map diagnostic] image_id={image_id} "
                f"label={label} "
                f"shape={tuple(diagnostic.shape)} "
                f"nan={torch.isnan(diagnostic).sum().item()} "
                f"posinf={torch.isposinf(diagnostic).sum().item()} "
                f"neginf={torch.isneginf(diagnostic).sum().item()} "
                f"finite={torch.isfinite(diagnostic).sum().item()}/{diagnostic.numel()}"
            )

            raw = as_2d_float_tensor(value)

            with Image.open(image_path) as image:
                original_size = tuple(int(x) for x in image.size)  # width, height
            payload = {
                "image_id": image_id,
                "image_path": str(image_path),
                "original_size_wh": original_size,
                "label": label,
                "ground_truth": 1,
                "prompt": prompt,
                "model_name": "BioViL-T",
                "model_type": "biovil_t",
                "method": "native_patch_text_cosine_similarity",
                "interpolation": args.interpolation,
                "similarity_map_raw": raw,
                "similarity_map_vis": minmax(raw),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(unbiased=False)),
            }
            torch.save(payload, output_path)
            print(f"[{position:03d}/{len(selected):03d}] saved {output_path.name}")

        manifest_records.append(
            {
                "image_id": image_id,
                "label": label,
                "prompt": prompt,
                "image_path": str(image_path),
                "heatmap_path": str(output_path),
                "map_height": int(raw.shape[0]),
                "map_width": int(raw.shape[1]),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(unbiased=False)),
            }
        )

    manifest = pd.DataFrame(manifest_records)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[Done] saved {len(manifest)} maps and {manifest_path}")


if __name__ == "__main__":
    main()
