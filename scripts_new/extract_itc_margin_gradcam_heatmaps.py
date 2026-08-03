#!/usr/bin/env python3
"""Extract ViT Grad-CAM maps for ALBEF's bare-prompt ITC classifier.

For each requested finding, this script uses the same score as
``zero_shot_eval_vindr_bare_prompts.py``::

    positive = "<finding>"
    negative = "no <finding>"
    target = (sim(image, positive) - sim(image, negative)) / temperature

The target uses only the ITC image/text embeddings.  ITM, XBERT multimodal
fusion, and cross-attention are not called.

One ``.pt`` file is written per image.  Each label entry contains the two ITC
similarities, margin/logit/probability, signed CAM, ReLU-positive CAM, and
normalized/upsampled visualization maps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import build_model_and_tokenizer, get_image_transform


def build_bare_prompt_pair(label: str) -> Tuple[str, str]:
    """Match the prompt construction used by the bare-prompt classifier."""
    clean = str(label).replace("_", " ").strip()
    if not clean:
        raise ValueError("Encountered an empty label name")
    if clean.casefold() == "no finding":
        return "No finding", "Finding"
    return clean, f"no {clean}"


@torch.no_grad()
def encode_prompt_pairs(
    model,
    tokenizer,
    labels: Sequence[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    pairs = [build_bare_prompt_pair(label) for label in labels]
    positive_prompts = [pair[0] for pair in pairs]
    negative_prompts = [pair[1] for pair in pairs]
    prompts = positive_prompts + negative_prompts

    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    output = model.text_encoder.bert(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True,
        mode="text",
    )
    features = model.text_proj(output.last_hidden_state[:, 0, :])
    features = F.normalize(features, dim=-1)
    n_labels = len(labels)
    return (
        features[:n_labels],
        features[n_labels:],
        positive_prompts,
        negative_prompts,
    )


def get_temperature(model, override: Optional[float]) -> float:
    if override is not None:
        temperature = float(override)
    elif hasattr(model, "temp"):
        value = model.temp.detach().float().cpu()
        if value.numel() != 1:
            raise ValueError(f"Expected scalar model.temp, got {tuple(value.shape)}")
        temperature = float(value.item())
    else:
        raise AttributeError("Model has no scalar temp; pass --temperature")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be finite and > 0, got {temperature}")
    return temperature


class VisualAttentionCapture:
    """Capture the output and gradient of a ViT attention-dropout module."""

    def __init__(self, attention_module: torch.nn.Module):
        attn_drop = getattr(attention_module, "attn_drop", None)
        if attn_drop is None:
            raise AttributeError("Visual attention module has no .attn_drop")
        self.attention_map: Optional[torch.Tensor] = None
        self.handle = attn_drop.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Expected attn_drop output to be a tensor")
        self.attention_map = output
        if output.requires_grad:
            output.retain_grad()

    def reset(self) -> None:
        self.attention_map = None

    def close(self) -> None:
        self.handle.remove()


def compute_gradcam(capture: VisualAttentionCapture) -> Dict[str, torch.Tensor]:
    attention = capture.attention_map
    gradient = None if attention is None else attention.grad
    if attention is None or gradient is None:
        raise RuntimeError("Final-layer attention or its gradient was not captured")
    if attention.ndim != 4 or gradient.shape != attention.shape:
        raise ValueError(
            "Expected matching (B,heads,tokens,tokens) tensors; "
            f"attention={tuple(attention.shape)}, gradient={tuple(gradient.shape)}"
        )
    if attention.shape[0] != 1:
        raise ValueError("Heatmap extraction requires one image at a time")

    cls_attention = attention[0, :, 0, 1:].detach().float()
    cls_gradient = gradient[0, :, 0, 1:].detach().float()
    num_patches = cls_attention.shape[-1]
    grid = math.isqrt(num_patches)
    if grid * grid != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patches into a square grid")

    per_head_signed = cls_attention * cls_gradient
    signed = per_head_signed.mean(dim=0).reshape(grid, grid)
    positive = torch.relu(per_head_signed).mean(dim=0).reshape(grid, grid)

    positive_max = positive.max()
    if float(positive_max) > 0:
        normalized = positive / positive_max
    else:
        normalized = torch.zeros_like(positive)

    return {
        "cam_signed_raw": signed.cpu().float(),
        "cam_positive_raw": positive.cpu().float(),
        "cam_vis": normalized.cpu().float(),
    }


def upsample_cam(cam: torch.Tensor, image_res: int) -> torch.Tensor:
    return F.interpolate(
        cam[None, None],
        size=(image_res, image_res),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0, 1).cpu().float()


def infer_mask_path(mask_root: Path, image_id: str) -> Path:
    # Match zero_shot_eval_vindr_bare_prompts.py exactly (no lowercasing).
    path = mask_root / image_id[:2] / f"{image_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found for {image_id}: {path}")
    return path


def load_view_image(
    images_root: Path,
    image_id: str,
    view_type: str,
    mask_root: Optional[Path],
) -> Tuple[Image.Image, Path, Optional[Path]]:
    image_path = images_root / f"{image_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for {image_id}: {image_path}")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    if view_type == "original":
        return image, image_path, None
    if mask_root is None:
        raise ValueError("--mask_root is required for lung/heart views")

    mask_path = infer_mask_path(mask_root, image_id)
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if image.size != mask.size:
        raise ValueError(
            f"Image/mask size mismatch for {image_id}: image={image.size}, mask={mask.size}"
        )
    # This deliberately matches the classification evaluator's masking operation.
    image = Image.composite(image, Image.new("RGB", image.size), mask)
    return image, image_path, mask_path


def load_selection(
    labels_csv: Path,
    target_labels: Sequence[str],
    max_images: Optional[int],
    positive_only_label: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(labels_csv)
    if df.shape[1] < 2:
        raise ValueError("Labels CSV must contain image_id and label columns")
    id_col = str(df.columns[0])
    missing = [label for label in target_labels if label not in df.columns]
    if missing:
        raise ValueError(f"Target labels absent from CSV: {missing}")
    if positive_only_label is not None:
        if positive_only_label not in df.columns:
            raise ValueError(f"Positive-only label absent: {positive_only_label}")
        df = df[df[positive_only_label] == 1]
    if max_images is not None:
        df = df.iloc[:max_images]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No images remain after filtering")
    return df, id_col


def extract(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    mask_root = Path(args.mask_root) if args.mask_root else None
    for path in (config_path, checkpoint_path, labels_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if args.view_type != "original" and (mask_root is None or not mask_root.is_dir()):
        raise FileNotFoundError(f"Valid --mask_root required for {args.view_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=str(config_path),
        ckpt_path=str(checkpoint_path),
        device=args.device,
    )
    model.eval()
    temperature = get_temperature(model, args.temperature)
    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)

    blocks = getattr(model.visual_encoder, "blocks", None)
    if blocks is None:
        raise AttributeError("model.visual_encoder has no .blocks")
    layer_index = len(blocks) - 1
    attention_module = getattr(blocks[layer_index], "attn", None)
    if attention_module is None:
        raise AttributeError("Final visual block has no .attn module")
    visual_attention = VisualAttentionCapture(attention_module)

    df, id_col = load_selection(
        labels_csv, args.target_labels, args.max_images, args.positive_only_label
    )
    positive_text, negative_text, positive_prompts, negative_prompts = (
        encode_prompt_pairs(
            model, tokenizer, args.target_labels, device, args.max_text_length
        )
    )
    prompt_data = {
        label: {"positive": positive, "negative": negative}
        for label, positive, negative in zip(
            args.target_labels, positive_prompts, negative_prompts
        )
    }

    print(
        f"[ITC] view={args.view_type} final_attention_layer={layer_index} "
        f"temperature={temperature}"
    )
    for label, pair in prompt_data.items():
        print(f"[Prompt] {label!r}: {pair['positive']!r} vs {pair['negative']!r}")

    records = []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ITC margin Grad-CAM"):
            image_id = str(row[id_col])
            output_path = output_dir / f"{image_id}.pt"
            if output_path.exists() and not args.overwrite:
                records.append({
                    "image_id": image_id,
                    "heatmap_path": str(output_path),
                    "status": "exists_skipped",
                    "view_type": args.view_type,
                })
                continue

            image, image_path, mask_path = load_view_image(
                images_root, image_id, args.view_type, mask_root
            )
            image_tensor = transform(image).unsqueeze(0).to(device)
            out = {
                "__metadata__": {
                    "image_id": image_id,
                    "view_type": args.view_type,
                    "image_path": str(image_path),
                    "mask_path": None if mask_path is None else str(mask_path),
                    "checkpoint": str(checkpoint_path),
                    "config": str(config_path),
                    "method": "albef_itc_margin_final_self_attention_gradcam",
                    "implementation_version": "2.1-margin-attn-drop-hook",
                    "vit_layer_index": layer_index,
                    "num_vit_layers": len(blocks),
                    "image_res": image_res,
                    "temperature": temperature,
                    "target": "(positive_similarity-negative_similarity)/temperature",
                    "uses_itm": False,
                }
            }
            record = {
                "image_id": image_id,
                "heatmap_path": str(output_path),
                "status": "saved",
                "view_type": args.view_type,
            }

            for label_index, label in enumerate(args.target_labels):
                visual_attention.reset()
                model.zero_grad(set_to_none=True)
                image_embeds = model.visual_encoder(image_tensor)
                image_feature = F.normalize(
                    model.vision_proj(image_embeds[:, 0, :]), dim=-1
                )
                positive_similarity = (image_feature * positive_text[label_index]).sum()
                negative_similarity = (image_feature * negative_text[label_index]).sum()
                raw_margin = positive_similarity - negative_similarity
                classification_logit = raw_margin / temperature
                positive_probability = torch.sigmoid(classification_logit)
                classification_logit.backward()

                cams = compute_gradcam(visual_attention)
                cams["cam_vis_up"] = upsample_cam(cams["cam_vis"], image_res)
                out[label] = {
                    "ground_truth": float(row[label]),
                    "positive_prompt": positive_prompts[label_index],
                    "negative_prompt": negative_prompts[label_index],
                    "positive_similarity": float(positive_similarity.detach().cpu()),
                    "negative_similarity": float(negative_similarity.detach().cpu()),
                    "margin": float(raw_margin.detach().cpu()),
                    "classification_logit": float(classification_logit.detach().cpu()),
                    "positive_probability": float(positive_probability.detach().cpu()),
                    **cams,
                }
                record[f"y::{label}"] = float(row[label])
                record[f"score::{label}"] = float(positive_probability.detach().cpu())
                record[f"margin::{label}"] = float(raw_margin.detach().cpu())

            torch.save(out, output_path)
            records.append(record)
    finally:
        visual_attention.close()

    index_path = output_dir / "itc_margin_gradcam_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    manifest = {
        "method": "albef_itc_margin_final_self_attention_gradcam",
        "implementation_version": "2.1-margin-attn-drop-hook",
        "uses_itm": False,
        "view_type": args.view_type,
        "target_labels": args.target_labels,
        "prompts": prompt_data,
        "temperature": temperature,
        "vit_layer_index": layer_index,
        "num_vit_layers": len(blocks),
        "num_images": len(df),
        "checkpoint": str(checkpoint_path),
        "index_file": str(index_path),
    }
    with (output_dir / "itc_margin_gradcam_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[Output] {index_path}")
    print("[Done] ITC-only margin heatmap extraction complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ALBEF ITC bare-prompt margin ViT Grad-CAM heatmaps"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--view_type", choices=("original", "lung", "heart"), default="original"
    )
    parser.add_argument("--mask_root", default=None)
    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=["Cardiomegaly"],
        help='Labels to explain, e.g. --target_labels Cardiomegaly "Pleural effusion"',
    )
    parser.add_argument("--max_text_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--positive_only_label",
        default=None,
        help="Optional qualitative subset only; leave unset for unbiased/full extraction",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
