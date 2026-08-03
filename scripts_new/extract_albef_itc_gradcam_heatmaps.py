#!/usr/bin/env python3
"""Extract standard ALBEF ITC Grad-CAM heatmaps on VinDr-CXR.

For each requested finding, the backpropagation target is the single positive
ITC cosine similarity

    sim(image, "<finding>")

The heatmap is computed from the final visual-transformer block's self-attention
probabilities.  It uses gradient-weighted [CLS]-to-patch attention, ReLU, and a
mean over attention heads.  Negative prompts, ITM, XBERT fusion, and
cross-attention are not used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import build_model_and_tokenizer, get_image_transform


def parse_labels(values: Sequence[str]) -> List[str]:
    """Accept space-separated labels and comma-separated label lists."""
    labels: List[str] = []
    for value in values:
        labels.extend(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise ValueError("At least one --target_labels value is required")
    return list(dict.fromkeys(labels))


@torch.no_grad()
def encode_positive_prompts(
    model,
    tokenizer,
    labels: Sequence[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Encode one bare positive disease prompt per label."""
    prompts = [str(label).replace("_", " ").strip() for label in labels]
    if any(not prompt for prompt in prompts):
        raise ValueError("An empty target label was supplied")

    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    text_output = model.text_encoder.bert(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True,
        mode="text",
    )
    text_features = model.text_proj(text_output.last_hidden_state[:, 0, :])
    return F.normalize(text_features, dim=-1), prompts


def get_final_visual_attention(model):
    """Return and validate ALBEF's final ViT self-attention module."""
    visual_encoder = getattr(model, "visual_encoder", None)
    blocks = getattr(visual_encoder, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise AttributeError("model.visual_encoder.blocks is unavailable")

    attention = getattr(blocks[-1], "attn", None)
    if attention is None:
        raise AttributeError("The final visual block has no .attn module")

    required = ("get_attention_map", "get_attn_gradients")
    missing = [name for name in required if not callable(getattr(attention, name, None))]
    if missing or not hasattr(attention, "save_attention"):
        raise RuntimeError(
            "The visual attention module does not provide ALBEF's native "
            "attention-saving API. Missing: "
            + ", ".join(missing + (["save_attention"] if not hasattr(attention, "save_attention") else []))
        )

    return attention, len(blocks) - 1, len(blocks)


def normalize_positive_map(cam: torch.Tensor) -> torch.Tensor:
    maximum = cam.max()
    if torch.isfinite(maximum) and float(maximum) > 0:
        return cam / maximum
    return torch.zeros_like(cam)


def compute_standard_itc_gradcam(
    model,
    image: torch.Tensor,
    text_feature: torch.Tensor,
    visual_attention,
) -> Dict[str, torch.Tensor | float]:
    """Compute final-layer ALBEF ITC attention Grad-CAM for one pair."""
    if image.shape[0] != 1:
        raise ValueError("ITC Grad-CAM extraction requires batch size 1")

    model.zero_grad(set_to_none=True)
    image_embeds = model.visual_encoder(image)
    image_feature = F.normalize(
        model.vision_proj(image_embeds[:, 0, :]), dim=-1
    )
    text_feature = F.normalize(text_feature.reshape(1, -1), dim=-1)
    similarity = (image_feature * text_feature).sum()
    similarity.backward()

    attention = visual_attention.get_attention_map()
    gradients = visual_attention.get_attn_gradients()
    if attention is None or gradients is None:
        raise RuntimeError(
            "Final-layer attention or its gradient was not captured. Ensure the "
            "ALBEF ViT registers a hook when save_attention=True."
        )
    if attention.ndim != 4 or gradients.shape != attention.shape:
        raise ValueError(
            "Expected matching (B,heads,tokens,tokens) tensors, got "
            f"attention={tuple(attention.shape)}, gradients={tuple(gradients.shape)}"
        )

    # One image: final-layer [CLS] query (0) attending to patch keys (1:).
    cls_attention = attention[0, :, 0, 1:].detach().float()
    cls_gradient = gradients[0, :, 0, 1:].detach().float()
    num_patches = cls_attention.shape[-1]
    grid = math.isqrt(num_patches)
    if grid * grid != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch tokens to a square grid")

    per_head_signed = cls_attention * cls_gradient
    cam_signed = per_head_signed.mean(dim=0).reshape(grid, grid)
    # ReLU preserves only evidence supporting the positive disease similarity.
    cam_positive = torch.relu(per_head_signed).mean(dim=0).reshape(grid, grid)
    cam_vis = normalize_positive_map(cam_positive)

    return {
        "positive_similarity": float(similarity.detach().cpu()),
        "cam_signed_raw": cam_signed.cpu(),
        "cam_raw": cam_positive.cpu(),
        "cam_vis": cam_vis.cpu(),
    }


def upsample_cam(cam: torch.Tensor, image_res: int) -> torch.Tensor:
    return F.interpolate(
        cam[None, None],
        size=(image_res, image_res),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0, 1).cpu().float()


def load_view_image(
    images_root: Path,
    image_id: str,
    view_type: str,
    mask_root: Optional[Path],
) -> Tuple[Image.Image, Path, Optional[Path]]:
    image_path = images_root / f"{image_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    if view_type == "original":
        return image, image_path, None
    if mask_root is None:
        raise ValueError("--mask_root is required for lung and heart views")

    mask_path = mask_root / image_id[:2] / f"{image_id}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if image.size != mask.size:
        raise ValueError(
            f"Image/mask size mismatch for {image_id}: {image.size} vs {mask.size}"
        )
    image = Image.composite(image, Image.new("RGB", image.size), mask)
    return image, image_path, mask_path


def load_dataframe(
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
            raise ValueError(f"Positive-only label absent from CSV: {positive_only_label}")
        df = df[df[positive_only_label].astype(float) == 1.0]
    if max_images is not None:
        df = df.iloc[:max_images]
    return df.reset_index(drop=True), id_col


def extract(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    checkpoint = Path(args.checkpoint)
    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    mask_root = Path(args.mask_root) if args.mask_root else None
    target_labels = parse_labels(args.target_labels)

    for path, description in (
        (config_path, "config"),
        (checkpoint, "checkpoint"),
        (labels_csv, "labels CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if args.view_type != "original" and (mask_root is None or not mask_root.is_dir()):
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=str(config_path),
        ckpt_path=str(checkpoint),
        device=args.device,
    )
    model.eval()
    visual_attention, layer_index, num_layers = get_final_visual_attention(model)
    visual_attention.save_attention = True

    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)
    text_features, positive_prompts = encode_positive_prompts(
        model, tokenizer, target_labels, device, args.max_text_length
    )
    df, id_col = load_dataframe(
        labels_csv, target_labels, args.max_images, args.positive_only_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Method] standard ALBEF ITC attention Grad-CAM")
    print(f"[Target] positive similarity only; prompts={positive_prompts}")
    print(f"[ViT] final self-attention block={layer_index} of {num_layers}")
    print(f"[Data] view={args.view_type} images={len(df)}")

    records = []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ITC Grad-CAM"):
            image_id = str(row[id_col])
            out_path = output_dir / f"{image_id}.pt"
            if out_path.exists() and not args.overwrite:
                record = {"image_id": image_id, "heatmap_path": str(out_path), "status": "exists_skipped"}
                for label in target_labels:
                    record[f"y::{label}"] = float(row[label])
                records.append(record)
                continue

            pil_image, image_path, mask_path = load_view_image(
                images_root, image_id, args.view_type, mask_root
            )
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            output = {
                "__metadata__": {
                    "image_id": image_id,
                    "view_type": args.view_type,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else None,
                    "method": "standard_albef_itc_final_self_attention_gradcam",
                    "target": "single_positive_itc_cosine_similarity",
                    "uses_negative_prompt": False,
                    "uses_itm": False,
                    "vit_layer_index": layer_index,
                    "num_vit_layers": num_layers,
                    "image_res": image_res,
                }
            }
            for label_index, label in enumerate(target_labels):
                result = compute_standard_itc_gradcam(
                    model, image_tensor, text_features[label_index], visual_attention
                )
                result["cam_vis_up"] = upsample_cam(result["cam_vis"], image_res)
                result["prompt"] = positive_prompts[label_index]
                result["ground_truth"] = float(row[label])
                output[label] = result

            torch.save(output, out_path)
            record = {
                "image_id": image_id,
                "heatmap_path": str(out_path),
                "status": "saved",
                "view_type": args.view_type,
                "mask_path": str(mask_path) if mask_path else "",
            }
            for label in target_labels:
                record[f"y::{label}"] = float(row[label])
                record[f"similarity::{label}"] = output[label]["positive_similarity"]
            records.append(record)
    finally:
        visual_attention.save_attention = False

    index_path = output_dir / "albef_itc_gradcam_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    manifest = {
        "method": "standard_albef_itc_final_self_attention_gradcam",
        "checkpoint": str(checkpoint),
        "view_type": args.view_type,
        "target_labels": target_labels,
        "positive_prompts": dict(zip(target_labels, positive_prompts)),
        "uses_negative_prompt": False,
        "uses_itm": False,
        "vit_layer_index": layer_index,
        "num_images": len(df),
        "index_file": str(index_path),
    }
    with (output_dir / "albef_itc_gradcam_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[Output] {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract standard final-self-attention ALBEF ITC Grad-CAM maps"
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
        help='Examples: Cardiomegaly or Cardiomegaly "Pleural effusion"',
    )
    parser.add_argument("--max_text_length", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--positive_only_label",
        default=None,
        help="Debug/qualitative subset only; leave unset for FROC extraction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
