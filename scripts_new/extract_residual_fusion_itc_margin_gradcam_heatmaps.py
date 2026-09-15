#!/usr/bin/env python3
"""Extract ITC-margin ViT Grad-CAM maps from the residual-fusion multi-view ALBEF model.

Methodology intentionally matches the FC-fusion extractor used in this project:
  * original/lung/heart inputs are processed by three separate ViT branches
  * the final fused ITC bare-prompt margin is the backward target
  * ViT self-attention is hooked at --vit_layer (default -2)
  * branch CAMs use CLS->patch attention * gradient
  * the composite fused CAM sums raw positive branch evidence, then normalizes once

The fusion-specific forward path is the only architectural difference.
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
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import yaml

from models.model_pretrain_multiview_residual_fusion import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed

DEFAULT_BERT_CACHE = (
    "/home/woody/iwi5/iwi5362h/.cache/huggingface/"
    "models--bert-base-uncased/snapshots/"
    "86b5e0934494bd15c9632b12f734a8a67f723594"
)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[len("module."):]: value for key, value in state.items()}
    return state


def interpolate_multiview_positional_embeddings(state, model: ALBEF) -> None:
    pairs = (
        ("visual_encoder_original.pos_embed", model.visual_encoder_original),
        ("visual_encoder_lung.pos_embed", model.visual_encoder_lung),
        ("visual_encoder_heart.pos_embed", model.visual_encoder_heart),
        ("visual_encoder_original_m.pos_embed", model.visual_encoder_original_m),
        ("visual_encoder_lung_m.pos_embed", model.visual_encoder_lung_m),
        ("visual_encoder_heart_m.pos_embed", model.visual_encoder_heart_m),
    )
    for key, encoder in pairs:
        if key in state:
            state[key] = interpolate_pos_embed(state[key], encoder)


def load_fused_checkpoint(model: ALBEF, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module_prefix(checkpoint.get("model", checkpoint))
    interpolate_multiview_positional_embeddings(state, model)
    msg = model.load_state_dict(state, strict=True)
    print(f"[Checkpoint] Loaded residual-fusion weights: {msg}")


def build_model_and_tokenizer(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    text_encoder: str = "bert-base-uncased",
    tokenizer_name_or_path: str = DEFAULT_BERT_CACHE,
):
    config = load_config(config_path)
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    model = ALBEF(
        config=config,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        init_deit=False,
    )
    load_fused_checkpoint(model, checkpoint_path)
    model = model.to(device_torch)
    model.eval()
    return model, tokenizer, config, device_torch


def get_image_transform(image_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_bare_prompt_pair(label: str) -> Tuple[str, str]:
    clean = str(label).replace("_", " ").strip()
    if not clean:
        raise ValueError("Encountered an empty label name")
    if clean.casefold() == "no finding":
        return "No finding", "Finding"
    return clean, f"no {clean}"


@torch.no_grad()
def encode_prompt_pairs(model, tokenizer, labels: Sequence[str], device, max_length: int):
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
    features = F.normalize(model.text_proj(output.last_hidden_state[:, 0, :]), dim=-1)
    n_labels = len(labels)
    return features[:n_labels], features[n_labels:], positive_prompts, negative_prompts


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
        raise RuntimeError("Attention or attention gradient was not captured")
    if attention.ndim != 4 or gradient.shape != attention.shape:
        raise ValueError(
            "Expected matching (B, heads, tokens, tokens) tensors; "
            f"attention={tuple(attention.shape)}, gradient={tuple(gradient.shape)}"
        )
    if attention.shape[0] != 1:
        raise ValueError("Heatmap extraction requires batch size 1")

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
    normalized = positive / positive_max if float(positive_max) > 0 else torch.zeros_like(positive)
    return {
        "cam_signed_raw": signed.cpu().float(),
        "cam_positive_raw": positive.cpu().float(),
        "cam_vis": normalized.cpu().float(),
    }


def build_fused_cam(cam_original, cam_lung, cam_heart):
    signed = (
        cam_original["cam_signed_raw"]
        + cam_lung["cam_signed_raw"]
        + cam_heart["cam_signed_raw"]
    )
    positive = (
        cam_original["cam_positive_raw"]
        + cam_lung["cam_positive_raw"]
        + cam_heart["cam_positive_raw"]
    )
    positive_max = positive.max()
    normalized = positive / positive_max if float(positive_max) > 0 else torch.zeros_like(positive)
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
    path = mask_root / image_id[:2] / f"{image_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found for {image_id}: {path}")
    return path


def load_three_view_images(images_root, image_id, lung_mask_root, heart_mask_root):
    image_path = images_root / f"{image_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for {image_id}: {image_path}")
    with Image.open(image_path) as handle:
        original = handle.convert("RGB")
    lung_mask_path = infer_mask_path(lung_mask_root, image_id)
    heart_mask_path = infer_mask_path(heart_mask_root, image_id)
    with Image.open(lung_mask_path) as handle:
        lung_mask = handle.convert("L")
    with Image.open(heart_mask_path) as handle:
        heart_mask = handle.convert("L")
    if not (original.size == lung_mask.size == heart_mask.size):
        raise ValueError(
            f"Size mismatch for {image_id}: image={original.size}, "
            f"lung_mask={lung_mask.size}, heart_mask={heart_mask.size}"
        )
    black = Image.new("RGB", original.size)
    lung_view = Image.composite(original, black, lung_mask)
    heart_view = Image.composite(original, black, heart_mask)
    return original, lung_view, heart_view, image_path, lung_mask_path, heart_mask_path


def load_selection(labels_csv, target_labels, max_images, positive_only_label):
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


def resolve_layer_index(num_layers: int, vit_layer: int) -> int:
    index = vit_layer if vit_layer >= 0 else num_layers + vit_layer
    if index < 0 or index >= num_layers:
        raise ValueError(f"Resolved vit layer index {index} out of range for {num_layers} layers")
    return index


def extract(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    lung_mask_root = Path(args.lung_mask_root)
    heart_mask_root = Path(args.heart_mask_root)
    output_dir = Path(args.output_dir)

    for path in (config_path, checkpoint_path, labels_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (images_root, lung_mask_root, heart_mask_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
        text_encoder=args.text_encoder,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    )
    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)
    temperature = get_temperature(model, args.temperature)

    blocks_o = getattr(model.visual_encoder_original, "blocks", None)
    blocks_l = getattr(model.visual_encoder_lung, "blocks", None)
    blocks_h = getattr(model.visual_encoder_heart, "blocks", None)
    if blocks_o is None or blocks_l is None or blocks_h is None:
        raise AttributeError("One or more visual encoders have no .blocks attribute")
    if not (len(blocks_o) == len(blocks_l) == len(blocks_h)):
        raise ValueError("The three ViT branches do not have the same depth")
    layer_index = resolve_layer_index(len(blocks_o), args.vit_layer)

    cap_o = VisualAttentionCapture(blocks_o[layer_index].attn)
    cap_l = VisualAttentionCapture(blocks_l[layer_index].attn)
    cap_h = VisualAttentionCapture(blocks_h[layer_index].attn)

    df, id_col = load_selection(
        labels_csv, args.target_labels, args.max_images, args.positive_only_label
    )
    positive_text, negative_text, positive_prompts, negative_prompts = encode_prompt_pairs(
        model, tokenizer, args.target_labels, device, args.max_text_length
    )
    prompt_data = {
        label: {"positive": pos, "negative": neg}
        for label, pos, neg in zip(args.target_labels, positive_prompts, negative_prompts)
    }

    print(
        f"[ITC residual-fusion] vit_layer_index={layer_index} "
        f"temperature={temperature:.6f}"
    )
    records = []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="residual-fusion ITC margin Grad-CAM"):
            image_id = str(row[id_col])
            output_path = output_dir / f"{image_id}.pt"
            if output_path.exists() and not args.overwrite:
                records.append({"image_id": image_id, "heatmap_path": str(output_path), "status": "exists_skipped"})
                continue

            original_img, lung_img, heart_img, image_path, lung_mask_path, heart_mask_path = load_three_view_images(
                images_root, image_id, lung_mask_root, heart_mask_root
            )
            original_tensor = transform(original_img).unsqueeze(0).to(device)
            lung_tensor = transform(lung_img).unsqueeze(0).to(device)
            heart_tensor = transform(heart_img).unsqueeze(0).to(device)

            out = {
                "__metadata__": {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "lung_mask_path": str(lung_mask_path),
                    "heart_mask_path": str(heart_mask_path),
                    "checkpoint": str(checkpoint_path),
                    "config": str(config_path),
                    "method": "albef_multiview_residual_itc_margin_self_attention_gradcam",
                    "implementation_version": "1.0-controlled-fc-compatible",
                    "vit_layer_index": layer_index,
                    "num_vit_layers": len(blocks_o),
                    "image_res": image_res,
                    "temperature": temperature,
                    "target": "(positive_similarity-negative_similarity)/temperature",
                    "uses_itm": False,
                    "fusion_type": "original_anchored_cls_conditioned_sigmoid_residual_fusion",
                    "composite_cam_definition": "sum raw positive branch CAMs, normalize once",
                }
            }
            record = {"image_id": image_id, "heatmap_path": str(output_path), "status": "saved"}

            for label_index, label in enumerate(args.target_labels):
                cap_o.reset(); cap_l.reset(); cap_h.reset()
                model.zero_grad(set_to_none=True)

                z_original = model.visual_encoder_original(original_tensor)
                z_lung = model.visual_encoder_lung(lung_tensor)
                z_heart = model.visual_encoder_heart(heart_tensor)

                image_embeds, residual_gates = model.view_fusion(
                    z_original, z_lung, z_heart, return_gates=True
                )

                image_feature = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
                positive_similarity = (image_feature * positive_text[label_index]).sum()
                negative_similarity = (image_feature * negative_text[label_index]).sum()
                raw_margin = positive_similarity - negative_similarity
                classification_logit = raw_margin / temperature
                positive_probability = torch.sigmoid(classification_logit)
                classification_logit.backward()

                cam_original = compute_gradcam(cap_o)
                cam_lung = compute_gradcam(cap_l)
                cam_heart = compute_gradcam(cap_h)
                cam_fused = build_fused_cam(cam_original, cam_lung, cam_heart)
                for cam in (cam_original, cam_lung, cam_heart, cam_fused):
                    cam["cam_vis_up"] = upsample_cam(cam["cam_vis"], image_res)

                out[label] = {
                    "ground_truth": float(row[label]),
                    "positive_prompt": positive_prompts[label_index],
                    "negative_prompt": negative_prompts[label_index],
                    "positive_similarity": float(positive_similarity.detach().cpu()),
                    "negative_similarity": float(negative_similarity.detach().cpu()),
                    "margin": float(raw_margin.detach().cpu()),
                    "classification_logit": float(classification_logit.detach().cpu()),
                    "positive_probability": float(positive_probability.detach().cpu()),
                    "original": cam_original,
                    "lung": cam_lung,
                    "heart": cam_heart,
                    "fused": cam_fused,
                    "residual_gates": {
                        "lung": float(residual_gates[0, 0].detach().cpu()),
                        "heart": float(residual_gates[0, 1].detach().cpu()),
                    },
                }
                record[f"y::{label}"] = float(row[label])
                record[f"score::{label}"] = float(positive_probability.detach().cpu())
                record[f"margin::{label}"] = float(raw_margin.detach().cpu())
                record[f"gate_lung::{label}"] = float(residual_gates[0, 0].detach().cpu())
                record[f"gate_heart::{label}"] = float(residual_gates[0, 1].detach().cpu())

            torch.save(out, output_path)
            records.append(record)
    finally:
        cap_o.close(); cap_l.close(); cap_h.close()

    index_path = output_dir / "residual_fusion_itc_margin_gradcam_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    manifest = {
        "method": "albef_multiview_residual_itc_margin_self_attention_gradcam",
        "implementation_version": "1.0-controlled-fc-compatible",
        "uses_itm": False,
        "fusion_type": "original_anchored_cls_conditioned_sigmoid_residual_fusion",
        "target_labels": args.target_labels,
        "prompts": prompt_data,
        "temperature": temperature,
        "vit_layer_index": layer_index,
        "num_vit_layers": len(blocks_o),
        "num_images": len(df),
        "checkpoint": str(checkpoint_path),
        "index_file": str(index_path),
        "composite_cam_definition": "sum raw positive branch CAMs, normalize once",
    }
    with (output_dir / "residual_fusion_itc_margin_gradcam_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[Output] {index_path}")
    print("[Done] residual-fusion ITC-margin heatmap extraction complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract residual-fusion ALBEF ITC-margin ViT Grad-CAM heatmaps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--lung_mask_root", required=True)
    parser.add_argument("--heart_mask_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_labels", nargs="+", default=["Cardiomegaly"])
    parser.add_argument("--max_text_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--positive_only_label", default=None)
    parser.add_argument("--vit_layer", type=int, default=-2, help="Default -2 = second-to-last ViT block")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_BERT_CACHE)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
