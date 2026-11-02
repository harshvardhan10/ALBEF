#!/usr/bin/env python
"""
Extract ALBEF patch–text similarity heatmaps (per-image × per-label) for VinDr-CXR.

Pipeline:
- Load VinDr image-level labels CSV:
    columns: image_id, <label_1>, <label_2>, ...
- Reuse ALBEF + tokenizer + config from src.build_model_and_tokenizer
- Compute multi-prompt text embeddings for each label
- For each image:
    - Resize to config["image_res"]
    - Get ViT patch token embeddings (16x16 grid)
    - Compute cosine similarity heatmap per label
    - Upsample to image_res x image_res
    - Save heatmaps as .pt with keys = label names

Output:
- One .pt file per image:  {label_name: (H, W) tensor}
- A CSV index: heatmap_index.csv mapping image_id -> .pt path
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_embeddings,
)


def infer_png_path(images_root, image_id):
    png_path = images_root / f"{image_id}.png"
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"No PNG found for image_id={image_id} at {png_path}")


def get_patch_embeddings(
    model,
    img_tensor,
    device,
    patch_grid= 16,
):
    """
    img_tensor: (1, 3, H, W) on device
    Returns:
        patch_feats: (H', W', D) where H'=W'=patch_grid
    """
    img_tensor = img_tensor.to(device, non_blocking=True)

    with torch.no_grad():
        image_embeds = model.visual_encoder(img_tensor)  # (1, N+1, 768)
        image_embeds = model.vision_proj(image_embeds)   # (1, N+1, D)
        patch_tokens = image_embeds[:, 1:, :]            # (1, N, D)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

    _, N, D = patch_tokens.shape
    expected_N = patch_grid * patch_grid
    if N != expected_N:
        raise ValueError(
            f"Expected {expected_N} patches for grid {patch_grid}x{patch_grid}, got {N}."
        )

    patch_tokens = patch_tokens.view(1, patch_grid, patch_grid, D)
    return patch_tokens.squeeze(0)  # (H', W', D)


def compute_heatmap(
    patch_feats,   # (H', W', D)
    text_feat,     # (D,)
    upsample_size,
):
    """
    Compute patch–text similarity and upsample to upsample_size x upsample_size.
    Returns heatmap as np.ndarray (H, W) in [0, 1].
    """
    H, W, D = patch_feats.shape
    text_feat = text_feat.view(1, 1, D)  # (1, 1, D)

    # similarity per patch: (H, W)
    sim = (patch_feats * text_feat).sum(dim=-1)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)

    sim_4d = sim.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
    sim_up = F.interpolate(
        sim_4d,
        size=(upsample_size, upsample_size),
        mode="bilinear",
        align_corners=False,
    )
    sim_up = sim_up.squeeze(0).squeeze(0)   # (H_up, W_up)

    return sim_up.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ALBEF localization heatmaps for VinDr-CXR."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to ALBEF config YAML (e.g., configs/Pretrain.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ALBEF checkpoint path (e.g., checkpoint_09.pth)")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="VinDr image-level labels CSV (image_id + label columns)")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root folder containing VinDr test PNGs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save heatmaps (.pt) and index CSV")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: limit number of images for quick debugging")
    parser.add_argument("--max_text_len", type=int, default=32,
                        help="Max token length for BERT text encoder")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model/tokenizer/config once and reuse
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )

    image_res = config["image_res"]
    transform = get_image_transform(image_res)

    # Load label CSV to get image_ids + label names
    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]
    label_cols = list(df.columns[1:])
    print(f"[Data] Found {len(df)} rows, {len(label_cols)} labels")

    # For debugging a small batch
    if args.max_images is not None:
        df = df.iloc[: args.max_images].reset_index(drop=True)
        print(f"[Data] Limiting to {len(df)} images (max_images={args.max_images})")

    df["__has_png__"] = df[id_col].apply(
        lambda x: (images_root / f"{x}.png").exists()
    )
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] After PNG filter: {len(df)} images")

    image_ids = df[id_col].tolist()
    label_names = label_cols

    # Multi-prompt text embeddings
    label_embs = get_label_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=label_names,
        device=device,
        max_length=args.max_text_len,
    )  # (L, D)
    label_embs = label_embs.to(device)
    print("[Text] Label embeddings shape:", label_embs.shape)

    heatmap_index_records = []

    for idx, image_id in enumerate(image_ids, start=1):
        try:
            img_path = infer_png_path(images_root, image_id)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

        try:
            patch_feats = get_patch_embeddings(
                model=model,
                img_tensor=img_tensor,
                device=device,
                patch_grid=16,  # 256 / 16 = 16;
            )
        except Exception as e:
            print(f"[ERROR] Patch features failed for {image_id}: {e}")
            continue

        heatmaps = {}
        for j, label in enumerate(label_names):
            text_feat = label_embs[j]  # (D,)
            hmap = compute_heatmap(
                patch_feats=patch_feats,
                text_feat=text_feat,
                upsample_size=image_res,
            )
            heatmaps[label] = torch.from_numpy(hmap)  # (H, W) tensor

        out_path = output_dir / f"{image_id}.pt"
        torch.save(heatmaps, out_path)
        heatmap_index_records.append(
            {"image_id": image_id, "heatmap_path": str(out_path)}
        )

        if idx % 50 == 0 or idx == len(image_ids):
            print(f"[Heatmaps] Processed {idx}/{len(image_ids)} images")

    # Save index CSV
    index_df = pd.DataFrame(heatmap_index_records)
    index_path = output_dir / "heatmap_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[Output] Saved heatmap index to: {index_path}")


if __name__ == "__main__":
    main()
