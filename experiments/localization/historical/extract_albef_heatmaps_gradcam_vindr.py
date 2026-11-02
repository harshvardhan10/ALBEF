"""
Gradient-based (Grad-CAM style) localization for ALBEF on VinDr-CXR.

For each image and each label:
  - Compute a scalar image–text score using CLS and label text embedding.
  - Backpropagate gradients to the patch features.
  - Compute a Grad-CAM heatmap over the patches.
  - Upsample to image_res x image_res.
  - Save per-image .pt with {label_name: (H, W) tensor}.
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


def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"No PNG found for image_id={image_id} at {png_path}")
    return png_path


def gradcam_for_single_label(
    model,
    img_tensor: torch.Tensor,    # (1,3,H,W)
    text_feat: torch.Tensor,     # (D,), normalized
    device: torch.device,
    patch_grid: int,
) -> torch.Tensor:
    """
    Compute Grad-CAM over ViT patch tokens for a single label.

    Steps:
      1. Forward: image -> visual_encoder -> vision_proj -> tokens
      2. Define scalar score = cosine(image_CLS, text_feat)
      3. Backward: d(score)/d(patch_tokens)
      4. Grad-CAM: weights = mean over spatial gradients
         heatmap[h,w] = ReLU( Σ_k weights[k] * patch_feats[h,w,k] )
    """
    model.zero_grad()

    img_tensor = img_tensor.to(device, non_blocking=True)
    img_tensor.requires_grad_(True)

    image_embeds = model.visual_encoder(img_tensor)   # (1, N+1, 768)
    image_embeds = model.vision_proj(image_embeds)    # (1, N+1, D)

    # CLS and patches
    image_cls = image_embeds[:, 0, :]                 # (1, D)
    patch_tokens = image_embeds[:, 1:, :]             # (1, N, D)

    # normalize CLS for cosine similarity
    image_cls_norm = F.normalize(image_cls, dim=-1)   # (1, D)
    text_feat = text_feat.to(device)
    text_feat = text_feat.view(1, -1)                 # (1, D)

    # ---- Scalar score for this label ----
    # score = cosine(image_cls, text_feat)
    score = (image_cls_norm * text_feat).sum(dim=-1)  # (1,)
    score = score.squeeze(0)                          # scalar

    # ---- Backward: gradients w.r.t patch tokens ----
    patch_tokens.retain_grad()
    score.backward(retain_graph=False)

    grads = patch_tokens.grad      # (1, N, D)
    feats = patch_tokens           # (1, N, D)

    # ---- Grad-CAM: weights over channels ----
    # average gradient over spatial dimension N
    weights = grads.mean(dim=1)    # (1, D)
    weights = weights.squeeze(0)   # (D,)

    # apply weights to patch features
    feats = feats.squeeze(0)       # (N, D)
    cam = (feats * weights).sum(dim=-1)  # (N,)

    # reshape to (H', W')
    N = cam.shape[0]
    grid_auto = int(np.sqrt(N))
    if grid_auto * grid_auto != N:
        raise ValueError(f"Cannot reshape N={N} into square grid.")
    if patch_grid is None:
        patch_grid = grid_auto

    if patch_grid * patch_grid != N:
        raise ValueError(
            f"patch_grid={patch_grid} but N={N} (should be {patch_grid*patch_grid})."
        )

    cam = cam.view(patch_grid, patch_grid)  # (H', W')

    # ReLU and normalize
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-6)

    return cam   # (H', W') on device


def upsample_cam(cam: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    cam: (H', W') on device
    returns: (target_size, target_size) on CPU
    """
    cam_4d = cam.unsqueeze(0).unsqueeze(0)   # (1,1,H',W')
    cam_up = F.interpolate(
        cam_4d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    cam_up = cam_up.squeeze(0).squeeze(0)    # (H, W)
    return cam_up.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract ALBEF Grad-CAM heatmaps for VinDr-CXR."
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
                        help="Directory to save Grad-CAM heatmaps (.pt) and index CSV")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: limit number of images for quick debugging")
    parser.add_argument("--max_text_len", type=int, default=32,
                        help="Max token length for BERT text encoder")
    parser.add_argument("--only_labels", type=str, nargs="*", default=None,
                        help="Optional: subset of labels to compute Grad-CAM for")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model/tokenizer/config ----
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )
    image_res = config["image_res"]
    transform = get_image_transform(image_res)

    # ---- Load label CSV to get image_ids + label_names ----
    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]
    label_cols = list(df.columns[1:])
    print(f"[Data] Found {len(df)} rows, {len(label_cols)} labels")

    if args.max_images is not None:
        df = df.iloc[: args.max_images].reset_index(drop=True)
        print(f"[Data] Limiting to {len(df)} images (max_images={args.max_images})")

    # Filter to rows that have a PNG
    def has_png(row):
        img_id = row[id_col]
        return (images_root / f"{img_id}.png").exists()

    df["__has_png__"] = df.apply(has_png, axis=1)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] After PNG filter: {len(df)} images")

    image_ids = df[id_col].tolist()
    label_names = label_cols

    # Optional: restrict to subset of labels
    if args.only_labels is not None:
        missing = [lb for lb in args.only_labels if lb not in label_names]
        if missing:
            raise ValueError(f"Requested labels not in CSV: {missing}")
        label_names = args.only_labels
        print(f"[Data] Restricting to labels: {label_names}")

    # ---- Multi-prompt text embeddings (shared with zero-shot) ----
    all_label_embs = get_label_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=label_cols,
        device=device,
        max_length=args.max_text_len,
    )  # (L_all, D)
    all_label_embs = F.normalize(all_label_embs, dim=-1)

    # Map label -> embedding index
    label_to_idx = {lb: i for i, lb in enumerate(label_cols)}

    # Prepare subset embedding tensor
    subset_indices = [label_to_idx[lb] for lb in label_names]
    subset_embs = all_label_embs[subset_indices]  # (L_sub, D)

    # ---- Process images ----
    gradcam_index_records = []
    patch_grid = 16  # ViT patch grid; can compute dynamically if needed

    for idx, image_id in enumerate(image_ids, start=1):
        try:
            img_path = infer_png_path(images_root, image_id)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0)  # (1,3,H,W)

        heatmaps = {}
        for j, label in enumerate(label_names):
            text_feat = subset_embs[j]  # (D,)
            cam = gradcam_for_single_label(
                model=model,
                img_tensor=img_tensor,
                text_feat=text_feat,
                device=device,
                patch_grid=patch_grid,
            )
            cam_up = upsample_cam(cam, target_size=image_res)  # (H,W)

            heatmaps[label] = cam_up  # keep as tensor for saving

        out_path = output_dir / f"{image_id}.pt"
        # convert to tensors (already are), but ensure CPU/float
        heatmaps_cpu = {k: v.float().cpu() for k, v in heatmaps.items()}
        torch.save(heatmaps_cpu, out_path)
        gradcam_index_records.append(
            {"image_id": image_id, "heatmap_path": str(out_path)}
        )

        if idx % 20 == 0 or idx == len(image_ids):
            print(f"[Grad-CAM] Processed {idx}/{len(image_ids)} images")

    # ---- Save index CSV ----
    index_df = pd.DataFrame(gradcam_index_records)
    index_path = output_dir / "gradcam_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[Output] Saved Grad-CAM index to: {index_path}")


if __name__ == "__main__":
    main()
