"""
Proper hook-based Grad-CAM for ALBEF on VinDr-CXR.

For each image and label:
  - Compute score = cosine(CLS(image), text_embedding(label)).
  - Backprop to last ViT block to get Grad-CAM activations.
  - Build CAM over patches and upsample to image_res x image_res.
  - Save per-image .pt with {label_name: heatmap_tensor(H,W)}.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_embeddings,
)
from albef_gradcam import (
    register_vit_gradcam_hooks,
    remove_vit_gradcam_hooks,
    generate_albef_gradcam,
    upsample_cam,
)


def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found for image_id={image_id}: {png_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract ALBEF Grad-CAM heatmaps on VinDr-CXR."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="ALBEF config YAML (e.g., configs/Pretrain.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ALBEF checkpoint path (e.g., checkpoint_09.pth)")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="VinDr image-level labels CSV (image_id + label columns)")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root folder with 256x256 VinDr PNGs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save Grad-CAM .pt files and index CSV")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: limit number of images (debug)")
    parser.add_argument("--max_text_len", type=int, default=32,
                        help="Max token length for BERT text encoder")
    parser.add_argument("--only_labels", type=str, nargs="*", default=None,
                        help="Optional: subset of labels to compute Grad-CAM for")
    parser.add_argument("--patch_grid", type=int, default=None,
                        help="Optional: ViT patch grid size (e.g., 16). "
                             "If None, inferred from number of patches.")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model, tokenizer, config ----
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )
    image_res = config["image_res"]
    transform = get_image_transform(image_res)
    model.eval()

    # ---- Register Grad-CAM hooks ----
    handles = register_vit_gradcam_hooks(model)
    print("[Grad-CAM] Hooks registered.")

    # ---- Load CSV & determine labels ----
    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]
    all_label_cols = list(df.columns[1:])
    print(f"[Data] {len(df)} rows, {len(all_label_cols)} labels in CSV.")

    if args.max_images is not None:
        df = df.iloc[: args.max_images].reset_index(drop=True)
        print(f"[Data] Restricting to first {len(df)} images (max_images={args.max_images}).")

    # Filter rows that actually have PNGs
    def has_png(row):
        return (images_root / f"{row[id_col]}.png").exists()

    df["__has_png__"] = df.apply(has_png, axis=1)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] After PNG filter: {len(df)} images remain.")

    image_ids = df[id_col].tolist()
    label_cols = all_label_cols

    # Optional: subset of labels
    if args.only_labels is not None:
        missing = [lb for lb in args.only_labels if lb not in label_cols]
        if missing:
            raise ValueError(f"Requested labels not in CSV: {missing}")
        label_cols = args.only_labels
        print(f"[Data] Restricting to labels: {label_cols}")

    # ---- Precompute text embeddings for ALL labels in CSV ----
    print("[Text] Computing label text embeddings...")
    all_label_embs = get_label_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=all_label_cols,
        device=device,
        max_length=args.max_text_len,
    )  # (L_all, D)
    all_label_embs = F.normalize(all_label_embs, dim=-1)
    print("[Text] Done.")

    # Map label name -> index into all_label_embs
    label_to_idx = {lb: i for i, lb in enumerate(all_label_cols)}

    # Prepare subset embedding tensor
    subset_indices = [label_to_idx[lb] for lb in label_cols]
    subset_embs = all_label_embs[subset_indices]  # (L_sub, D)

    # ---- Process images ----
    index_records = []

    for idx_img, image_id in enumerate(image_ids, start=1):
        try:
            img_path = infer_png_path(images_root, image_id)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0)  # (1,3,H,W)

        heatmaps = {}
        for j, label in enumerate(label_cols):
            text_feat = subset_embs[j]  # (D,)

            cam_patch = generate_albef_gradcam(
                model=model,
                img_tensor=img_tensor,
                text_feat=text_feat,
                device=device,
                patch_grid=args.patch_grid,
            )  # (H', W')

            cam_up = upsample_cam(cam_patch, target_size=image_res)  # (H,W)
            heatmaps[label] = cam_up  # tensor

        out_path = output_dir / f"{image_id}.pt"
        heatmaps_cpu = {k: v.float().cpu() for k, v in heatmaps.items()}
        torch.save(heatmaps_cpu, out_path)
        index_records.append(
            {"image_id": image_id, "heatmap_path": str(out_path)}
        )

        if idx_img % 20 == 0 or idx_img == len(image_ids):
            print(f"[Grad-CAM] Processed {idx_img}/{len(image_ids)} images")

    # ---- Save index CSV ----
    index_df = pd.DataFrame(index_records)
    index_path = output_dir / "gradcam_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[Output] Saved Grad-CAM index to: {index_path}")

    # ---- Remove hooks ----
    remove_vit_gradcam_hooks(handles)
    print("[Grad-CAM] Hooks removed.")


if __name__ == "__main__":
    main()
