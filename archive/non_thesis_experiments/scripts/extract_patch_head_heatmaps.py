"""
Extract A5 learned patch-head heatmaps for VinDr-CXR localization/FROC.

This script keeps compatibility with compute_froc_from_gradcam_heatmaps.py by:
  - writing crossattn_gradcam_index.csv
  - saving per-image .pt files structured as out_obj[label][cam_key]
  - saving the learned A5 map under both:
        patch_pred_up
        cam_vis_up        # alias, so FROC can remain unchanged

Recommended FROC usage:
  python compute_froc_from_gradcam_heatmaps.py \
    --heatmaps_dir <A5_HEATMAP_DIR> \
    --cam_key cam_vis_up \
    ...

Or explicitly:
  --cam_key patch_pred_up
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src import (
    build_model_and_tokenizer,
    get_image_transform,
)

from anatomy_prior.attention_extract import (
    enable_crossattn_attention_saving_for_anatomy,
    extract_raw_crossattn_for_anatomy_loss,
)
from anatomy_prior.token_utils import build_token_mask

from anatomy_prior.patch_head import (
    build_patch_head_from_config,
    upsample_patch_vector,
    patch_vector_to_grid
)

# ============================================================
# Utilities copied/adapted from existing extraction script
# ============================================================

def parse_layers(s: str):
    s = s.strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_labels(s: str):
    if s is None or s.strip() == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def minmax_norm_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    xmin = x.amin(dim=(-2, -1), keepdim=True)
    xmax = x.amax(dim=(-2, -1), keepdim=True)
    return (x - xmin) / (xmax - xmin).clamp_min(eps)


def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found for image_id={image_id}: {png_path}")
    return png_path


def load_image_ids_and_labels(
    labels_csv: Path,
    images_root: Path,
    only_labels=None,
    max_images=None,
    positive_only_label=None,
):
    df = pd.read_csv(labels_csv)

    id_col = df.columns[0]
    all_label_cols = list(df.columns[1:])

    print(f"[Data] Loaded CSV: {labels_csv}", flush=True)
    print(f"[Data] Original rows: {len(df)}", flush=True)
    print(f"[Data] Label columns: {len(all_label_cols)}", flush=True)

    if only_labels is not None:
        missing = [lb for lb in only_labels if lb not in all_label_cols]
        if missing:
            raise ValueError(f"Requested labels not in CSV: {missing}")
        label_cols = only_labels
    else:
        label_cols = all_label_cols

    if positive_only_label is not None:
        if positive_only_label not in all_label_cols:
            raise ValueError(
                f"--positive_only_label={positive_only_label} not found in CSV labels."
            )
        before = len(df)
        df = df[df[positive_only_label] == 1].reset_index(drop=True)
        print(
            f"[Data] positive_only_label={positive_only_label}: {before} -> {len(df)} rows",
            flush=True,
        )

    df["__has_png__"] = df[id_col].apply(
        lambda x: (images_root / f"{str(x)}.png").exists()
    )
    before_png = len(df)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] PNG filter: {before_png} -> {len(df)} images", flush=True)

    if len(df) == 0:
        raise RuntimeError(f"No valid PNG images found under: {images_root}")

    if max_images is not None:
        df = df.iloc[:max_images].reset_index(drop=True)
        print(f"[Data] max_images={max_images}, using {len(df)} images", flush=True)

    image_ids = df[id_col].astype(str).tolist()
    print(f"[Data] Final number of images: {len(image_ids)}", flush=True)
    print(f"[Data] Labels to extract: {label_cols}", flush=True)
    return df, id_col, label_cols, image_ids


def load_a5_patch_head(ckpt_path: Path, config: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "patch_head" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain checkpoint['patch_head']. "
            "Use an A5 checkpoint produced by Pretrain_anatomy_prior_A5_patch_head.py."
        )

    patch_head = build_patch_head_from_config(config)
    patch_head.load_state_dict(ckpt["patch_head"], strict=True)
    patch_head = patch_head.to(device)
    patch_head.eval()

    print(
        f"[A5] Loaded patch_head: num_patches={patch_head.num_patches}, "
        f"num_layers={patch_head.num_layers}, normalization={patch_head.normalization}",
        flush=True,
    )
    return patch_head


def extract_patch_head_maps_for_checkpoint(
    config_path: Path,
    ckpt_path: Path,
    labels_csv: Path,
    images_root: Path,
    output_dir: Path,
    layers_to_use,
    only_labels=None,
    max_images=None,
    device_str="cuda",
    max_length=32,
    positive_only_label=None,
    overwrite=False,
    save_attn_debug=True,
):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("[Config] A5 patch-head heatmap extraction", flush=True)
    print(f"[Config] config_path       = {config_path}", flush=True)
    print(f"[Config] ckpt_path         = {ckpt_path}", flush=True)
    print(f"[Config] labels_csv        = {labels_csv}", flush=True)
    print(f"[Config] images_root       = {images_root}", flush=True)
    print(f"[Config] output_dir        = {output_dir}", flush=True)
    print(f"[Config] layers_to_use     = {layers_to_use}", flush=True)
    print(f"[Config] only_labels       = {only_labels}", flush=True)
    print(f"[Config] max_images        = {max_images}", flush=True)
    print(f"[Config] positive_only     = {positive_only_label}", flush=True)
    print("=" * 80, flush=True)

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=str(config_path),
        ckpt_path=str(ckpt_path),
        device=device_str,
    )
    model.eval()

    patch_head = load_a5_patch_head(ckpt_path=ckpt_path, config=config, device=device)

    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)

    print(f"[Model] image_res={image_res}", flush=True)
    print(f"[Model] device={device}", flush=True)

    df, id_col, label_cols, image_ids = load_image_ids_and_labels(
        labels_csv=labels_csv,
        images_root=images_root,
        only_labels=only_labels,
        max_images=max_images,
        positive_only_label=positive_only_label,
    )

    enable_crossattn_attention_saving_for_anatomy(model, layers=layers_to_use)
    print("[A5] Enabled raw cross-attention saving for patch-head extraction.", flush=True)

    index_records = []

    for idx_img, image_id in enumerate(tqdm(image_ids, desc="Extracting A5 patch-head maps"), start=1):
        img_path = infer_png_path(images_root, image_id)
        out_path = output_dir / f"{image_id}.pt"

        if out_path.exists() and not overwrite:
            index_records.append(
                {"image_id": image_id, "heatmap_path": str(out_path), "status": "exists_skipped"}
            )
            continue

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        out_obj = {}

        for label in label_cols:
            label_text = str(label).lower().strip()

            text_input = tokenizer(
                [label_text],
                padding='longest',
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            target_token_ids = tokenizer(
                label_text,
                add_special_tokens=False,
            ).input_ids

            target_token_mask = build_token_mask(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                target_token_ids=target_token_ids,
            )

            if not target_token_mask.any():
                raise RuntimeError(
                    f"Target token mask is empty for label='{label}' and label_text='{label_text}'."
                )

            model.zero_grad(set_to_none=True)
            patch_head.zero_grad(set_to_none=True)

            with torch.enable_grad():
                _ = model(img_tensor, text_input, alpha=0.0)

                attn_patch = extract_raw_crossattn_for_anatomy_loss(
                    model=model,
                    text_token_mask=target_token_mask,
                    layers_to_use=layers_to_use,
                    remove_image_cls=True,
                    normalize_patches=True,
                )

                attn_patch_detached = attn_patch.detach()

            with torch.no_grad():
                patch_pred = patch_head(attn_patch_detached)

                # [1,N] -> [S,S] and [image_res,image_res]
                patch_pred_grid = patch_vector_to_grid(patch_pred).squeeze(0).detach().float().cpu()
                patch_pred_up = upsample_patch_vector(patch_pred, target_size=image_res).squeeze(0)
                patch_pred_up = patch_pred_up.detach().float().cpu()

                # Visual/minmax versions for connected components.
                patch_pred_vis = minmax_norm_torch(patch_pred_grid.unsqueeze(0)).squeeze(0).cpu()
                patch_pred_up_vis = minmax_norm_torch(patch_pred_up.unsqueeze(0)).squeeze(0).cpu()

                label_payload = {
                    # A5 learned map
                    "patch_pred_raw": patch_pred_grid,         # usually 16 x 16, probability mass
                    "patch_pred_vis": patch_pred_vis,          # usually 16 x 16, minmax visual
                    "patch_pred_up": patch_pred_up_vis,        # image_res x image_res

                    # Compatibility alias: compute_froc default --cam_key cam_vis_up works unchanged.
                    "cam_vis_up": patch_pred_up_vis,

                    "layers_to_use": layers_to_use,
                    "source": "A5_patch_head_from_detached_raw_crossattention",
                }

                if save_attn_debug:
                    attn_grid = patch_vector_to_grid(attn_patch_detached).squeeze(0).detach().float().cpu()
                    attn_up = upsample_patch_vector(attn_patch_detached, target_size=image_res).squeeze(0)
                    attn_up = attn_up.detach().float().cpu()
                    label_payload.update(
                        {
                            "attn_raw": attn_grid,
                            "attn_up": minmax_norm_torch(attn_up.unsqueeze(0)).squeeze(0).cpu(),
                        }
                    )

                out_obj[label] = label_payload

        torch.save(out_obj, out_path)

        row_record = {"image_id": image_id, "heatmap_path": str(out_path), "status": "saved"}
        for lb in label_cols:
            if lb in df.columns:
                matched = df[df[id_col].astype(str) == image_id]
                if len(matched) == 1:
                    row_record[f"y::{lb}"] = float(matched.iloc[0][lb])
        index_records.append(row_record)

        if idx_img % 50 == 0 or idx_img == len(image_ids):
            print(f"[A5] Processed {idx_img}/{len(image_ids)} images", flush=True)

    index_df = pd.DataFrame(index_records)
    # Keep this exact file name because compute_froc_from_gradcam_heatmaps.py expects it.
    index_path = output_dir / "crossattn_gradcam_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"[Output] Saved index to: {index_path}", flush=True)
    print("[Done] A5 patch-head heatmap extraction complete.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract A5 patch-head heatmaps for VinDr-CXR."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to A5 config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to A5 checkpoint .pth.")
    parser.add_argument("--labels_csv", type=str, required=True, help="VinDr image-level labels CSV.")
    parser.add_argument("--images_root", type=str, required=True, help="Directory containing VinDr PNGs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for .pt maps and index CSV.")
    parser.add_argument("--layers_to_use", type=str, default="8", help='Cross-attention layers, e.g. "8".')
    parser.add_argument("--only_labels", type=str, default="Cardiomegaly", help='Comma-separated labels.')
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument(
        "--positive_only_label",
        type=str,
        default=None,
        help="Optional debug filter only. Do NOT use this for FROC, which needs all images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_save_attn_debug",
        action="store_true",
        help="Do not save original raw-attention debug maps alongside patch-head maps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layers_to_use = parse_layers(args.layers_to_use)
    only_labels = parse_labels(args.only_labels)

    extract_patch_head_maps_for_checkpoint(
        config_path=Path(args.config),
        ckpt_path=Path(args.checkpoint),
        labels_csv=Path(args.labels_csv),
        images_root=Path(args.images_root),
        output_dir=Path(args.output_dir),
        layers_to_use=layers_to_use,
        only_labels=only_labels,
        max_images=args.max_images,
        device_str=args.device,
        max_length=args.max_length,
        positive_only_label=args.positive_only_label,
        overwrite=args.overwrite,
        save_attn_debug=not args.no_save_attn_debug,
    )


if __name__ == "__main__":
    main()
