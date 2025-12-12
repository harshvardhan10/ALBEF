# extract_albef_crossattn_gradcam_vindr.py

import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_inputs,   # <- new helper
)
from albef_crossattn_gradcam import (
    register_albef_crossattn_gradcam_hooks,
    remove_albef_crossattn_gradcam_hooks,
    generate_albef_crossattn_gradcam,
)
from albef_gradcam import upsample_cam  # reuse your upsampling

def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found for image_id={image_id}: {png_path}")
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract ALBEF cross-attention Grad-CAM heatmaps on VinDr-CXR."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_text_len", type=int, default=32)
    parser.add_argument("--only_labels", type=str, nargs="*", default=None)
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

    # ---- Register cross-attn hooks ----
    handles = register_albef_crossattn_gradcam_hooks(model)
    print("[CrossAttn-GradCAM] Hooks registered.")

    # ---- Load CSV & determine labels ----
    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]
    all_label_cols = list(df.columns[1:])
    print(f"[Data] {len(df)} rows, {len(all_label_cols)} labels in CSV.")

    if args.max_images is not None:
        df = df.iloc[: args.max_images].reset_index(drop=True)

    def has_png(row):
        return (images_root / f"{row[id_col]}.png").exists()

    df["__has_png__"] = df.apply(has_png, axis=1)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] After PNG filter: {len(df)} images remain.")

    image_ids = df[id_col].tolist()
    label_cols = all_label_cols

    if args.only_labels is not None:
        missing = [lb for lb in args.only_labels if lb not in label_cols]
        if missing:
            raise ValueError(f"Requested labels not in CSV: {missing}")
        label_cols = args.only_labels
        print(f"[Data] Restricting to labels: {label_cols}")

    # ---- Precompute text inputs per label ----
    input_ids_dict, attn_mask_dict, token_mask_dict = get_label_text_inputs(
        tokenizer=tokenizer,
        labels=label_cols,
        max_length=args.max_text_len,
    )

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
        for label in label_cols:
            input_ids = input_ids_dict[label]          # (1,T)
            attn_mask = attn_mask_dict[label]          # (1,T)
            text_token_mask = token_mask_dict[label]   # (1,T)

            cam_patch = generate_albef_crossattn_gradcam(
                model=model,
                img_tensor=img_tensor,
                input_ids=input_ids,
                attention_mask=attn_mask,
                device=device,
                text_token_mask=text_token_mask,
                layers_to_use=None
            )

            cam_up = upsample_cam(cam_patch, target_size=image_res)
            heatmaps[label] = cam_up

        out_path = output_dir / f"{image_id}.pt"
        heatmaps_cpu = {k: v.float().cpu() for k, v in heatmaps.items()}
        torch.save(heatmaps_cpu, out_path)
        index_records.append({"image_id": image_id, "heatmap_path": str(out_path)})

        if idx_img % 20 == 0 or idx_img == len(image_ids):
            print(f"[CrossAttn-GradCAM] Processed {idx_img}/{len(image_ids)} images")

    index_df = pd.DataFrame(index_records)
    index_path = output_dir / "crossattn_gradcam_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[Output] Saved cross-attn Grad-CAM index to: {index_path}")

    remove_albef_crossattn_gradcam_hooks(handles)
    print("[CrossAttn-GradCAM] Hooks removed.")


if __name__ == "__main__":
    main()
