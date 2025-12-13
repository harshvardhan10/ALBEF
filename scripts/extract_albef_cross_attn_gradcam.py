import argparse
from pathlib import Path

import torch
import pandas as pd
from PIL import Image

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_inputs,
)
from albef_crossattn_gradcam import (
    register_albef_crossattn_gradcam_hooks,
    remove_albef_crossattn_gradcam_hooks,
    generate_albef_crossattn_gradcam,
    enable_crossattn_attention_saving
)
from albef_gradcam import upsample_cam


def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found for image_id={image_id}: {png_path}")
    return png_path


def parse_layers(s: str):
    """
    Parse --layers_to_use argument.
    Examples:
      "8" -> [8]
      "8,9,10,11" -> [8,9,10,11]
    """
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    return [int(s)]


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

    # Paper-faithful default: multimodal "3rd" layer => encoder layer 8 in your model
    parser.add_argument("--layers_to_use", type=str, default="8",
                        help='Comma-separated encoder layer indices for cross-attn CAM, e.g. "8" or "8,9,10,11"')

    # Prefer getter methods (if your modules provide them). If False, always use hooks.
    parser.add_argument("--prefer_getters", action="store_true",
                        help="Prefer get_attention_map()/get_attn_gradients() if available.")

    args = parser.parse_args()

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers_to_use = parse_layers(args.layers_to_use)
    print(f"[Config] layers_to_use={layers_to_use}, prefer_getters={args.prefer_getters}")

    # ---- Build model, tokenizer, config ----
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )
    image_res = config["image_res"]
    transform = get_image_transform(image_res)
    model.eval()

    # Enable attention_map/attn_gradients storage for the grounding layers
    enable_crossattn_attention_saving(model, layers=layers_to_use)

    # ---- Hooks (needed for fallback mode) ----
    handles = register_albef_crossattn_gradcam_hooks(model)
    print("[CrossAttn-GradCAM] Hooks registered (used as fallback if getters unavailable).")

    # ---- Load CSV & determine labels ----
    df = pd.read_csv(args.labels_csv)
    id_col = df.columns[0]
    all_label_cols = list(df.columns[1:])
    print(f"[Data] {len(df)} rows, {len(all_label_cols)} labels in CSV.")

    if args.max_images is not None:
        df = df.iloc[: args.max_images].reset_index(drop=True)

    df["__has_png__"] = df[id_col].apply(lambda x: (images_root / f"{x}.png").exists())
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] After PNG filter: {len(df)} images remain.")

    image_ids = df[id_col].astype(str).tolist()
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

        out_obj = {}  # label -> dict(cams)

        for label in label_cols:
            input_ids = input_ids_dict[label]          # (1,T)
            attn_mask = attn_mask_dict[label]          # (1,T)
            text_token_mask = token_mask_dict[label]   # (1,T) or (T,)

            cams = generate_albef_crossattn_gradcam(
                model=model,
                img_tensor=img_tensor,
                input_ids=input_ids,
                attention_mask=attn_mask,
                device=device,
                text_token_mask=text_token_mask,
                layers_to_use=layers_to_use,
                prefer_getters=args.prefer_getters,
            )
            # cams: {"cam_raw": (16,16), "cam_vis": (16,16)}

            # Upsample VIS map to image_res for thresholding / overlays
            cam_vis_up = upsample_cam(cams["cam_vis"], target_size=image_res)

            out_obj[label] = {
                "cam_raw": cams["cam_raw"].cpu(),
                "cam_vis": cams["cam_vis"].cpu(),
                "cam_vis_up": cam_vis_up.float().cpu(),   # (image_res, image_res)
            }

        out_path = output_dir / f"{image_id}.pt"
        torch.save(out_obj, out_path)
        index_records.append({"image_id": image_id, "heatmap_path": str(out_path)})

        if idx_img % 20 == 0 or idx_img == len(image_ids):
            print(f"[CrossAttn-GradCAM] Processed {idx_img}/{len(image_ids)} images")

    index_df = pd.DataFrame(index_records)
    index_path = output_dir / "crossattn_gradcam_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"[Output] Saved index to: {index_path}")

    remove_albef_crossattn_gradcam_hooks(handles)
    print("[CrossAttn-GradCAM] Hooks removed.")


if __name__ == "__main__":
    main()
