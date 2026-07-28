import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_inputs,
)

from albef_crossattn_gradcam import (
    register_albef_crossattn_gradcam_hooks,
    remove_albef_crossattn_gradcam_hooks,
    generate_albef_crossattn_gradcam,
    enable_crossattn_attention_saving,
)

from albef_gradcam import upsample_cam


SUPPORTED_VIEWS = ("original", "lung_only", "heart_only")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz")


def parse_layers(s: str):
    """
    Parse --layers_to_use argument.
    Examples:
      "8" -> [8]
      "8,9,10,11" -> [8,9,10,11]
    """
    s = s.strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_labels(s: str):
    """
    Parse --only_labels argument.
    Example:
      "Cardiomegaly"
      "Cardiomegaly,Pleural effusion"
    """
    if s is None or s.strip() == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def infer_png_path(images_root: Path, image_id: str) -> Path:
    png_path = images_root / f"{image_id}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"PNG not found for image_id={image_id}: {png_path}")
    return png_path


def infer_mask_path(mask_root: Path, image_id: str) -> Path:
    """
    Find a mask stored directly under ``mask_root``.

    Supported formats are common image formats, .npy, and .npz. For .npz,
    either a single array or an array under the key ``mask`` is expected.
    """
    for extension in MASK_EXTENSIONS:
        candidate = mask_root / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Mask not found for image_id={image_id} under {mask_root}. "
        f"Expected one of: {', '.join(image_id + ext for ext in MASK_EXTENSIONS)}"
    )


def load_binary_mask(mask_path: Path, image_size) -> np.ndarray:
    """Load a mask, resize it with nearest-neighbour interpolation, and binarize it."""
    suffix = mask_path.suffix.lower()

    if suffix == ".npy":
        mask = np.load(mask_path)
    elif suffix == ".npz":
        with np.load(mask_path) as data:
            if "mask" in data:
                mask = data["mask"]
            elif len(data.files) == 1:
                mask = data[data.files[0]]
            else:
                raise ValueError(
                    f"{mask_path} contains multiple arrays and no 'mask' key: "
                    f"{data.files}"
                )
    else:
        mask = np.asarray(Image.open(mask_path).convert("L"))

    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2-D mask for {mask_path}, got shape {mask.shape}"
        )

    # PIL expects (width, height), which is also Image.size.
    if (mask.shape[1], mask.shape[0]) != image_size:
        mask_pil = Image.fromarray(mask.astype(np.float32), mode="F")
        mask = np.asarray(
            mask_pil.resize(image_size, resample=Image.Resampling.NEAREST)
        )

    # Works for masks encoded as bool, 0/1, 0/255, or probabilities.
    finite_mask = np.nan_to_num(mask.astype(np.float32), nan=0.0)
    threshold = 0.5 if finite_mask.max(initial=0.0) <= 1.0 else 127.5
    return finite_mask > threshold


def apply_anatomy_view(
    image: Image.Image,
    image_id: str,
    view: str,
    mask_root: Path,
) -> Image.Image:
    """Return the original image or an RGB image with pixels outside the mask zeroed."""
    if view == "original":
        return image

    if mask_root is None:
        raise ValueError(f"--mask_root is required when --view={view}")

    mask_path = infer_mask_path(mask_root, image_id)
    mask = load_binary_mask(mask_path, image.size)

    image_array = np.asarray(image, dtype=np.uint8)
    masked_array = image_array * mask[..., None].astype(np.uint8)
    return Image.fromarray(masked_array, mode="RGB")


def load_image_ids_and_labels(
    labels_csv: Path,
    images_root: Path,
    only_labels=None,
    max_images=None,
    positive_only_label=None,
    view="original",
    mask_root=None,
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
            f"[Data] positive_only_label={positive_only_label}: "
            f"{before} -> {len(df)} rows",
            flush=True,
        )

    df["__has_png__"] = df[id_col].apply(
        lambda x: (images_root / f"{str(x)}.png").exists()
    )

    before_png = len(df)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(
        f"[Data] PNG filter: {before_png} -> {len(df)} images",
        flush=True,
    )

    if len(df) == 0:
        raise RuntimeError(f"No valid PNG images found under: {images_root}")

    if view != "original":
        if mask_root is None:
            raise ValueError(f"--mask_root is required when --view={view}")
        if not mask_root.exists():
            raise FileNotFoundError(f"Mask root not found: {mask_root}")

        missing_masks = []
        for image_id in df[id_col].astype(str):
            try:
                infer_mask_path(mask_root, image_id)
            except FileNotFoundError:
                missing_masks.append(image_id)

        if missing_masks:
            preview = ", ".join(missing_masks[:10])
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} {view} masks under {mask_root}. "
                f"First missing image IDs: {preview}. Heatmap extraction was "
                "stopped to avoid mixing original and masked views."
            )
        print(
            f"[Data] Mask check: found all {len(df)} masks under {mask_root}",
            flush=True,
        )

    if max_images is not None:
        df = df.iloc[:max_images].reset_index(drop=True)
        print(f"[Data] max_images={max_images}, using {len(df)} images", flush=True)

    image_ids = df[id_col].astype(str).tolist()

    print(f"[Data] Final number of images: {len(image_ids)}", flush=True)
    print(f"[Data] Labels to extract: {label_cols}", flush=True)

    return df, id_col, label_cols, image_ids


def extract_heatmaps_for_checkpoint(
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
    view="original",
    mask_root=None,
):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    if view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported view={view!r}; choose from {SUPPORTED_VIEWS}")

    if view != "original":
        if mask_root is None:
            raise ValueError(f"--mask_root is required when --view={view}")
        if not mask_root.exists():
            raise FileNotFoundError(f"Mask root not found: {mask_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("[Config] Cross-attention Grad-CAM heatmap extraction", flush=True)
    print(f"[Config] config_path       = {config_path}", flush=True)
    print(f"[Config] ckpt_path         = {ckpt_path}", flush=True)
    print(f"[Config] labels_csv        = {labels_csv}", flush=True)
    print(f"[Config] images_root       = {images_root}", flush=True)
    print(f"[Config] view              = {view}", flush=True)
    print(f"[Config] mask_root         = {mask_root}", flush=True)
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
        view=view,
        mask_root=mask_root,
    )

    input_ids_dict, attn_mask_dict, token_mask_dict = get_label_text_inputs(
        tokenizer=tokenizer,
        labels=label_cols,
        max_length=max_length,
    )

    enable_crossattn_attention_saving(model, layers=layers_to_use)

    handles = register_albef_crossattn_gradcam_hooks(model)
    print("[CrossAttn-GradCAM] Hooks registered.", flush=True)

    index_records = []

    try:
        for idx_img, image_id in enumerate(tqdm(image_ids, desc="Extracting heatmaps"), start=1):
            img_path = infer_png_path(images_root, image_id)

            out_path = output_dir / f"{image_id}.pt"

            if out_path.exists() and not overwrite:
                index_records.append(
                    {
                        "image_id": image_id,
                        "heatmap_path": str(out_path),
                        "status": "exists_skipped",
                        "view": view,
                        "mask_root": (
                            str(mask_root) if mask_root is not None else None
                        ),
                    }
                )
                continue

            img_pil = Image.open(img_path).convert("RGB")
            img_pil = apply_anatomy_view(
                image=img_pil,
                image_id=image_id,
                view=view,
                mask_root=mask_root,
            )
            img_tensor = transform(img_pil).unsqueeze(0)

            out_obj = {
                "__metadata__": {
                    "image_id": image_id,
                    "view": view,
                    "mask_root": str(mask_root) if mask_root is not None else None,
                }
            }

            for label in label_cols:
                input_ids = input_ids_dict[label]
                attn_mask = attn_mask_dict[label]
                text_token_mask = token_mask_dict[label]

                cams = generate_albef_crossattn_gradcam(
                    model=model,
                    img_tensor=img_tensor,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    device=device,
                    text_token_mask=text_token_mask,
                    layers_to_use=layers_to_use,
                    prefer_getters=True,
                )

                cam_raw = cams["cam_raw"].detach().float().cpu()
                cam_vis = cams["cam_vis"].detach().float().cpu()

                cam_vis_up = upsample_cam(cam_vis, target_size=image_res)
                cam_vis_up = cam_vis_up.detach().float().cpu()

                out_obj[label] = {
                    "cam_raw": cam_raw,             # usually 16 x 16
                    "cam_vis": cam_vis,             # usually 16 x 16, normalized/visual form
                    "cam_vis_up": cam_vis_up,       # image_res x image_res
                    "layers_to_use": layers_to_use,
                }

            torch.save(out_obj, out_path)

            row_record = {
                "image_id": image_id,
                "heatmap_path": str(out_path),
                "status": "saved",
                "view": view,
                "mask_root": str(mask_root) if mask_root is not None else None,
            }

            for lb in label_cols:
                if lb in df.columns:
                    matched = df[df[id_col].astype(str) == image_id]
                    if len(matched) == 1:
                        row_record[f"y::{lb}"] = float(matched.iloc[0][lb])

            index_records.append(row_record)

            if idx_img % 50 == 0 or idx_img == len(image_ids):
                print(
                    f"[CrossAttn-GradCAM] Processed {idx_img}/{len(image_ids)} images",
                    flush=True,
                )

    finally:
        remove_albef_crossattn_gradcam_hooks(handles)
        print("[CrossAttn-GradCAM] Hooks removed.", flush=True)

    index_df = pd.DataFrame(index_records)
    index_path = output_dir / "crossattn_gradcam_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"[Output] Saved index to: {index_path}", flush=True)
    print("[Done] Heatmap extraction complete.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ALBEF cross-attention Grad-CAM heatmaps for VinDr-CXR."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ALBEF config YAML.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ALBEF checkpoint .pth.",
    )

    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="VinDr image-level labels CSV.",
    )

    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Directory containing VinDr test PNG images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where .pt heatmaps and index CSV will be saved.",
    )

    parser.add_argument(
        "--view",
        type=str,
        choices=SUPPORTED_VIEWS,
        default="original",
        help=(
            "Image view supplied to the model. lung_only and heart_only zero "
            "pixels outside the corresponding binary mask."
        ),
    )

    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help=(
            "Directory containing masks named <image_id>.<extension>. Required "
            "for lung_only and heart_only; ignored for original."
        ),
    )

    parser.add_argument(
        "--layers_to_use",
        type=str,
        default="8",
        help='Cross-attention layers to use, e.g. "8" or "8,9,10,11".',
    )

    parser.add_argument(
        "--only_labels",
        type=str,
        default="Cardiomegaly",
        help='Comma-separated labels to extract, e.g. "Cardiomegaly,Pleural effusion".',
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of images for debugging.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device, e.g. "cuda" or "cpu".',
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Maximum text token length.",
    )

    parser.add_argument(
        "--positive_only_label",
        type=str,
        default=None,
        help=(
            "Optional: restrict image set to positives of this label. "
            "Do NOT use this for FROC; FROC needs all images."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing heatmap .pt files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layers_to_use = parse_layers(args.layers_to_use)
    only_labels = parse_labels(args.only_labels)

    extract_heatmaps_for_checkpoint(
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
        view=args.view,
        mask_root=Path(args.mask_root) if args.mask_root is not None else None,
    )


if __name__ == "__main__":
    main()
