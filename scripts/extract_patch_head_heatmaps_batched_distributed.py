"""
Batched + multi-GPU A5 patch-head heatmap extraction for VinDr-CXR.

This is a faster replacement for the per-image extract_patch_head_heatmaps.py.
It keeps compatibility with compute_froc_from_gradcam_heatmaps.py by:
  - writing crossattn_gradcam_index.csv
  - saving per-image .pt files structured as out_obj[label][cam_key]
  - saving the learned A5 map under both patch_pred_up and cam_vis_up

Recommended 2-GPU usage:
  python -u -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    --use_env \
    scripts/extract_patch_head_heatmaps_batched_distributed.py \
    --config ./configs/Pretrain_A5_patch_head.yaml \
    --checkpoint ./output_anatomy_A5_patch_head_cm_lambda_0.01/checkpoint_last.pth \
    --labels_csv /path/to/vindr/image_labels_test.csv \
    --images_root /path/to/vindr/test_pngs \
    --output_dir ./heatmaps_A5_last_patch_head \
    --only_labels Cardiomegaly \
    --layers_to_use 8 \
    --batch_size 16 \
    --num_workers 4 \
    --device cuda \
    --overwrite \
    --no_save_attn_debug
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    patch_vector_to_grid,
)


# ============================================================
# Distributed utilities
# ============================================================

def init_distributed_from_env(device_str: str):
    """
    Supports:
      - non-distributed single process
      - torch.distributed.launch --use_env
      - torchrun / torch.distributed.run

    Returns: rank, world_size, local_rank, device
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    distributed = world_size > 1

    if device_str.startswith("cuda") and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier_if_needed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# General utilities
# ============================================================

def parse_layers(s: str):
    s = str(s).strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_labels(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def minmax_norm_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x can be [B,H,W], [H,W], etc. Normalizes independently over last two dims.
    """
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

    df[id_col] = df[id_col].astype(str)
    image_ids = df[id_col].tolist()
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


# ============================================================
# Dataset
# ============================================================

class VinDrImageDataset(Dataset):
    def __init__(self, rows_df: pd.DataFrame, id_col: str, images_root: Path, transform):
        self.df = rows_df.reset_index(drop=True)
        self.id_col = id_col
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx][self.id_col])
        img_path = infer_png_path(self.images_root, image_id)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil)
        return img_tensor, image_id


# ============================================================
# Saving helpers
# ============================================================

def build_y_lookup(df: pd.DataFrame, id_col: str, label_cols: List[str]) -> Dict[str, Dict[str, float]]:
    lookup = {}
    for _, row in df.iterrows():
        image_id = str(row[id_col])
        lookup[image_id] = {}
        for lb in label_cols:
            if lb in df.columns:
                lookup[image_id][f"y::{lb}"] = float(row[lb])
    return lookup


def make_index_record(image_id: str, out_path: Path, status: str, y_lookup: Dict[str, Dict[str, float]]):
    rec = {
        "image_id": str(image_id),
        "heatmap_path": str(out_path),
        "status": status,
    }
    rec.update(y_lookup.get(str(image_id), {}))
    return rec


# ============================================================
# Main extraction
# ============================================================

def extract_patch_head_maps_for_checkpoint_batched(
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
    batch_size=16,
    num_workers=4,
    amp=False,
):
    rank, world_size, local_rank, device = init_distributed_from_env(device_str=device_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_needed(world_size)

    if is_main_process(rank):
        print("=" * 80, flush=True)
        print("[Config] Batched A5 patch-head heatmap extraction", flush=True)
        print(f"[Config] config_path       = {config_path}", flush=True)
        print(f"[Config] ckpt_path         = {ckpt_path}", flush=True)
        print(f"[Config] labels_csv        = {labels_csv}", flush=True)
        print(f"[Config] images_root       = {images_root}", flush=True)
        print(f"[Config] output_dir        = {output_dir}", flush=True)
        print(f"[Config] layers_to_use     = {layers_to_use}", flush=True)
        print(f"[Config] only_labels       = {only_labels}", flush=True)
        print(f"[Config] max_images        = {max_images}", flush=True)
        print(f"[Config] positive_only     = {positive_only_label}", flush=True)
        print(f"[Config] batch_size        = {batch_size}", flush=True)
        print(f"[Config] num_workers       = {num_workers}", flush=True)
        print(f"[Config] amp               = {amp}", flush=True)
        print(f"[Dist] world_size={world_size}", flush=True)
        print("=" * 80, flush=True)

    # Build model separately on each GPU/rank. No DDP is needed for extraction.
    model, tokenizer, config, build_device = build_model_and_tokenizer(
        config_path=str(config_path),
        ckpt_path=str(ckpt_path),
        device=str(device),
    )
    model = model.to(device)
    model.eval()

    patch_head = load_a5_patch_head(ckpt_path=ckpt_path, config=config, device=device)

    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)

    if is_main_process(rank):
        print(f"[Model] image_res={image_res}", flush=True)
        print(f"[Model] device={device}", flush=True)

    # Load full list on every rank; split deterministically by row position.
    df, id_col, label_cols, image_ids = load_image_ids_and_labels(
        labels_csv=labels_csv,
        images_root=images_root,
        only_labels=only_labels,
        max_images=max_images,
        positive_only_label=positive_only_label,
    )

    y_lookup = build_y_lookup(df=df, id_col=id_col, label_cols=label_cols)

    # Rank split: rank 0 processes rows 0, world_size, ...; rank 1 processes 1, world_size, ...
    rank_df = df.iloc[rank::world_size].reset_index(drop=True)

    # Split skipped vs to-process before creating DataLoader.
    index_records = []
    process_rows = []
    for _, row in rank_df.iterrows():
        image_id = str(row[id_col])
        out_path = output_dir / f"{image_id}.pt"
        if out_path.exists() and not overwrite:
            index_records.append(make_index_record(image_id, out_path, "exists_skipped", y_lookup))
        else:
            process_rows.append(row)

    process_df = pd.DataFrame(process_rows).reset_index(drop=True) if process_rows else rank_df.iloc[:0].copy()

    print(
        f"[Rank {rank}/{world_size}] total_assigned={len(rank_df)} "
        f"to_process={len(process_df)} skipped={len(index_records)} device={device}",
        flush=True,
    )

    enable_crossattn_attention_saving_for_anatomy(model, layers=layers_to_use)
    print(f"[Rank {rank}] Enabled raw cross-attention saving.", flush=True)

    dataset = VinDrImageDataset(
        rows_df=process_df,
        id_col=id_col,
        images_root=images_root,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )

    # Precompute target token ids per label. Text batches are still created per batch because B changes.
    target_token_ids_by_label = {
        label: tokenizer(str(label).lower().strip(), add_special_tokens=False).input_ids
        for label in label_cols
    }

    pbar = tqdm(
        loader,
        desc=f"Rank {rank} extracting A5 maps",
        disable=not is_main_process(rank),
    )

    for batch_idx, (images, batch_image_ids) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        batch_image_ids = [str(x) for x in batch_image_ids]
        B = images.shape[0]

        # One output object per image in this batch.
        out_objs = [{} for _ in range(B)]

        for label in label_cols:
            label_text = str(label).lower().strip()
            texts = [label_text] * B

            text_input = tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            ).to(device)

            target_token_mask = build_token_mask(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                target_token_ids=target_token_ids_by_label[label],
            )

            if not target_token_mask.any():
                raise RuntimeError(
                    f"Target token mask is empty for label='{label}' and label_text='{label_text}'."
                )

            model.zero_grad(set_to_none=True)
            patch_head.zero_grad(set_to_none=True)

            # In the current ALBEF/XBERT code, attention saving registers hooks on
            # attention tensors. That requires grad tracking. We do not call backward(),
            # so this is still extraction only.
            with torch.enable_grad():
                if amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = model(images, text_input, alpha=0.0)
                else:
                    _ = model(images, text_input, alpha=0.0)

                attn_patch = extract_raw_crossattn_for_anatomy_loss(
                    model=model,
                    text_token_mask=target_token_mask,
                    layers_to_use=layers_to_use,
                    remove_image_cls=True,
                    normalize_patches=True,
                )
                attn_patch_detached = attn_patch.detach().float()

            with torch.no_grad():
                patch_pred = patch_head(attn_patch_detached)

                # [B,N] -> [B,S,S] and [B,image_res,image_res]
                patch_pred_grid = patch_vector_to_grid(patch_pred).detach().float().cpu()
                patch_pred_up = upsample_patch_vector(patch_pred, target_size=image_res)
                patch_pred_up = patch_pred_up.detach().float().cpu()

                patch_pred_vis = minmax_norm_torch(patch_pred_grid).cpu()
                patch_pred_up_vis = minmax_norm_torch(patch_pred_up).cpu()

                if save_attn_debug:
                    attn_grid = patch_vector_to_grid(attn_patch_detached).detach().float().cpu()
                    attn_up = upsample_patch_vector(attn_patch_detached, target_size=image_res)
                    attn_up = attn_up.detach().float().cpu()
                    attn_up_vis = minmax_norm_torch(attn_up).cpu()

                for b in range(B):
                    label_payload = {
                        "patch_pred_raw": patch_pred_grid[b],
                        "patch_pred_vis": patch_pred_vis[b],
                        "patch_pred_up": patch_pred_up_vis[b],
                        # Compatibility alias for unchanged FROC script.
                        "cam_vis_up": patch_pred_up_vis[b],
                        "layers_to_use": layers_to_use,
                        "source": "A5_patch_head_from_detached_raw_crossattention_batched",
                    }

                    if save_attn_debug:
                        label_payload.update(
                            {
                                "attn_raw": attn_grid[b],
                                "attn_up": attn_up_vis[b],
                            }
                        )

                    out_objs[b][label] = label_payload

            # Release graph references before next label/batch.
            del text_input, target_token_mask, attn_patch, attn_patch_detached, patch_pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save one .pt per image to keep compute_froc_from_gradcam_heatmaps.py unchanged.
        for image_id, out_obj in zip(batch_image_ids, out_objs):
            out_path = output_dir / f"{image_id}.pt"
            torch.save(out_obj, out_path)
            index_records.append(make_index_record(image_id, out_path, "saved", y_lookup))

        if batch_idx % 10 == 0:
            print(
                f"[Rank {rank}] processed_batches={batch_idx}/{len(loader)} "
                f"processed_images={min(batch_idx * int(batch_size), len(dataset))}/{len(dataset)}",
                flush=True,
            )

    # Write per-rank index first.
    rank_index_path = output_dir / f"crossattn_gradcam_index_rank{rank}.csv"
    pd.DataFrame(index_records).to_csv(rank_index_path, index=False)
    print(f"[Rank {rank}] Saved rank index: {rank_index_path}", flush=True)

    barrier_if_needed(world_size)

    # Merge rank indexes on rank 0 into the exact filename expected by your FROC script.
    if is_main_process(rank):
        all_idx = []
        for r in range(world_size):
            p = output_dir / f"crossattn_gradcam_index_rank{r}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing rank index: {p}")
            all_idx.append(pd.read_csv(p))

        index_df = pd.concat(all_idx, ignore_index=True)
        index_df = index_df.drop_duplicates(subset=["image_id"], keep="last")

        # Preserve original CSV image order where possible.
        order = {str(image_id): i for i, image_id in enumerate(image_ids)}
        index_df["__order__"] = index_df["image_id"].astype(str).map(order)
        index_df = index_df.sort_values("__order__").drop(columns=["__order__"])

        index_path = output_dir / "crossattn_gradcam_index.csv"
        index_df.to_csv(index_path, index=False)

        print(f"[Output] Saved merged index to: {index_path}", flush=True)
        print("[Done] Batched A5 patch-head heatmap extraction complete.", flush=True)

    barrier_if_needed(world_size)
    cleanup_distributed(world_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched/distributed extraction of A5 patch-head heatmaps for VinDr-CXR."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to A5 config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to A5 checkpoint .pth.")
    parser.add_argument("--labels_csv", type=str, required=True, help="VinDr image-level labels CSV.")
    parser.add_argument("--images_root", type=str, required=True, help="Directory containing VinDr PNGs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for .pt maps and index CSV.")
    parser.add_argument("--layers_to_use", type=str, default="8", help='Cross-attention layers, e.g. "8".')
    parser.add_argument("--only_labels", type=str, default="Cardiomegaly", help="Comma-separated labels.")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast during ALBEF forward. Optional; leave off if results differ.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layers_to_use = parse_layers(args.layers_to_use)
    only_labels = parse_labels(args.only_labels)

    extract_patch_head_maps_for_checkpoint_batched(
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
