"""
A6.0 sanity-control extractor: frozen-A0 identity patch-head heatmaps.

Purpose
-------
A6.0 is NOT a training script. It loads an A0 ALBEF checkpoint, builds a fresh
identity-initialized patch head, and extracts:

    attn_patch = raw ALBEF cross-attention over image patches
    patch_pred = identity_patch_head(attn_patch.detach())

At initialization, patch_pred should be approximately equal to attn_patch. This
lets you evaluate the post-hoc patch-head extraction pipeline while preserving
A0 global classification behavior, because no ALBEF weights are trained or saved.

Compatibility
-------------
The output is compatible with compute_froc_from_gradcam_heatmaps.py:
  - writes crossattn_gradcam_index.csv
  - saves per-image .pt files as out_obj[label][cam_key]
  - saves the identity patch-head map under both:
        patch_pred_up
        cam_vis_up        # alias so existing FROC command can use --cam_key cam_vis_up

Recommended launch on 2 GPUs:

python -u -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  --use_env \
  scripts/extract_a6_0_identity_patch_head_heatmaps.py \
  --config ./configs/Pretrain_A5_patch_head.yaml \
  --checkpoint ./output_A0/checkpoint_best.pth \
  --labels_csv /path/to/vindr/image_labels_test.csv \
  --images_root /path/to/vindr/test_pngs \
  --output_dir ./heatmaps_A6_0_A0_identity_patch_head \
  --only_labels Cardiomegaly \
  --layers_to_use 8 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --no_save_attn_debug

Important
---------
This script runs ALBEF forward under torch.enable_grad() because the ALBEF/XBERT
attention-saving code may register hooks on attention tensors. No backward() and
no optimizer step are performed, so checkpoint weights are not changed.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src import build_model_and_tokenizer, get_image_transform

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

def init_distributed_from_env(device_str: str = "cuda") -> Tuple[int, int, int, torch.device]:
    """
    Supports both single-process and torch.distributed.launch --use_env.
    Does NOT wrap the model in DDP. Each rank independently processes a shard.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")
        num_visible = torch.cuda.device_count()
        if local_rank >= num_visible:
            raise RuntimeError(
                "Invalid CUDA device ordinal. "
                f"LOCAL_RANK={local_rank}, torch.cuda.device_count()={num_visible}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}. "
                "If you launch with --nproc_per_node=2, request two GPUs in Slurm, e.g. "
                "#SBATCH --gres=gpu:a40:2."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")

    return rank, world_size, local_rank, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier_if_needed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


# ============================================================
# Parsing and small utilities
# ============================================================

def parse_layers(s: str) -> List[int]:
    s = str(s).strip()
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def parse_labels(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


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


def load_filtered_df(
    labels_csv: Path,
    images_root: Path,
    only_labels: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    positive_only_label: Optional[str] = None,
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
            raise ValueError(f"--positive_only_label={positive_only_label} not found in CSV labels")
        before = len(df)
        df = df[df[positive_only_label] == 1].reset_index(drop=True)
        print(f"[Data] positive_only_label={positive_only_label}: {before} -> {len(df)} rows", flush=True)

    df["__has_png__"] = df[id_col].apply(lambda x: (images_root / f"{str(x)}.png").exists())
    before_png = len(df)
    df = df[df["__has_png__"]].reset_index(drop=True)
    print(f"[Data] PNG filter: {before_png} -> {len(df)} images", flush=True)

    if len(df) == 0:
        raise RuntimeError(f"No valid PNG images found under: {images_root}")

    if max_images is not None:
        df = df.iloc[: int(max_images)].reset_index(drop=True)
        print(f"[Data] max_images={max_images}, using {len(df)} images", flush=True)

    df[id_col] = df[id_col].astype(str)
    print(f"[Data] Final number of images: {len(df)}", flush=True)
    print(f"[Data] Labels to extract: {label_cols}", flush=True)
    return df, id_col, label_cols


class VinDrImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_col: str, images_root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row[self.id_col])
        img_path = infer_png_path(self.images_root, image_id)
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, image_id


def collate_images(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    image_ids = [b[1] for b in batch]
    return images, image_ids


# ============================================================
# A6.0 extractor
# ============================================================

def build_identity_patch_head(config: dict, device: torch.device):
    """
    Fresh identity-initialized patch head. Does NOT load checkpoint['patch_head'].
    This is intentional for A6.0.
    """
    patch_head = build_patch_head_from_config(config)
    patch_head = patch_head.to(device)
    patch_head.eval()

    for p in patch_head.parameters():
        p.requires_grad = False

    print(
        f"[A6.0] Built fresh identity patch head: "
        f"num_patches={patch_head.num_patches}, "
        f"num_layers={patch_head.num_layers}, "
        f"normalization={patch_head.normalization}",
        flush=True,
    )
    return patch_head


def write_merged_index(output_dir: Path, world_size: int):
    parts = []
    for r in range(world_size):
        part_path = output_dir / f"crossattn_gradcam_index_rank{r}.csv"
        if part_path.exists():
            parts.append(pd.read_csv(part_path))
        else:
            print(f"[WARN] Missing rank index: {part_path}", flush=True)

    if not parts:
        raise RuntimeError("No rank index CSVs found; cannot build crossattn_gradcam_index.csv")

    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id", "heatmap_path"], keep="last")
    merged = merged.sort_values("image_id").reset_index(drop=True)
    index_path = output_dir / "crossattn_gradcam_index.csv"
    merged.to_csv(index_path, index=False)
    print(f"[Output] Saved merged index to: {index_path} ({len(merged)} rows)", flush=True)


def extract_a6_0_identity_patch_maps(
    config_path: Path,
    ckpt_path: Path,
    labels_csv: Path,
    images_root: Path,
    output_dir: Path,
    layers_to_use: List[int],
    only_labels: Optional[List[str]],
    max_images: Optional[int],
    device_str: str,
    max_length: int,
    positive_only_label: Optional[str],
    overwrite: bool,
    save_attn_debug: bool,
    batch_size: int,
    num_workers: int,
):
    rank, world_size, local_rank, device = init_distributed_from_env(device_str=device_str)

    if is_main_process(rank):
        print("=" * 80, flush=True)
        print("[Config] A6.0 identity patch-head heatmap extraction", flush=True)
        print(f"[Config] config_path       = {config_path}", flush=True)
        print(f"[Config] ckpt_path         = {ckpt_path}", flush=True)
        print(f"[Config] labels_csv        = {labels_csv}", flush=True)
        print(f"[Config] images_root       = {images_root}", flush=True)
        print(f"[Config] output_dir        = {output_dir}", flush=True)
        print(f"[Config] layers_to_use     = {layers_to_use}", flush=True)
        print(f"[Config] only_labels       = {only_labels}", flush=True)
        print(f"[Config] max_images        = {max_images}", flush=True)
        print(f"[Config] batch_size        = {batch_size}", flush=True)
        print(f"[Config] num_workers       = {num_workers}", flush=True)
        print(f"[Dist] rank/world/local    = {rank}/{world_size}/{local_rank}", flush=True)
        print(f"[Dist] CUDA_VISIBLE_DEVICES= {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print("=" * 80, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    # Load ALBEF A0 checkpoint. This function loads only checkpoint['model'] into ALBEF.
    model, tokenizer, config, _ = build_model_and_tokenizer(
        config_path=str(config_path),
        ckpt_path=str(ckpt_path),
        device=str(device),
    )
    model.eval()

    # Do not optimize anything. We intentionally do not save the model after extraction.
    # Keeping requires_grad=True avoids XBERT hook-registration crashes in unpatched repos.
    for p in model.parameters():
        p.requires_grad = True

    patch_head = build_identity_patch_head(config=config, device=device)

    image_res = int(config["image_res"])
    transform = get_image_transform(image_res)

    queue_size = int(config.get("queue_size", 65536))
    if queue_size % int(batch_size) != 0:
        raise ValueError(
            f"batch_size={batch_size} must divide queue_size={queue_size}, because "
            "ALBEF model_pretrain._dequeue_and_enqueue asserts queue_size % batch_size == 0. "
            "Use 8, 16, 32, or 64 for the usual queue_size=65536."
        )

    if is_main_process(rank):
        print(f"[Model] image_res={image_res}", flush=True)
        print(f"[Model] device={device}", flush=True)
        print("[A6.0] No training, no optimizer, no backward. A0 checkpoint weights are not modified.", flush=True)

    df, id_col, label_cols = load_filtered_df(
        labels_csv=labels_csv,
        images_root=images_root,
        only_labels=only_labels,
        max_images=max_images,
        positive_only_label=positive_only_label,
    )

    # Shard by rank. No DDP needed.
    df_rank = df.iloc[rank::world_size].reset_index(drop=True)
    if is_main_process(rank):
        print(f"[Data] world_size={world_size}; rank 0 images={len(df_rank)}", flush=True)
    else:
        print(f"[Data] rank {rank} images={len(df_rank)}", flush=True)

    dataset = VinDrImageDataset(df_rank, id_col=id_col, images_root=images_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_images,
    )

    # Enable saved raw attention tensors used by extract_raw_crossattn_for_anatomy_loss.
    enable_crossattn_attention_saving_for_anatomy(model, layers=layers_to_use)
    if is_main_process(rank):
        print("[A6.0] Enabled raw cross-attention saving.", flush=True)

    # Precompute target token ids once per label.
    label_token_ids = {}
    for label in label_cols:
        label_text = str(label).lower().strip()
        label_token_ids[label] = tokenizer(label_text, add_special_tokens=False).input_ids

    y_lookup = {}
    for _, row in df.iterrows():
        image_id = str(row[id_col])
        y_lookup[image_id] = {lb: float(row[lb]) for lb in label_cols if lb in df.columns}

    index_records = []
    pbar = tqdm(loader, desc=f"Rank {rank} extracting A6.0 maps", disable=not is_main_process(rank))

    for images_cpu, image_ids in pbar:
        real_B = images_cpu.shape[0]
        model_B = int(batch_size)

        images = images_cpu.to(device, non_blocking=True)

        # Pad final partial batch to avoid ALBEF queue assertion.
        if real_B < model_B:
            pad_n = model_B - real_B
            pad_images = images[-1:].expand(pad_n, -1, -1, -1).contiguous()
            images_for_model = torch.cat([images, pad_images], dim=0)
        else:
            images_for_model = images

        B = images_for_model.shape[0]

        # Skip whole batch if all outputs already exist and not overwriting.
        out_paths = [output_dir / f"{image_id}.pt" for image_id in image_ids]
        if (not overwrite) and all(p.exists() for p in out_paths):
            for image_id, out_path in zip(image_ids, out_paths):
                row_record = {"image_id": image_id, "heatmap_path": str(out_path), "status": "exists_skipped"}
                for lb in label_cols:
                    if image_id in y_lookup and lb in y_lookup[image_id]:
                        row_record[f"y::{lb}"] = y_lookup[image_id][lb]
                index_records.append(row_record)
            continue

        batch_out_objs = [{} for _ in range(real_B)]

        for label in label_cols:
            label_text = str(label).lower().strip()
            texts = [label_text] * B

            text_input = tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            target_token_mask = build_token_mask(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                target_token_ids=label_token_ids[label],
            )

            if not target_token_mask.any():
                raise RuntimeError(f"Target token mask is empty for label='{label}'")

            model.zero_grad(set_to_none=True)

            # Needed in unpatched ALBEF/XBERT because attention tensors may register hooks.
            # No backward() is called.
            with torch.enable_grad():
                _ = model(images_for_model, text_input, alpha=0.0)

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

                patch_pred_grid = patch_vector_to_grid(patch_pred)  # [B,S,S]
                patch_pred_up = upsample_patch_vector(patch_pred, target_size=image_res)  # [B,H,W]

                patch_pred_vis = minmax_norm_torch(patch_pred_grid)
                patch_pred_up_vis = minmax_norm_torch(patch_pred_up)

                if save_attn_debug:
                    attn_grid = patch_vector_to_grid(attn_patch_detached)
                    attn_up = upsample_patch_vector(attn_patch_detached, target_size=image_res)
                    attn_up_vis = minmax_norm_torch(attn_up)

                # Save only real images, not padded duplicates.
                for b in range(real_B):
                    label_payload = {
                        "patch_pred_raw": patch_pred_grid[b].detach().float().cpu(),
                        "patch_pred_vis": patch_pred_vis[b].detach().float().cpu(),
                        "patch_pred_up": patch_pred_up_vis[b].detach().float().cpu(),
                        "cam_vis_up": patch_pred_up_vis[b].detach().float().cpu(),
                        "layers_to_use": layers_to_use,
                        "source": "A6_0_A0_checkpoint_identity_patch_head_from_detached_raw_crossattention",
                    }

                    if save_attn_debug:
                        label_payload.update(
                            {
                                "attn_raw": attn_grid[b].detach().float().cpu(),
                                "attn_up": attn_up_vis[b].detach().float().cpu(),
                            }
                        )

                    batch_out_objs[b][label] = label_payload

        # Write one .pt per image.
        for image_id, out_path, out_obj in zip(image_ids, out_paths, batch_out_objs):
            if out_path.exists() and not overwrite:
                status = "exists_skipped"
            else:
                torch.save(out_obj, out_path)
                status = "saved"

            row_record = {"image_id": image_id, "heatmap_path": str(out_path), "status": status}
            for lb in label_cols:
                if image_id in y_lookup and lb in y_lookup[image_id]:
                    row_record[f"y::{lb}"] = y_lookup[image_id][lb]
            index_records.append(row_record)

    rank_index_path = output_dir / f"crossattn_gradcam_index_rank{rank}.csv"
    pd.DataFrame(index_records).to_csv(rank_index_path, index=False)
    print(f"[Rank {rank}] Saved rank index: {rank_index_path} ({len(index_records)} rows)", flush=True)

    barrier_if_needed(world_size)

    if is_main_process(rank):
        write_merged_index(output_dir=output_dir, world_size=world_size)
        print("[Done] A6.0 identity patch-head heatmap extraction complete.", flush=True)

    barrier_if_needed(world_size)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="A6.0: extract identity patch-head heatmaps from frozen/unmodified A0 ALBEF checkpoint."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML. A5 config is OK for patch_head defaults.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to A0 checkpoint_best.pth or other ALBEF checkpoint.")
    parser.add_argument("--labels_csv", type=str, required=True, help="VinDr image-level labels CSV.")
    parser.add_argument("--images_root", type=str, required=True, help="Directory containing VinDr PNGs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for .pt maps and index CSV.")
    parser.add_argument("--layers_to_use", type=str, default="8", help='Cross-attention layers, e.g. "8" or "8,9,10,11".')
    parser.add_argument("--only_labels", type=str, default="Cardiomegaly", help="Comma-separated labels.")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--positive_only_label", type=str, default=None, help="Debug only. Do NOT use for FROC.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_save_attn_debug", action="store_true", help="Do not save original attention debug maps.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    layers_to_use = parse_layers(args.layers_to_use)
    only_labels = parse_labels(args.only_labels)

    extract_a6_0_identity_patch_maps(
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
    )


if __name__ == "__main__":
    main()
