'''
A6.1 training script:
Frozen A0 ALBEF backbone + train only an identity-initialized patch head.

Purpose:
  - Preserve A0 global classification by freezing all ALBEF parameters.
  - Train a lightweight patch prediction head from detached raw cross-attention.
  - Apply anatomy support loss to the patch-head map.
  - Add identity regularization so the patch head does not collapse to a generic
    cardiac prior and stays close to the original image-specific attention map.

Expected use:
  python -m torch.distributed.launch --nproc_per_node=2 --use_env \
      Pretrain_anatomy_prior_A6_1_frozen_A0_patch_head.py \
      --config configs/Pretrain_A6_1_frozen_A0_patch_head.yaml \
      --checkpoint /path/to/A0/checkpoint_best.pth \
      --output_dir output_A6_1_frozen_A0_patch_head

Checkpoint format:
  checkpoint['model']      = frozen A0 ALBEF state_dict
  checkpoint['patch_head'] = trained A6.1 patch head

This keeps existing A5/A6 patch-head heatmap extractors compatible.
'''

import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Subset

from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler

from anatomy_prior.attention_extract import (
    enable_crossattn_attention_saving_for_anatomy,
    extract_raw_crossattn_for_anatomy_loss,
)

from anatomy_prior.losses import (
    resize_prior_to_patch_mask,
    support_outside_loss,
)

from anatomy_prior.token_utils import build_token_mask
from anatomy_prior.patch_head import build_patch_head_from_config


# ============================================================
# Helpers
# ============================================================

def get_model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def build_support_weights_from_captions(text, config, device):
    """
    Builds per-sample weights for anatomy support loss.

    For A6.1 use:
      support_mode: positive_only
      anatomy_target_phrase: cardiomegaly
    """
    support_mode = str(config.get("support_mode", "positive_only")).lower().strip()
    target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    if support_mode == "all_cardiomegaly_captions":
        support_mode = "all_target_captions"

    if support_mode != "none" and target_phrase == "":
        raise ValueError("anatomy_target_phrase must be set when support_mode is not 'none'.")

    positive_weight = float(config.get("positive_caption_weight", 1.0))
    uncertain_weight = float(config.get("uncertain_caption_weight", 0.5))

    weights = []
    for t in text:
        t_lower = str(t).lower().strip()
        exact_positive = t_lower == target_phrase
        contains_target = target_phrase in t_lower
        uncertain_target = contains_target and ("uncertain" in t_lower)

        if support_mode == "none":
            weight = 0.0
        elif support_mode == "all_target_captions":
            weight = 1.0 if contains_target else 0.0
        elif support_mode == "uncertainty_weighted":
            if exact_positive:
                weight = positive_weight
            elif uncertain_target:
                weight = uncertain_weight
            else:
                weight = 0.0
        elif support_mode == "positive_only":
            weight = 1.0 if exact_positive else 0.0
        else:
            raise ValueError(
                f"Unknown support_mode={support_mode}. Expected: none, all_target_captions, "
                "uncertainty_weighted, positive_only."
            )
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_dummy_anatomy_prior_mask(target_phrase, batch_size, height, width, device):
    """
    Same fixed anatomy prior as previous experiments.
    Cardiomegaly: central cardiac silhouette proxy.
    """
    target_phrase = str(target_phrase).lower().strip()
    prior_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=device)

    if target_phrase == "cardiomegaly":
        h1, h2 = int(0.35 * height), int(0.75 * height)
        w1, w2 = int(0.25 * width), int(0.75 * width)
        prior_mask[:, :, h1:h2, w1:w2] = 1.0
    elif target_phrase == "pleural effusion":
        h1, h2 = int(0.55 * height), int(0.93 * height)
        w1_r, w2_r = int(0.07 * width), int(0.45 * width)
        w1_l, w2_l = int(0.55 * width), int(0.93 * width)
        prior_mask[:, :, h1:h2, w1_r:w2_r] = 1.0
        prior_mask[:, :, h1:h2, w1_l:w2_l] = 1.0
    else:
        raise ValueError(
            f"No dummy anatomy prior mask defined for anatomy_target_phrase='{target_phrase}'. "
            "Supported: cardiomegaly, pleural effusion."
        )

    return prior_mask


def weighted_mean(values, weights, eps=1e-8):
    weights = weights.float().clamp_min(0.0)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def module_grad_norm(module):
    total = 0.0
    found = False
    for p in module.parameters():
        if p.grad is not None:
            found = True
            total += float(p.grad.detach().float().norm(2).cpu()) ** 2
    return total ** 0.5 if found else 0.0


def module_param_norm(module):
    total = 0.0
    for p in module.parameters():
        total += float(p.detach().float().norm(2).cpu()) ** 2
    return total ** 0.5


def sync_module_gradients(module):
    """
    Average standalone patch_head gradients across ranks.

    Also handles ranks that had no active support samples by filling None grads
    with zeros before all_reduce. This keeps all ranks synchronized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def distributed_sum_scalar(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    return x.detach().clone()


def load_model_albef_state(model, state_dict, strict=False):
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[Checkpoint:A6.1] load_state_dict strict={strict}: {msg}")


def extract_model_state_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_a0_weights(model, ckpt_path):
    print(f"[Checkpoint:A6.1] Loading frozen A0 ALBEF from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_model_state_from_checkpoint(checkpoint)

    # Interpolate pos embeddings if needed, matching existing ALBEF scripts.
    if "visual_encoder.pos_embed" in state_dict:
        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
    if "visual_encoder_m.pos_embed" in state_dict:
        state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )

    load_model_albef_state(model, state_dict, strict=False)
    return checkpoint


# ============================================================
# Identity regularization
# ============================================================

def patch_identity_loss(patch_pred, attn_patch_detached, active_weights, mode="mse", eps=1e-8):
    """
    Keeps the learned patch-head output close to the input attention map.

    Why: support_outside_loss alone can be satisfied by producing a generic
    central cardiac prior. Identity regularization discourages collapse by
    preserving image-specific structure from the original attention map.

    Args:
      patch_pred: [B,N], normalized prediction from patch_head
      attn_patch_detached: [B,N], normalized detached raw attention
      active_weights: [B], usually support_weights
      mode: mse or kl_pred_to_attn
    """
    mode = str(mode).lower().strip()
    w = active_weights.float().clamp_min(0.0)

    if mode == "none":
        return torch.zeros((), dtype=patch_pred.dtype, device=patch_pred.device)

    if mode == "mse":
        per_sample = ((patch_pred - attn_patch_detached) ** 2).mean(dim=-1)
    elif mode in {"kl", "kl_pred_to_attn"}:
        p = patch_pred.clamp_min(eps)
        q = attn_patch_detached.clamp_min(eps)
        per_sample = (p * (p.log() - q.log())).sum(dim=-1)
    else:
        raise ValueError("identity_reg_mode must be one of: mse, kl_pred_to_attn, none")

    return weighted_mean(per_sample, w)


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, data_loader, optimizer, tokenizer, epoch, device, config, args, patch_head):
    # Frozen ALBEF: eval mode, no trainable params. Patch head trains.
    model.eval()
    patch_head.train()
    raw_model = get_model_without_ddp(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_total", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_support", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_support_weighted", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_identity", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_identity_weighted", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("attn_inside", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("attn_outside", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("patch_pred_inside", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("patch_pred_outside", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("support_active", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("support_weight", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("patch_head_grad_norm", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))

    header = f"A6.1 Train Epoch: [{epoch}]"
    print_freq = int(config.get("print_freq", 50))

    lambda_support = float(config.get("lambda_support", 0.01))
    beta_identity = float(config.get("beta_identity", config.get("lambda_identity", 0.1)))
    identity_reg_mode = str(config.get("identity_reg_mode", "mse"))
    anatomy_layers = config.get("anatomy_layers", [8])
    anatomy_target_phrase = str(config.get("anatomy_target_phrase", "cardiomegaly")).lower().strip()

    target_token_ids = tokenizer(anatomy_target_phrase, add_special_tokens=False).input_ids

    if utils.is_main_process():
        print(f"[A6.1] Frozen ALBEF, train patch_head only")
        print(f"[A6.1] lambda_support={lambda_support}")
        print(f"[A6.1] beta_identity={beta_identity}")
        print(f"[A6.1] identity_reg_mode={identity_reg_mode}")
        print(f"[A6.1] anatomy_target_phrase={anatomy_target_phrase}")
        print(f"[A6.1] target_token_ids={target_token_ids}")
        print(f"[A6.1] anatomy_layers={anatomy_layers}")
        print(f"[A6.1] patch_head_param_norm={module_param_norm(patch_head):.6f}")

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad(set_to_none=True)

        image = image.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=int(config.get("max_text_len", 25)),
            return_tensors="pt",
        ).to(device)

        # Default values
        loss_total = torch.zeros((), device=device, requires_grad=True)
        loss_support = torch.zeros((), device=device)
        loss_identity = torch.zeros((), device=device)
        loss_support_weighted = torch.zeros((), device=device)
        loss_identity_weighted = torch.zeros((), device=device)
        mean_attn_inside = torch.zeros((), device=device)
        mean_attn_outside = torch.zeros((), device=device)
        mean_patch_inside = torch.zeros((), device=device)
        mean_patch_outside = torch.zeros((), device=device)
        mean_support_weight = torch.zeros((), device=device)
        support_active = torch.zeros((image.shape[0],), dtype=torch.bool, device=device)
        support_weights = torch.zeros((image.shape[0],), dtype=torch.float32, device=device)

        attn_patch = None
        attn_patch_detached = None
        patch_pred = None
        prior_patch = None
        anatomy_prior_mask = None

        target_token_mask = build_token_mask(
            input_ids=text_input.input_ids,
            attention_mask=text_input.attention_mask,
            target_token_ids=target_token_ids,
        )
        token_active = target_token_mask.any(dim=1)

        support_weights = build_support_weights_from_captions(
            text=text,
            config=config,
            device=device,
        )
        support_weights = support_weights * token_active.float()
        support_active = support_weights > 0
        mean_support_weight = support_weights.mean().detach()

        # Let all ranks know whether any rank has active support samples.
        local_active_count = support_active.float().sum()
        global_active_count = distributed_sum_scalar(local_active_count)
        has_global_active = float(global_active_count.item()) > 0.0

        if support_active.sum() > 0:
            # Important: even though ALBEF parameters are frozen, set image.requires_grad_(True)
            # so existing XBERT attention hook code can register hooks on attention_probs.
            # We detach attn_patch before patch_head, so no ALBEF/image gradients are used.
            image_for_forward = image.detach().requires_grad_(True)

            with torch.enable_grad():
                _ = model(image_for_forward, text_input, alpha=0.0)

                attn_patch = extract_raw_crossattn_for_anatomy_loss(
                    model=raw_model,
                    text_token_mask=target_token_mask,
                    layers_to_use=anatomy_layers,
                    remove_image_cls=True,
                    normalize_patches=True,
                )

                attn_patch_detached = attn_patch.detach()

            # Patch head is the only trainable component.
            patch_pred = patch_head(attn_patch_detached)

            B, C, H, W = image.shape
            anatomy_prior_mask = build_dummy_anatomy_prior_mask(
                target_phrase=anatomy_target_phrase,
                batch_size=B,
                height=H,
                width=W,
                device=device,
            )
            prior_patch = resize_prior_to_patch_mask(
                prior_mask=anatomy_prior_mask,
                num_patches=patch_pred.shape[-1],
            )

            loss_support = support_outside_loss(
                attn_patch=patch_pred,
                prior_patch=prior_patch,
                active_mask=support_weights,
            )
            loss_identity = patch_identity_loss(
                patch_pred=patch_pred,
                attn_patch_detached=attn_patch_detached,
                active_weights=support_weights,
                mode=identity_reg_mode,
            )

            loss_support_weighted = lambda_support * loss_support
            loss_identity_weighted = beta_identity * loss_identity
            loss_total = loss_support_weighted + loss_identity_weighted

            attn_inside = (attn_patch_detached * prior_patch).sum(dim=-1)
            attn_outside = (attn_patch_detached * (1.0 - prior_patch)).sum(dim=-1)
            patch_inside = (patch_pred.detach() * prior_patch).sum(dim=-1)
            patch_outside = (patch_pred.detach() * (1.0 - prior_patch)).sum(dim=-1)

            mean_attn_inside = weighted_mean(attn_inside, support_weights).detach()
            mean_attn_outside = weighted_mean(attn_outside, support_weights).detach()
            mean_patch_inside = weighted_mean(patch_inside, support_weights).detach()
            mean_patch_outside = weighted_mean(patch_outside, support_weights).detach()

        if i == 0 and utils.is_main_process():
            print("[DEBUG:A6.1] image:", image.shape)
            print("[DEBUG:A6.1] text_input.input_ids:", text_input.input_ids.shape)
            print("[DEBUG:A6.1] support_active:", support_active.shape)
            print("[DEBUG:A6.1] active samples:", int(support_active.sum().item()), "/", support_active.numel())
            print("[DEBUG:A6.1] global_active_count:", float(global_active_count.cpu()))
            print("[DEBUG:A6.1] loss_support_raw:", float(loss_support.detach().cpu()))
            print("[DEBUG:A6.1] loss_support_weighted:", float(loss_support_weighted.detach().cpu()))
            print("[DEBUG:A6.1] loss_identity_raw:", float(loss_identity.detach().cpu()))
            print("[DEBUG:A6.1] loss_identity_weighted:", float(loss_identity_weighted.detach().cpu()))
            print("[DEBUG:A6.1] support_mode:", config.get("support_mode"))
            print("[DEBUG:A6.1] support_weights mean:", float(support_weights.mean().detach().cpu()))
            print("[DEBUG:A6.1] support_weights unique:", torch.unique(support_weights.detach().cpu()))
            print("[DEBUG:A6.1] first 10 raw texts:")
            for idx, t in enumerate(text[:10]):
                print(f"  {idx}: {t}")
            if attn_patch is not None:
                print("[DEBUG:A6.1] attn_patch:", attn_patch.shape)
                print("[DEBUG:A6.1] attn_patch.requires_grad before detach:", attn_patch.requires_grad)
            if attn_patch_detached is not None:
                print("[DEBUG:A6.1] attn_patch_detached.requires_grad:", attn_patch_detached.requires_grad)
            if patch_pred is not None:
                print("[DEBUG:A6.1] patch_pred:", patch_pred.shape)
                print("[DEBUG:A6.1] patch_pred.requires_grad:", patch_pred.requires_grad)
                print("[DEBUG:A6.1] patch_pred row-sum min/max:",
                      float(patch_pred.detach().sum(dim=-1).min().cpu()),
                      float(patch_pred.detach().sum(dim=-1).max().cpu()))
            if prior_patch is not None:
                print("[DEBUG:A6.1] prior_patch:", prior_patch.shape)
            print("[DEBUG:A6.1] attn_inside/outside:",
                  float(mean_attn_inside.cpu()), float(mean_attn_outside.cpu()))
            print("[DEBUG:A6.1] patch_pred_inside/outside:",
                  float(mean_patch_inside.cpu()), float(mean_patch_outside.cpu()))

        loss_total.backward()

        if args.distributed:
            sync_module_gradients(patch_head)

        patch_head_grad_norm = module_grad_norm(patch_head)

        if i == 0 and utils.is_main_process():
            print("[DEBUG:A6.1] patch_head_grad_norm_after_backward:", patch_head_grad_norm)
            for name, p in patch_head.named_parameters():
                gnorm = None if p.grad is None else float(p.grad.detach().float().norm(2).cpu())
                print(f"[DEBUG:A6.1] grad_norm patch_head.{name}: {gnorm}")

        # Avoid AdamW weight decay moving identity parameters on batches where no rank had active signal.
        if has_global_active:
            optimizer.step()

        metric_logger.update(loss_total=float(loss_total.detach().cpu()))
        metric_logger.update(loss_support=loss_support.item())
        metric_logger.update(loss_support_weighted=loss_support_weighted.item())
        metric_logger.update(loss_identity=loss_identity.item())
        metric_logger.update(loss_identity_weighted=loss_identity_weighted.item())
        metric_logger.update(attn_inside=mean_attn_inside.item())
        metric_logger.update(attn_outside=mean_attn_outside.item())
        metric_logger.update(patch_pred_inside=mean_patch_inside.item())
        metric_logger.update(patch_pred_outside=mean_patch_outside.item())
        metric_logger.update(support_active=support_active.float().mean().item())
        metric_logger.update(support_weight=mean_support_weight.item())
        metric_logger.update(patch_head_grad_norm=patch_head_grad_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if config.get("debug_max_batches", None) is not None:
            if i + 1 >= int(config["debug_max_batches"]):
                if utils.is_main_process():
                    print(f"[DEBUG:A6.1] Stopping after {i + 1} batches because debug_max_batches is set.")
                break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ============================================================
# Checkpointing
# ============================================================

def save_a6_checkpoint(args, model_without_ddp, patch_head, optimizer, scheduler, config, epoch, best_loss, is_best):
    save_obj = {
        "model": model_without_ddp.state_dict(),
        "patch_head": patch_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "a6_base_checkpoint": str(args.checkpoint),
        "experiment": "A6.1_frozen_A0_patch_head",
    }
    last_path = os.path.join(args.output_dir, "checkpoint_last.pth")
    torch.save(save_obj, last_path)
    if is_best:
        best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
        torch.save(save_obj, best_path)


def auto_resume_from_output_dir(args, model, patch_head, optimizer, scheduler):
    ckpt_path = Path(args.output_dir) / "checkpoint_last.pth"
    if not ckpt_path.exists():
        return 0, float("inf")

    print(f"[AutoResume:A6.1] Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    load_model_albef_state(model, checkpoint["model"], strict=False)
    patch_head.load_state_dict(checkpoint["patch_head"], strict=True)
    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("lr_scheduler", None) is not None:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    print(f"[AutoResume:A6.1] Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss


# ============================================================
# Main
# ============================================================

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = int(config.get("schedular", {}).get("epochs", config.get("epochs", 10)))

    print("Creating dataset")
    datasets = [create_dataset("pretrain", config)]

    train_subset_size = config.get("train_subset_size", None)
    if train_subset_size is not None:
        train_subset_size = int(train_subset_size)
        full_dataset_size = len(datasets[0])
        if train_subset_size > full_dataset_size:
            raise ValueError(f"train_subset_size={train_subset_size} > full dataset size={full_dataset_size}")
        subset_seed = int(config.get("train_subset_seed", 42))
        rng = np.random.default_rng(subset_seed)
        subset_indices = rng.permutation(full_dataset_size)[:train_subset_size].tolist()
        datasets = [Subset(datasets[0], subset_indices)]
        if utils.is_main_process():
            print(f"[DATASET] Using subset: {train_subset_size}/{full_dataset_size} with seed={subset_seed}")
    else:
        if utils.is_main_process():
            print(f"[DATASET] Using full dataset: {len(datasets[0])} samples")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[int(config["batch_size"])],
        num_workers=[int(config.get("num_workers", 4))],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Creating frozen A0 ALBEF model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    if args.checkpoint == "":
        raise ValueError("A6.1 requires --checkpoint pointing to A0-best ALBEF checkpoint.")

    load_a0_weights(model, args.checkpoint)
    model = model.to(device)

    # Freeze every ALBEF parameter. Only patch_head is trainable.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    enable_crossattn_attention_saving_for_anatomy(
        model,
        layers=config.get("anatomy_layers", [8]),
    )

    patch_head = build_patch_head_from_config(config).to(device)
    patch_head.train()
    for p in patch_head.parameters():
        p.requires_grad = True

    if utils.is_main_process():
        n_trainable_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_patch = sum(p.numel() for p in patch_head.parameters() if p.requires_grad)
        print(f"[A6.1] Trainable ALBEF params: {n_trainable_model}")
        print(f"[A6.1] Trainable patch_head params: {n_patch}")
        print(f"[A6.1] patch_head: {patch_head}")

    lr = float(config.get("patch_head_lr", config.get("optimizer", {}).get("lr", 1e-4)))
    weight_decay = float(config.get("patch_head_weight_decay", config.get("optimizer", {}).get("weight_decay", 0.02)))
    optimizer = torch.optim.AdamW(patch_head.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    use_scheduler = bool(config.get("use_patch_head_scheduler", False))
    if use_scheduler:
        # Reuse existing scheduler interface if requested.
        arg_sche = utils.AttrDict(config["schedular"])
        arg_sche["lr"] = float(arg_sche["lr"])
        arg_sche["warmup_lr"] = float(arg_sche["warmup_lr"])
        arg_sche["min_lr"] = float(arg_sche["min_lr"])
        scheduler, _ = create_scheduler(arg_sche, optimizer)

    best_loss = float("inf")
    if getattr(args, "auto_resume", False):
        start_epoch, best_loss = auto_resume_from_output_dir(args, model, patch_head, optimizer, scheduler)

    model_without_ddp = model
    # Do not wrap frozen ALBEF in DDP. We only use distributed data splitting and
    # manually average standalone patch_head gradients.

    print("Start A6.1 frozen-backbone patch-head training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if scheduler is not None and epoch > 0:
            warmup_steps = int(config.get("schedular", {}).get("warmup_epochs", 0))
            scheduler.step(epoch + warmup_steps)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            tokenizer=tokenizer,
            epoch=epoch,
            device=device,
            config=config,
            args=args,
            patch_head=patch_head,
        )

        if utils.is_main_process():
            train_loss_total = float(train_stats.get("loss_total", 0.0))
            is_best = train_loss_total < best_loss
            if is_best:
                best_loss = train_loss_total

            log_stats = {
                "epoch": epoch,
                "train_loss_total": train_loss_total,
                "train_loss_support": float(train_stats.get("loss_support", 0.0)),
                "train_loss_support_weighted": float(train_stats.get("loss_support_weighted", 0.0)),
                "train_loss_identity": float(train_stats.get("loss_identity", 0.0)),
                "train_loss_identity_weighted": float(train_stats.get("loss_identity_weighted", 0.0)),
                "train_attn_inside": float(train_stats.get("attn_inside", 0.0)),
                "train_attn_outside": float(train_stats.get("attn_outside", 0.0)),
                "train_patch_pred_inside": float(train_stats.get("patch_pred_inside", 0.0)),
                "train_patch_pred_outside": float(train_stats.get("patch_pred_outside", 0.0)),
                "train_support_active": float(train_stats.get("support_active", 0.0)),
                "train_support_weight": float(train_stats.get("support_weight", 0.0)),
                "train_patch_head_grad_norm": float(train_stats.get("patch_head_grad_norm", 0.0)),
                "patch_head_param_norm": module_param_norm(patch_head),
                "best_loss": best_loss,
                "lr": float(train_stats.get("lr", optimizer.param_groups[0]["lr"])),
                "a6_base_checkpoint": str(args.checkpoint),
            }

            save_a6_checkpoint(
                args=args,
                model_without_ddp=model_without_ddp,
                patch_head=patch_head,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_loss=best_loss,
                is_best=is_best,
            )

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Pretrain_A6_1_frozen_A0_patch_head.yaml")
    parser.add_argument("--checkpoint", default="", help="Required: A0-best checkpoint path")
    parser.add_argument("--output_dir", default="output_A6_1_frozen_A0_patch_head/")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--auto_resume", default=True, type=bool)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
