"""
A5 pretraining script:
Identity-initialized patch prediction head trained with anatomy support loss
while detaching ALBEF cross-attention maps.

Key A5 behavior:
  - ALBEF losses train ALBEF normally.
  - Raw cross-attention is extracted exactly as in A3.
  - attn_patch is detached before entering patch_head.
  - support_outside_loss is applied to patch_pred, not attn_patch.
  - support-loss gradients update only patch_head parameters.
  - Checkpoints store ALBEF weights under checkpoint['model'] and patch head
    weights separately under checkpoint['patch_head'], keeping old zero-shot
    classification code compatible with checkpoint['model'].
"""

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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Subset

from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

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
# Support weights and dummy anatomy masks
# ============================================================

def build_support_weights_from_captions(text, config, device):
    """
    Builds per-sample weights for anatomy support loss.

    Disease-agnostic modes:
        none
        all_target_captions
        uncertainty_weighted
        positive_only

    Backward-compatible alias:
        all_cardiomegaly_captions -> all_target_captions
    """
    support_mode = config.get("support_mode", "all_target_captions")
    support_mode = str(support_mode).lower().strip()

    target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    if support_mode == "all_cardiomegaly_captions":
        support_mode = "all_target_captions"

    if support_mode != "none" and target_phrase == "":
        raise ValueError(
            "anatomy_target_phrase must be set when support_mode is not 'none'."
        )

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
                f"Unknown support_mode: {support_mode}. Expected one of: "
                "none, all_target_captions, uncertainty_weighted, positive_only, "
                "all_cardiomegaly_captions."
            )

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float, device=device)


def build_dummy_anatomy_prior_mask(target_phrase, batch_size, height, width, device):
    """
    Fixed anatomy prior proxies.

    Cardiomegaly:
        central cardiac silhouette proxy.
    Pleural Effusion:
        bilateral lower lung-base proxy.
    """
    target_phrase = str(target_phrase).lower().strip()

    prior_mask = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=device,
    )

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
            "Supported dummy masks: 'cardiomegaly', 'pleural effusion'."
        )

    return prior_mask


# ============================================================
# A5 helpers
# ============================================================

def get_model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def get_patch_head(model):
    raw_model = get_model_without_ddp(model)
    if not hasattr(raw_model, "patch_head"):
        raise AttributeError("A5 requires raw_model.patch_head, but it is missing.")
    return raw_model.patch_head


def albef_state_dict_without_patch_head(model_without_ddp):
    """
    Keep checkpoint['model'] loadable by old zero-shot/evaluation code that
    instantiates plain ALBEF. Patch head is saved separately.
    """
    return {
        k: v for k, v in model_without_ddp.state_dict().items()
        if not k.startswith("patch_head.")
    }


def module_grad_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.norm(2).item() ** 2)
    return float(total_sq ** 0.5)


def module_param_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        with torch.no_grad():
            total_sq += float(p.detach().float().norm(2).item() ** 2)
    return float(total_sq ** 0.5)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    return (values * w).sum() / w.sum().clamp_min(1.0)


def load_model_albef_state(model, state_dict, strict=False):
    """
    Load ALBEF weights into a model that may already contain patch_head.
    strict=False is intentional because old A0-A3 checkpoints do not have
    patch_head.* keys, and A5 checkpoint['model'] deliberately excludes them.
    """
    msg = model.load_state_dict(state_dict, strict=strict)
    if hasattr(msg, "missing_keys") or hasattr(msg, "unexpected_keys"):
        missing = list(getattr(msg, "missing_keys", []))
        unexpected = list(getattr(msg, "unexpected_keys", []))
        allowed_missing = [k for k in missing if k.startswith("patch_head.")]
        other_missing = [k for k in missing if not k.startswith("patch_head.")]
        if other_missing:
            print(f"[Checkpoint] Non-patch missing keys: {other_missing[:20]} ...")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys: {unexpected[:20]} ...")
        if allowed_missing:
            print(f"[Checkpoint] Missing patch_head keys are expected: {allowed_missing[:4]} ...")
    return msg


# ============================================================
# Training
# ============================================================

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args):
    model.train()
    raw_model = get_model_without_ddp(model)
    patch_head = get_patch_head(raw_model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # A5 anatomy-prior meters
    metric_logger.add_meter('loss_support', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_support_weighted', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_inside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_outside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('patch_pred_inside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('patch_pred_outside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('support_active', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('support_weight', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('patch_head_grad_norm', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = config.get("print_freq", 50)
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    lambda_support = float(config.get("lambda_support", 0.0))
    anatomy_layers = config.get("anatomy_layers", [8])
    anatomy_target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    target_token_ids = None
    if lambda_support > 0:
        target_token_ids = tokenizer(
            anatomy_target_phrase,
            add_special_tokens=False,
        ).input_ids

    if utils.is_main_process():
        print(f"[A5] lambda_support: {lambda_support}")
        print(f"[A5] patch_head: {patch_head}")
        print(f"[A5] patch_head_param_norm_init: {module_param_norm(patch_head):.6f}")
        if lambda_support > 0:
            print(f"[A5] target phrase: {anatomy_target_phrase}")
            print(f"[A5] target token ids: {target_token_ids}")
            print(f"[A5] anatomy layers: {anatomy_layers}")
            print(f"[A5] support mode: {config.get('support_mode')}")
        else:
            print("[A5] lambda_support=0, patch head will not receive support-loss updates.")

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad(set_to_none=True)

        image = image.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            padding='longest',
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        # ------------------------------------------------------------
        # Original ALBEF losses. These train ALBEF normally.
        # ------------------------------------------------------------
        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)

        # ------------------------------------------------------------
        # Default A5 values
        # ------------------------------------------------------------
        loss_support = torch.zeros((), device=image.device)
        loss_support_weighted = torch.zeros((), device=image.device)
        mean_attn_inside = torch.zeros((), device=image.device)
        mean_attn_outside = torch.zeros((), device=image.device)
        mean_patch_inside = torch.zeros((), device=image.device)
        mean_patch_outside = torch.zeros((), device=image.device)
        support_active = torch.zeros((image.shape[0],), dtype=torch.bool, device=image.device)
        mean_support_weight = torch.zeros((), device=image.device)

        attn_patch = None
        attn_patch_detached = None
        patch_pred = None
        prior_patch = None
        anatomy_prior_mask = None
        support_weights = torch.zeros((image.shape[0],), dtype=torch.float, device=image.device)

        # ------------------------------------------------------------
        # A5 anatomy-prior support branch
        # ------------------------------------------------------------
        if lambda_support > 0:
            target_token_mask = build_token_mask(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                target_token_ids=target_token_ids,
            )

            token_active = target_token_mask.any(dim=1)

            support_weights = build_support_weights_from_captions(
                text=text,
                config=config,
                device=image.device,
            )

            # Only allow support loss where the target token exists in tokenized text.
            support_weights = support_weights * token_active.float()
            mean_support_weight = support_weights.mean().detach()
            support_active = support_weights > 0

            if support_active.sum() > 0:
                # Raw ALBEF cross-attention from the forward pass above.
                attn_patch = extract_raw_crossattn_for_anatomy_loss(
                    model=raw_model,
                    text_token_mask=target_token_mask,
                    layers_to_use=anatomy_layers,
                    remove_image_cls=True,
                    normalize_patches=True,
                )

                # Critical A5 detach: support loss cannot backprop into ALBEF attention,
                # image encoder, text encoder, or global representation.
                attn_patch_detached = attn_patch.detach()

                # Only the FC patch head receives anatomy-support gradients.
                patch_pred = patch_head(attn_patch_detached)

                B, C, H, W = image.shape
                anatomy_prior_mask = build_dummy_anatomy_prior_mask(
                    target_phrase=anatomy_target_phrase,
                    batch_size=B,
                    height=H,
                    width=W,
                    device=image.device,
                )

                prior_patch = resize_prior_to_patch_mask(
                    prior_mask=anatomy_prior_mask,
                    num_patches=patch_pred.shape[-1],
                )

                # Same loss as A3, but applied to learned patch_pred.
                loss_support = support_outside_loss(
                    attn_patch=patch_pred,
                    prior_patch=prior_patch,
                    active_mask=support_weights,
                )
                loss_support_weighted = lambda_support * loss_support

                attn_inside = (attn_patch.detach() * prior_patch).sum(dim=-1)
                attn_outside = (attn_patch.detach() * (1.0 - prior_patch)).sum(dim=-1)
                patch_inside = (patch_pred.detach() * prior_patch).sum(dim=-1)
                patch_outside = (patch_pred.detach() * (1.0 - prior_patch)).sum(dim=-1)

                mean_attn_inside = weighted_mean(attn_inside, support_weights).detach()
                mean_attn_outside = weighted_mean(attn_outside, support_weights).detach()
                mean_patch_inside = weighted_mean(patch_inside, support_weights).detach()
                mean_patch_outside = weighted_mean(patch_outside, support_weights).detach()

        # ------------------------------------------------------------
        # Debug print before backward for first batch
        # ------------------------------------------------------------
        if i == 0 and utils.is_main_process():
            print("[DEBUG:A5] image:", image.shape)
            print("[DEBUG:A5] text_input.input_ids:", text_input.input_ids.shape)
            print("[DEBUG:A5] support_active:", support_active.shape)
            print("[DEBUG:A5] active samples:", int(support_active.sum().item()), "/", support_active.numel())
            print("[DEBUG:A5] loss_support_raw:", float(loss_support.detach().cpu()))
            print("[DEBUG:A5] loss_support_weighted:", float(loss_support_weighted.detach().cpu()))
            print("[DEBUG:A5] support_mode:", config.get("support_mode"))
            print("[DEBUG:A5] support_weights mean:", float(support_weights.mean().detach().cpu()))
            print("[DEBUG:A5] support_weights unique:", torch.unique(support_weights.detach().cpu()))
            print("[DEBUG:A5] first 10 raw texts:")
            for idx, t in enumerate(text[:10]):
                print(f"  {idx}: {t}")

            if attn_patch is not None:
                print("[DEBUG:A5] attn_patch:", attn_patch.shape)
                print("[DEBUG:A5] attn_patch.requires_grad before detach:", attn_patch.requires_grad)
            if attn_patch_detached is not None:
                print("[DEBUG:A5] attn_patch_detached.requires_grad:", attn_patch_detached.requires_grad)
            if patch_pred is not None:
                print("[DEBUG:A5] patch_pred:", patch_pred.shape)
                print("[DEBUG:A5] patch_pred.requires_grad:", patch_pred.requires_grad)
                print("[DEBUG:A5] patch_pred row-sum min/max:",
                      float(patch_pred.detach().sum(dim=-1).min().cpu()),
                      float(patch_pred.detach().sum(dim=-1).max().cpu()))
            if prior_patch is not None:
                print("[DEBUG:A5] prior_patch:", prior_patch.shape)
            if anatomy_prior_mask is not None:
                print("[DEBUG:A5] anatomy_prior_mask:", anatomy_prior_mask.shape)
            print("[DEBUG:A5] attn_inside/outside:",
                  float(mean_attn_inside.cpu()), float(mean_attn_outside.cpu()))
            print("[DEBUG:A5] patch_pred_inside/outside:",
                  float(mean_patch_inside.cpu()), float(mean_patch_outside.cpu()))

        # ------------------------------------------------------------
        # Final A5 loss
        # ------------------------------------------------------------
        loss = loss_mlm + loss_ita + loss_itm + loss_support_weighted

        loss.backward()

        patch_head_grad_norm = module_grad_norm(patch_head)

        if i == 0 and utils.is_main_process():
            print("[DEBUG:A5] patch_head_grad_norm_after_backward:", patch_head_grad_norm)
            for name, p in patch_head.named_parameters():
                gnorm = None if p.grad is None else float(p.grad.detach().float().norm(2).cpu())
                print(f"[DEBUG:A5] grad_norm patch_head.{name}: {gnorm}")

        optimizer.step()

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_support=loss_support.item())
        metric_logger.update(loss_support_weighted=loss_support_weighted.item())
        metric_logger.update(attn_inside=mean_attn_inside.item())
        metric_logger.update(attn_outside=mean_attn_outside.item())
        metric_logger.update(patch_pred_inside=mean_patch_inside.item())
        metric_logger.update(patch_pred_outside=mean_patch_outside.item())
        metric_logger.update(support_active=support_active.float().mean().item())
        metric_logger.update(support_weight=mean_support_weight.item())
        metric_logger.update(patch_head_grad_norm=patch_head_grad_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        if config.get("debug_max_batches", None) is not None:
            if i + 1 >= int(config["debug_max_batches"]):
                if utils.is_main_process():
                    print(f"[DEBUG:A5] Stopping after {i + 1} batches because debug_max_batches is set.")
                break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ============================================================
# Checkpoint loading / saving
# ============================================================

def auto_resume_from_output_dir(args, model, optimizer, lr_scheduler, config, device):
    """
    A5 compact output policy: prefer checkpoint_last.pth. Falls back to the
    latest checkpoint_XX.pth only for compatibility with older output dirs.
    """
    output_dir = Path(args.output_dir)
    last_ckpt = output_dir / "checkpoint_last.pth"

    if last_ckpt.exists():
        latest_ckpt = last_ckpt
    else:
        ckpts = sorted(output_dir.glob("checkpoint_*.pth"))
        if not ckpts:
            return 0, float("inf")
        latest_ckpt = ckpts[-1]

    print(f"[AutoResume:A5] Loading {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location='cpu')

    state_dict = checkpoint['model']
    load_model_albef_state(model, state_dict, strict=False)

    raw_model = get_model_without_ddp(model)
    if "patch_head" in checkpoint:
        raw_model.patch_head.load_state_dict(checkpoint["patch_head"], strict=True)
        print("[AutoResume:A5] Loaded patch_head state.")
    else:
        print("[AutoResume:A5] WARNING: checkpoint has no patch_head state; using identity init.")

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    best_loss = float(checkpoint.get('best_loss', float("inf")))
    print(f"[AutoResume:A5] Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss


def save_a5_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, config, epoch, best_loss, is_best):
    save_obj = {
        'model': albef_state_dict_without_patch_head(model_without_ddp),
        'patch_head': model_without_ddp.patch_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'config': config,
        'epoch': epoch,
        'best_loss': best_loss,
        'a5_note': 'checkpoint[model] excludes patch_head.*; checkpoint[patch_head] stores A5 head',
    }

    last_path = os.path.join(args.output_dir, 'checkpoint_last.pth')
    torch.save(save_obj, last_path)

    if is_best:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        torch.save(save_obj, best_path)


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
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]

    train_subset_size = config.get("train_subset_size", None)
    if train_subset_size is not None:
        train_subset_size = int(train_subset_size)
        full_dataset_size = len(datasets[0])
        if train_subset_size > full_dataset_size:
            raise ValueError(
                f"train_subset_size={train_subset_size} is larger than full dataset size={full_dataset_size}"
            )
        subset_seed = int(config.get("train_subset_seed", 42))
        rng = np.random.default_rng(subset_seed)
        subset_indices = rng.permutation(full_dataset_size)[:train_subset_size].tolist()
        datasets = [Subset(datasets[0], subset_indices)]
        if utils.is_main_process():
            print(
                f"[DATASET] Using subset: {train_subset_size}/{full_dataset_size} "
                f"samples with train_subset_seed={subset_seed}"
            )
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
        batch_size=[config['batch_size']],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Creating ALBEF model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    # Attach the A5 head before optimizer creation so create_optimizer includes it.
    patch_head = build_patch_head_from_config(config)
    model.add_module("patch_head", patch_head)
    print(f"[A5] Attached patch_head with num_patches={patch_head.num_patches}, "
          f"num_layers={patch_head.num_layers}, normalization={patch_head.normalization}")

    if bool(config.get("freeze_albef_for_a5", False)):
        for name, p in model.named_parameters():
            if not name.startswith("patch_head."):
                p.requires_grad = False
        print("[A5] freeze_albef_for_a5=True: only patch_head parameters are trainable.")
    else:
        print("[A5] freeze_albef_for_a5=False: ALBEF is trained by original ALBEF losses; support branch is detached.")

    model = model.to(device)

    if config.get("lambda_support", 0.0) > 0:
        enable_crossattn_attention_saving_for_anatomy(
            model,
            layers=config.get("anatomy_layers", [8]),
        )
    else:
        if utils.is_main_process():
            print("[A5] lambda_support=0, cross-attention saving disabled.")

    arg_opt = utils.AttrDict(config['optimizer'])
    arg_opt["lr"] = float(arg_opt["lr"])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche["lr"] = float(arg_sche["lr"])
    arg_sche["warmup_lr"] = float(arg_sche["warmup_lr"])
    arg_sche["min_lr"] = float(arg_sche["min_lr"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    best_loss = float("inf")

    if args.checkpoint:
        print(f"[Checkpoint:A5] Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        if args.resume:
            load_model_albef_state(model, state_dict, strict=False)
            if isinstance(checkpoint, dict) and "patch_head" in checkpoint:
                model.patch_head.load_state_dict(checkpoint["patch_head"], strict=True)
                print("[Checkpoint:A5] Loaded patch_head state.")
            else:
                print("[Checkpoint:A5] No patch_head state found; keeping identity-initialized head.")

            if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if isinstance(checkpoint, dict) and "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = int(checkpoint.get('epoch', -1)) + 1 if isinstance(checkpoint, dict) else 0
            best_loss = float(checkpoint.get('best_loss', best_loss)) if isinstance(checkpoint, dict) else best_loss
            print(f"[Checkpoint:A5] Resumed training from epoch {start_epoch}, best_loss={best_loss:.6f}")
        else:
            # Weight-only load from old A0/A3/A4 or A5 checkpoint.
            if 'visual_encoder.pos_embed' in state_dict:
                state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
                    state_dict['visual_encoder.pos_embed'], model.visual_encoder
                )
            if 'visual_encoder_m.pos_embed' in state_dict:
                state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
                    state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m
                )
            load_model_albef_state(model, state_dict, strict=False)
            if isinstance(checkpoint, dict) and "patch_head" in checkpoint:
                model.patch_head.load_state_dict(checkpoint["patch_head"], strict=True)
                print("[Checkpoint:A5] Loaded patch_head state from checkpoint.")
            else:
                print("[Checkpoint:A5] Loaded ALBEF weights only; patch_head remains identity-initialized.")
    else:
        if getattr(args, "auto_resume", False):
            start_epoch, best_loss = auto_resume_from_output_dir(
                args, model, optimizer, lr_scheduler, config, device
            )
        else:
            print("[Checkpoint:A5] No checkpoint provided and auto-resume disabled. Starting from scratch.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,  # patch_head is unused in batches with no active support samples
        )
        model_without_ddp = model.module

    print("Start A5 training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model,
            data_loader,
            optimizer,
            tokenizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            config,
            args,
        )

        if utils.is_main_process():
            loss_mlm = float(train_stats.get('loss_mlm', 0.0))
            loss_ita = float(train_stats.get('loss_ita', 0.0))
            loss_itm = float(train_stats.get('loss_itm', 0.0))
            loss_support = float(train_stats.get('loss_support', 0.0))
            loss_support_weighted = float(train_stats.get('loss_support_weighted', 0.0))

            # Keep old best-loss convention: best by ALBEF pretraining losses only.
            # checkpoint_last is also saved every epoch, so you can evaluate both.
            train_loss_total = loss_mlm + loss_ita + loss_itm
            train_loss_total_with_support = train_loss_total + loss_support_weighted

            is_best = train_loss_total < best_loss
            if is_best:
                best_loss = train_loss_total

            log_stats = {
                'epoch': epoch,
                'train_loss_mlm': loss_mlm,
                'train_loss_ita': loss_ita,
                'train_loss_itm': loss_itm,
                'train_loss_total': train_loss_total,
                'train_loss_total_with_support': train_loss_total_with_support,
                'train_loss_support': loss_support,
                'train_loss_support_weighted': loss_support_weighted,
                'train_attn_inside': float(train_stats.get('attn_inside', 0.0)),
                'train_attn_outside': float(train_stats.get('attn_outside', 0.0)),
                'train_patch_pred_inside': float(train_stats.get('patch_pred_inside', 0.0)),
                'train_patch_pred_outside': float(train_stats.get('patch_pred_outside', 0.0)),
                'train_support_active': float(train_stats.get('support_active', 0.0)),
                'train_support_weight': float(train_stats.get('support_weight', 0.0)),
                'train_patch_head_grad_norm': float(train_stats.get('patch_head_grad_norm', 0.0)),
                'patch_head_param_norm': module_param_norm(model_without_ddp.patch_head),
                'best_loss': best_loss,
                'lr': float(train_stats.get('lr', optimizer.param_groups[0]["lr"])),
            }

            # Compact output policy: only best + last + config.yaml + log.txt.
            save_a5_checkpoint(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
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
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain_A5_patch_head.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain_A5_patch_head/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--auto_resume', default=True, type=bool,
                        help='automatically resume from checkpoint_last.pth in output_dir if no explicit checkpoint is given')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
