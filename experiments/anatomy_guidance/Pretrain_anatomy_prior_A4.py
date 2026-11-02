'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Subset
)
import torch.backends.cudnn as cudnn
import torch.distributed as dist

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


def build_support_weights_from_captions(text, config, device):
    """
    Builds per-sample weights for anatomy support loss.

    Modified it for compatibility with both Cardiomegaly and PE

    Disease-agnostic modes:
        "none"
        "all_target_captions"
        "uncertainty_weighted"
        "positive_only"

    For current experiments:
        anatomy_target_phrase: "cardiomegaly"
        anatomy_target_phrase: "pleural effusion"
    """

    support_mode = config.get("support_mode", "all_target_captions")
    support_mode = str(support_mode).lower().strip()

    target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    # ------------------------------------------------------------------
    # Backward compatibility with Cardiomegaly YAMLs
    # ------------------------------------------------------------------
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
                f"Unknown support_mode: {support_mode}. "
                "Expected one of: none, all_target_captions, "
                "uncertainty_weighted, positive_only, "
                "all_cardiomegaly_captions."
            )

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float, device=device)




def unwrap_model(model):
    """Return the underlying module if model is wrapped in DDP/DataParallel."""
    return model.module if hasattr(model, "module") else model


def strip_module_prefix(state_dict):
    """Make checkpoints saved with DistributedDataParallel compatible with normal models."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_state_dict_flexible(model, checkpoint_path, device="cpu", interpolate_pos=True):
    """
    Load a checkpoint into an ALBEF model. Supports checkpoints saved as:
        {'model': state_dict, ...}, {'state_dict': state_dict}, or raw state_dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)

    if interpolate_pos:
        if "visual_encoder.pos_embed" in state_dict:
            state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )
        if "visual_encoder_m.pos_embed" in state_dict and hasattr(model, "visual_encoder_m"):
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
            )

    msg = model.load_state_dict(state_dict, strict=False)
    return msg


def get_global_image_text_features(model, image, text_input):
    """
    Extract normalized global image and text features from an ALBEF-style model.

    These are the global image/text representations used to build a batch-wise
    image-text similarity matrix for classification-preservation distillation.
    """
    model = unwrap_model(model)

    image_embeds = model.visual_encoder(image)
    if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]

    image_feat = F.normalize(
        model.vision_proj(image_embeds[:, 0, :]),
        dim=-1,
    )

    if hasattr(model.text_encoder, "bert"):
        text_output = model.text_encoder.bert(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True,
            mode="text",
        )
    else:
        try:
            text_output = model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
                mode="text",
            )
        except TypeError:
            text_output = model.text_encoder(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
            )

    text_feat = F.normalize(
        model.text_proj(text_output.last_hidden_state[:, 0, :]),
        dim=-1,
    )

    return image_feat, text_feat


def classification_preservation_loss(
    student_model,
    teacher_model,
    image,
    text_input,
    temperature=0.07,
):
    """
    Preserve the A0 teacher's global image-text matching behaviour.

    For each batch, the frozen A0 teacher produces image-to-text and text-to-image
    similarity distributions. The A4 student is penalized if its similarity
    distributions drift away from the teacher.
    """
    student_image_feat, student_text_feat = get_global_image_text_features(
        student_model,
        image,
        text_input,
    )

    with torch.no_grad():
        teacher_image_feat, teacher_text_feat = get_global_image_text_features(
            teacher_model,
            image,
            text_input,
        )

    student_sim_i2t = student_image_feat @ student_text_feat.t()
    student_sim_t2i = student_text_feat @ student_image_feat.t()

    teacher_sim_i2t = teacher_image_feat @ teacher_text_feat.t()
    teacher_sim_t2i = teacher_text_feat @ teacher_image_feat.t()

    temperature = float(temperature)

    teacher_prob_i2t = F.softmax(teacher_sim_i2t / temperature, dim=-1)
    teacher_prob_t2i = F.softmax(teacher_sim_t2i / temperature, dim=-1)

    student_log_prob_i2t = F.log_softmax(student_sim_i2t / temperature, dim=-1)
    student_log_prob_t2i = F.log_softmax(student_sim_t2i / temperature, dim=-1)

    loss_i2t = F.kl_div(
        student_log_prob_i2t,
        teacher_prob_i2t,
        reduction="batchmean",
    )
    loss_t2i = F.kl_div(
        student_log_prob_t2i,
        teacher_prob_t2i,
        reduction="batchmean",
    )

    return 0.5 * (loss_i2t + loss_t2i)

def build_dummy_anatomy_prior_mask(target_phrase, batch_size, height, width, device):
    """
    Builds fixed dummy anatomy masks for anatomy-aware support regularization.

    Cardiomegaly:
        central lower chest / heart region.

    Pleural Effusion:
        bilateral lower lung-base regions near the costophrenic angles.
    """

    target_phrase = str(target_phrase).lower().strip()

    prior_mask = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=device,
    )

    if target_phrase == "cardiomegaly":
        # Central cardiac silhouette proxy
        h1, h2 = int(0.35 * height), int(0.75 * height)
        w1, w2 = int(0.25 * width), int(0.75 * width)

        prior_mask[:, :, h1:h2, w1:w2] = 1.0

    elif target_phrase == "pleural effusion":
        # Bilateral lower pleural/lung-base proxy
        h1, h2 = int(0.55 * height), int(0.93 * height)

        # Right lower lung in image coordinates
        w1_r, w2_r = int(0.07 * width), int(0.45 * width)

        # Left lower lung in image coordinates
        w1_l, w2_l = int(0.55 * width), int(0.93 * width)

        prior_mask[:, :, h1:h2, w1_r:w2_r] = 1.0
        prior_mask[:, :, h1:h2, w1_l:w2_l] = 1.0

    else:
        raise ValueError(
            f"No dummy anatomy prior mask defined for anatomy_target_phrase='{target_phrase}'. "
            "Supported dummy masks: 'cardiomegaly', 'pleural effusion'."
        )

    return prior_mask


def train(model, teacher_model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    raw_model = unwrap_model(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # Anatomy-prior meters
    metric_logger.add_meter('loss_support', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_inside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_outside', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('support_active', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # A2
    metric_logger.add_meter('support_weight', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    # A4 classification-preservation meters
    metric_logger.add_meter('loss_cls_preserve', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cls_preserve_w', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_support_w', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = config.get("print_freq", 50)
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    lambda_support = float(config.get("lambda_support", 0.0))
    use_cls_preserve = bool(config.get("use_classification_preservation", False))
    lambda_cls_preserve = float(config.get("lambda_cls_preserve", 0.0))
    cls_preserve_temperature = float(config.get("cls_preserve_temperature", 0.07))
    anatomy_layers = config.get("anatomy_layers", [8])

    anatomy_target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()
    # Build once, not every batch
    target_token_ids = None

    if lambda_support > 0:
        target_token_ids = tokenizer(
            anatomy_target_phrase,
            add_special_tokens=False,
        ).input_ids

    if utils.is_main_process():
        print(f"[AnatomyPrior] lambda_support: {lambda_support}")
        if lambda_support > 0:
            print(f"[AnatomyPrior] target phrase: {anatomy_target_phrase}")
            print(f"[AnatomyPrior] target token ids: {target_token_ids}")
            print(f"[AnatomyPrior] layers: {anatomy_layers}")
        else:
            print("[AnatomyPrior] disabled for this run.")

        if use_cls_preserve:
            print(f"[A4] classification preservation enabled: lambda_cls_preserve={lambda_cls_preserve}, temperature={cls_preserve_temperature}")
        else:
            print("[A4] classification preservation disabled.")

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)

        text_input = tokenizer(
            text,
            padding='longest',
            truncation=True,
            max_length=25,
            return_tensors="pt"
        ).to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        # ------------------------------------------------------------
        # Original ALBEF losses
        # ------------------------------------------------------------
        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)

        # ------------------------------------------------------------
        # Default anatomy-prior values
        # ------------------------------------------------------------
        loss_support = torch.zeros((), device=image.device)
        mean_inside_mass = torch.zeros((), device=image.device)
        mean_outside_mass = torch.zeros((), device=image.device)
        support_active = torch.zeros((image.shape[0],), dtype=torch.bool, device=image.device)
        mean_support_weight = torch.zeros((), device=image.device)
        loss_cls_preserve = torch.zeros((), device=image.device)

        attn_patch = None
        prior_patch = None
        anatomy_prior_mask = None

        # ------------------------------------------------------------
        # Anatomy-prior support regularization
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

            # Only allow support loss where Cardiomegaly token was found
            support_weights = support_weights * token_active.float()

            mean_support_weight = support_weights.mean().detach()

            support_active = support_weights > 0

            if support_active.sum() > 0:
                attn_patch = extract_raw_crossattn_for_anatomy_loss(
                    model=raw_model,
                    text_token_mask=target_token_mask,
                    layers_to_use=anatomy_layers,
                    remove_image_cls=True,
                    normalize_patches=True,
                )

                # ------------------------------------------------------------
                # Temporary disease-specific dummy anatomy support proxy.
                # Cardiomegaly: central heart region.
                # Pleural Effusion: bilateral lower lung-base regions.
                # ------------------------------------------------------------
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
                    num_patches=attn_patch.shape[-1],
                )

                loss_support = support_outside_loss(
                    attn_patch=attn_patch,
                    prior_patch=prior_patch,
                    active_mask=support_weights,
                )

                inside_mass = (attn_patch * prior_patch).sum(dim=-1)
                outside_mass = (attn_patch * (1.0 - prior_patch)).sum(dim=-1)

                # mean_inside_mass = inside_mass[support_active].mean().detach()
                # mean_outside_mass = outside_mass[support_active].mean().detach()

                w = support_weights.clamp_min(0.0)

                mean_inside_mass = (inside_mass * w).sum() / w.sum().clamp_min(1.0)
                mean_outside_mass = (outside_mass * w).sum() / w.sum().clamp_min(1.0)

                mean_inside_mass = mean_inside_mass.detach()
                mean_outside_mass = mean_outside_mass.detach()

        # ------------------------------------------------------------
        # Debug print for first batch
        # ------------------------------------------------------------
        if i == 0 and utils.is_main_process():
            print("[DEBUG] image:", image.shape)
            print("[DEBUG] text_input.input_ids:", text_input.input_ids.shape)
            print("[DEBUG] support_active:", support_active.shape)
            print("[DEBUG] active samples:", int(support_active.sum().item()), "/", support_active.numel())
            print("[DEBUG] loss_support:", float(loss_support.detach().cpu()))
            print("[DEBUG] use_classification_preservation:", use_cls_preserve)
            print("[DEBUG] lambda_cls_preserve:", lambda_cls_preserve)

            if lambda_support > 0:
                print("[DEBUG] support_mode:", config.get("support_mode"))
                print("[DEBUG] support_weights mean:", float(support_weights.mean().detach().cpu()))
                print("[DEBUG] support_weights unique:", torch.unique(support_weights.detach().cpu()))
                print("[DEBUG] first 10 raw texts:")
                for idx, t in enumerate(text[:10]):
                    print(f"  {idx}: {t}")

            if attn_patch is not None:
                print("[DEBUG] attn_patch:", attn_patch.shape)
                print("[DEBUG] attn_patch requires_grad:", attn_patch.requires_grad)

            if prior_patch is not None:
                print("[DEBUG] prior_patch:", prior_patch.shape)

            if anatomy_prior_mask is not None:
                print("[DEBUG] cardiac_prior_mask:", anatomy_prior_mask.shape)

        # ------------------------------------------------------------
        # A4 classification-preservation distillation
        # ------------------------------------------------------------
        if use_cls_preserve and lambda_cls_preserve > 0.0:
            if teacher_model is None:
                raise RuntimeError(
                    "use_classification_preservation=True but teacher_model is None. "
                    "Set classification_teacher_checkpoint in the config."
                )
            teacher_model.eval()
            loss_cls_preserve = classification_preservation_loss(
                student_model=model,
                teacher_model=teacher_model,
                image=image,
                text_input=text_input,
                temperature=cls_preserve_temperature,
            )

        # ------------------------------------------------------------
        # Final A4 loss
        # ------------------------------------------------------------
        loss = (
            loss_mlm
            + loss_ita
            + loss_itm
            + lambda_support * loss_support
            + lambda_cls_preserve * loss_cls_preserve
        )

        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_support=loss_support.item())
        metric_logger.update(loss_support_w=(lambda_support * loss_support).item())
        metric_logger.update(loss_cls_preserve=loss_cls_preserve.item())
        metric_logger.update(loss_cls_preserve_w=(lambda_cls_preserve * loss_cls_preserve).item())
        metric_logger.update(attn_inside=mean_inside_mass.item())
        metric_logger.update(attn_outside=mean_outside_mass.item())
        metric_logger.update(support_active=support_active.float().mean().item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(support_weight=mean_support_weight.item())

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        # Short pilot run
        if config.get("debug_max_batches", None) is not None:
            if i + 1 >= config["debug_max_batches"]:
                if utils.is_main_process():
                    print(f"[DEBUG] Stopping after {i + 1} batches because debug_max_batches is set.")
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # return raw floats so we can compute best_loss properly
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def auto_resume_from_output_dir(args, model, optimizer, lr_scheduler, config, device):
    """
    If args.checkpoint is empty and args.auto_resume is True, try to find the latest
    checkpoint_XX.pth in args.output_dir and resume from it.
    Returns (start_epoch, best_loss).
    """
    output_dir = Path(args.output_dir)

    # Prefer checkpoint_last.pth when epoch checkpoints are disabled.
    last_ckpt = output_dir / "checkpoint_last.pth"
    if last_ckpt.exists():
        latest_ckpt = last_ckpt
    else:
        ckpts = sorted(output_dir.glob("checkpoint_*.pth"))
        if not ckpts:
            return 0, float("inf")
        latest_ckpt = ckpts[-1]

    print(f"Auto-resume: found checkpoint {latest_ckpt}, loading...")

    checkpoint = torch.load(latest_ckpt, map_location='cpu')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = checkpoint.get('epoch', -1) + 1
    best_loss = checkpoint.get('best_loss', float("inf"))

    print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]

    # ------------------------------------------------------------
    # Fixed training subset for controlled A0-A3 experiments
    # ------------------------------------------------------------
    train_subset_size = config.get("train_subset_size", None)

    if train_subset_size is not None:
        train_subset_size = int(train_subset_size)
        full_dataset_size = len(datasets[0])

        if train_subset_size > full_dataset_size:
            raise ValueError(
                f"train_subset_size={train_subset_size} is larger than "
                f"full dataset size={full_dataset_size}"
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
        datasets, samplers,
        batch_size=[config['batch_size']],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None]
    )[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    model = model.to(device)

    # Enable saving raw cross-attention maps for anatomy-prior support loss
    if config.get("lambda_support", 0.0) > 0:
        enable_crossattn_attention_saving_for_anatomy(
            model,
            layers=config.get("anatomy_layers", [8]),
        )
    else:
        if utils.is_main_process():
            print("[AnatomyPrior] lambda_support=0, cross-attention saving disabled.")

    # ------------------------------------------------------------
    # A4 frozen A0 teacher for classification preservation
    # ------------------------------------------------------------
    teacher_model = None
    if config.get("use_classification_preservation", False):
        teacher_ckpt = str(config.get("classification_teacher_checkpoint", "")).strip()
        if teacher_ckpt == "":
            raise ValueError(
                "use_classification_preservation=True requires "
                "classification_teacher_checkpoint in the config."
            )
        if utils.is_main_process():
            print(f"[A4] Creating frozen A0 teacher from: {teacher_ckpt}")

        teacher_model = ALBEF(
            config=config,
            text_encoder=args.text_encoder,
            tokenizer=tokenizer,
            init_deit=True,
        )
        load_msg = load_state_dict_flexible(
            teacher_model,
            teacher_ckpt,
            device="cpu",
            interpolate_pos=True,
        )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        if utils.is_main_process():
            print(f"[A4] Teacher loaded. load_state_dict message: {load_msg}")
            print("[A4] Teacher frozen and excluded from optimizer.")

    arg_opt = utils.AttrDict(config['optimizer'])
    arg_opt["lr"] = float(arg_opt["lr"])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche["lr"] = float(arg_sche["lr"])
    arg_sche["warmup_lr"] = float(arg_sche["warmup_lr"])
    arg_sche["min_lr"] = float(arg_sche["min_lr"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # ---------- Checkpoint loading / resume logic ----------
    best_loss = float("inf")

    if args.checkpoint:
        # Explicit checkpoint path provided
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if args.resume:
            # Full resume: optimizer, scheduler, epoch, best_loss
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', best_loss)
            model.load_state_dict(state_dict)
            print(f"Resumed training from epoch {start_epoch}, best_loss={best_loss:.6f}")
        else:
            # Weight-only load (e.g., finetuning from pretrain)
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict['visual_encoder.pos_embed'], model.visual_encoder
            )
            m_pos_embed_reshaped = interpolate_pos_embed(
                state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m
            )
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {args.checkpoint} (no optimizer/scheduler resume).")
    else:
        # No explicit checkpoint: try automatic resume if enabled
        if getattr(args, "auto_resume", False):
            start_epoch, best_loss = auto_resume_from_output_dir(
                args, model, optimizer, lr_scheduler, config, device
            )
        else:
            print("No checkpoint provided and auto-resume disabled. Starting from scratch.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model, teacher_model, data_loader, optimizer, tokenizer,
            epoch, warmup_steps, device, lr_scheduler, config
        )

        if utils.is_main_process():
            # Compute total training loss for best-model tracking
            loss_mlm = float(train_stats.get('loss_mlm', 0.0))
            loss_ita = float(train_stats.get('loss_ita', 0.0))
            loss_itm = float(train_stats.get('loss_itm', 0.0))
            train_loss_total = loss_mlm + loss_ita + loss_itm
            train_objective_total = (
                train_loss_total
                + float(config.get('lambda_support', 0.0)) * float(train_stats.get('loss_support', 0.0))
                + float(config.get('lambda_cls_preserve', 0.0)) * float(train_stats.get('loss_cls_preserve', 0.0))
            )

            is_best = train_loss_total < best_loss
            if is_best:
                best_loss = train_loss_total

            log_stats = {
                'epoch': epoch,
                'train_loss_mlm': loss_mlm,
                'train_loss_ita': loss_ita,
                'train_loss_itm': loss_itm,
                'train_loss_total': train_loss_total,
                'train_objective_total': train_objective_total,
                'train_loss_support': float(train_stats.get('loss_support', 0.0)),
                'train_loss_support_weighted': float(train_stats.get('loss_support_w', 0.0)),
                'train_loss_cls_preserve': float(train_stats.get('loss_cls_preserve', 0.0)),
                'train_loss_cls_preserve_weighted': float(train_stats.get('loss_cls_preserve_w', 0.0)),
                'train_attn_inside': float(train_stats.get('attn_inside', 0.0)),
                'train_attn_outside': float(train_stats.get('attn_outside', 0.0)),
                'train_support_active': float(train_stats.get('support_active', 0.0)),
                'best_loss': best_loss,
                'lr': float(train_stats.get('lr', optimizer.param_groups[0]["lr"])),
                'train_support_weight': float(train_stats.get('support_weight', 0.0)),
            }

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_loss': best_loss,
            }

            # Save only compact checkpoints by default.
            # Set save_epoch_checkpoints: true in the YAML if you also want checkpoint_XX.pth.
            if bool(config.get('save_epoch_checkpoints', False)):
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth'))

            # always keep "last"
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))

            # update "best" if this is the best so far
            if is_best:
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--auto_resume', default=True, type=bool,
                        help='automatically resume from latest checkpoint in output_dir if no explicit checkpoint is given')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
