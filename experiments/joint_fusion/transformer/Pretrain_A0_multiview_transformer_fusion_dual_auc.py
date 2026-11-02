#!/usr/bin/env python3
"""Train an original/lung/heart ALBEF model with 2-layer Transformer view fusion.

Architecture
------------
Each sample supplies three aligned versions of the same CXR:
    original, lung-masked, heart-masked.

Three ViT-B/16 branches produce [B, 257, 768] token sequences. For every
token position, the original/lung/heart tokens form a length-3 sequence. A
2-layer TransformerEncoder performs cross-view self-attention and the three
outputs are mean pooled to one 768-D fused token. The fused sequence is then
optimized with the standard ALBEF objectives:
    loss_total = loss_mlm + loss_ita + loss_itm

Initialization
--------------
For a fresh fusion run, the three visual branches are initialized from the
previously trained single-view A0 checkpoints:
    original checkpoint -> original ViT
    lung checkpoint     -> lung ViT
    heart checkpoint    -> heart ViT

The shared text encoder, text projection, vision projection, ITM head, and
temperature are initialized from one chosen single-view checkpoint (original by
default). Old single-view queues are NOT loaded. After assembly, all momentum
modules are reset from the assembled online modules.

Checkpoint policy
-----------------
The same dual validation policy is retained:
    checkpoint_best_cardiomegaly_auc.pth
    checkpoint_best_macro_auc_stable.pth

checkpoint_last.pth contains full optimizer/scheduler state for crash recovery
and is optionally removed after a clean finish.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import yaml

from dataset import create_loader, create_sampler
from dataset.utils import pre_caption
from models.model_pretrain_multiview_transformer_fusion import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import utils

from scripts_new.vindr_multiview_classification_validation import (
    DualAUCSelectionState,
    VinDrMultiViewClassificationValidationRunner,
    atomic_save_scores_npz,
    atomic_torch_save,
)


ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD = (0.26862954, 0.26130258, 0.27577711)
EXPERIMENT_NAME = "A0_multiview_transformer2_mean_fusion_dual_auc"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def raw_model(model):
    return model.module if hasattr(model, "module") else model


def _as_path_list(value: Any, field: str) -> list[Path]:
    if isinstance(value, (str, Path)):
        result = [Path(value)]
    elif isinstance(value, Sequence):
        result = [Path(item) for item in value]
    else:
        raise TypeError(f"{field} must be a path or list of paths")

    if not result:
        raise ValueError(f"{field} is empty")

    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"{field}: {path}")

    return result


def load_json_records(paths: Any, field: str) -> list[dict]:
    records: list[dict] = []
    for path in _as_path_list(paths, field):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON list in {path}")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}: record {index} is not a JSON object"
                )
        records.extend(payload)

    if not records:
        raise ValueError(f"No records loaded from {field}")
    return records


def _safe_relative_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value}")
    return path


class SynchronizedCXRTransform:
    """Apply exactly the same geometric/photometric augmentation to all views."""

    def __init__(self, config: dict):
        self.image_res = int(config["image_res"])
        aug_cfg = config.get("cxr_augmentation", {})
        self.enabled = bool(aug_cfg.get("enabled", True))
        self.degrees = float(aug_cfg.get("rotation_degrees", 5.0))
        self.translate = float(aug_cfg.get("translate_fraction", 0.02))
        self.scale_min = float(aug_cfg.get("scale_min", 0.98))
        self.scale_max = float(aug_cfg.get("scale_max", 1.02))
        self.brightness = float(aug_cfg.get("brightness", 0.10))
        self.contrast = float(aug_cfg.get("contrast", 0.10))
        self.photometric_probability = float(
            aug_cfg.get("photometric_probability", 0.5)
        )

        if self.image_res <= 0:
            raise ValueError("image_res must be positive")
        if self.degrees < 0:
            raise ValueError("rotation_degrees must be >= 0")
        if not 0 <= self.translate <= 1:
            raise ValueError("translate_fraction must be in [0,1]")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("Invalid scale range")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness/contrast must be >= 0")
        if not 0 <= self.photometric_probability <= 1:
            raise ValueError("photometric_probability must be in [0,1]")

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if np.isclose(low, high):
            return float(low)
        return float(torch.empty(1).uniform_(low, high).item())

    def _resize(self, image: Image.Image) -> Image.Image:
        return TF.resize(
            image,
            [self.image_res, self.image_res],
            interpolation=InterpolationMode.BICUBIC,
        )

    def __call__(
        self,
        original: Image.Image,
        lung: Image.Image,
        heart: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = [
            self._resize(original),
            self._resize(lung),
            self._resize(heart),
        ]

        if self.enabled:
            angle = self._uniform(-self.degrees, self.degrees)
            max_dx = self.translate * self.image_res
            max_dy = self.translate * self.image_res
            translate_xy = [
                int(round(self._uniform(-max_dx, max_dx))),
                int(round(self._uniform(-max_dy, max_dy))),
            ]
            scale = self._uniform(self.scale_min, self.scale_max)

            views = [
                TF.affine(
                    image,
                    angle=angle,
                    translate=translate_xy,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                for image in views
            ]

            if (
                (self.brightness > 0 or self.contrast > 0)
                and float(torch.rand(1).item())
                < self.photometric_probability
            ):
                brightness_factor = (
                    self._uniform(
                        max(0.0, 1.0 - self.brightness),
                        1.0 + self.brightness,
                    )
                    if self.brightness > 0
                    else 1.0
                )
                contrast_factor = (
                    self._uniform(
                        max(0.0, 1.0 - self.contrast),
                        1.0 + self.contrast,
                    )
                    if self.contrast > 0
                    else 1.0
                )

                # Use one sampled operation order for all three views.
                if bool(torch.randint(0, 2, (1,)).item()):
                    views = [
                        TF.adjust_contrast(
                            TF.adjust_brightness(image, brightness_factor),
                            contrast_factor,
                        )
                        for image in views
                    ]
                else:
                    views = [
                        TF.adjust_brightness(
                            TF.adjust_contrast(image, contrast_factor),
                            brightness_factor,
                        )
                        for image in views
                    ]

        tensors = []
        for image in views:
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, ALBEF_MEAN, ALBEF_STD)
            tensors.append(tensor)

        return tensors[0], tensors[1], tensors[2]


class MultiViewCXRPretrainDataset(Dataset):
    """Paired original/lung/heart MIMIC-CXR pretraining dataset.

    The existing CheXmask finalization creates separate lung and heart
    manifests. They must describe the same source images in the same order.
    Each record supplies its own `mask_relpath`, allowing lung and heart mask
    relative paths to differ safely.
    """

    def __init__(
        self,
        *,
        lung_ann_files,
        heart_ann_files,
        lung_mask_root: str | Path,
        heart_mask_root: str | Path,
        transform: SynchronizedCXRTransform,
        max_words: int,
        image_key: str = "image",
        caption_key: str = "caption",
        mask_key: str = "mask_relpath",
        verify_files_at_start: bool = False,
    ):
        self.lung_records = load_json_records(
            lung_ann_files,
            "lung_train_file",
        )
        self.heart_records = load_json_records(
            heart_ann_files,
            "heart_train_file",
        )

        if len(self.lung_records) != len(self.heart_records):
            raise ValueError(
                "Lung and heart manifests have different lengths: "
                f"{len(self.lung_records)} vs {len(self.heart_records)}"
            )

        self.lung_mask_root = Path(lung_mask_root)
        self.heart_mask_root = Path(heart_mask_root)
        if not self.lung_mask_root.is_dir():
            raise FileNotFoundError(
                f"lung_mask_root not found: {self.lung_mask_root}"
            )
        if not self.heart_mask_root.is_dir():
            raise FileNotFoundError(
                f"heart_mask_root not found: {self.heart_mask_root}"
            )

        self.transform = transform
        self.max_words = int(max_words)
        self.image_key = str(image_key)
        self.caption_key = str(caption_key)
        self.mask_key = str(mask_key)

        required = {self.image_key, self.caption_key, self.mask_key}
        for index, (lung_item, heart_item) in enumerate(
            zip(self.lung_records, self.heart_records)
        ):
            missing_lung = required - lung_item.keys()
            missing_heart = required - heart_item.keys()
            if missing_lung or missing_heart:
                raise KeyError(
                    f"Manifest record {index}: "
                    f"missing lung={sorted(missing_lung)}, "
                    f"missing heart={sorted(missing_heart)}"
                )

            lung_image = str(lung_item[self.image_key])
            heart_image = str(heart_item[self.image_key])
            if lung_image != heart_image:
                raise ValueError(
                    f"Manifest misalignment at index {index}: "
                    f"lung image={lung_image}, heart image={heart_image}"
                )

            if lung_item[self.caption_key] != heart_item[self.caption_key]:
                raise ValueError(
                    f"Caption mismatch between manifests at index {index}"
                )

        if verify_files_at_start:
            self._verify_all_files()

        print(
            "[MultiViewCXRPretrainDataset] "
            f"paired records={len(self.lung_records):,}",
            flush=True,
        )

    def _paths_for_index(
        self,
        index: int,
    ) -> tuple[Path, Path, Path]:
        lung_item = self.lung_records[index]
        heart_item = self.heart_records[index]

        image_path = Path(str(lung_item[self.image_key])).expanduser()
        lung_rel = _safe_relative_path(
            str(lung_item[self.mask_key]),
            "lung mask_relpath",
        )
        heart_rel = _safe_relative_path(
            str(heart_item[self.mask_key]),
            "heart mask_relpath",
        )
        return (
            image_path,
            self.lung_mask_root / lung_rel,
            self.heart_mask_root / heart_rel,
        )

    def _verify_all_files(self) -> None:
        for index in range(len(self)):
            image_path, lung_path, heart_path = self._paths_for_index(index)
            for path, name in (
                (image_path, "image"),
                (lung_path, "lung mask"),
                (heart_path, "heart mask"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {name} at record {index}: {path}"
                    )
            if index == 0 or (index + 1) % 10000 == 0:
                print(
                    f"[Dataset verify] {index + 1:,}/{len(self):,}",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.lung_records)

    def __getitem__(self, index: int):
        lung_item = self.lung_records[index]
        image_path, lung_path, heart_path = self._paths_for_index(index)

        with Image.open(image_path) as handle:
            original = handle.convert("RGB")

        with Image.open(lung_path) as handle:
            lung_mask = handle.convert("L")

        with Image.open(heart_path) as handle:
            heart_mask = handle.convert("L")

        if lung_mask.size != original.size:
            raise ValueError(
                f"Lung mask size mismatch at index={index}: "
                f"image={original.size}, mask={lung_mask.size}"
            )
        if heart_mask.size != original.size:
            raise ValueError(
                f"Heart mask size mismatch at index={index}: "
                f"image={original.size}, mask={heart_mask.size}"
            )

        black = Image.new("RGB", original.size, (0, 0, 0))
        lung = Image.composite(original, black, lung_mask)
        heart = Image.composite(original, black, heart_mask)

        original_t, lung_t, heart_t = self.transform(
            original,
            lung,
            heart,
        )

        caption_value = lung_item[self.caption_key]
        if isinstance(caption_value, list):
            if not caption_value:
                raise ValueError(f"Empty caption list at index={index}")
            caption_value = random.choice(caption_value)

        caption = pre_caption(
            str(caption_value),
            self.max_words,
        )

        return original_t, lung_t, heart_t, caption


def build_dataset(config: dict) -> MultiViewCXRPretrainDataset:
    required = [
        "lung_train_file",
        "heart_train_file",
        "lung_mask_root",
        "heart_mask_root",
    ]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(
            "Multi-view fusion config is missing: " + ", ".join(missing)
        )

    return MultiViewCXRPretrainDataset(
        lung_ann_files=config["lung_train_file"],
        heart_ann_files=config["heart_train_file"],
        lung_mask_root=config["lung_mask_root"],
        heart_mask_root=config["heart_mask_root"],
        transform=SynchronizedCXRTransform(config),
        max_words=int(config.get("max_words", 30)),
        image_key=config.get("image_key", "image"),
        caption_key=config.get("caption_key", "caption"),
        mask_key=config.get("mask_key", "mask_relpath"),
        verify_files_at_start=bool(
            config.get("verify_training_files_at_start", False)
        ),
    )


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    args,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    for name in ("loss_mlm", "loss_ita", "loss_itm", "loss_total"):
        metric_logger.add_meter(
            name,
            utils.SmoothedValue(window_size=50, fmt="{value:.4f}"),
        )
    metric_logger.add_meter(
        "lr",
        utils.SmoothedValue(window_size=50, fmt="{value:.6f}"),
    )

    header = f"A0-multiview-transformer2-mean Epoch [{epoch}]"
    print_freq = int(config.get("print_freq", 50))
    step_size = 100
    warmup_iterations = int(warmup_steps) * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
        )
    ):
        if len(batch) != 4:
            raise ValueError(
                "Expected batch=(original, lung, heart, text); "
                f"received {len(batch)} items"
            )

        image_original, image_lung, image_heart, text = batch

        optimizer.zero_grad(set_to_none=True)

        image_original = image_original.to(
            device,
            non_blocking=True,
        )
        image_lung = image_lung.to(
            device,
            non_blocking=True,
        )
        image_heart = image_heart.to(
            device,
            non_blocking=True,
        )

        text_input = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=int(config.get("max_text_len", 32)),
            return_tensors="pt",
        ).to(device)

        if epoch > 0:
            alpha = float(config["alpha"])
        else:
            alpha = float(config["alpha"]) * min(
                1.0,
                step / max(1, len(data_loader)),
            )

        if device.type == "cuda" and step == 0:
            torch.cuda.reset_peak_memory_stats(device)

        loss_mlm, loss_ita, loss_itm = model(
            image_original,
            image_lung,
            image_heart,
            text_input,
            alpha=alpha,
        )
        loss_total = loss_mlm + loss_ita + loss_itm

        numeric_every = int(config.get("numeric_debug_every", 100))
        if (
            numeric_every > 0
            and step % numeric_every == 0
            and utils.is_main_process()
        ):
            current_model = raw_model(model)
            fusion_norm, fusion_absmax = current_model.fusion_parameter_stats()
            print(
                f"[NUMERIC] epoch={epoch} step={step} "
                f"loss_mlm={loss_mlm.item():.6f} "
                f"loss_ita={loss_ita.item():.6f} "
                f"loss_itm={loss_itm.item():.6f} "
                f"fusion_param_norm={fusion_norm:.6f} "
                f"fusion_param_absmax={fusion_absmax:.6f} "
                f"temp={current_model.temp.item():.6f}",
                flush=True,
            )

        if device.type == "cuda" and step == 0 and utils.is_main_process():
            print(
                "[GPU MEMORY after forward] "
                f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB "
                f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB "
                f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
                flush=True,
            )

        if not torch.isfinite(loss_total):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}: "
                f"mlm={loss_mlm.item()}, "
                f"ita={loss_ita.item()}, "
                f"itm={loss_itm.item()}"
            )

        loss_total.backward()

        if device.type == "cuda" and step == 0 and utils.is_main_process():
            print(
                "[GPU MEMORY after backward] "
                f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB "
                f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB "
                f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
                flush=True,
            )

        grad_clip = config.get("grad_clip_norm")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )

        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_total=loss_total.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if step == 0 and utils.is_main_process():
            print(
                f"[DEBUG] original={tuple(image_original.shape)}",
                flush=True,
            )
            print(
                f"[DEBUG] lung={tuple(image_lung.shape)}",
                flush=True,
            )
            print(
                f"[DEBUG] heart={tuple(image_heart.shape)}",
                flush=True,
            )
            print(
                f"[DEBUG] loss_total={loss_total.item():.6f}",
                flush=True,
            )

        # Preserve the scheduler convention of the existing ALBEF script.
        if (
            epoch == 0
            and step % step_size == 0
            and step <= warmup_iterations
        ):
            scheduler.step(step // step_size)

        debug_batches = config.get("debug_max_batches")
        if (
            debug_batches is not None
            and step + 1 >= int(debug_batches)
        ):
            break

    metric_logger.synchronize_between_processes()
    print(
        "Averaged stats:",
        metric_logger.global_avg(),
        flush=True,
    )
    return {
        key: meter.global_avg
        for key, meter in metric_logger.meters.items()
    }


def _load_checkpoint_state(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint state at {path}")

    # Be tolerant of checkpoints saved directly from DDP.
    if state and all(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :]: value
            for key, value in state.items()
        }

    config = checkpoint.get("config", {})
    return state, config


def _substate(state: dict, prefix: str) -> dict:
    prefix_dot = prefix + "."
    result = {
        key[len(prefix_dot) :]: value
        for key, value in state.items()
        if key.startswith(prefix_dot)
    }
    if not result:
        raise KeyError(f"No parameters found for prefix {prefix!r}")
    return result


def _load_single_view_visual_encoder(
    *,
    target_encoder,
    checkpoint_path: str | Path,
    view_name: str,
) -> None:
    state, _ = _load_checkpoint_state(checkpoint_path)
    visual_state = _substate(state, "visual_encoder")

    if "pos_embed" in visual_state:
        visual_state["pos_embed"] = interpolate_pos_embed(
            visual_state["pos_embed"],
            target_encoder,
        )

    msg = target_encoder.load_state_dict(
        visual_state,
        strict=True,
    )
    print(
        f"[Init] {view_name} visual encoder <- {checkpoint_path}: {msg}",
        flush=True,
    )


def _load_shared_from_single_view_checkpoint(
    *,
    model: ALBEF,
    checkpoint_path: str | Path,
    source_name: str,
) -> None:
    state, _ = _load_checkpoint_state(checkpoint_path)

    module_map = {
        "text_encoder": model.text_encoder,
        "vision_proj": model.vision_proj,
        "text_proj": model.text_proj,
        "itm_head": model.itm_head,
    }
    for prefix, module in module_map.items():
        msg = module.load_state_dict(
            _substate(state, prefix),
            strict=True,
        )
        print(
            f"[Init shared/{source_name}] {prefix}: {msg}",
            flush=True,
        )

    if "temp" not in state:
        raise KeyError(
            f"{checkpoint_path} does not contain learned temperature 'temp'"
        )
    with torch.no_grad():
        model.temp.copy_(state["temp"].reshape_as(model.temp))

    print(
        f"[Init shared/{source_name}] temp={float(model.temp):.8f}",
        flush=True,
    )


def initialize_from_three_single_view_checkpoints(
    *,
    model: ALBEF,
    original_checkpoint: str | Path,
    lung_checkpoint: str | Path,
    heart_checkpoint: str | Path,
    shared_components_from: str = "original",
) -> None:
    checkpoints = {
        "original": Path(original_checkpoint),
        "lung": Path(lung_checkpoint),
        "heart": Path(heart_checkpoint),
    }

    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_original,
        checkpoint_path=checkpoints["original"],
        view_name="original",
    )
    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_lung,
        checkpoint_path=checkpoints["lung"],
        view_name="lung",
    )
    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_heart,
        checkpoint_path=checkpoints["heart"],
        view_name="heart",
    )

    shared_components_from = str(shared_components_from).lower().strip()
    if shared_components_from not in checkpoints:
        raise ValueError(
            "shared_components_from must be original, lung, or heart"
        )

    _load_shared_from_single_view_checkpoint(
        model=model,
        checkpoint_path=checkpoints[shared_components_from],
        source_name=shared_components_from,
    )

    # The Transformer fusion module keeps its fresh initialization.
    # Momentum copies must now mirror the newly assembled online network.
    model.reset_momentum_from_online()

    # image_queue/text_queue intentionally remain fresh random normalized
    # buffers from __init__, because old queues contain single-view features.
    model.queue_ptr.zero_()

    print(
        "[Init] Three-view model assembled; momentum reset; "
        "single-view queues were not reused.",
        flush=True,
    )


def _interpolate_fused_positional_embeddings(
    state: dict,
    model: ALBEF,
) -> None:
    pairs = (
        ("visual_encoder_original.pos_embed", model.visual_encoder_original),
        ("visual_encoder_lung.pos_embed", model.visual_encoder_lung),
        ("visual_encoder_heart.pos_embed", model.visual_encoder_heart),
        (
            "visual_encoder_original_m.pos_embed",
            model.visual_encoder_original_m,
        ),
        ("visual_encoder_lung_m.pos_embed", model.visual_encoder_lung_m),
        ("visual_encoder_heart_m.pos_embed", model.visual_encoder_heart_m),
    )

    for key, encoder in pairs:
        if key in state:
            state[key] = interpolate_pos_embed(
                state[key],
                encoder,
            )


def load_fused_weights_only(
    model: ALBEF,
    path: str | Path,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if state and all(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :]: value
            for key, value in state.items()
        }

    _interpolate_fused_positional_embeddings(state, model)
    msg = model.load_state_dict(state, strict=True)
    print(
        f"[Checkpoint] Loaded fused model weights from {path}: {msg}",
        flush=True,
    )


def resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
) -> Tuple[int, float, DualAUCSelectionState]:
    checkpoint = torch.load(path, map_location="cpu")
    required = {"model", "optimizer", "lr_scheduler"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Cannot resume from {path}: missing {sorted(missing)}. "
            "Resume from checkpoint_last.pth."
        )

    state = checkpoint["model"]
    _interpolate_fused_positional_embeddings(state, model)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])

    selection_state = DualAUCSelectionState.from_dict(
        checkpoint.get("validation_selection_state")
    )
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_loss", float("inf"))),
        selection_state,
    )


def build_validation_runner(config):
    val_cfg = config.get("vindr_validation", {})
    if not bool(val_cfg.get("enabled", False)):
        return None

    required = [
        "labels_csv",
        "images_root",
        "lung_mask_root",
        "heart_mask_root",
    ]
    missing = [key for key in required if not val_cfg.get(key)]
    if missing:
        raise ValueError(
            "vindr_validation is enabled but missing: "
            + ", ".join(missing)
        )

    return VinDrMultiViewClassificationValidationRunner(
        labels_csv=val_cfg["labels_csv"],
        images_root=val_cfg["images_root"],
        lung_mask_root=val_cfg["lung_mask_root"],
        heart_mask_root=val_cfg["heart_mask_root"],
        image_res=int(config["image_res"]),
        batch_size=int(val_cfg.get("batch_size", 32)),
        num_workers=int(val_cfg.get("num_workers", 4)),
        label_name=val_cfg.get("label_name", "Cardiomegaly"),
        max_images=val_cfg.get("max_images"),
        max_text_length=int(val_cfg.get("max_text_length", 32)),
        threshold_steps=int(val_cfg.get("threshold_steps", 200)),
        min_positive_per_label=int(
            val_cfg.get("min_positive_per_label", 5)
        ),
        min_negative_per_label=int(
            val_cfg.get("min_negative_per_label", 5)
        ),
        macro_auc_labels=val_cfg.get("macro_auc_labels"),
    )


def save_last_checkpoint(
    *,
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    config: dict,
    epoch: int,
    best_loss: float,
    selection_state: DualAUCSelectionState,
    latest_validation: Optional[dict],
) -> None:
    payload = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "validation_selection_state": selection_state.to_dict(),
        "latest_validation": latest_validation,
        "experiment": EXPERIMENT_NAME,
        "checkpoint_purpose": "temporary_resume_checkpoint",
        "architecture": (
            "3x ViT-B/16 -> per-position 3-view sequence -> 2-layer "
            "TransformerEncoder -> mean pool views -> LayerNorm -> "
            "shared ALBEF ITC+ITM+MLM"
        ),
    }
    atomic_torch_save(
        payload,
        output_dir / "checkpoint_last.pth",
    )


def save_best_model_checkpoint(
    *,
    output_path: Path,
    model,
    config: dict,
    epoch: int,
    validation_metrics: dict,
    selection_state: DualAUCSelectionState,
) -> None:
    payload = {
        "model": raw_model(model).state_dict(),
        "config": config,
        "epoch": int(epoch),
        "validation": validation_metrics,
        "validation_selection_state": selection_state.to_dict(),
        "experiment": EXPERIMENT_NAME,
        "checkpoint_purpose": (
            "multiview_fused_model_selected_on_vindr_train_validation"
        ),
        "architecture": (
            "3x ViT-B/16 -> per-position 3-view sequence -> 2-layer "
            "TransformerEncoder -> mean pool views -> LayerNorm -> "
            "shared ALBEF ITC+ITM+MLM"
        ),
    }
    atomic_torch_save(payload, output_path)


def _resolve_initial_checkpoint(
    args_value: str,
    config: dict,
    view_name: str,
) -> str:
    if args_value:
        return args_value

    init_cfg = config.get("initial_checkpoints", {})
    value = init_cfg.get(view_name, "")
    return str(value) if value else ""


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)
    else:
        # Do not let ALBEF's Slurm auto-detection force env:// DDP for a
        # deliberately single-process smoke/debug run.
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        print(
            "Not using distributed mode (explicit --distributed false)",
            flush=True,
        )

    device = torch.device(args.device)

    seed = int(args.seed) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = bool(
        config.get("cudnn_deterministic", True)
    )

    dataset = build_dataset(config)

    subset_size = config.get("train_subset_size")
    if subset_size is not None:
        subset_size = int(subset_size)
        if subset_size > len(dataset):
            raise ValueError(
                f"train_subset_size={subset_size} > {len(dataset)}"
            )
        rng = np.random.default_rng(
            int(config.get("train_subset_seed", 42))
        )
        indices = rng.permutation(len(dataset))[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        print(
            f"[Dataset] Fixed subset: {subset_size}",
            flush=True,
        )
    else:
        print(
            f"[Dataset] Full paired manifest: {len(dataset)} records",
            flush=True,
        )

    datasets = [dataset]
    if args.distributed:
        samplers = create_sampler(
            datasets,
            [True],
            utils.get_world_size(),
            utils.get_rank(),
        )
    else:
        samplers = [None]

    loader = create_loader(
        datasets,
        samplers,
        batch_size=[int(config["batch_size"])],
        num_workers=[int(config.get("num_workers", 8))],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # The fresh fusion experiment is initialized from the three A0
    # checkpoints, so downloading/loading DeiT here would only be overwritten.
    model = ALBEF(
        config=config,
        text_encoder=args.text_encoder,
        tokenizer=tokenizer,
        init_deit=False,
    ).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "checkpoint_last.pth"

    # Preserve the original priority:
    # explicit --resume > explicit --checkpoint weights-only > auto-resume.
    auto_resume_path = (
        last_path
        if (
            args.auto_resume
            and last_path.exists()
            and not args.resume
            and not args.checkpoint
        )
        else None
    )
    will_resume = bool(args.resume or auto_resume_path is not None)

    if not will_resume:
        if args.checkpoint:
            load_fused_weights_only(model, args.checkpoint)
        else:
            original_checkpoint = _resolve_initial_checkpoint(
                args.original_checkpoint,
                config,
                "original",
            )
            lung_checkpoint = _resolve_initial_checkpoint(
                args.lung_checkpoint,
                config,
                "lung",
            )
            heart_checkpoint = _resolve_initial_checkpoint(
                args.heart_checkpoint,
                config,
                "heart",
            )
            missing = [
                name
                for name, value in (
                    ("original", original_checkpoint),
                    ("lung", lung_checkpoint),
                    ("heart", heart_checkpoint),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Fresh multi-view fusion training requires three "
                    "single-view checkpoints. Missing: "
                    + ", ".join(missing)
                    + ". Supply CLI arguments or config.initial_checkpoints."
                )

            shared_from = str(
                config.get("initial_checkpoints", {}).get(
                    "shared_components_from",
                    "original",
                )
            )
            initialize_from_three_single_view_checkpoints(
                model=model,
                original_checkpoint=original_checkpoint,
                lung_checkpoint=lung_checkpoint,
                heart_checkpoint=heart_checkpoint,
                shared_components_from=shared_from,
            )

    opt_cfg = utils.AttrDict(config["optimizer"])
    opt_cfg["lr"] = float(opt_cfg["lr"])
    optimizer = create_optimizer(opt_cfg, model)

    sched_cfg = utils.AttrDict(config["schedular"])
    for key in ("lr", "warmup_lr", "min_lr"):
        sched_cfg[key] = float(sched_cfg[key])
    scheduler, _ = create_scheduler(sched_cfg, optimizer)

    start_epoch = 0
    best_loss = float("inf")
    selection_state = DualAUCSelectionState()

    if args.resume:
        resume_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else last_path
        )
        start_epoch, best_loss, selection_state = resume_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
        )
        print(
            f"[Resume] {resume_path} -> epoch {start_epoch}",
            flush=True,
        )
    elif auto_resume_path is not None:
        start_epoch, best_loss, selection_state = resume_checkpoint(
            auto_resume_path,
            model,
            optimizer,
            scheduler,
        )
        print(
            f"[Auto-resume] {auto_resume_path} -> epoch {start_epoch}",
            flush=True,
        )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )

    validation_runner = None
    if utils.is_main_process():
        validation_runner = build_validation_runner(config)

    val_cfg = config.get("vindr_validation", {})
    validation_enabled = bool(val_cfg.get("enabled", False))
    validate_every = max(
        1,
        int(val_cfg.get("validate_every", 1)),
    )
    cardio_min_delta = float(
        val_cfg.get("cardiomegaly_min_delta", 0.0005)
    )
    macro_min_delta = float(
        val_cfg.get("macro_min_delta", 0.0005)
    )

    early_cfg = val_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_patience = max(
        1,
        int(early_cfg.get("patience", 5)),
    )
    early_min_epochs = max(
        1,
        int(early_cfg.get("min_epochs", 15)),
    )

    max_epoch = int(config["schedular"]["epochs"])
    warmup_steps = int(config["schedular"]["warmup_epochs"])
    started = time.time()
    completed_cleanly = False

    try:
        for epoch in range(start_epoch, max_epoch):
            if epoch > 0:
                scheduler.step(epoch + warmup_steps)

            stats = train_one_epoch(
                model,
                loader,
                optimizer,
                tokenizer,
                epoch,
                warmup_steps,
                device,
                scheduler,
                config,
                args,
            )

            if dist_ready():
                dist.barrier()

            stop_training = False

            if utils.is_main_process():
                total = float(stats["loss_total"])
                best_loss = min(best_loss, total)

                validation_metrics = None
                improved_cardio = False
                improved_macro = False

                should_validate = (
                    validation_enabled
                    and validation_runner is not None
                    and (
                        (epoch + 1) % validate_every == 0
                        or epoch == max_epoch - 1
                    )
                )

                if should_validate:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    metrics_obj, outputs = validation_runner.evaluate(
                        model=raw_model(model),
                        tokenizer=tokenizer,
                        device=device,
                        epoch=epoch,
                        return_outputs=True,
                    )
                    validation_metrics = metrics_obj.to_dict()

                    (
                        improved_cardio,
                        improved_macro,
                    ) = selection_state.consider(
                        metrics_obj,
                        cardiomegaly_min_delta=cardio_min_delta,
                        macro_min_delta=macro_min_delta,
                    )

                    if improved_cardio:
                        save_best_model_checkpoint(
                            output_path=(
                                output_dir
                                / "checkpoint_best_cardiomegaly_auc.pth"
                            ),
                            model=model,
                            config=config,
                            epoch=epoch,
                            validation_metrics=validation_metrics,
                            selection_state=selection_state,
                        )
                        atomic_save_scores_npz(
                            (
                                output_dir
                                / "best_cardiomegaly_auc_scores.npz"
                            ),
                            outputs=outputs,
                            metrics=metrics_obj,
                        )
                        print(
                            "[Checkpoint] New best Cardiomegaly AUC "
                            f"epoch={epoch}: "
                            f"{metrics_obj.cardiomegaly_auc:.6f}",
                            flush=True,
                        )

                    if improved_macro:
                        save_best_model_checkpoint(
                            output_path=(
                                output_dir
                                / "checkpoint_best_macro_auc_stable.pth"
                            ),
                            model=model,
                            config=config,
                            epoch=epoch,
                            validation_metrics=validation_metrics,
                            selection_state=selection_state,
                        )
                        atomic_save_scores_npz(
                            (
                                output_dir
                                / "best_macro_auc_stable_scores.npz"
                            ),
                            outputs=outputs,
                            metrics=metrics_obj,
                        )
                        print(
                            "[Checkpoint] New best stable-label macro AUC "
                            f"epoch={epoch}: "
                            f"{metrics_obj.macro_auc_stable:.6f}",
                            flush=True,
                        )

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    stop_training = bool(
                        early_enabled
                        and (epoch + 1) >= early_min_epochs
                        and (
                            selection_state.validations_without_improvement
                            >= early_patience
                        )
                    )

                save_last_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    best_loss=best_loss,
                    selection_state=selection_state,
                    latest_validation=validation_metrics,
                )

                row = {
                    "epoch": int(epoch),
                    "experiment": EXPERIMENT_NAME,
                    **{
                        f"train_{key}": float(value)
                        for key, value in stats.items()
                    },
                    "best_train_loss": float(best_loss),
                    "validation_ran": bool(should_validate),
                    "improved_cardiomegaly_auc": bool(
                        improved_cardio
                    ),
                    "improved_macro_auc_stable": bool(
                        improved_macro
                    ),
                    "best_cardiomegaly_auc": (
                        None
                        if not np.isfinite(
                            selection_state.best_cardiomegaly_auc
                        )
                        else float(
                            selection_state.best_cardiomegaly_auc
                        )
                    ),
                    "best_cardiomegaly_epoch": int(
                        selection_state.best_cardiomegaly_epoch
                    ),
                    "best_macro_auc_stable": (
                        None
                        if not np.isfinite(
                            selection_state.best_macro_auc
                        )
                        else float(selection_state.best_macro_auc)
                    ),
                    "best_macro_auc_stable_epoch": int(
                        selection_state.best_macro_epoch
                    ),
                    "validations_without_improvement": int(
                        selection_state.validations_without_improvement
                    ),
                    "early_stop": bool(stop_training),
                }

                if validation_metrics is not None:
                    row.update(
                        {
                            "val_cardiomegaly_auc": validation_metrics[
                                "cardiomegaly_auc"
                            ],
                            "val_macro_auc_stable": validation_metrics[
                                "macro_auc_stable"
                            ],
                            "val_macro_auc_all_evaluable": (
                                validation_metrics[
                                    "macro_auc_all_evaluable"
                                ]
                            ),
                            "val_micro_auc": validation_metrics[
                                "micro_auc"
                            ],
                            "val_num_valid_auc_labels": (
                                validation_metrics[
                                    "num_valid_auc_labels"
                                ]
                            ),
                            "val_per_label_auc": validation_metrics[
                                "per_label_auc"
                            ],
                            "val_skipped_auc_labels": (
                                validation_metrics[
                                    "skipped_auc_labels"
                                ]
                            ),
                            "val_cardiomegaly_best_f1": (
                                validation_metrics[
                                    "cardiomegaly_best_f1"
                                ]
                            ),
                            "val_cardiomegaly_best_threshold": (
                                validation_metrics[
                                    "cardiomegaly_best_threshold"
                                ]
                            ),
                            "val_cardiomegaly_score_mean": (
                                validation_metrics[
                                    "cardiomegaly_score_mean"
                                ]
                            ),
                            "val_cardiomegaly_score_std": (
                                validation_metrics[
                                    "cardiomegaly_score_std"
                                ]
                            ),
                        }
                    )

                with open(
                    output_dir / "log.txt",
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(json.dumps(row) + "\n")

            if dist_ready():
                stop_tensor = torch.tensor(
                    [1 if stop_training else 0],
                    device=device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
                dist.barrier()

            if stop_training:
                if utils.is_main_process():
                    print(
                        "[Early stopping] Neither Cardiomegaly AUC nor "
                        "stable-label macro AUC improved for "
                        f"{early_patience} validations after "
                        f"min_epochs={early_min_epochs}.",
                        flush=True,
                    )
                break

        completed_cleanly = True

    finally:
        if (
            completed_cleanly
            and utils.is_main_process()
            and not bool(
                config.get(
                    "retain_last_checkpoint_after_success",
                    False,
                )
            )
            and last_path.exists()
        ):
            last_path.unlink()
            print(
                "[Checkpoint] Deleted checkpoint_last.pth after "
                "successful completion; best checkpoints remain.",
                flush=True,
            )

    elapsed = int(time.time() - started)
    print(
        "Training time",
        str(datetime.timedelta(seconds=elapsed)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    # Fused checkpoint: use for weight-only restart or explicit full resume.
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--auto_resume", type=str2bool, default=True)

    # Fresh fusion initialization. CLI overrides config.initial_checkpoints.
    parser.add_argument("--original_checkpoint", default="")
    parser.add_argument("--lung_checkpoint", default="")
    parser.add_argument("--heart_checkpoint", default="")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--text_encoder",
        default="bert-base-uncased",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument(
        "--distributed",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "config.yaml",
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )

    main(args, config)
