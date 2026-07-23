"""
A0 ALBEF pretraining for original, lung-only, and heart-only views with
online VinDr checkpoint selection.

At the end of each epoch, rank 0 evaluates the current in-memory model on one
fixed VinDr-train validation subset:

1. Primary checkpoint criterion: Cardiomegaly ROC-AUC.
2. Tie-breaker within an AUC tolerance band: mean FROC sensitivity at the
   configured FP/image targets.

Only two checkpoints are retained:

* checkpoint_last.pth: full model/optimizer/scheduler state for resuming.
* checkpoint_best_val.pth: model-only checkpoint selected by VinDr validation.

The selected checkpoint's lightweight per-image validation scores are saved to
best_val_cardiomegaly_scores.npz. These scores are later used to normalize or
calibrate original/lung/heart cosine similarities before ensembling.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
import yaml

from dataset import create_dataset, create_loader, create_sampler
from dataset.randaugment import RandomAugment
from dataset.chexmask_cached_mask_dataset import CheXmaskCachedMaskPretrainDataset
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import utils

from scripts.vindr_online_validation import (
    ValidationSelectionState,
    VinDrOnlineValidationRunner,
    atomic_save_validation_scores_npz,
    atomic_torch_save,
)


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


def build_dataset(config):
    """Build the correct pretraining dataset for the configured image view.

    ``original`` uses ALBEF's standard pretraining dataset and transformation.
    ``lung`` and ``heart`` use the compact mask-cache dataset, applying the
    binary mask before the otherwise unchanged ALBEF augmentation pipeline.
    """
    view_name = str(config.get("view_name", "")).lower().strip()

    if view_name == "original":
        if config.get("mask_root") not in (None, ""):
            print(
                "[Dataset] view=original: ignoring mask_root from config.",
                flush=True,
            )
        dataset = create_dataset("pretrain", config)
        print("[Dataset] Using standard ALBEF original-CXR dataset.", flush=True)
        return dataset

    if view_name not in {"lung", "heart"}:
        raise ValueError(
            f"Unsupported view_name={view_name!r}. Expected original, lung, or heart."
        )

    mask_root = config.get("mask_root")
    if not mask_root:
        raise ValueError(f"mask_root is required for view_name={view_name!r}.")

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    pretrain_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                int(config["image_res"]),
                scale=(0.2, 1.0),
                interpolation=Image.Resampling.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CheXmaskCachedMaskPretrainDataset(
        ann_files=config["train_file"],
        transform=pretrain_transform,
        mask_root=mask_root,
        max_words=int(config.get("max_words", 30)),
        image_key=config.get("image_key", "image"),
        caption_key=config.get("caption_key", "caption"),
        mask_key=config.get("mask_key", "mask_relpath"),
    )
    print(
        f"[Dataset] Using cached-mask pretraining dataset for view={view_name}.",
        flush=True,
    )
    return dataset


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
            name, utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
        )
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )

    header = f"A0-{config['view_name']} Epoch [{epoch}]"
    print_freq = int(config.get("print_freq", 50))
    step_size = 100
    warmup_iterations = int(warmup_steps) * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for step, (image, text) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer.zero_grad(set_to_none=True)
        image = image.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=int(config.get("max_text_len", 25)),
            return_tensors="pt",
        ).to(device)

        if epoch > 0:
            alpha = float(config["alpha"])
        else:
            alpha = float(config["alpha"]) * min(
                1.0, step / max(1, len(data_loader))
            )

        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)
        loss_total = loss_mlm + loss_ita + loss_itm
        loss_total.backward()

        grad_clip = config.get("grad_clip_norm")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_total=loss_total.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if step == 0 and utils.is_main_process():
            print(f"[DEBUG] view={config['view_name']}", flush=True)
            print(f"[DEBUG] image={tuple(image.shape)}", flush=True)
            print(f"[DEBUG] loss_total={loss_total.item():.6f}", flush=True)

        if epoch == 0 and step % step_size == 0 and step <= warmup_iterations:
            scheduler.step(step // step_size)

        debug_batches = config.get("debug_max_batches")
        if debug_batches is not None and step + 1 >= int(debug_batches):
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg(), flush=True)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def load_weights_only(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint

    for key, encoder in (
        ("visual_encoder.pos_embed", model.visual_encoder),
        ("visual_encoder_m.pos_embed", model.visual_encoder_m),
    ):
        if key in state:
            state[key] = interpolate_pos_embed(state[key], encoder)

    print(model.load_state_dict(state, strict=False), flush=True)


def resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
) -> Tuple[int, float, ValidationSelectionState]:
    checkpoint = torch.load(path, map_location="cpu")

    required = {"model", "optimizer", "lr_scheduler"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Cannot resume from {path}: missing {sorted(missing)}. "
            "Use checkpoint_last.pth for --resume, or load a model-only "
            "checkpoint with --resume false --checkpoint <path>."
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])

    validation_state = ValidationSelectionState.from_dict(
        checkpoint.get("validation_selection_state")
    )

    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_loss", float("inf"))),
        validation_state,
    )


def build_validation_runner(config):
    val_cfg = config.get("vindr_validation", {})
    enabled = bool(val_cfg.get("enabled", False))
    if not enabled:
        return None

    required = [
        "classification_labels_csv",
        "localization_annotations_csv",
        "images_root",
        "view_type",
    ]
    missing = [key for key in required if not val_cfg.get(key)]
    if missing:
        raise ValueError(
            "vindr_validation is enabled but these keys are missing: "
            + ", ".join(missing)
        )

    return VinDrOnlineValidationRunner(
        classification_labels_csv=val_cfg["classification_labels_csv"],
        localization_labels_csv=val_cfg.get("localization_labels_csv"),
        localization_annotations_csv=val_cfg["localization_annotations_csv"],
        images_root=val_cfg["images_root"],
        view_type=val_cfg["view_type"],
        mask_root=val_cfg.get("mask_root"),
        image_res=int(config["image_res"]),
        label_name=val_cfg.get("label_name", "Cardiomegaly"),
        classification_batch_size=int(val_cfg.get("batch_size", 64)),
        classification_num_workers=int(val_cfg.get("num_workers", 4)),
        max_classification_images=val_cfg.get("max_classification_images"),
        max_localization_images=val_cfg.get("max_localization_images"),
        layers_to_use=val_cfg.get("layers_to_use", [8]),
        max_text_length=int(val_cfg.get("max_text_length", 32)),
        cam_key=val_cfg.get("cam_key", "cam_vis"),
        heatmap_threshold=float(val_cfg.get("heatmap_threshold", 0.50)),
        min_box_area_frac=float(val_cfg.get("min_box_area_frac", 0.002)),
        score_mode=val_cfg.get("score_mode", "max"),
        match_mode=val_cfg.get("match_mode", "quadrant"),
        iou_threshold=float(val_cfg.get("iou_threshold", 0.1)),
        froc_targets=val_cfg.get("froc_targets", [0.10, 0.25, 0.50]),
        threshold_steps=int(val_cfg.get("threshold_steps", 200)),
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
    validation_state: ValidationSelectionState,
    latest_validation: Optional[dict],
) -> None:
    payload = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "validation_selection_state": validation_state.to_dict(),
        "latest_validation": latest_validation,
        "experiment": f"A0_{config['view_name']}_vindr_val",
        "checkpoint_purpose": "resume_training",
    }
    atomic_torch_save(payload, output_dir / "checkpoint_last.pth")


def save_best_validation_checkpoint(
    *,
    output_dir: Path,
    model,
    config: dict,
    epoch: int,
    validation_metrics: dict,
    validation_state: ValidationSelectionState,
) -> None:
    score_std = float(validation_metrics["score_std"])
    payload = {
        "model": raw_model(model).state_dict(),
        "config": config,
        "epoch": int(epoch),
        "validation": validation_metrics,
        "validation_selection_state": validation_state.to_dict(),
        "selection_rule": {
            "primary": "cardiomegaly_auc",
            "secondary": "mean_froc_sensitivity",
            "auc_tolerance": float(
                config["vindr_validation"].get("auc_tolerance", 0.002)
            ),
            "localization_min_delta": float(
                config["vindr_validation"].get("localization_min_delta", 0.0)
            ),
        },
        "score_calibration": {
            "scope": "Cardiomegaly classification score",
            "raw_score_mean": float(validation_metrics["score_mean"]),
            "raw_score_std": score_std,
            "recommended_baseline_transform": "z = (raw_score - mean) / std",
            "validation_scores_file": "best_val_cardiomegaly_scores.npz",
            "note": (
                "This stores calibration data only. Cross-view ensemble weights "
                "must be fitted after all selected view checkpoints are available."
            ),
        },
        "experiment": f"A0_{config['view_name']}_vindr_val",
        "checkpoint_purpose": "best_vindr_validation_model_only",
    }
    atomic_torch_save(payload, output_dir / "checkpoint_best_val.pth")


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = int(args.seed) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    dataset = build_dataset(config)

    subset_size = config.get("train_subset_size")
    if subset_size is not None:
        subset_size = int(subset_size)
        if subset_size > len(dataset):
            raise ValueError(f"train_subset_size={subset_size} > {len(dataset)}")
        rng = np.random.default_rng(int(config.get("train_subset_seed", 42)))
        indices = rng.permutation(len(dataset))[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        print(f"[Dataset] Fixed subset: {subset_size}", flush=True)
    else:
        print(f"[Dataset] Full manifest: {len(dataset)} records", flush=True)

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
    model = ALBEF(
        config=config,
        text_encoder=args.text_encoder,
        tokenizer=tokenizer,
        init_deit=True,
    ).to(device)

    opt_cfg = utils.AttrDict(config["optimizer"])
    opt_cfg["lr"] = float(opt_cfg["lr"])
    optimizer = create_optimizer(opt_cfg, model)

    sched_cfg = utils.AttrDict(config["schedular"])
    for key in ("lr", "warmup_lr", "min_lr"):
        sched_cfg[key] = float(sched_cfg[key])
    scheduler, _ = create_scheduler(sched_cfg, optimizer)

    output_dir = Path(args.output_dir)
    last_path = output_dir / "checkpoint_last.pth"
    start_epoch = 0
    best_loss = float("inf")
    validation_state = ValidationSelectionState()

    if args.resume:
        resume_path = Path(args.checkpoint) if args.checkpoint else last_path
        start_epoch, best_loss, validation_state = resume_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
        )
    elif args.checkpoint:
        load_weights_only(model, args.checkpoint)
    elif args.auto_resume and last_path.exists():
        start_epoch, best_loss, validation_state = resume_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
        )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )

    # Only rank 0 owns the validation data loader and performs Grad-CAM.
    validation_runner = None
    if utils.is_main_process():
        validation_runner = build_validation_runner(config)

    val_cfg = config.get("vindr_validation", {})
    validation_enabled = bool(val_cfg.get("enabled", False))
    validate_every = max(1, int(val_cfg.get("validate_every", 1)))
    auc_tolerance = float(val_cfg.get("auc_tolerance", 0.002))
    localization_min_delta = float(val_cfg.get("localization_min_delta", 0.0))

    max_epoch = int(config["schedular"]["epochs"])
    warmup_steps = int(config["schedular"]["warmup_epochs"])
    started = time.time()

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

        # All ranks must finish the epoch before rank 0 starts validation on the
        # unwrapped model. Non-zero ranks wait at the final barrier below.
        if dist_ready():
            dist.barrier()

        if utils.is_main_process():
            total = float(stats["loss_total"])
            best_loss = min(best_loss, total)  # diagnostic only; not selection.

            validation_metrics = None
            selected_this_epoch = False
            selection_reason = "validation not run"

            should_validate = (
                validation_enabled
                and validation_runner is not None
                and ((epoch + 1) % validate_every == 0 or epoch == max_epoch - 1)
            )

            if should_validate:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                validation_obj, classification_outputs = validation_runner.evaluate(
                    model=raw_model(model),
                    tokenizer=tokenizer,
                    device=device,
                    epoch=epoch,
                    return_classification_outputs=True,
                )
                validation_metrics = validation_obj.to_dict()

                selected_this_epoch, selection_reason = validation_state.consider(
                    validation_obj,
                    auc_tolerance=auc_tolerance,
                    localization_min_delta=localization_min_delta,
                )

                if selected_this_epoch:
                    scores_path = output_dir / "best_val_cardiomegaly_scores.npz"
                    atomic_save_validation_scores_npz(
                        scores_path,
                        outputs=classification_outputs,
                        metrics=validation_obj,
                    )
                    save_best_validation_checkpoint(
                        output_dir=output_dir,
                        model=model,
                        config=config,
                        epoch=epoch,
                        validation_metrics=validation_metrics,
                        validation_state=validation_state,
                    )
                    print(
                        f"[Checkpoint] Updated checkpoint_best_val.pth at epoch "
                        f"{epoch}: {selection_reason}",
                        flush=True,
                    )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            save_last_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_loss=best_loss,
                validation_state=validation_state,
                latest_validation=validation_metrics,
            )

            row = {
                "epoch": int(epoch),
                "view_name": config["view_name"],
                **{f"train_{key}": float(value) for key, value in stats.items()},
                "best_train_loss": float(best_loss),
                "validation_ran": bool(should_validate),
                "selected_this_epoch": bool(selected_this_epoch),
                "selection_reason": selection_reason,
                "selected_epoch": int(validation_state.selected_epoch),
                "selected_auc": (
                    None
                    if not np.isfinite(validation_state.selected_auc)
                    else float(validation_state.selected_auc)
                ),
                "selected_localization_score": (
                    None
                    if not np.isfinite(validation_state.selected_localization_score)
                    else float(validation_state.selected_localization_score)
                ),
            }
            if validation_metrics is not None:
                row.update(
                    {
                        "val_cardiomegaly_auc": validation_metrics[
                            "cardiomegaly_auc"
                        ],
                        "val_cardiomegaly_best_f1": validation_metrics[
                            "cardiomegaly_best_f1"
                        ],
                        "val_cardiomegaly_best_threshold": validation_metrics[
                            "cardiomegaly_best_threshold"
                        ],
                        "val_localization_score": validation_metrics[
                            "localization_score"
                        ],
                        "val_froc": validation_metrics["froc_sensitivities"],
                        "val_score_min": validation_metrics["score_min"],
                        "val_score_max": validation_metrics["score_max"],
                        "val_score_mean": validation_metrics["score_mean"],
                        "val_score_std": validation_metrics["score_std"],
                    }
                )

            with open(output_dir / "log.txt", "a") as handle:
                handle.write(json.dumps(row) + "\n")

        if dist_ready():
            dist.barrier()

    elapsed = int(time.time() - started)
    print("Training time", str(datetime.timedelta(seconds=elapsed)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    main(args, config)
