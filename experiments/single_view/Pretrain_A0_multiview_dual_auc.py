"""A0 ALBEF pretraining for original, lung-only, and heart-only CXR views.

Checkpoint policy
-----------------
After each configured validation epoch, the current in-memory model is evaluated
on a fixed VinDr-train multi-label validation set. Two model-only checkpoints are
maintained independently:

* checkpoint_best_cardiomegaly_auc.pth
* checkpoint_best_macro_auc_stable.pth

A full checkpoint_last.pth is overwritten each epoch only for crash recovery and
resume. By default it is deleted after successful completion, leaving exactly the
two requested best-validation checkpoints.

Early stopping is triggered only when neither Cardiomegaly AUC nor stable-label
macro AUC has improved for the configured patience, and only after min_epochs
has been reached. Macro AUC over all evaluable labels is logged but is not used
for checkpoint selection.
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
from torchvision.transforms import InterpolationMode
import yaml

from dataset import create_loader, create_sampler
from dataset.caption_dataset import pretrain_dataset
from dataset.chexmask_cached_mask_dataset import CheXmaskCachedMaskPretrainDataset
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import utils

from scripts.vindr_classification_validation import (
    DualAUCSelectionState,
    VinDrClassificationValidationRunner,
    atomic_save_scores_npz,
    atomic_torch_save,
)


ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD = (0.26862954, 0.26130258, 0.27577711)


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


def build_cxr_pretrain_transform(config):
    """Conservative augmentation that preserves global thoracic anatomy.

    No random crop and no horizontal flip are used. This avoids removing chest
    boundaries needed for Cardiomegaly and avoids contradicting laterality terms
    in the paired radiology report.
    """
    image_res = int(config["image_res"])
    aug_cfg = config.get("cxr_augmentation", {})
    enabled = bool(aug_cfg.get("enabled", True))

    operations = [
        transforms.Resize(
            (image_res, image_res),
            interpolation=Image.Resampling.BICUBIC,
        )
    ]

    if enabled:
        degrees = float(aug_cfg.get("rotation_degrees", 5.0))
        translate = float(aug_cfg.get("translate_fraction", 0.02))
        scale_min = float(aug_cfg.get("scale_min", 0.98))
        scale_max = float(aug_cfg.get("scale_max", 1.02))

        if degrees > 0 or translate > 0 or (scale_min, scale_max) != (1.0, 1.0):
            operations.append(
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=(translate, translate),
                    scale=(scale_min, scale_max),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            )

        brightness = float(aug_cfg.get("brightness", 0.10))
        contrast = float(aug_cfg.get("contrast", 0.10))
        photometric_probability = float(
            aug_cfg.get("photometric_probability", 0.5)
        )
        if brightness > 0 or contrast > 0:
            operations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=0.0,
                            hue=0.0,
                        )
                    ],
                    p=photometric_probability,
                )
            )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(ALBEF_MEAN, ALBEF_STD),
        ]
    )
    return transforms.Compose(operations)


def build_dataset(config):
    """Build original/lung/heart datasets with one identical CXR transform."""
    view_name = str(config.get("view_name", "")).lower().strip()
    if view_name not in {"original", "lung", "heart"}:
        raise ValueError(
            f"Unsupported view_name={view_name!r}; expected original, lung, or heart"
        )

    transform = build_cxr_pretrain_transform(config)
    max_words = int(config.get("max_words", 30))

    if view_name == "original":
        dataset = pretrain_dataset(
            ann_file=config["train_file"],
            transform=transform,
            max_words=max_words,
        )
        print(
            "[Dataset] Original CXR pretraining with conservative CXR transforms.",
            flush=True,
        )
        return dataset

    mask_root = config.get("mask_root")
    if not mask_root:
        raise ValueError(f"mask_root is required for view_name={view_name}")

    dataset = CheXmaskCachedMaskPretrainDataset(
        ann_files=config["train_file"],
        transform=transform,
        mask_root=mask_root,
        max_words=max_words,
        image_key=config.get("image_key", "image"),
        caption_key=config.get("caption_key", "caption"),
        mask_key=config.get("mask_key", "mask_relpath"),
    )
    print(
        f"[Dataset] {view_name}-only pretraining with the same CXR transforms.",
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
            max_length=int(config.get("max_text_len", 32)),
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

        if not torch.isfinite(loss_total):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}: "
                f"mlm={loss_mlm.item()}, ita={loss_ita.item()}, "
                f"itm={loss_itm.item()}"
            )

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

        # Preserve the original ALBEF scheduler convention.
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
) -> Tuple[int, float, DualAUCSelectionState]:
    checkpoint = torch.load(path, map_location="cpu")
    required = {"model", "optimizer", "lr_scheduler"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Cannot resume from {path}: missing {sorted(missing)}. "
            "Resume from checkpoint_last.pth."
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    state = DualAUCSelectionState.from_dict(
        checkpoint.get("validation_selection_state")
    )
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_loss", float("inf"))),
        state,
    )


def build_validation_runner(config):
    val_cfg = config.get("vindr_validation", {})
    if not bool(val_cfg.get("enabled", False)):
        return None

    required = ["labels_csv", "images_root", "view_type"]
    missing = [key for key in required if not val_cfg.get(key)]
    if missing:
        raise ValueError(
            "vindr_validation is enabled but missing: " + ", ".join(missing)
        )

    return VinDrClassificationValidationRunner(
        labels_csv=val_cfg["labels_csv"],
        images_root=val_cfg["images_root"],
        view_type=val_cfg["view_type"],
        mask_root=val_cfg.get("mask_root"),
        image_res=int(config["image_res"]),
        batch_size=int(val_cfg.get("batch_size", 64)),
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
        "experiment": f"A0_{config['view_name']}_dual_auc",
        "checkpoint_purpose": "temporary_resume_checkpoint",
    }
    atomic_torch_save(payload, output_dir / "checkpoint_last.pth")


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
        "experiment": f"A0_{config['view_name']}_dual_auc",
        "checkpoint_purpose": "model_selected_on_vindr_train_validation",
        "score_calibration": {
            "validation_scores_saved_separately": True,
            "recommended_initial_transform": (
                "per-label z-score using validation mean/std; fit ensemble "
                "weights only after all view checkpoints are selected"
            ),
        },
    }
    atomic_torch_save(payload, output_path)


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = int(args.seed) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prefer reproducibility for a controlled thesis comparison.
    cudnn.benchmark = False
    cudnn.deterministic = bool(config.get("cudnn_deterministic", True))

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
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "checkpoint_last.pth"
    start_epoch = 0
    best_loss = float("inf")
    selection_state = DualAUCSelectionState()

    if args.resume:
        resume_path = Path(args.checkpoint) if args.checkpoint else last_path
        start_epoch, best_loss, selection_state = resume_checkpoint(
            resume_path, model, optimizer, scheduler
        )
    elif args.checkpoint:
        load_weights_only(model, args.checkpoint)
    elif args.auto_resume and last_path.exists():
        start_epoch, best_loss, selection_state = resume_checkpoint(
            last_path, model, optimizer, scheduler
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
    validate_every = max(1, int(val_cfg.get("validate_every", 1)))
    cardio_min_delta = float(val_cfg.get("cardiomegaly_min_delta", 0.0005))
    macro_min_delta = float(val_cfg.get("macro_min_delta", 0.0005))

    early_cfg = val_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_patience = max(1, int(early_cfg.get("patience", 5)))
    early_min_epochs = max(1, int(early_cfg.get("min_epochs", 15)))

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
                best_loss = min(best_loss, total)  # diagnostic only

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

                    improved_cardio, improved_macro = selection_state.consider(
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
                            output_dir / "best_cardiomegaly_auc_scores.npz",
                            outputs=outputs,
                            metrics=metrics_obj,
                        )
                        print(
                            f"[Checkpoint] New best Cardiomegaly AUC at "
                            f"epoch={epoch}: {metrics_obj.cardiomegaly_auc:.6f}",
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
                            output_dir / "best_macro_auc_stable_scores.npz",
                            outputs=outputs,
                            metrics=metrics_obj,
                        )
                        print(
                            f"[Checkpoint] New best stable-label macro AUC at "
                            f"epoch={epoch}: "
                            f"{metrics_obj.macro_auc_stable:.6f}",
                            flush=True,
                        )

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    stop_training = bool(
                        early_enabled
                        and (epoch + 1) >= early_min_epochs
                        and selection_state.validations_without_improvement
                        >= early_patience
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
                    "view_name": config["view_name"],
                    **{
                        f"train_{key}": float(value)
                        for key, value in stats.items()
                    },
                    "best_train_loss": float(best_loss),
                    "validation_ran": bool(should_validate),
                    "improved_cardiomegaly_auc": bool(improved_cardio),
                    "improved_macro_auc_stable": bool(improved_macro),
                    "best_cardiomegaly_auc": (
                        None
                        if not np.isfinite(selection_state.best_cardiomegaly_auc)
                        else float(selection_state.best_cardiomegaly_auc)
                    ),
                    "best_cardiomegaly_epoch": int(
                        selection_state.best_cardiomegaly_epoch
                    ),
                    "best_macro_auc_stable": (
                        None
                        if not np.isfinite(selection_state.best_macro_auc)
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
                            "val_macro_auc_all_evaluable": validation_metrics[
                                "macro_auc_all_evaluable"
                            ],
                            "val_micro_auc": validation_metrics["micro_auc"],
                            "val_num_valid_auc_labels": validation_metrics[
                                "num_valid_auc_labels"
                            ],
                            "val_per_label_auc": validation_metrics[
                                "per_label_auc"
                            ],
                            "val_skipped_auc_labels": validation_metrics[
                                "skipped_auc_labels"
                            ],
                            "val_cardiomegaly_best_f1": validation_metrics[
                                "cardiomegaly_best_f1"
                            ],
                            "val_cardiomegaly_best_threshold": validation_metrics[
                                "cardiomegaly_best_threshold"
                            ],
                            "val_cardiomegaly_score_mean": validation_metrics[
                                "cardiomegaly_score_mean"
                            ],
                            "val_cardiomegaly_score_std": validation_metrics[
                                "cardiomegaly_score_std"
                            ],
                        }
                    )

                with open(output_dir / "log.txt", "a") as handle:
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
                        f"[Early stopping] Neither Cardiomegaly AUC nor "
                        f"stable-label macro AUC improved for "
                        f"{early_patience} validations after min_epochs="
                        f"{early_min_epochs}.",
                        flush=True,
                    )
                break

        completed_cleanly = True
    finally:
        if (
            completed_cleanly
            and utils.is_main_process()
            and not bool(config.get("retain_last_checkpoint_after_success", False))
            and last_path.exists()
        ):
            last_path.unlink()
            print(
                "[Checkpoint] Deleted checkpoint_last.pth after successful "
                "completion; the two best model checkpoints remain.",
                flush=True,
            )

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
