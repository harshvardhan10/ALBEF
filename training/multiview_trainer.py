"""One configurable multiview trainer; legacy scripts select fusion through configuration.

Loss, scheduler, initialization, resume and dual-checkpoint selection follow the
original source. Console diagnostics are standardized across fusion variants.
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
from torch.utils.data import Subset
import utils
from training.cli import str2bool
from training.multiview_checkpoints import (
    raw_model, _load_checkpoint_state, _substate, _load_single_view_visual_encoder,
    _load_shared_from_single_view_checkpoint, initialize_from_three_single_view_checkpoints,
    _interpolate_fused_positional_embeddings, load_fused_weights_only, resume_checkpoint,
    _resolve_initial_checkpoint,
)
from evaluation.checkpoint_roles import DualAUCSelectionState
from evaluation.artifacts import atomic_torch_save, atomic_save_scores_npz
from evaluation.experiment_metadata import repository_state


def build_dataset(config):
    from dataset.multiview_cxr_dataset import build_dataset as build
    return build(config)


def experiment_metadata(config):
    fusion = config["view_fusion_type"]
    descriptions = {
        "fc": "3x ViT-B/16 -> token concat -> Linear(2304,768) -> LayerNorm -> shared ALBEF ITC+ITM+MLM",
        "mean": "3x ViT-B/16 -> tokenwise arithmetic mean -> shared ALBEF ITC+ITM+MLM",
        "gated": "3x ViT-B/16 -> sample-specific softmax view gates -> LayerNorm -> shared ALBEF ITC+ITM+MLM",
        "residual": "3x ViT-B/16 -> original-anchored gated residual fusion -> shared ALBEF ITC+ITM+MLM",
        "transformer": "3x ViT-B/16 -> per-position 3-view sequence -> 2-layer TransformerEncoder -> mean pool views -> LayerNorm -> shared ALBEF ITC+ITM+MLM",
        "cross_attention": "3x ViT-B/16 -> original query + [lung,heart] memory per position -> 2-layer anchored cross-attention Transformer -> fused original token -> shared ALBEF ITC+ITM+MLM",
    }
    experiment_names = {"fc": "A0_multiview_fc_fusion_dual_auc", "transformer": "A0_multiview_transformer2_mean_fusion_dual_auc"}
    if fusion not in descriptions:
        raise ValueError("Unknown fusion: " + str(fusion))
    return {"name": experiment_names.get(fusion, "A0_multiview_" + fusion + "_fusion_dual_auc"), "architecture": descriptions[fusion]}


def experiment_name(config):
    return experiment_metadata(config)["name"]


def cleanup_last_checkpoint(output_dir, config):
    output_dir = Path(output_dir)
    last = output_dir/"checkpoint_last.pth"
    if not last.exists() or config.get("retain_last_checkpoint_after_success", False):
        return
    guarded = config["view_fusion_type"] in ("mean", "gated", "residual")
    if guarded and not all((output_dir/name).exists() for name in ("checkpoint_best_cardiomegaly_auc.pth", "checkpoint_best_macro_auc_stable.pth")):
        print("[WARNING] Retaining checkpoint_last.pth because validation-selected best checkpoints are missing.", flush=True)
        return
    last.unlink()
    print("[Checkpoint] Deleted checkpoint_last.pth after successful completion; best checkpoints remain.", flush=True)


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


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

    header = f"{experiment_name(config)} Epoch [{epoch}]"
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


def build_validation_runner(config):
    from scripts_new.vindr_multiview_classification_validation import VinDrMultiViewClassificationValidationRunner
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
        "provenance": config.get("_run_metadata", repository_state()),
        "experiment": experiment_name(config),
        "checkpoint_purpose": "temporary_resume_checkpoint",
        "architecture": (
            experiment_metadata(config)["architecture"]
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
        "provenance": config.get("_run_metadata", repository_state()),
        "checkpoint_role": Path(output_path).stem.replace("checkpoint_", ""),
        "validation_selection_state": selection_state.to_dict(),
        "experiment": experiment_name(config),
        "checkpoint_purpose": (
            "multiview_fused_model_selected_on_vindr_train_validation"
        ),
        "architecture": (
            experiment_metadata(config)["architecture"]
        ),
    }
    atomic_torch_save(payload, output_path)


def main(args, config):
    from dataset import create_loader, create_sampler
    from models.model_pretrain_multiview import ALBEF
    from models.tokenization_bert import BertTokenizer
    from optim import create_optimizer
    from scheduler import create_scheduler
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
                            metadata={"checkpoint_role": "best_cardiomegaly_auc", "resolved_config": config},
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
                            metadata={"checkpoint_role": "best_macro_auc_stable", "resolved_config": config},
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
                    "experiment": experiment_name(config),
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
        if completed_cleanly and utils.is_main_process():
            cleanup_last_checkpoint(output_dir, config)

    elapsed = int(time.time() - started)
    print(
        "Training time",
        str(datetime.timedelta(seconds=elapsed)),
        flush=True,
    )
