"""
Train one independent A0 ALBEF model for one CXR view.

Run this same script separately for:
    - original CXR
    - lung-only CXR
    - bone-suppressed CXR
    - heart-only CXR

The selected view is controlled only by the JSON manifest referenced by
`train_file` in the YAML config. No multi-view forward pass is performed.

This preserves the standard A0 objective:

    loss = loss_mlm + loss_ita + loss_itm

and saves only:
    checkpoint_best.pth
    checkpoint_last.pth
    config.yaml
    log.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.utils.data import Subset

from dataset import create_dataset, create_loader, create_sampler
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import utils


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch: int,
    warmup_steps: int,
    device: torch.device,
    scheduler,
    config: Dict[str, Any],
    args,
):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_mlm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_total", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    view_name = str(config.get("view_name", "unspecified"))
    header = f"A0 {view_name} | Train Epoch: [{epoch}]"

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

        # Standard A0 ALBEF forward.
        loss_mlm, loss_ita, loss_itm = model(
            image,
            text_input,
            alpha=alpha,
        )
        loss_total = loss_mlm + loss_ita + loss_itm

        loss_total.backward()

        grad_clip_norm = config.get("grad_clip_norm", None)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip_norm),
            )

        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_total=loss_total.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if step == 0 and utils.is_main_process():
            print(f"[DEBUG:A0-{view_name}] image shape: {tuple(image.shape)}")
            print(
                f"[DEBUG:A0-{view_name}] text shape: "
                f"{tuple(text_input.input_ids.shape)}"
            )
            print(f"[DEBUG:A0-{view_name}] loss_mlm: {loss_mlm.item():.6f}")
            print(f"[DEBUG:A0-{view_name}] loss_ita: {loss_ita.item():.6f}")
            print(f"[DEBUG:A0-{view_name}] loss_itm: {loss_itm.item():.6f}")
            print(f"[DEBUG:A0-{view_name}] total: {loss_total.item():.6f}")

        if (
            epoch == 0
            and step % step_size == 0
            and step <= warmup_iterations
        ):
            scheduler.step(step // step_size)

        debug_max_batches = config.get("debug_max_batches", None)
        if (
            debug_max_batches is not None
            and step + 1 >= int(debug_max_batches)
        ):
            if utils.is_main_process():
                print(
                    f"[DEBUG:A0-{view_name}] Stopping after "
                    f"{step + 1} batches."
                )
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {
        name: meter.global_avg
        for name, meter in metric_logger.meters.items()
    }


def load_weights_only(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    if "visual_encoder.pos_embed" in state_dict:
        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"],
            model.visual_encoder,
        )
    if "visual_encoder_m.pos_embed" in state_dict:
        state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"],
            model.visual_encoder_m,
        )

    message = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] Weight-only load: {message}")


def resume_from_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))

    print(
        f"[Resume] {checkpoint_path} -> epoch={start_epoch}, "
        f"best_loss={best_loss:.6f}"
    )
    return start_epoch, best_loss


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    config,
    epoch: int,
    best_loss: float,
    is_best: bool,
):
    save_obj = {
        "model": model_without_ddp(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "view_name": str(config.get("view_name", "unspecified")),
        "experiment": "A0_single_view",
    }

    torch.save(save_obj, output_dir / "checkpoint_last.pth")
    if is_best:
        torch.save(save_obj, output_dir / "checkpoint_best.pth")


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = int(args.seed) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    view_name = str(config.get("view_name", "unspecified"))
    max_epoch = int(config["schedular"]["epochs"])
    warmup_steps = int(config["schedular"]["warmup_epochs"])

    print("=" * 80)
    print(f"[A0] View: {view_name}")
    print(f"[A0] train_file: {config['train_file']}")
    print(f"[A0] subset size: {config.get('train_subset_size')}")
    print(f"[A0] subset seed: {config.get('train_subset_seed', 42)}")
    print("=" * 80)

    # The manifest selects the view. The regular pretrain dataset remains unchanged.
    datasets = [create_dataset("pretrain", config)]

    train_subset_size = config.get("train_subset_size", None)
    if train_subset_size is not None:
        train_subset_size = int(train_subset_size)
        full_size = len(datasets[0])

        if train_subset_size > full_size:
            raise ValueError(
                f"train_subset_size={train_subset_size} > "
                f"full dataset size={full_size}"
            )

        subset_seed = int(config.get("train_subset_seed", 42))
        rng = np.random.default_rng(subset_seed)
        subset_indices = rng.permutation(full_size)[:train_subset_size].tolist()
        datasets = [Subset(datasets[0], subset_indices)]

        if utils.is_main_process():
            print(
                f"[Dataset] Fixed subset {train_subset_size}/{full_size}, "
                f"seed={subset_seed}"
            )

    if args.distributed:
        samplers = create_sampler(
            datasets,
            [True],
            utils.get_world_size(),
            utils.get_rank(),
        )
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

    model = ALBEF(
        config=config,
        text_encoder=args.text_encoder,
        tokenizer=tokenizer,
        init_deit=True,
    ).to(device)

    arg_opt = utils.AttrDict(config["optimizer"])
    arg_opt["lr"] = float(arg_opt["lr"])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config["schedular"])
    arg_sche["lr"] = float(arg_sche["lr"])
    arg_sche["warmup_lr"] = float(arg_sche["warmup_lr"])
    arg_sche["min_lr"] = float(arg_sche["min_lr"])
    scheduler, _ = create_scheduler(arg_sche, optimizer)

    start_epoch = 0
    best_loss = float("inf")

    last_checkpoint = Path(args.output_dir) / "checkpoint_last.pth"

    if args.resume:
        resume_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else last_checkpoint
        )
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume requested but checkpoint not found: {resume_path}"
            )
        start_epoch, best_loss = resume_from_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
        )
    elif args.checkpoint:
        # Optional common initialization checkpoint for all views.
        load_weights_only(model, args.checkpoint)
    elif args.auto_resume and last_checkpoint.exists():
        start_epoch, best_loss = resume_from_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
        )
    else:
        print("[Checkpoint] Starting a fresh A0 run.")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )

    output_dir = Path(args.output_dir)
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            scheduler.step(epoch + warmup_steps)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            tokenizer=tokenizer,
            epoch=epoch,
            warmup_steps=warmup_steps,
            device=device,
            scheduler=scheduler,
            config=config,
            args=args,
        )

        if utils.is_main_process():
            train_loss = float(train_stats["loss_total"])
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss

            log_stats = {
                "epoch": int(epoch),
                "view_name": view_name,
                "train_loss_mlm": float(train_stats["loss_mlm"]),
                "train_loss_ita": float(train_stats["loss_ita"]),
                "train_loss_itm": float(train_stats["loss_itm"]),
                "train_loss_total": train_loss,
                "best_loss": best_loss,
                "lr": float(train_stats["lr"]),
            }

            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                best_loss=best_loss,
                is_best=is_best,
            )

            with open(output_dir / "log.txt", "a") as handle:
                handle.write(json.dumps(log_stats) + "\n")

        if distributed_ready():
            dist.barrier()

    elapsed = time.time() - start_time
    print(
        "Training time",
        str(datetime.timedelta(seconds=int(elapsed))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/Pretrain_A0_lung.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Optional initialization checkpoint. Leave empty to match a fresh "
            "A0 run; use the same initialization for every view if supplied."
        ),
    )
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--train_file_override",
        default="",
        help="Optional runtime manifest path, e.g. a node-local copy.",
    )
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the canonical, reproducible configuration. A node-local runtime
    # manifest may be injected only for faster I/O during this Slurm job.
    with open(output_dir / "config_source.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if args.train_file_override:
        config = dict(config)
        config["train_file"] = [str(Path(args.train_file_override).resolve())]
        print(f"[Runtime] train_file override: {config['train_file']}")

    with open(output_dir / "config_runtime.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    # Keep config.yaml canonical for compatibility with existing tooling.
    with open(output_dir / "config.yaml", "w") as handle:
        with open(args.config, "r") as source:
            handle.write(source.read())

    main(args, config)
