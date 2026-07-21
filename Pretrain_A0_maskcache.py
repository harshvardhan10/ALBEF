"""
Standard A0 ALBEF pretraining using compact cached binary masks.

A one-bit cached mask is applied to the original PIL image before the
unchanged ALBEF RandomResizedCrop / horizontal flip / RandAugment pipeline.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
import yaml

from dataset import create_loader, create_sampler
from dataset.randaugment import RandomAugment
from ALBEF.dataset.chexmask_cached_mask_dataset import CheXmaskCachedMaskPretrainDataset
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
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def raw_model(model):
    return model.module if hasattr(model, "module") else model


def build_dataset(config):
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

    return CheXmaskCachedMaskPretrainDataset(
        ann_files=config["train_file"],
        transform=pretrain_transform,
        mask_root=config["mask_root"],
        max_words=int(config.get("max_words", 30)),
        image_key=config.get("image_key", "image"),
        caption_key=config.get("caption_key", "caption"),
        mask_key=config.get("mask_key", "mask_relpath"),
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
            print(f"[DEBUG] view={config['view_name']}")
            print(f"[DEBUG] image={tuple(image.shape)}")
            print(f"[DEBUG] loss_total={loss_total.item():.6f}")

        if epoch == 0 and step % step_size == 0 and step <= warmup_iterations:
            scheduler.step(step // step_size)

        debug_batches = config.get("debug_max_batches")
        if debug_batches is not None and step + 1 >= int(debug_batches):
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
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

    print(model.load_state_dict(state, strict=False))


def resume_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_loss", float("inf"))),
    )


def save_checkpoint(
    output_dir,
    model,
    optimizer,
    scheduler,
    config,
    epoch,
    best_loss,
    is_best,
):
    obj = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "experiment": f"A0_maskcache_{config['view_name']}",
    }
    torch.save(obj, output_dir / "checkpoint_last.pth")
    if is_best:
        torch.save(obj, output_dir / "checkpoint_best.pth")


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
        print(f"[Dataset] Fixed subset: {subset_size}")
    else:
        print(f"[Dataset] Full manifest: {len(dataset)} records")

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

    if args.resume:
        resume_path = Path(args.checkpoint) if args.checkpoint else last_path
        start_epoch, best_loss = resume_checkpoint(
            resume_path, model, optimizer, scheduler
        )
    elif args.checkpoint:
        load_weights_only(model, args.checkpoint)
    elif args.auto_resume and last_path.exists():
        start_epoch, best_loss = resume_checkpoint(
            last_path, model, optimizer, scheduler
        )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )

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

        if utils.is_main_process():
            total = float(stats["loss_total"])
            is_best = total < best_loss
            best_loss = min(best_loss, total)

            save_checkpoint(
                output_dir,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_loss,
                is_best,
            )

            row = {
                "epoch": epoch,
                "view_name": config["view_name"],
                **{f"train_{k}": float(v) for k, v in stats.items()},
                "best_loss": best_loss,
            }
            with open(output_dir / "log.txt", "a") as handle:
                handle.write(json.dumps(row) + "\n")

        if dist_ready():
            dist.barrier()

    elapsed = int(time.time() - started)
    print("Training time", str(datetime.timedelta(seconds=elapsed)))


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
