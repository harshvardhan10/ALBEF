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
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

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

        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)

        loss = loss_mlm + loss_ita + loss_itm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

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
            model, data_loader, optimizer, tokenizer,
            epoch, warmup_steps, device, lr_scheduler, config
        )

        if utils.is_main_process():
            # Compute total training loss for best-model tracking
            loss_mlm = float(train_stats.get('loss_mlm', 0.0))
            loss_ita = float(train_stats.get('loss_ita', 0.0))
            loss_itm = float(train_stats.get('loss_itm', 0.0))
            train_loss_total = loss_mlm + loss_ita + loss_itm

            is_best = train_loss_total < best_loss
            if is_best:
                best_loss = train_loss_total

            log_stats = {
                'epoch': epoch,
                'train_loss_mlm': loss_mlm,
                'train_loss_ita': loss_ita,
                'train_loss_itm': loss_itm,
                'train_loss_total': train_loss_total,
                'best_loss': best_loss,
                'lr': float(train_stats.get('lr', optimizer.param_groups[0]["lr"])),
            }

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_loss': best_loss,
            }

            # epoch-specific checkpoint (as before)
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
