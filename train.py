import pdb

import logging
import argparse
import os
import random
import numpy as np
import math
import timm

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP
from timm.data.mixup import Mixup
import timm.optim.optim_factory as optim_factory

from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from models.vit import VisionTransformer, CONFIGS

logger = logging.getLogger(__name__)

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    config = CONFIGS[args.model_type]

    num_classes = 1000

    model = VisionTransformer(config, args.img_size, num_classes=num_classes)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%f" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", choices=["ViT-S_16", "ViT-T_16"],
                        default="ViT-S_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="./exp/clean", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training (per GPU).")
    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--mixup", action="store_true",
                        help="Whether to use MixUp or not.")

    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-1, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--num_epochs", default=90, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local-rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


def train(args, model):
    is_master = args.local_rank in [-1, 0]
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader, train_sampler = get_loader(args)

    def param_groups_weight_decay(
            model: torch.nn.Module,
            weight_decay=1e-4,
            no_weight_decay_list=()
    ):
        no_weight_decay_list = set(no_weight_decay_list)
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    param_groups = param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    all_stat_dict = {}
    all_stat_dict['lr'] = []
    all_stat_dict['train/loss'] = []
    all_stat_dict['val/loss'] = []
    all_stat_dict['val/prec@1'] = []

    if args.resume:
        stat_dict_pth = torch.load(os.path.join(args.resume, 'all_stat_dict.pth'))
        for k in all_stat_dict.keys():
            all_stat_dict[k] = stat_dict_pth[k]

        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    t_total = math.ceil(len(train_loader.dataset) / (args.train_batch_size * 8)) * args.num_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    loss_fct = torch.nn.CrossEntropyLoss()
    mixup_fn = Mixup(mixup_alpha=0.2, label_smoothing=0., num_classes=1000)

    if is_master:
        logger.info("***** Running training *****")
        logger.info("Total optimization steps = %d", t_total)
        logger.info("Batch size per GPU = %d", args.train_batch_size)
        logger.info("Total train batch size (w. parallel, distributed) = %d",
                    args.train_batch_size * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    print(f'start: {args.start_epoch}')
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        ep_stat_dict = {}
        ep_stat_dict['train/loss'] = []

        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        model.train()
        model.zero_grad()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            if args.mixup:
                x, y = mixup_fn(x, y)

            logits = model(x)
            loss = loss_fct(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ep_stat_dict['train/loss'].append(loss.item())

            if step == len(train_loader) - 1:
                last_lr = scheduler.get_last_lr()[0]
                all_stat_dict['lr'].append(last_lr)
            scheduler.step()

        all_stat_dict['train/loss'].append(np.asarray(ep_stat_dict['train/loss']).mean())
        if is_master:
            accuracy, eval_loss = valid(args, model, test_loader)
            all_stat_dict['val/loss'].append(eval_loss)
            all_stat_dict['val/prec@1'].append(accuracy)
            torch.save(all_stat_dict, os.path.join(args.output_dir, 'all_stat_dict.pth'))

            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(to_save, os.path.join(args.output_dir, 'checkpoint.pth'))


def valid(args, model, test_loader):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    ncorrect, loss, nseen = 0, 0, 0

    model.eval()
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            loss += eval_loss
            ncorrect += (torch.argmax(logits, dim=-1) == y).sum()
            nseen += x.shape[0]

    accuracy = 1. * ncorrect.cpu().item() / nseen
    loss = loss.cpu().item() / nseen
    return accuracy, loss


if __name__ == "__main__":
    main()
