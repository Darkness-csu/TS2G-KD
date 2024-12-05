# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset_d2vfm
from engine import train_one_epoch_r50_d2vfm
from models import build_r50_d2vfm


def get_args_parser():
    parser = argparse.ArgumentParser('R50 D2VFM Distillation', add_help=False)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--lr_drop', default=2, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')


    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--construct_func', default=['MSE'], type=str, choices=['MSE', 'MAE', 'Cosine'])
    parser.add_argument('--construct_loss_coef', default=2, type=float) #-->construct loss

    # dataset parameters
    parser.add_argument('--distill_dataset', default='biomedclip', choices=['biomedclip','clip','plip','gigapath','dinov2'])
    parser.add_argument('--data_root', default='/home/commonfile/wsi/gc-filter/filter-features', type=str) 
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #new
    parser.add_argument('--pretrained_detector_r50_weights', default='', help='load r50 weights from pth')
    
    #adapter adapter_type
    parser.add_argument('--needle', default=1, type=int)
    parser.add_argument('--adapter_type', default=['mpa'], type=str, choices=['p', 'm', 'ma', 'mp', 'mpa'])
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_r50_d2vfm(args)
    model.to(device)
    model_without_ddp = model
    #new

    for name, layer in model_without_ddp.named_modules():
        if 'adapter' not in name:
            layer.eval()
    
    for name, param in model_without_ddp.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
   
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            print(name)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, n_parameters))
    # print('number of params:', n_parameters)

    dataset_train = build_dataset_d2vfm(args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if p.requires_grad],
            "lr": args.lr,
        },
        
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if not p.requires_grad],
            "lr": args.lr * 0,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    
    output_dir = Path(args.output_dir)
    
    #load resnet50 weights from detector's backbone
    pretrained_detector_r50_weights = args.pretrained_detector_r50_weights
    if not os.path.exists(pretrained_detector_r50_weights):
        raise ValueError(f'Wrong path for pretrained_detector_r50_weights:{pretrained_detector_r50_weights}')
    
    r50_weights_dict = torch.load(pretrained_detector_r50_weights, map_location=device)
    
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(r50_weights_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    print('Successfully loading pretrained weights.')
    
    #Training
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_r50_d2vfm(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('D2VFM Distillation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
