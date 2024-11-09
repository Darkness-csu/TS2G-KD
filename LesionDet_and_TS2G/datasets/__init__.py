# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .tct import build as build_tct #new
from .tct_IF import build as build_tct_IF#new
from .tct_d2vfm import build as build_tct_d2vfm

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_dataset_tct(image_set, args):
    if args.dataset_file in ['tct', 'tct_IF'] :
        return build_tct(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_dataset_tct_IF(image_set, args):
    if args.dataset_file == 'tct_IF':
        if image_set in ['train_biomed', 'train_clip', 'train_plip']:
            return build_tct_IF(image_set, args)
        elif image_set == 'val':
            return build_tct('val_IF', args)
        else:
            raise ValueError(f'{image_set} not supported')
    raise ValueError(f'dataset {args.dataset_file} not supported')
    
def build_dataset_d2vfm(args):
    return build_tct_d2vfm(args)
