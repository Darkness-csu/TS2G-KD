# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch

import torch.utils.data
import os

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T

class D2VFM_Detection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, cache_mode=False, local_rank=0, local_size=1):
        super(D2VFM_Detection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
    
    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        path = coco.loadImgs(img_id)[0]['file_name']
        feat_p = os.path.join(self.root, path.replace('.jpg', '.pt'))
        img = self.get_image(path)
        #print(img.size)
        feat = torch.load(feat_p, map_location='cpu')
        target = {'image_id': img_id, 'annotations': []}
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, feat


def make_d2vfm_transfoms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        normalize,
    ])



#new
def build(args):
    root = Path(args.data_root)
    assert root.exists(), f'provided root {root} does not exist'
    
    distill_dataset = args.distill_dataset

    PATHS = {
        "biomedclip": (root / 'biomed2-meanmil', root / 'train1839wsi_91950_det.json'),
        "plip": (root / 'plip-meanmil', root / 'train1839wsi_91950_det.json'),
        "clip": (root / 'clip1-meanmil-50', root / 'train_distill_coco_clip.json'),
        "gigapath": (root / 'gigapath1-meanmil-50', root / 'train_distill_coco_gigapath.json'),
        "dinov2": (root / 'dino1-meanmil-50', root / 'train_distill_coco_dino.json'),
    }

    img_folder, ann_file = PATHS[distill_dataset]
    dataset = D2VFM_Detection(img_folder, ann_file, transforms=make_d2vfm_transfoms(),
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
