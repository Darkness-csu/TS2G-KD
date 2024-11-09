# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build
from .deformable_detr_IF import build_IF
from .deformable_detr_IF_v2 import build_IF_V2
from .r50_d2vfm import build_r50_d2vfm

def build_model(args):
    return build(args)

def build_model_IF(args):
    return build_IF(args)

def build_model_IF_V2(args):
    return build_IF_V2(args)

def build_model_R50_D2VFM(args):
    return build_r50_d2vfm(args)
