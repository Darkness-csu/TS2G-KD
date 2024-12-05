#!/usr/bin/env bash

set -x

EXP_DIR=exps/tct/r50_deformable_detr_240420
LOAD=pretrained_pth/r50_deformable_detr-checkpoint_tct.pth
PY_ARGS=${@:1}

python -u main_tct.py \
    --output_dir ${EXP_DIR} \
    --load ${LOAD} \
    ${PY_ARGS}
