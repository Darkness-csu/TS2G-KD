#!/usr/bin/env bash

set -x

EXP_DIR=exps/tct_IF_v2/clip/r50_deformable_detr_240520
LOAD=exps/tct/r50_deformable_detr_240420/checkpoint0049.pth
VFM='clip'
PY_ARGS=${@:1}

python -u main_tct_IF_v2.py \
    --output_dir ${EXP_DIR} \
    --load ${LOAD} \
    --extractor ${VFM} \
    ${PY_ARGS}
