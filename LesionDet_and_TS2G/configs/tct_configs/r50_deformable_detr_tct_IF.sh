#!/usr/bin/env bash

set -x

EXP_DIR=exps/tct_IF/r50_deformable_detr_cosine_240526
LOAD=exps/tct/r50_deformable_detr_240420/checkpoint0049.pth
VFM='biomed'
OUT_DIM=768
CONSTRUCT='Cosine'
PY_ARGS=${@:1}

python -u main_tct_IF.py \
    --output_dir ${EXP_DIR} \
    --load ${LOAD} \
    --extractor ${VFM} \
    --adapter_out_dim ${OUT_DIM} \
    --construct_func ${CONSTRUCT} \
    ${PY_ARGS}
