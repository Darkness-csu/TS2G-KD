#!/usr/bin/env bash

set -x

DATA_ROOT=/home1/wsi/gc-filter/filter-features
DISTILL_DATASET=plip
LOAD=r50_weights/deformable_detr_gc_50epoch_r50.pth
NEEDLE=1
BATCH_SIZE=10
EPOCHS=4
LR=0.002
ADAPTER=p
DISTILL_LOSS=MSE
EXP_DIR=exps/tct_d2vfm/${DISTILL_DATASET}_${DISTILL_LOSS}_g${ADAPTER}_ep${EPOCHS}

PY_ARGS=${@:1}

python -u main_r50_d2vfm.py \
    --data_root ${DATA_ROOT} \
    --distill_dataset ${DISTILL_DATASET} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --output_dir ${EXP_DIR} \
    --pretrained_detector_r50_weights ${LOAD} \
    --needle ${NEEDLE} \
    --adapter_type ${ADAPTER} \
    --construct_func ${DISTILL_LOSS} \
    ${PY_ARGS}

#GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/tct_d2vfm_configs/d2vfm_distill.sh