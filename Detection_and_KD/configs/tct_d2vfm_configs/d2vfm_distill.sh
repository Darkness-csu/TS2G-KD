#!/usr/bin/env bash

set -x


DATA_ROOT=/home1/wsi/gc-filter/filter-features
DISTILL_DATASET=$1 #biomedclip
NEEDLE=$2 #1
BATCH_SIZE=$3 #10
EPOCHS=$4 #4
LR=$5 #0.002
ADAPTER=$6 #mpa
DISTILL_LOSS=$7 #MAE
EXP_DIR=$8 #exps/tct_d2vfm/${DISTILL_DATASET}-${DISTILL_LOSS}-g${ADAPTER}-ep${EPOCHS}
DETECTION_METHOD=$9 # deformable_detr, dino, dab_detr, faster_rcnn

if [ ${DETECTION_METHOD} == "dino" ]; then
    LOAD=r50_weights/dino_gc_12epoch_r50.pth
elif [ ${DETECTION_METHOD} == "deformable_detr" ]; then
    LOAD=r50_weights/deformable_detr_gc_50epoch_r50.pth
elif [ ${DETECTION_METHOD} == "dab_detr" ]; then
    LOAD=r50_weights/dab_detr_gc_50epoch_r50.pth
elif [ ${DETECTION_METHOD} == "faster_rcnn" ]; then
    LOAD=r50_weights/faster_rcnn_fpn_gc_12epoch_r50.pth

else
    echo "not supported detection method ${DETECTION_METHOD}"
fi


PY_ARGS=${@:10}

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