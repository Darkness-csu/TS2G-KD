#!/usr/bin/env bash

set -x

DISTILL_DATASET=plip # 'biomedclip','clip','plip','gigapath'
NEEDLE=1
BATCH_SIZE=10
EPOCHS=4 
LR=0.002
ADAPTER=mpa # 'p', 'm', 'ma', 'mp', 'mpa'
DISTILL_LOSS=MAE # 'MSE', 'MAE', 'Cosine'
EXP_DIR=/home1/hjl/r50_weights/${DISTILL_DATASET}_${DISTILL_LOSS}_g${ADAPTER}_ep${EPOCHS}

bash_file=./configs/tct_d2vfm_configs/d2vfm_distill.sh
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ${bash_file} ${DISTILL_DATASET} ${NEEDLE} ${BATCH_SIZE} ${EPOCHS} ${LR} ${ADAPTER} ${DISTILL_LOSS} ${EXP_DIR}
