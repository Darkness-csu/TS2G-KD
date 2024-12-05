#!/usr/bin/env bash

set -x

DISTILL_DATASET=plip # 'biomedclip','clip','plip','gigapath'
NUM_DIM=512 # 512 768 1536
EPOCHS=4 
ADAPTER=mpa # 'p', 'm', 'ma', 'mp', 'mpa'
DISTILL_LOSS=MAE # 'MSE', 'MAE', 'Cosine'

DATASET=gc

FEATURE_NAME=${DISTILL_DATASET}_${DISTILL_LOSS}_g${ADAPTER}_ep${EPOCHS}
SEED=2024 # 2024 2025 2026
cd MIL
python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --num_dim=${NUM_DIM} --seed=${SEED} --cpus 0 1 --test_mode