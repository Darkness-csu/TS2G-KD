#!/usr/bin/env bash

#VFM=biomed2 # clip1 biomed2 plip1 gigapath1 dino1 mae1
#NUM_DIM=768 # 512 768 1536
D2VFM=gigapath_MAE_gmpa_ep6
DATASET=gc

FEATURE_NAME=${D2VFM}
SEED=2026 # 2024 2025 2026

python benchmark.py ${FEATURE_NAME} --dataset=${DATASET} --seed=${SEED} --gpus 0 1