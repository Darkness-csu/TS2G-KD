#!/usr/bin/env bash

set -x

BS=16
BACKBONE=biomedclip # 'biomedclip','clip','plip','gigapath'
ADAPTER=mpa # 'p', 'm', 'ma', 'mp', 'mpa'
DISTILL_LOSS=MAE # 'MSE', 'MAE', 'Cosine'
EPOCHS=4

WSI_DIR=/home1/xxx/TCT_smear_pt_xxx
OUT_DIR=/home1/wsi/gc-all-features/distill
CKP_DIR="/home1/xxx/r50_weights/${BACKBONE}_${DISTILL_LOSS}_g${ADAPTER}_ep${EPOCHS}"

python -m torch.distributed.launch --nproc_per_node=4 extract_features_d2vfm.py --multi_gpu \
    --batch_size ${BS} \
    --wsi_root ${WSI_DIR} \
    --output_dir ${OUT_DIR} \
    --ckp_dir ${CKP_DIR} \