#!/usr/bin/env bash

set -x

BS=24
WSI_DIR=/home1/lgj/TCT_smear_pt_lgj
OUT_DIR=/home1/lgj/gc-all-features/distill/test
CKP_DIR=/home/ligaojie/LungCancer/Deformable-DETR-main/exps/tct_d2vfm/plip_MSE_gp_ep4

python -m torch.distributed.launch --nproc_per_node=4 extract_features_d2vfm.py --multi_gpu \
    --batch_size ${BS} \
    --wsi_root ${WSI_DIR} \
    --output_dir ${OUT_DIR} \
    --ckp_dir ${CKP_DIR} \

