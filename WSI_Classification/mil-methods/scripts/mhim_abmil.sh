#!/bin/bash


echo "提取特征方式 ：$1"
echo "输入维度 ：$3"
GPU_ID=0

cd ../
DATASET_PATH=../extract-features/result-final-gc-features/$1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME="$1-mhim(abmil)-gc-trainval"
TEACHER_INIT=./output-model/mil-methods/$1-abmil-$2-trainval
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --label_path=$LABEL_PATH --datasets=$2 --cv_fold=1 --input_dim=$3 --teacher_init=$TEACHER_INIT --title=$TITLE_NAME --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2024 --wandb

CHECKPOINT_PATH="output-model/mil-methods/$1-mhim(abmil)-$2-trainval"
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$3 --model=pure --baseline=attn --seed=2024 
