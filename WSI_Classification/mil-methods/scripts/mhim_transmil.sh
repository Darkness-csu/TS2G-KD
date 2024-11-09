#!/bin/bash


# echo "提取特征方式 ：$1"
# echo "VFM ：$2"
# echo "输入维度 ：$3"
# echo "蒸馏损失 ：$4"
echo "阈值 ：$1"
GPU_ID=3

FEATURE_NAME=plip_vit_b-32
#VFM=$2
NUM_DIM=512
#DLOSS=$4
THR=$1

cd ../
DATASET_PATH=/home1/lgj/TCT_2625/VFM_extracted/result-final-tct-features/$FEATURE_NAME
#DATASET_PATH=/home1/lgj/TCT_2625/deformable_adapter_extracted_$DLOSS/$VFM/$FEATURE_NAME
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=$FEATURE_NAME-mhim_transmil-gc-trainval
#TITLE_NAME=deformable_adapter_extracted_$DLOSS-by_$VFM-$FEATURE_NAME-mhim_transmil-gc-trainval
TEACHER_INIT=./output-model/mil-methods/$FEATURE_NAME-transmil-gc-trainval
#TEACHER_INIT=./output-model/mil-methods/deformable_adapter_extracted-by_$VFM-$FEATURE_NAME-transmil-gc-trainval
#CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH  --label_path=$LABEL_PATH --datasets=gc --cv_fold=1 --input_dim=$NUM_DIM --teacher_init=$TEACHER_INIT --train_val --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title=$TITLE_NAME --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2024

CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$TITLE_NAME
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$NUM_DIM --model=pure --baseline=selfattn --seed=2024 --threshold=$THR