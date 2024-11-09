#!/bin/bash


# echo "提取特征方式 ：$1"
# echo "VFM ：$2"
# echo "输入维度 ：$3"
# echo "蒸馏损失 ：$4"
# echo "阈值 ：$5"
GPU_ID=1

#FEATURE_NAME=plip_vit_b-32
FEATURE_NAME=plip_MAE_gmp_ep4
NUM_DIM=512


cd ../
#DATASET_PATH=/home1/lgj/TCT_2625/VFM_extracted/result-final-tct-features/$FEATURE_NAME
#DATASET_PATH=/home1/lgj/TCT_2625/deformable_adapter_extracted_$DLOSS/$VFM/$FEATURE_NAME
DATASET_PATH=/home1/wsi/gc-all-features/distill/$FEATURE_NAME
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=$FEATURE_NAME-dsmil-gc-trainval
#TITLE_NAME=deformable_adapter_extracted_$DLOSS-by_$VFM-$FEATURE_NAME-abmil-gc-trainval
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=gc --input_dim=$NUM_DIM --cv_fold=1 --title=$TITLE_NAME --model=pure --baseline=dsmil --train_val --seed=2024

CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$TITLE_NAME
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --ckp_path=$CHECKPOINT_PATH --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$NUM_DIM --model=pure --baseline=dsmil --seed=2024
