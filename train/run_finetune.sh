#!/bin/bash
# Fine-tune Qwen2-VL with LoRA.
#
# Defaults: r=32, alpha=64, aux cls on (weight 0.05), label smoothing 0.05
#
# Usage:
#   bash run_finetune.sh                     # Default: competition_v2 balanced data
#   bash run_finetune.sh qwen35              # Qwen3.5-generated data
#   bash run_finetune.sh qwen35_v2           # Qwen3.5 + sample dataset annotations
#   bash run_finetune.sh qwen35_v2_stage1    # Stage 1 only
#   bash run_finetune.sh qwen35_v2_stage2    # Stage 2 only
#   bash run_finetune.sh 72b_balanced        # 72B-labeled data
#   bash run_finetune.sh custom --train_data path/to/train.json --val_data path/to/val.json

DATASET_NAME=${1:-competition_v2}
shift 2>/dev/null

case $DATASET_NAME in
    competition_v2)
        TRAIN_DATA="dataset/train_competition/train_balanced.json"
        VAL_DATA="dataset/train_competition/val_balanced.json"
        ;;
    competition_v2_coco)
        TRAIN_DATA="dataset/train_competition/train_with_coco_balanced.json"
        VAL_DATA="dataset/train_competition/val_with_coco_balanced.json"
        ;;
    best)
        TRAIN_DATA="dataset/train_competition/train_best.json"
        VAL_DATA="dataset/train_competition/val_best.json"
        ;;
    72b_v2)
        TRAIN_DATA="dataset/train_competition/train_72b_v2_only.json"
        VAL_DATA="dataset/train_competition/val_72b_v2_only.json"
        ;;
    combined)
        TRAIN_DATA="dataset/train_competition/train_combined_best.json"
        VAL_DATA="dataset/train_competition/val_combined_best.json"
        ;;
    72b_balanced)
        TRAIN_DATA="dataset/finetune_72b/train_balanced.json"
        VAL_DATA="dataset/finetune_72b/val_balanced.json"
        ;;
    qwen35)
        TRAIN_DATA="dataset/finetune_qwen35/train.json"
        VAL_DATA="dataset/finetune_qwen35/val.json"
        ;;
    qwen35_v2)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val.json"
        ;;
    qwen35_v2_stage1)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train_stage1.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val_stage1.json"
        ;;
    qwen35_v2_stage2)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train_stage2.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val_stage2.json"
        ;;
    sample)
        TRAIN_DATA="dataset/finetune_sample/train.json"
        VAL_DATA="dataset/finetune_sample/val.json"
        ;;
    *)
        TRAIN_DATA=""
        VAL_DATA=""
        ;;
esac

python train/finetune_qwen2vl_stage2.py \
    --dataset_name $DATASET_NAME \
    --train_data ${TRAIN_DATA} \
    --val_data ${VAL_DATA} \
    --epochs 5 \
    --lr 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --gradient_accumulation 4 \
    --batch_size 2 \
    --stage1_aux_cls \
    --stage1_aux_cls_weight 0.05 \
    --label_smoothing 0.05 \
    "$@"
