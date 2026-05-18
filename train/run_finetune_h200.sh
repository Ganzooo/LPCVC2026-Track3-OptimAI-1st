#!/bin/bash
# Distributed fine-tuning on 4× H200 (140GB each).
#
# Usage:
#   bash run_finetune_h200.sh                  # Default: qwen35_18k
#   bash run_finetune_h200.sh qwen35_18k       # Full 18k dataset
#   bash run_finetune_h200.sh qwen35_v2        # 1.9k + sample_dataset annotations
#   bash run_finetune_h200.sh qwen35_18k --epochs 10

set -e

DATASET_NAME=${1:-qwen35_18k}
shift 2>/dev/null

case $DATASET_NAME in
    qwen35)
        TRAIN_DATA="dataset/finetune_qwen35/train.json"
        VAL_DATA="dataset/finetune_qwen35/val.json"
        ;;
    qwen35_v2)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val.json"
        ;;
    qwen35_18k)
        TRAIN_DATA="dataset/finetune_qwen35_18k/train.json"
        VAL_DATA="dataset/finetune_qwen35_18k/val.json"
        ;;
    qwen35_18kv2)
        TRAIN_DATA="dataset/finetune_qwen35_18kv2/train.json"
        VAL_DATA="dataset/finetune_qwen35_18kv2/val.json"
        ;;
    qwen35_40k)
        TRAIN_DATA="dataset/finetune_qwen35_40k/train.json"
        VAL_DATA="dataset/finetune_qwen35_40k/val.json"
        ;;
    qwen35_40kv2)
        TRAIN_DATA="dataset/finetune_qwen35_40kv2/train.json"
        VAL_DATA="dataset/finetune_qwen35_40kv2/val.json"
        ;;
    qwen35_40k_cri)
        TRAIN_DATA="dataset/finetune_qwen35_40k_cri/train.json"
        VAL_DATA="dataset/finetune_qwen35_40k_cri/val.json"
        ;;
    qwen35_60k)
        TRAIN_DATA="dataset/finetune_qwen35_60k/train.json"
        VAL_DATA="dataset/finetune_qwen35_60k/val.json"
        ;;
    qwen35_64k)
        TRAIN_DATA="dataset/finetune_qwen35_64k/train.json"
        VAL_DATA="dataset/finetune_qwen35_64k/val.json"
        ;;
    qwen35_18k_rich)
        TRAIN_DATA="dataset/finetune_qwen35_18k_rich/train.json"
        VAL_DATA="dataset/finetune_qwen35_18k_rich/val.json"
        ;;
    qwen35_18k_rich_compact)
        TRAIN_DATA="dataset/finetune_qwen35_18k_rich_compact/train.json"
        VAL_DATA="dataset/finetune_qwen35_18k_rich_compact/val.json"
        ;;
    qwen35_v2_stage1)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train_stage1.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val_stage1.json"
        ;;
    qwen35_v2_stage2)
        TRAIN_DATA="dataset/finetune_qwen35_v2/train_stage2.json"
        VAL_DATA="dataset/finetune_qwen35_v2/val_stage2.json"
        ;;
    *)
        echo "Unknown dataset: $DATASET_NAME"
        echo "Usage: bash run_finetune_h200.sh [qwen35|qwen35_v2|qwen35_18k|qwen35_18kv2|qwen35_18k_rich|qwen35_40k|qwen35_40kv2|qwen35_40k_cri|qwen35_60k|qwen35_64k|qwen35_v2_stage1|qwen35_v2_stage2]"
        exit 1
        ;;
esac

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training file not found: $TRAIN_DATA"
    exit 1
fi

echo "================================================================"
echo "Distributed H200 training"
echo "================================================================"
echo "  GPUs: 4,5,6,7 (4× H200, 140GB each)"
echo "  Dataset: $DATASET_NAME"
echo "  Train: $TRAIN_DATA"
echo "  Val: $VAL_DATA"
echo ""

# Count samples for time estimate
TRAIN_SIZE=$(python3 -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
echo "  Training samples: $TRAIN_SIZE"
echo ""

# Set distributed env
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
# PyTorch 2.x SDPA flags
export TORCH_CUDNN_V8_API_ENABLED=1
# Avoid NCCL hang on some systems
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train/finetune_qwen2vl_stage2.py \
    --dataset_name ${DATASET_NAME}_h200 \
    --train_data ${TRAIN_DATA} \
    --val_data ${VAL_DATA} \
    --epochs 5 \
    --lr 3e-4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --batch_size 6 \
    --gradient_accumulation 1 \
    --max_length 2048 \
    --attn_impl sdpa \
    --stage1_aux_cls \
    --stage1_aux_cls_weight 0.05 \
    --label_smoothing 0.05 \
    "$@"
