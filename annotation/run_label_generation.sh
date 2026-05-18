#!/bin/bash
# =============================================================
# Run this on your LOCAL machine after the server is running
# =============================================================

# Set your server IP here
SERVER_IP="YOUR_SERVER_IP"
SERVER_URL="http://${SERVER_IP}:8000/v1"

# Step 1: Install openai client (only dependency needed locally)
pip install openai

# Step 2: Test server connection
echo "Testing server connection..."
curl -s http://${SERVER_IP}:8000/v1/models | python3 -m json.tool
echo ""

# Step 3: Generate training labels
# - 1000 per class = 2000 images total
# - Uses the 2-stage pipeline (Stage1: image analysis, Stage2: JSON synthesis)
# - Only keeps predictions that match ground truth
# - Saves progress to progress.jsonl (resumable)
python annotation/generate_training_labels_qwen35.py \
    --server ${SERVER_URL} \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --sample_n 1000 \
    --output_dir finetune_data_labeled

# Step 4: After label generation is done, fine-tune
# python finetune_qwen2vl.py \
#     --train_data finetune_data_labeled/train.json \
#     --val_data finetune_data_labeled/val.json \
#     --epochs 3
