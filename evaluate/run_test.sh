#!/bin/bash
set -e
# Test model and evaluate (full 3-step pipeline)
#
# Pipeline:
#   1. Inference         - Run model on 50 sample images (test_qwen2vl.py)
#   2. Basic eval        - Detection + per-criterion accuracy (evaluate.py)
#   3. Server estimate   - Add Evidence semantic similarity → predicted server score (evaluate_server_score.py)
#
# Usage:
#   bash run_test.sh [model_path] [output_name]
#   bash run_test.sh output/finetune/{run_name}/final results_my_run
#   bash run_test.sh Qwen/Qwen2-VL-2B-Instruct baseline
#   bash run_test.sh best_weights/{run_name}/final results_best
#
# Skip server score estimate (faster, no sentence-transformers needed):
#   SKIP_SERVER_SCORE=1 bash run_test.sh ...

MODEL=${1:-output/finetune/my_run/final}
OUTPUT_NAME=${2:-local_test}
RESULTS_DIR="output/eval/$OUTPUT_NAME"
ANNOTATION="dataset/sample_dataset/annotation.json"

echo "================================================================"
echo "Model:   $MODEL"
echo "Output:  $RESULTS_DIR"
echo "================================================================"
echo ""

# Step 1: Inference
echo "[Step 1/3] Running inference on 50 sample images..."
python evaluate/test_qwen2vl.py \
    --model "$MODEL" \
    --dataset_dir dataset/sample_dataset \
    --output_dir "$RESULTS_DIR"

python annotation/reparse_results.py "$RESULTS_DIR"

# Step 2: Basic evaluation (Detection + Criterion accuracy)
echo ""
echo "[Step 2/3] Evaluating Detection + Per-Criterion accuracy..."
python evaluate/evaluate.py \
    --results_dir "$RESULTS_DIR" \
    --annotation_path "$ANNOTATION" \
    --postprocess

# Step 3: Server score estimate (adds Evidence semantic similarity)
if [ "$SKIP_SERVER_SCORE" = "1" ]; then
    echo ""
    echo "[Step 3/3] SKIPPED (SKIP_SERVER_SCORE=1)"
else
    echo ""
    echo "[Step 3/3] Estimating server score (Evidence similarity + combined formula)..."
    python evaluate/evaluate_server_score.py \
        --results_dir "$RESULTS_DIR" \
        --annotation_path "$ANNOTATION" \
        --postprocess
fi

echo ""
echo "================================================================"
echo "Done! Results in: $RESULTS_DIR"
echo "================================================================"
