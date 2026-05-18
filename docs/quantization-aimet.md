# Quantization (AIMET → QNN)

We use **Qualcomm AIMET Pro** to quantize the merged Qwen2-VL-2B-Instruct
model to **W4A16** and export a **QNN binary** for the Snapdragon 8 Gen 5
mobile device target.

Toolkit documentation: <https://quic.github.io/aimet-pages/>

## Files in `quantize/`

| File                                | Purpose |
| ----------------------------------- | ------- |
| `quantize/prepare_calibration_data.py` | Produces a calibration JSON from the same distribution as training (mixed real + fake). |
| `quantize/prepare_submission.py`       | Packages the QNN binary + tokenizer + required `.raw` tensors into the LPCVC submission zip. |
| `quantize/run_submit.sh`               | Orchestrates the final packaging step. |

## Workflow

1. Merge the LoRA adapter into the base model:
   ```bash
   python train/merge_lora.py \
       --lora_path <path-to-final-lora> \
       --output_dir merged
   ```
2. Run the AIMET quantization workflow against `merged/` per the
   AIMET documentation (W4A16 weights, QNN export target).
3. Package the resulting binary:
   ```bash
   bash quantize/run_submit.sh OptimAI
   ```

The exact AIMET command sequence is toolkit-specific and is covered by
the upstream AIMET docs.
