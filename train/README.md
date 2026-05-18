# Stage 2 — Training

LoRA fine-tuning of Qwen2-VL-2B-Instruct.

## Usage

```bash
bash run_finetune_h200.sh qwen35_64k    # multi-GPU H200
bash run_finetune.sh      qwen35_64k    # single-GPU
```

## Scripts

| Script                       | Purpose |
| ---------------------------- | ------- |
| `finetune_qwen2vl_stage2.py` | Main training entry point. |
| `merge_lora.py`              | Merge the trained LoRA adapter into the base model. |
| `run_finetune.sh`            | Single-GPU launcher. |
| `run_finetune_h200.sh`       | Distributed launcher (3–4× H200). |

## Output

By default, runs write to `output/finetune/<run_name>/`. The final adapter
lives at `output/finetune/<run_name>/final/`. Pass that path to
`evaluate/run_test.sh` for host-side scoring, or to `merge_lora.py` to
produce a merged model for quantization.
