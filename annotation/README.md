# Stage 1 — Annotation

Generate the teacher-labelled training dataset.

## Pipeline

1. Launch a Qwen3.5-VL-35B-A3B teacher (vLLM, separate host).
2. Run `run_label_generation.sh` from this folder to call the teacher
   across your image pool. Output: per-class JSONL annotations.
3. (Optional) Add synthetic images via `generate_janus_images.py` and
   `generate_infinity_images.py`.
4. Merge sample-dataset annotations via `add_sample_dataset.py`.
5. Split into Stage 1 / Stage 2 files with `split_by_stage.py`.

## Scripts

| Script                              | Purpose |
| ----------------------------------- | ------- |
| `generate_training_labels_qwen35.py` | Teacher labelling via OpenAI-compatible vLLM endpoint. |
| `generate_janus_images.py`           | Synthetic image generation (JanusPro). |
| `generate_infinity_images.py`        | Synthetic image generation (Infinity-2B/8B). |
| `add_generated_images_to_finetune.py`| Merge synthetic images into the training set. |
| `add_sample_dataset.py`              | Add competition sample-dataset annotations. |
| `extract_criterion_labels.py`        | Extract per-criterion binary labels from Stage 2 JSON. |
| `split_by_stage.py`                  | Produce `train_stage1.json` / `train_stage2.json`. |
| `reparse_results.py`                 | Cleanup helper for malformed JSON output. |
| `run_label_generation.sh`            | One-command launch (set `SERVER_IP` first). |

See [`docs/dataset-preparation.md`](../docs/dataset-preparation.md) for the
full end-to-end walkthrough.
