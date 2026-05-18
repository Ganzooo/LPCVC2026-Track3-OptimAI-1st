# LPCVC2026-Track3-OptimAI-1st

🥇 **1st Place, LPCVC 2026 Track 3 — AI-Generated Image Detection**
ECV Workshop @ CVPR 2026, Denver

| Model           | TPS    | Server Score |
| --------------- | ------ | ------------ |
| OptimAI (ours)  | 30.48  | **0.827**    |

[Official Results](https://lpcv.ai/2026LPCVC/winners/) · [Competition Page](https://lpcv.ai/2026LPCVC/tracks/track3/)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Released Assets](#3-released-assets)
4. [Quick Start](#4-quick-start)
5. [Reproducing the Winning Result](#5-reproducing-the-winning-result)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Acknowledgments](#8-acknowledgments)

---

## 1. Overview

A LoRA-fine-tuned Qwen2-VL-2B-Instruct model, quantized via Qualcomm AIMET
to W4A16 and exported as a QNN binary for the Snapdragon 8 Gen 5 mobile
device target.

```
raw images → annotation → train → evaluate → quantize → submit
```

---

## 2. Repository Structure

```
LPCVC2026-Track3-OptimAI-1st/
├── annotation/        # Stage 1: teacher labels + synthetic image generation
├── train/             # Stage 2: LoRA fine-tuning
├── evaluate/          # Host-side evaluation
├── quantize/          # Stage 3: AIMET → QNN packaging
├── prompts/           # Stage 1 / Stage 2 prompt templates
├── docs/              # Detailed walk-throughs
├── best_weights/      # Local-only (use HuggingFace or GitHub Release for artifacts)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 3. Released Assets

| Asset                         | Location |
| ----------------------------- | -------- |
| Final LoRA adapter            | 🤗 [`<HF_USER>/Qwen2VL-2B-AIGID-LoRA`](https://huggingface.co/) |
| Merged FP16 model (optional)  | 🤗 [`<HF_USER>/Qwen2VL-2B-AIGID-merged`](https://huggingface.co/) |
| Training dataset (JSONL)      | 🤗 [`<HF_USER>/LPCVC2026-Track3-Finetune-64k`](https://huggingface.co/datasets/) |
| Submission binary             | 📦 GitHub Release: `v1.0-lpcvc2026-winner` |

> Replace `<HF_USER>` with the team's HuggingFace handle when publishing.

---

## 4. Quick Start

### 4.1 Environment

```bash
conda create -n lpcvc_t3 python=3.10 -y && conda activate lpcvc_t3
pip install -r requirements.txt
```

### 4.2 Stage 1 — Annotation

```bash
# Launch a Qwen3.5-VL-35B-A3B teacher on a separate vLLM server (see annotation/README.md)
bash annotation/run_label_generation.sh
```

### 4.3 Stage 2 — Training

```bash
huggingface-cli download <HF_USER>/LPCVC2026-Track3-Finetune-64k \
    --repo-type dataset --local-dir dataset/finetune_qwen35_64k
bash train/run_finetune_h200.sh qwen35_64k
```

### 4.4 Host evaluation

```bash
bash evaluate/run_test.sh output/finetune/<run>/final result_my_run
```

### 4.5 Stage 3 — Quantization

```bash
python train/merge_lora.py --lora_path <lora> --output_dir merged
# Run the AIMET pipeline against the merged model — see docs/quantization-aimet.md
bash quantize/run_submit.sh OptimAI
```

---

## 5. Reproducing the Winning Result

Final training configuration:

| Parameter              | Value          |
| ---------------------- | -------------- |
| LoRA r                 | 64             |
| LoRA alpha             | 128            |
| Learning rate          | 3e-4           |
| Batch size             | 4 per GPU × 3  |
| Gradient accumulation  | 2              |
| Epochs                 | 5              |
| Label smoothing        | 0.05           |
| Attention              | sdpa           |
| Sequence length        | 2048           |
| Image resolution       | 342 × 512      |

Run `bash train/run_finetune_h200.sh qwen35_64k`. Expected server score: ~0.827.

---

## 6. License

MIT — see [LICENSE](LICENSE).

Third-party components:
- Qwen2-VL-2B-Instruct (Apache 2.0, Alibaba)
- Qualcomm AIMET (BSD-3)

---

## 7. Citation

```bibtex
@inproceedings{optimai2026lpcvc,
  title     = {OptimAI: 1st Place Solution for LPCVC 2026 Track 3},
  author    = {{OptimAI Team}},
  booktitle = {ECV Workshop @ CVPR},
  year      = {2026}
}
```

---

## 8. Acknowledgments

LPCVC organizers, Qualcomm AI Research, Alibaba Qwen team, Meta Infinity,
DeepSeek JanusPro.
