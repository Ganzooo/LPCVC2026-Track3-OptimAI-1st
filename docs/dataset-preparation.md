# Dataset Preparation

This guide reproduces the 64k training dataset
(`dataset/finetune_qwen35_64k/`) used to fine-tune the winning model.

## Prerequisites

- Source image pool: a directory of real photographs (e.g. COCO `train2017`)
  plus AI-generated images (e.g. DiffusionDB, Chameleon, AIGIQA).
- A separate machine with ≥4× H200 or A100 80GB GPUs to host the
  Qwen3.5-VL-35B-A3B teacher.
- `vllm`, `openai` Python clients on the labelling client machine.

## Step 1 — Launch the teacher

On the teacher machine:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 1}'
```

## Step 2 — Generate base teacher labels

On the client machine, set `SERVER_IP` in `annotation/run_label_generation.sh`,
then:

```bash
bash annotation/run_label_generation.sh
```

This produces `dataset/finetune_qwen35/{train,val}.json`.

## Step 3 — Generate synthetic AI images

```bash
python annotation/generate_janus_images.py    --output_dir dataset/aigen/additional/janus
python annotation/generate_infinity_images.py --output_dir dataset/aigen/additional/infinity
```

## Step 4 — Add the synthetic images to the training set

```bash
python annotation/add_generated_images_to_finetune.py \
    --base_dataset dataset/finetune_qwen35 \
    --additional_dirs dataset/aigen/additional/janus dataset/aigen/additional/infinity \
    --output_dir dataset/finetune_qwen35_64k
```

## Step 5 — Add competition sample-dataset annotations

```bash
python annotation/add_sample_dataset.py \
    --server http://<TEACHER_IP>:8000/v1 \
    --base_train dataset/finetune_qwen35_64k/train.json \
    --base_val   dataset/finetune_qwen35_64k/val.json \
    --output_dir dataset/finetune_qwen35_64k
```

## Step 6 — Split into Stage 1 / Stage 2 training files

```bash
python annotation/split_by_stage.py \
    --input_dir dataset/finetune_qwen35_64k
```

Outputs: `train_stage1.json`, `train_stage2.json`.

## Result

```
dataset/finetune_qwen35_64k/
├── train.json
├── val.json
├── train_stage1.json
├── train_stage2.json
└── image_labels.json
```

The ready-made splits are also available on HuggingFace:
`<HF_USER>/LPCVC2026-Track3-Finetune-64k`.
