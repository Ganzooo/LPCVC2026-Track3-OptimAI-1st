# Host-Side Evaluation

Run inference + scoring on the competition sample dataset (50 images)
before submitting to the LPCVC server.

## Usage

```bash
bash run_test.sh <model_path> <output_name>
# example:
bash run_test.sh output/finetune/qwen35_64k_r64/final result_repro
```

## Pipeline

1. `test_qwen2vl.py` — runs inference (Stage 1 + Stage 2) on every image
   in `dataset/sample_dataset/`.
2. `evaluate.py` — Detection accuracy + per-criterion accuracy.
3. `evaluate_server_score.py` — Adds Evidence semantic similarity and
   reports an estimated server score using the official scoring formula.

To skip the server-score estimator (faster, no `sentence-transformers`
needed):

```bash
SKIP_SERVER_SCORE=1 bash run_test.sh <model_path> <output_name>
```

Results land in `output/eval/<output_name>/`.
