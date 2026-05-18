# Stage 3 — Quantization & Submission

We use Qualcomm AIMET Pro to quantize the merged model to W4A16 and
export a QNN binary for Snapdragon 8 Gen 5.

See [`docs/quantization-aimet.md`](../docs/quantization-aimet.md) for the
workflow overview.

## Scripts

| Script                       | Purpose |
| ---------------------------- | ------- |
| `prepare_calibration_data.py` | Produces the calibration JSON used by AIMET. |
| `prepare_submission.py`       | Packages the QNN binary + tokenizer + `.raw` tensors into the submission zip. |
| `run_submit.sh`               | End-to-end packaging entry point. |

## Submission shape

```
team_name.zip/
└── team_name/
    ├── ar128-ar1-cl2048/
    │   └── weight_sharing_model_1_of_1.serialized.bin
    ├── serialized_binaries/
    │   └── veg.serialized.bin
    ├── embedding_weights_151936x1536.raw
    ├── inputs.json
    ├── mask.raw
    ├── position_ids_cos.raw
    ├── position_ids_sin.raw
    └── tokenizer.json
```
