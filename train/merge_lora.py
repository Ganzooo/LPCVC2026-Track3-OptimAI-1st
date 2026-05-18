"""
Merge LoRA adapter weights into the base model for quantization.

The Qualcomm Qwen2-VL tutorial environment uses `transformers==4.45.0`, so the
export here intentionally writes files in that older Hugging Face layout:

- raw base-model config/tokenizer/processor files copied from the base snapshot
- merged weights with `model.visual.*` remapped back to `visual.*`

This avoids load failures such as:
    AttributeError: 'dict' object has no attribute 'to_dict'

Usage:
    python merge_lora.py --lora_path output_finetune/final --output_dir merged_model
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file as save_safetensors
from transformers import Qwen2VLForConditionalGeneration
from transformers.utils import cached_file


BASE_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

OPTIONAL_BASE_MODEL_FILES = [
    "chat_template.json",
    "chat_template.jinja",
    "special_tokens_map.json",
]


def copy_base_model_files(base_model_name: str, output_dir: Path) -> None:
    """Copy raw base-model metadata files in the exact layout the server expects."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in BASE_MODEL_FILES + OPTIONAL_BASE_MODEL_FILES:
        try:
            src = cached_file(base_model_name, filename)
        except OSError:
            if filename in BASE_MODEL_FILES:
                raise RuntimeError(
                    f"Required base-model file missing: {filename}. "
                    f"Cannot build a Qualcomm-compatible merged checkpoint."
                ) from None
            continue
        shutil.copy2(src, output_dir / filename)

    # Remove newer-format files that can confuse debugging when reusing an output dir.
    stale_path = output_dir / "processor_config.json"
    if stale_path.exists():
        stale_path.unlink()


def export_legacy_compatible_weights(model, output_dir: Path) -> None:
    """
    Export weights in the naming/layout expected by transformers 4.45.0.

    Newer transformers save the vision tower as `model.visual.*`, while the
    Qualcomm environment expects `visual.*`. Text tower keys already match.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    legacy_state_dict = {}

    for key, value in state_dict.items():
        legacy_key = key

        # Newer transformers nest the text tower under model.language_model.*.
        if key.startswith("model.language_model."):
            legacy_key = "model." + key[len("model.language_model."):]
        elif key.startswith("language_model."):
            legacy_key = "model." + key[len("language_model."):]
        elif key.startswith("model.visual."):
            legacy_key = "visual." + key[len("model.visual."):]

        # lm_head is tied to embed_tokens for this model and may be omitted safely.
        if legacy_key == "lm_head.weight":
            continue

        legacy_state_dict[legacy_key] = value.detach().cpu().contiguous()

    save_safetensors(
        legacy_state_dict,
        str(output_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="output_finetune/final",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merged_model",
        help="Output directory for merged model",
    )
    args = parser.parse_args()

    with open(os.path.join(args.lora_path, "adapter_config.json"), encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
    output_dir = Path(args.output_dir)

    print(f"Base model: {base_model_name}")
    print(f"LoRA adapter: {args.lora_path}")

    print("Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    export_legacy_compatible_weights(model, output_dir)
    copy_base_model_files(base_model_name, output_dir)

    print()
    print(f"Done! Merged model saved to: {args.output_dir}")
    print()
    print("Notes:")
    print("  - Exported metadata matches the Qualcomm tutorial's transformers 4.45.0 layout")
    print("  - Vision weights were renamed from model.visual.* to visual.* for compatibility")
    print()
    print("Next steps:")
    print(f"  1. Copy {args.output_dir}/ to your H200 server")
    print("  2. In the tutorial notebooks, change model_id to point to this merged model")
    print("  3. Run Example1A -> 1B -> 2A -> 2B")
    print("  4. Package submission with prepare_submission.py")


if __name__ == "__main__":
    main()
