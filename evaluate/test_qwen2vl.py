"""
Host-side 2-stage inference pipeline for LPCVC 2026 Track 3.
Replicates the on-device pipeline using Qwen2-VL-2B-Instruct via HuggingFace transformers.

Usage:
    python test_qwen2vl.py [--model MODEL_NAME] [--dataset_dir DIR] [--output_dir DIR]
"""

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Host-side Qwen2-VL inference for AIGC detection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--dataset_dir", type=str, default="dataset/sample dataset",
                        help="Path to dataset with Real/ and Fake/ subdirectories")
    parser.add_argument("--prompts_dir", type=str,
                        default="26LPCVC_Track3_Sample_Solution/dataset/prompts",
                        help="Path to prompts directory containing stage1.txt and stage2.txt")
    parser.add_argument("--output_dir", type=str, default="output/results",
                        help="Directory to save inference results")
    parser.add_argument("--inp_h", type=int, default=342, help="Image input height")
    parser.add_argument("--inp_w", type=int, default=512, help="Image input width")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling (0=disabled). Must explicitly override model "
                             "generation_config which has top_k=1 (forces greedy decoding).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling. None = system random.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detected if not set)")
    return parser.parse_args()


def load_model(model_name, device):
    """Load Qwen2-VL model and processor. Supports LoRA adapters."""
    print(f"Loading model: {model_name}")

    # Check if this is a LoRA adapter directory
    is_lora = os.path.exists(os.path.join(model_name, "adapter_config.json"))

    if is_lora:
        import json
        from peft import PeftModel
        with open(os.path.join(model_name, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
        print(f"  LoRA adapter detected, base model: {base_model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.bfloat16
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()
        model.eval()
        print(f"  LoRA merged, loaded on auto, dtype={dtype}")
        return model, processor

    processor = AutoProcessor.from_pretrained(model_name)

    # Determine dtype and device_map
    if device == "cpu":
        dtype = torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to("cpu")
    elif device == "auto":
        dtype = torch.bfloat16
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        dtype = torch.bfloat16
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    model.eval()
    print(f"Model loaded on {device}, dtype={dtype}")
    return model, processor


def stage1_inference(model, processor, image_path, prompt, inp_h, inp_w, gen_kwargs):
    """Stage 1: Image + text prompt -> free-text analysis."""
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
                "resized_height": inp_h,
                "resized_width": inp_w,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated tokens (skip input tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text


def stage2_inference(model, processor, stage1_output, stage2_prompt, gen_kwargs):
    """Stage 2: Stage1 output + stage2 prompt -> structured JSON."""
    # Clean stage1 output (match inference_script.py behavior)
    clean_text = stage1_output.replace("[BEGIN]:", "").replace("[END]", "").strip()

    combined_prompt = (
        f"*** INSTRUCTIONS ***\n{stage2_prompt}\n\n"
        f"*** ANALYSIS DATA TO PROCESS ***\n{clean_text}\n\n"
    )

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": combined_prompt,
            },
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text


def fix_trailing_commas(json_str):
    """Remove trailing commas before } or ] which are invalid in JSON."""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


def parse_json_output(text):
    """Try to extract JSON from model output text."""
    # Try to find JSON block in markdown code fence
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(fix_trailing_commas(match.group(1)))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(fix_trailing_commas(match.group(0)))
        except json.JSONDecodeError:
            pass

    return None


def main():
    args = parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "auto"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    # Load prompts
    stage1_path = os.path.join(args.prompts_dir, "stage1.txt")
    stage2_path = os.path.join(args.prompts_dir, "stage2.txt")
    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1_prompt = f.read().strip()
    with open(stage2_path, "r", encoding="utf-8") as f:
        stage2_prompt = f.read().strip()

    print(f"Stage 1 prompt ({len(stage1_prompt)} chars): {stage1_prompt[:100]}...")
    print(f"Stage 2 prompt ({len(stage2_prompt)} chars): {stage2_prompt[:100]}...")

    # Load model
    model, processor = load_model(args.model, args.device)

    # Set seed if specified (for reproducible sampling variation)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Generation parameters (match genie config; explicitly override model's
    # generation_config which sets top_k=1, forcing greedy decoding regardless
    # of temperature/top_p).
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k if args.top_k > 0 else None,  # None = disabled in HF
        "do_sample": True,
    }

    # Find all images
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(args.dataset_dir, "**", ext), recursive=True))
    image_files.sort()
    print(f"\nFound {len(image_files)} images")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    all_results = {}
    total_time = 0

    for i, img_path in enumerate(image_files):
        rel_path = os.path.relpath(img_path, args.dataset_dir)
        img_name = Path(img_path).stem
        # Remove "Copy of " prefix if present (to match annotation keys)
        annotation_key = img_name.replace("Copy of ", "")
        label = "Real" if "Real" in rel_path else "Fake"

        print(f"\n[{i+1}/{len(image_files)}] {rel_path} (label={label})")

        start = time.time()

        # Stage 1: Image analysis
        print("  Stage 1: Analyzing image...")
        try:
            s1_output = stage1_inference(
                model, processor, img_path, stage1_prompt,
                args.inp_h, args.inp_w, gen_kwargs
            )
            print(f"  Stage 1 output ({len(s1_output)} chars): {s1_output[:150]}...")
        except Exception as e:
            print(f"  Stage 1 FAILED: {e}")
            continue

        # Stage 2: JSON synthesis
        print("  Stage 2: Synthesizing JSON...")
        try:
            s2_output = stage2_inference(
                model, processor, s1_output, stage2_prompt, gen_kwargs
            )
            print(f"  Stage 2 output ({len(s2_output)} chars): {s2_output[:150]}...")
        except Exception as e:
            print(f"  Stage 2 FAILED: {e}")
            continue

        elapsed = time.time() - start
        total_time += elapsed

        # Parse JSON
        parsed = parse_json_output(s2_output)
        if parsed:
            print(f"  Parsed JSON: overall_likelihood={parsed.get('overall_likelihood', 'N/A')}")
        else:
            print(f"  WARNING: Could not parse JSON from output")

        # Save individual result
        result = {
            "image": rel_path,
            "annotation_key": annotation_key,
            "ground_truth_label": label,
            "stage1_output": s1_output,
            "stage2_raw_output": s2_output,
            "parsed_json": parsed,
            "inference_time_sec": round(elapsed, 2),
        }
        all_results[annotation_key] = result

        # Save per-image result in subfolder
        each_image_dir = output_dir / "each_image_result"
        each_image_dir.mkdir(parents=True, exist_ok=True)
        result_file = each_image_dir / f"{annotation_key}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Time: {elapsed:.1f}s")

    # Save summary (without full results to keep it clean)
    image_summary = []
    for key, r in all_results.items():
        pred = r.get("parsed_json", {})
        image_summary.append({
            "image": r["image"],
            "annotation_key": key,
            "ground_truth": r["ground_truth_label"],
            "prediction": pred.get("overall_likelihood", "PARSE_FAIL") if pred else "PARSE_FAIL",
            "time_sec": r["inference_time_sec"],
        })

    summary = {
        "model": args.model,
        "total_images": len(image_files),
        "successful": len(all_results),
        "total_time_sec": round(total_time, 2),
        "avg_time_per_image_sec": round(total_time / max(len(all_results), 1), 2),
        "images": image_summary,
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also save full results for evaluate.py compatibility
    full_results = {"results": all_results}
    full_results_file = output_dir / "each_image_result" / "_full_results.json"
    with open(full_results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(all_results)}/{len(image_files)} images")
    print(f"Total time: {total_time:.1f}s, Avg: {total_time/max(len(all_results),1):.1f}s/image")
    print(f"Results saved to:")
    print(f"  {output_dir}/summary.json (overview)")
    print(f"  {output_dir}/each_image_result/ (per-image details)")
    print(f"\nRun evaluate.py to compute accuracy against ground truth.")


if __name__ == "__main__":
    main()
