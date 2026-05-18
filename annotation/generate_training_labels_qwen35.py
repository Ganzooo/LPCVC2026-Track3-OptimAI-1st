"""
Generate 2-stage training data using Qwen3.5-VL as teacher model.

Creates TWO training conversations per image:
  - Stage 1: image + stage1_prompt → free-text analysis
  - Stage 2: stage2_prompt + stage1_output → JSON output

The teacher (Qwen3.5-35B-A3B) generates high-quality labels that are then
used to fine-tune the student (Qwen2-VL-2B). Only samples where the teacher's
prediction matches the ground truth are kept.

Qwen3.5 may produce <think>...</think> reasoning blocks — these are stripped
from the final training data so the student learns to produce direct outputs.

Usage:
    # Generate from aigen shards (default)
    python annotation/generate_training_labels_qwen35.py \
        --server http://localhost:8000/v1 \
        --sample_n 500 \
        --output_dir dataset/finetune_qwen35

    # Specific shards only
    python annotation/generate_training_labels_qwen35.py \
        --server http://localhost:8000/v1 \
        --shards shard_0 shard_1 \
        --sample_n 200

    # Resume from a specific index
    python annotation/generate_training_labels_qwen35.py \
        --server http://localhost:8000/v1 \
        --batch_start 100
"""

import argparse
import base64
import csv
import json
import os
import re
import random
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2-stage training labels with Qwen3.5-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--server", type=str, default="http://11.11.50.6:8000/v1",
                        help="vLLM server URL (e.g. http://11.11.50.6:8000/v1)")
    parser.add_argument("--aigen_dir", type=str, default="dataset/aigen/train")
    parser.add_argument("--shards", type=str, nargs="*", default=None,
                        help="Specific shard names (e.g. shard_0 shard_1). Default: all shards")
    parser.add_argument("--target_n", type=int, default=20000,
                        help="Target kept samples per class (real/fake). Stop early when reached.")
    parser.add_argument("--oversample_ratio", type=float, default=1.5,
                        help="Sample target_n * ratio images per class (buffer for rejects)")
    parser.add_argument("--output_dir", type=str, default="dataset/finetune_qwen35_40k")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_start", type=int, default=0,
                        help="Resume from this index")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries per image on failure")
    parser.add_argument("--max_stage1_tokens", type=int, default=500,
                        help="Max approximate tokens for Stage 1 output (competition limit is 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inc_dataset", action="store_true",
                        help="Incremental mode: extend an existing dataset")
    parser.add_argument("--prev_dataset", type=str, default=None,
                        help="Path to previous dataset folder (e.g., dataset/finetune_qwen35_40k)")
    parser.add_argument("--additional_dir", type=str, default="dataset/aigen/additional",
                        help="Additional fake-only image folder (all images treated as AI-generated)")
    return parser.parse_args()


STAGE1_PROMPT = open(
    "26LPCVC_Track3_Sample_Solution/dataset/prompts/stage1.txt"
).read().strip()

STAGE2_PROMPT = open(
    "26LPCVC_Track3_Sample_Solution/dataset/prompts/stage2.txt"
).read().strip()

# Extra instruction appended only to the TEACHER query (not saved in training data).
# Instructs the teacher to produce concise outputs within the token limit.
STAGE1_LENGTH_CONSTRAINT = (
    "\n\nCRITICAL CONSTRAINT: Keep your entire analysis under 380 tokens. "
    "For each of the 8 criteria, write ONE specific sentence (~15-20 words) "
    "describing concrete visual details: name colors, light direction, body parts, "
    "textures, materials. Avoid generic phrases like 'appears unnatural' — "
    "describe WHAT is unnatural and WHY. "
    "Do NOT use markdown headers or extra whitespace. "
    "End with: 'The image is REAL.' or 'The image is AI-GENERATED.'"
)


def approx_tokens(text):
    """Rough token count: words * 1.35 (close to GPT/Qwen tokenizers)."""
    import re as _re
    return int(len(_re.findall(r'\S+', text)) * 1.35)


def extract_response(message):
    """Extract the actual response from a Qwen3.5 message.

    When vLLM uses --reasoning-parser qwen3, the <think> block is split into
    message.reasoning_content and the answer goes into message.content.
    If content is None (model only produced thinking), fall back to
    reasoning_content with think tags stripped.
    Without --reasoning-parser, everything is in message.content.
    """
    content = message.content or ""
    # vLLM with --reasoning-parser uses "reasoning" (not "reasoning_content")
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
    finish = getattr(message, "finish_reason", None)

    # Debug: show what we got
    if not content and not reasoning:
        # Try to dump the raw message for debugging
        try:
            raw = message.to_dict() if hasattr(message, "to_dict") else vars(message)
            print(f"    [DEBUG] Empty response. finish_reason={finish}, raw keys={list(raw.keys())}")
            for k, v in raw.items():
                v_str = repr(v)[:200] if v else "None"
                print(f"    [DEBUG]   {k}: {v_str}")
        except Exception as e:
            print(f"    [DEBUG] Cannot inspect message: {e}")

    if content:
        # Content exists — strip any leftover <think> tags (in case no reasoning parser)
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return cleaned if cleaned else content

    # content is None — reasoning parser consumed everything into reasoning_content
    if reasoning:
        cleaned = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
        return cleaned if cleaned else reasoning

    raise ValueError("Both content and reasoning_content are empty")


def fix_json_string(s):
    """Fix common LLM JSON issues: trailing commas, unquoted string values."""
    # Fix trailing commas
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)

    # Fix unquoted string values line by line
    # Split each line into "key_part" and "value_part" at the first `: ` after the key
    lines = s.split('\n')
    fixed_lines = []
    key_re = re.compile(r'^(\s*"[^"]+"\s*:\s*)(.*)$')
    for line in lines:
        m = key_re.match(line)
        if m:
            prefix = m.group(1)
            rest = m.group(2)
            # Only fix if value is not already valid JSON start
            stripped = rest.lstrip()
            if stripped and stripped[0] not in '"{[0123456789-' and \
               not stripped.startswith(('true', 'false', 'null')):
                # Unquoted string value — quote it
                value = rest.rstrip()
                trailing_comma = ''
                if value.endswith(','):
                    value = value[:-1].rstrip()
                    trailing_comma = ','
                value = value.replace('"', '\\"')
                line = f'{prefix}"{value}"{trailing_comma}'
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def parse_json_output(text):
    """Extract JSON from model output, handling common malformations."""
    candidates = []

    # Try code-fenced JSON first
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        candidates.append(match.group(1))

    # Try bare JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        # Try as-is first
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try with fixes
        try:
            return json.loads(fix_json_string(candidate))
        except json.JSONDecodeError:
            pass

    return None


def image_to_base64(image_path):
    ext = Path(image_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def run_two_stage_server(client, model_name, image_path):
    """Run 2-stage pipeline via vLLM server, return both stage outputs."""
    img_b64 = image_to_base64(image_path)

    # Disable thinking mode to produce direct answers for training data
    extra = {"chat_template_kwargs": {"enable_thinking": False}}

    # Stage 1: Image + prompt → free-text analysis
    # Teacher gets a length-constrained version of the prompt to produce short outputs
    teacher_stage1_prompt = STAGE1_PROMPT + STAGE1_LENGTH_CONSTRAINT
    s1_response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": teacher_stage1_prompt},
            ],
        }],
        max_tokens=4096,
        temperature=0.1,
        top_p=0.1,
        extra_body=extra,
    )
    s1_output = extract_response(s1_response.choices[0].message)

    # Stage 2: Text-only → JSON
    clean_text = s1_output.replace("[BEGIN]:", "").replace("[END]", "").strip()
    combined_prompt = (
        f"*** INSTRUCTIONS ***\n{STAGE2_PROMPT}\n\n"
        f"*** ANALYSIS DATA TO PROCESS ***\n{clean_text}\n\n"
    )
    s2_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": combined_prompt}],
        max_tokens=4096,
        temperature=0.1,
        top_p=0.1,
        extra_body=extra,
    )
    s2_output = extract_response(s2_response.choices[0].message)
    parsed = parse_json_output(s2_output)

    return s1_output, s2_output, parsed


def postprocess_likelihood(parsed):
    """Fix overall_likelihood based on per-criterion scores."""
    if parsed and "per_criterion" in parsed:
        aigc_count = sum(
            1 for c in parsed["per_criterion"]
            if c.get("aigc score", c.get("score", 0)) == 1
        )
        if aigc_count >= 2:
            parsed["overall_likelihood"] = "AI-Generated"
        elif aigc_count == 0:
            parsed["overall_likelihood"] = "Real"
    return parsed


def build_image_label_manifest(samples):
    manifest = []
    for sample in samples:
        image_path = None
        for content in sample["messages"][0]["content"]:
            if content.get("type") == "image":
                image_path = content.get("image")
                break
        if not image_path:
            continue
        manifest.append({
            "file_name": Path(image_path).name,
            "image_path": image_path,
            "label": "Fake" if sample.get("is_fake") else "Real",
            "is_fake": bool(sample.get("is_fake", False)),
        })
    manifest.sort(key=lambda item: item["file_name"])
    return manifest


def collect_additional_fakes(additional_dir):
    """Collect all images from a fake-only folder (recursive)."""
    additional_dir = Path(additional_dir)
    if not additional_dir.exists():
        print(f"  Additional fake dir not found: {additional_dir}")
        return []
    fakes = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        fakes.extend(str(p) for p in additional_dir.rglob(ext))
    print(f"  {additional_dir.name}: {len(fakes)} additional fake images")
    return fakes


def load_previous_dataset(prev_path):
    """Load previously generated dataset. Returns (stage1_pairs, stage2_pairs, used_images)."""
    prev_path = Path(prev_path)
    stage1_pairs, stage2_pairs = [], []
    used_images = set()
    for split in ("train.json", "val.json"):
        fp = prev_path / split
        if not fp.exists():
            continue
        with open(fp) as f:
            samples = json.load(f)
        for s in samples:
            if s.get("stage") == 1:
                stage1_pairs.append(s)
                for c in s["messages"][0]["content"]:
                    if c.get("type") == "image":
                        used_images.add(c.get("image", ""))
            elif s.get("stage") == 2:
                stage2_pairs.append(s)
    kept_real = sum(1 for p in stage1_pairs if not p.get("is_fake"))
    kept_fake = sum(1 for p in stage1_pairs if p.get("is_fake"))
    print(f"  Previous dataset: {len(stage1_pairs)} stage1, {len(stage2_pairs)} stage2")
    print(f"  Previous kept: {kept_real} real, {kept_fake} fake ({len(used_images)} unique images)")
    return stage1_pairs, stage2_pairs, used_images, kept_real, kept_fake


def collect_images(aigen_dir, shards):
    """Collect all real/fake image paths from shards."""
    all_real, all_fake = [], []
    shard_dirs = [aigen_dir / s for s in shards] if shards else sorted(aigen_dir.glob("shard_*"))

    for shard_dir in shard_dirs:
        labels_file = shard_dir / "labels.csv"
        images_dir = shard_dir / "images"
        if not labels_file.exists() or not images_dir.exists():
            print(f"  Skipping {shard_dir.name}: missing labels.csv or images/")
            continue
        count = 0
        with open(labels_file) as f:
            for row in csv.DictReader(f):
                img_path = images_dir / row["image_name"]
                if not img_path.exists():
                    continue
                if int(row["label"]) == 0:
                    all_real.append(str(img_path))
                else:
                    all_fake.append(str(img_path))
                count += 1
        print(f"  {shard_dir.name}: {count} images")

    return all_real, all_fake


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Generate training data with Qwen3.5-VL teacher")
    print(f"{'='*60}")
    print(f"Server: {args.server}")
    print(f"Output: {output_dir}")
    print(f"Target kept per class: {args.target_n}")
    print(f"Oversample ratio: {args.oversample_ratio}x (will sample {int(args.target_n * args.oversample_ratio)} per class)")
    print(f"Max Stage 1 tokens: {args.max_stage1_tokens}")
    print(f"")

    # Load previous dataset if incremental mode
    prev_stage1, prev_stage2 = [], []
    used_images = set()
    prev_real = prev_fake = 0
    if args.inc_dataset:
        if not args.prev_dataset:
            raise ValueError("--inc_dataset requires --prev_dataset")
        print(f"\nLoading previous dataset: {args.prev_dataset}")
        prev_stage1, prev_stage2, used_images, prev_real, prev_fake = \
            load_previous_dataset(args.prev_dataset)
        print(f"  Need {args.target_n - prev_real} more real, {args.target_n - prev_fake} more fake\n")

    # Collect images
    aigen_dir = Path(args.aigen_dir)
    print(f"Scanning shards in {aigen_dir}...")
    all_real, all_fake = collect_images(aigen_dir, args.shards)
    print(f"\nAvailable from shards: {len(all_real)} real, {len(all_fake)} fake")

    # Collect additional fake-only images (these will be FORCE-INCLUDED, not diluted)
    extra_fakes = []
    if args.additional_dir:
        extra_fakes = collect_additional_fakes(args.additional_dir)

    # Filter out images already used in previous dataset
    if used_images:
        before_real, before_fake, before_extra = len(all_real), len(all_fake), len(extra_fakes)
        all_real = [p for p in all_real if p not in used_images]
        all_fake = [p for p in all_fake if p not in used_images]
        extra_fakes = [p for p in extra_fakes if p not in used_images]
        print(f"Filtered already-used: real {before_real}->{len(all_real)}, "
              f"fake {before_fake}->{len(all_fake)}, additional {before_extra}->{len(extra_fakes)}")

    print(f"Available after filter: {len(all_real)} real, {len(all_fake)} fake (+{len(extra_fakes)} additional)")

    # Sample target_n * oversample_ratio per class to have a buffer for rejections
    # In inc mode, only need (target_n - prev_count) more per class
    need_real = max(0, args.target_n - prev_real)
    need_fake = max(0, args.target_n - prev_fake)
    pool_real = int(need_real * args.oversample_ratio)
    pool_fake = int(need_fake * args.oversample_ratio)
    random.seed(args.seed)
    sampled_real = random.sample(all_real, min(pool_real, len(all_real)))

    # Force-include ALL additional fakes (don't dilute them in the random sample)
    # then fill remaining quota from shard fakes
    sampled_fake_extra = list(extra_fakes)
    remaining_fake_quota = max(0, pool_fake - len(sampled_fake_extra))
    sampled_fake_shard = random.sample(all_fake, min(remaining_fake_quota, len(all_fake)))
    sampled_fake = sampled_fake_extra + sampled_fake_shard

    all_images = [(p, False) for p in sampled_real] + [(p, True) for p in sampled_fake]
    random.shuffle(all_images)
    print(f"Sampled pool: {len(sampled_real)} real + {len(sampled_fake)} fake "
          f"({len(sampled_fake_extra)} from additional/ + {len(sampled_fake_shard)} from shards) "
          f"= {len(all_images)} total")

    # Setup server
    from openai import OpenAI
    client = OpenAI(base_url=args.server, api_key="not-needed")
    models = client.models.list()
    available = [m.id for m in models.data]
    model_name = available[0] if available else args.model
    print(f"Model: {model_name}")
    print(f"")

    # Process images
    stage1_pairs = []
    stage2_pairs = []
    progress_file = output_dir / "progress.jsonl"
    # Start counters from prev dataset in incremental mode
    kept_real = prev_real
    kept_fake = prev_fake
    failed = skipped_wrong = skipped_long = 0

    for i, (img_path, is_fake) in enumerate(all_images):
        if i < args.batch_start:
            continue

        # Early stopping: per-class target reached
        if kept_real >= args.target_n and kept_fake >= args.target_n:
            print(f"\n*** TARGET REACHED: {kept_real} real + {kept_fake} fake — stopping early ***")
            break

        # Skip this class if already at target (but continue the loop to process the other class)
        if (not is_fake and kept_real >= args.target_n) or \
           (is_fake and kept_fake >= args.target_n):
            continue

        gt_label = "AI-Generated" if is_fake else "Real"
        print(f"\n[{i+1}/{len(all_images)}] {Path(img_path).name} (GT={gt_label}) "
              f"[kept: real={kept_real}/{args.target_n}, fake={kept_fake}/{args.target_n}]")

        for attempt in range(args.max_retries + 1):
            try:
                start = time.time()
                s1_out, s2_out, parsed = run_two_stage_server(
                    client, model_name, img_path
                )
                elapsed = time.time() - start

                if parsed is None:
                    print(f"  SKIP: JSON parse failed ({elapsed:.1f}s)")
                    print(f"    S2 (first 200): {s2_out[:200]}...")
                    if attempt < args.max_retries:
                        print(f"    Retrying ({attempt+1}/{args.max_retries})...")
                        continue
                    failed += 1
                    break

                parsed = postprocess_likelihood(parsed)
                pred_label = parsed.get("overall_likelihood", "Unknown")

                # Only keep if prediction matches ground truth
                gt_match = (is_fake and pred_label == "AI-Generated") or \
                           (not is_fake and pred_label in ("Real", "Uncertain"))

                if not gt_match:
                    skipped_wrong += 1
                    print(f"  SKIP: pred={pred_label} != GT={gt_label} ({elapsed:.1f}s)")
                    break

                # Length filter: reject outputs exceeding device token budget
                s1_tokens = approx_tokens(s1_out)
                if s1_tokens > args.max_stage1_tokens:
                    skipped_long += 1
                    print(f"  SKIP: stage1 too long ({s1_tokens} > {args.max_stage1_tokens} tokens) ({elapsed:.1f}s)")
                    break

                # Stage 1 training pair: image + stage1_prompt → free-text
                s1_conv = {
                    "stage": 1,
                    "is_fake": int(is_fake),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": STAGE1_PROMPT},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": s1_out},
                            ],
                        },
                    ],
                }
                stage1_pairs.append(s1_conv)

                # Stage 2 training pair: stage2_prompt + s1_output → JSON
                clean_s1 = s1_out.replace("[BEGIN]:", "").replace("[END]", "").strip()
                s2_prompt = (
                    f"*** INSTRUCTIONS ***\n{STAGE2_PROMPT}\n\n"
                    f"*** ANALYSIS DATA TO PROCESS ***\n{clean_s1}\n\n"
                )
                s2_conv = {
                    "stage": 2,
                    "is_fake": int(is_fake),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": s2_prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                # Use compact JSON (no indent) to minimize output tokens
                                {"type": "text", "text": json.dumps(parsed, separators=(",", ":"))},
                            ],
                        },
                    ],
                }
                stage2_pairs.append(s2_conv)

                if is_fake:
                    kept_fake += 1
                else:
                    kept_real += 1
                total_kept = kept_real + kept_fake
                print(f"  KEPT: pred={pred_label} s1_tokens={s1_tokens} ({elapsed:.1f}s) "
                      f"[real={kept_real}, fake={kept_fake}, total={total_kept}]")

                # Save progress
                with open(progress_file, "a") as f:
                    f.write(json.dumps({"s1": s1_conv, "s2": s2_conv}) + "\n")

                break  # success, no retry

            except Exception as e:
                if attempt < args.max_retries:
                    print(f"  ERROR: {e} — retrying ({attempt+1}/{args.max_retries})...")
                    time.sleep(2)
                else:
                    print(f"  ERROR: {e}")
                    failed += 1

    # Balance and split
    total_kept = kept_real + kept_fake
    print(f"\n{'='*60}")
    print(f"Results: kept={total_kept} (real={kept_real}, fake={kept_fake}), "
          f"skipped_wrong={skipped_wrong}, skipped_long={skipped_long}, failed={failed}")

    # Merge with previous dataset pairs (incremental mode)
    if args.inc_dataset:
        print(f"Merging prev ({len(prev_stage1)} stage1 + {len(prev_stage2)} stage2) "
              f"with new ({len(stage1_pairs)} stage1 + {len(stage2_pairs)} stage2)")
        stage1_pairs = prev_stage1 + stage1_pairs
        stage2_pairs = prev_stage2 + stage2_pairs

    s1_real = [p for p in stage1_pairs if not p.get("is_fake", False)]
    s1_fake = [p for p in stage1_pairs if p.get("is_fake", False)]
    print(f"Stage 1: {len(s1_real)} real, {len(s1_fake)} fake")

    # Combine both stages
    all_pairs = stage1_pairs + stage2_pairs
    random.shuffle(all_pairs)

    split = int(len(all_pairs) * args.val_split)
    val_data = all_pairs[:split]
    train_data = all_pairs[split:]

    # Save all outputs
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(output_dir / "train_stage1.json", "w") as f:
        json.dump(stage1_pairs, f, indent=2, ensure_ascii=False)
    with open(output_dir / "train_stage2.json", "w") as f:
        json.dump(stage2_pairs, f, indent=2, ensure_ascii=False)

    label_manifest = build_image_label_manifest(stage1_pairs)
    with open(output_dir / "image_labels.json", "w") as f:
        json.dump(label_manifest, f, indent=2, ensure_ascii=False)

    s1_count = len(stage1_pairs)
    s2_count = len(stage2_pairs)
    print(f"\nSaved to {output_dir}/:")
    print(f"  train.json:        {len(train_data)} samples (Stage1 + Stage2 mixed)")
    print(f"  val.json:          {len(val_data)} samples")
    print(f"  train_stage1.json: {s1_count} (Stage 1 only, {len(s1_real)} real / {len(s1_fake)} fake)")
    print(f"  train_stage2.json: {s2_count} (Stage 2 only)")
    print(f"  image_labels.json: {len(label_manifest)} image labels")
    print(f"\nFine-tune with:")
    print(f"  python train/finetune_qwen2vl_stage2.py \\")
    print(f"      --train_data {output_dir}/train.json \\")
    print(f"      --val_data {output_dir}/val.json \\")
    print(f"      --dataset_name qwen35 --epochs 5 --lr 1e-4 --stage1_aux_cls")


if __name__ == "__main__":
    main()
