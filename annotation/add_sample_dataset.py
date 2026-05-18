"""
Add the competition sample dataset (50 images) to an existing training dataset.

Strategy:
- Fake images (25): build training pairs directly from annotation.json ground truth
  (no server needed — 100% retention, high-quality evidence from competition organizers)
- Real images (25): generate via Qwen3.5-VL server (easy for the model, low skip rate)

Usage:
    python annotation/add_sample_dataset.py \
        --server http://HOST:8000/v1 \
        --sample_dir dataset/sample_dataset \
        --base_train dataset/finetune_qwen35/train.json \
        --base_val dataset/finetune_qwen35/val.json \
        --output_dir dataset/finetune_qwen35_v2
"""

import argparse
import base64
import json
import re
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, default="dataset/sample_dataset")
    parser.add_argument("--base_train", type=str, default="dataset/finetune_qwen35/train.json")
    parser.add_argument("--base_val", type=str, default="dataset/finetune_qwen35/val.json")
    parser.add_argument("--output_dir", type=str, default="dataset/finetune_qwen35_v2")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_stage1_tokens", type=int, default=490)
    return parser.parse_args()


STAGE1_PROMPT = open(
    "26LPCVC_Track3_Sample_Solution/dataset/prompts/stage1.txt"
).read().strip()

STAGE2_PROMPT = open(
    "26LPCVC_Track3_Sample_Solution/dataset/prompts/stage2.txt"
).read().strip()

STAGE1_LENGTH_CONSTRAINT = (
    "\n\nIMPORTANT CONSTRAINT: Keep your entire analysis under 450 tokens. "
    "Use concise bullet points (1-2 sentences per criterion, only if notable). "
    "Do NOT use markdown headers or extra whitespace. "
    "End with a single-line conclusion: 'The image is REAL.' or 'The image is AI-GENERATED.'"
)

CRITERION_ORDER = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common-Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]


def approx_tokens(text):
    return int(len(re.findall(r'\S+', text)) * 1.35)


def image_to_base64(image_path, max_h=342, max_w=512):
    """Encode image as base64, resizing to fit max_h×max_w to stay within context limit."""
    from PIL import Image
    import io
    img = Image.open(image_path).convert("RGB")
    # Resize keeping aspect ratio so neither dimension exceeds the limit
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def extract_response(message):
    content = message.content or ""
    reasoning = getattr(message, "reasoning", None) or \
                getattr(message, "reasoning_content", None) or ""
    if content:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return cleaned if cleaned else content
    if reasoning:
        cleaned = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
        return cleaned if cleaned else reasoning
    raise ValueError("Both content and reasoning are empty")


def fix_json_string(s):
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    lines = s.split('\n')
    fixed = []
    key_re = re.compile(r'^(\s*"[^"]+"\s*:\s*)(.*)$')
    for line in lines:
        m = key_re.match(line)
        if m:
            prefix, rest = m.group(1), m.group(2)
            stripped = rest.lstrip()
            if stripped and stripped[0] not in '"{[0123456789-' and \
               not stripped.startswith(('true', 'false', 'null')):
                value = rest.rstrip()
                trailing = ''
                if value.endswith(','):
                    value = value[:-1].rstrip()
                    trailing = ','
                value = value.replace('"', '\\"')
                line = f'{prefix}"{value}"{trailing}'
        fixed.append(line)
    return '\n'.join(fixed)


def parse_json_output(text):
    candidates = []
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        for s in [candidate, fix_json_string(candidate)]:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return None


def postprocess_likelihood(parsed):
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


# ── Fake images: build from annotation.json ─────────────────────────────────

def score_to_aigc(score):
    """Map competition score (0/1/2) → binary aigc score (0/1)."""
    return 1 if score >= 1 else 0


def build_stage1_text_from_annotation(ann_entry):
    """Reconstruct concise stage-1 free-text from annotation per-criterion evidence."""
    lines = []
    for criterion in ann_entry["per_criterion"]:
        name = criterion["criterion"]
        evidence = criterion["evidence"]
        score = criterion["score"]
        # Trim evidence to first sentence if very long
        first_sentence = re.split(r'(?<=[.!?])\s', evidence)[0]
        flag = "⚠" if score >= 1 else "✓"
        lines.append(f"- {name}: {first_sentence}")

    ol = ann_entry["overall_likelihood"]
    conclusion = "The image is AI-GENERATED." if ol == "AI-Generated" else "The image is REAL."
    return "\n".join(lines) + "\n\n" + conclusion


def build_stage2_json_from_annotation(ann_entry):
    """Convert annotation to stage-2 JSON format used in training."""
    per_criterion = []
    for criterion in ann_entry["per_criterion"]:
        per_criterion.append({
            "criterion": criterion["criterion"],
            "evidence": criterion["evidence"],
            "aigc score": score_to_aigc(criterion["score"]),
        })
    ol = ann_entry["overall_likelihood"]
    return {
        "per_criterion": per_criterion,
        "overall_likelihood": ol,
    }


def make_pairs_from_annotation(image_path, ann_entry, is_fake):
    """Build (stage1_conv, stage2_conv) from annotation.json entry."""
    s1_text = build_stage1_text_from_annotation(ann_entry)
    s2_json = build_stage2_json_from_annotation(ann_entry)

    s1_conv = {
        "stage": 1,
        "is_fake": is_fake,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": STAGE1_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": s1_text},
            ]},
        ],
    }

    clean_s1 = s1_text.replace("[BEGIN]:", "").replace("[END]", "").strip()
    s2_conv = {
        "stage": 2,
        "is_fake": is_fake,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"*** INSTRUCTIONS ***\n{STAGE2_PROMPT}\n\n"
                    f"*** ANALYSIS DATA TO PROCESS ***\n{clean_s1}\n\n"
                )},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(s2_json, indent=2)},
            ]},
        ],
    }
    return s1_conv, s2_conv


# ── Real images: generate via server ────────────────────────────────────────

def run_two_stage(client, model_name, image_path):
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    img_b64 = image_to_base64(image_path)

    s1_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_b64}},
            {"type": "text", "text": STAGE1_PROMPT + STAGE1_LENGTH_CONSTRAINT},
        ]}],
        max_tokens=4096, temperature=0.1, top_p=0.1, extra_body=extra,
    )
    s1_out = extract_response(s1_response.choices[0].message)

    clean_s1 = s1_out.replace("[BEGIN]:", "").replace("[END]", "").strip()
    s2_prompt = (
        f"*** INSTRUCTIONS ***\n{STAGE2_PROMPT}\n\n"
        f"*** ANALYSIS DATA TO PROCESS ***\n{clean_s1}\n\n"
    )
    s2_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": s2_prompt}],
        max_tokens=4096, temperature=0.1, top_p=0.1, extra_body=extra,
    )
    s2_out = extract_response(s2_response.choices[0].message)
    parsed = parse_json_output(s2_out)
    return s1_out, s2_out, parsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir)

    # Load annotation
    with open(sample_dir / "annotation.json") as f:
        annotation = json.load(f)

    # Map annotation keys to image files
    fake_dir = sample_dir / "Fake"
    real_dir = sample_dir / "Real"

    # Build a lookup: annotation_key → image path
    def find_image(folder, key):
        for img in folder.glob("*"):
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            stem = img.stem.replace("Copy of ", "")
            if stem == key:
                return img
        return None

    # ── Part 1: Fake images from annotation.json ─────────────────────────────
    print("=" * 60)
    print("Part 1: Fake images from annotation.json (no server needed)")
    print("=" * 60)

    fake_s1, fake_s2 = [], []
    for key, ann_entry in annotation.items():
        img_path = find_image(fake_dir, key)
        if img_path is None:
            print(f"  WARNING: image not found for {key}")
            continue

        s1_conv, s2_conv = make_pairs_from_annotation(img_path, ann_entry, is_fake=1)
        s1_tokens = approx_tokens(s1_conv["messages"][1]["content"][0]["text"])
        fake_s1.append(s1_conv)
        fake_s2.append(s2_conv)
        ol = ann_entry["overall_likelihood"]
        print(f"  BUILT: {key} | {ol} | s1={s1_tokens} tokens")

    print(f"\nFake pairs built: {len(fake_s1)} stage1 + {len(fake_s2)} stage2")

    # ── Part 2: Real images via Qwen3.5 server ───────────────────────────────
    print()
    print("=" * 60)
    print("Part 2: Real images via Qwen3.5-VL server")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(base_url=args.server, api_key="not-needed")
    models = client.models.list()
    model_name = models.data[0].id if models.data else "Qwen/Qwen3.5-35B-A3B"
    print(f"Model: {model_name}\n")

    real_images = sorted(real_dir.glob("*"))
    real_images = [p for p in real_images if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    print(f"Real images to process: {len(real_images)}")

    real_s1, real_s2 = [], []
    failed = skipped_wrong = skipped_long = 0

    for i, img_path in enumerate(real_images):
        print(f"\n[{i+1}/{len(real_images)}] {img_path.name}")

        for attempt in range(args.max_retries + 1):
            try:
                start = time.time()
                s1_out, s2_out, parsed = run_two_stage(client, model_name, img_path)
                elapsed = time.time() - start

                if parsed is None:
                    print(f"  SKIP: JSON parse failed ({elapsed:.1f}s)")
                    if attempt < args.max_retries:
                        print(f"    Retrying ({attempt+1}/{args.max_retries})...")
                        continue
                    failed += 1
                    break

                parsed = postprocess_likelihood(parsed)
                pred = parsed.get("overall_likelihood", "Unknown")

                if pred not in ("Real", "Uncertain"):
                    skipped_wrong += 1
                    print(f"  SKIP: model predicted {pred} for real image ({elapsed:.1f}s)")
                    break

                s1_tokens = approx_tokens(s1_out)
                if s1_tokens > args.max_stage1_tokens:
                    skipped_long += 1
                    print(f"  SKIP: too long ({s1_tokens} tokens) ({elapsed:.1f}s)")
                    break

                s1_conv = {
                    "stage": 1, "is_fake": 0,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": STAGE1_PROMPT},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": s1_out},
                        ]},
                    ],
                }
                clean_s1 = s1_out.replace("[BEGIN]:", "").replace("[END]", "").strip()
                s2_conv = {
                    "stage": 2, "is_fake": 0,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": (
                                f"*** INSTRUCTIONS ***\n{STAGE2_PROMPT}\n\n"
                                f"*** ANALYSIS DATA TO PROCESS ***\n{clean_s1}\n\n"
                            )},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": json.dumps(parsed, indent=2)},
                        ]},
                    ],
                }
                real_s1.append(s1_conv)
                real_s2.append(s2_conv)
                print(f"  KEPT: pred={pred} s1={s1_tokens}tok ({elapsed:.1f}s)")
                break

            except Exception as e:
                if attempt < args.max_retries:
                    print(f"  ERROR: {e} — retrying ({attempt+1}/{args.max_retries})...")
                    time.sleep(2)
                else:
                    print(f"  ERROR: {e}")
                    failed += 1

    print(f"\nReal pairs kept: {len(real_s1)} / {len(real_images)}")
    print(f"  skipped_wrong={skipped_wrong}, skipped_long={skipped_long}, failed={failed}")

    # ── Merge and save ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Merging and saving")
    print("=" * 60)

    new_pairs = fake_s1 + fake_s2 + real_s1 + real_s2

    with open(args.base_train) as f:
        base_train = json.load(f)
    with open(args.base_val) as f:
        base_val = json.load(f)

    import random
    random.seed(42)
    combined_train = base_train + new_pairs
    random.shuffle(combined_train)

    with open(output_dir / "train.json", "w") as f:
        json.dump(combined_train, f, indent=2, ensure_ascii=False)
    with open(output_dir / "val.json", "w") as f:
        json.dump(base_val, f, indent=2, ensure_ascii=False)

    print(f"\nNew pairs: {len(new_pairs)}")
    print(f"  Fake s1+s2: {len(fake_s1)+len(fake_s2)} (from annotation.json)")
    print(f"  Real s1+s2: {len(real_s1)+len(real_s2)} (from server)")
    print(f"Base train: {len(base_train)} → Combined: {len(combined_train)}")
    print(f"Val (unchanged): {len(base_val)}")
    print(f"\nSaved to {output_dir}/")
    print(f"\nFine-tune with:")
    print(f"  python train/finetune_qwen2vl_stage2.py \\")
    print(f"      --dataset_name qwen35_v2 \\")
    print(f"      --train_data {output_dir}/train.json \\")
    print(f"      --val_data {output_dir}/val.json \\")
    print(f"      --epochs 5 --lr 1e-4 --stage1_aux_cls")


if __name__ == "__main__":
    main()
