"""
Extract per-criterion binary labels from Stage 2 JSON and attach to Stage 1 entries.

The 40k dataset's progress.jsonl contains paired (s1, s2) entries for each image.
For each image, we parse the Stage 2 JSON output to extract 8 per-criterion
`aigc score` values, then attach them to the corresponding Stage 1 entry as
`criterion_labels: [0 or 1] * 8`.

This enables the per-criterion auxiliary classifier during training.

Usage:
    python annotation/extract_criterion_labels.py \
        --dataset_dir dataset/finetune_qwen35_40k \
        --output_dir dataset/finetune_qwen35_40k_cri
"""

import argparse
import json
import re
from pathlib import Path


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

# Alternative spellings accepted when matching (normalize via lowercase compare)
CRITERION_ALIASES = {
    "lighting & shadows consistency": 0,
    "lighting and shadows consistency": 0,
    "edges & boundaries": 1,
    "edges and boundaries": 1,
    "texture & resolution": 2,
    "texture and resolution": 2,
    "perspective & spatial relationships": 3,
    "perspective and spatial relationships": 3,
    "physical & common-sense logic": 4,
    "physical & common sense logic": 4,
    "physical and common-sense logic": 4,
    "physical and common sense logic": 4,
    "text & symbols": 5,
    "text and symbols": 5,
    "human & biological structure integrity": 6,
    "human and biological structure integrity": 6,
    "material & object details": 7,
    "material and object details": 7,
}


def parse_json_tolerant(text):
    """Parse JSON that may have trailing commas or unquoted evidence values."""
    # Try plain parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = text[start:end + 1]

    # Remove trailing commas before } or ]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def extract_labels_from_stage2(s2_entry):
    """Extract 8 per-criterion binary labels from Stage 2 assistant JSON output.

    Returns list of length 8 with values in {0, 1, -100}.
    -100 means the criterion couldn't be parsed.
    """
    labels = [-100] * 8

    assistant_text = ""
    for msg in s2_entry.get("messages", []):
        if msg.get("role") == "assistant":
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    assistant_text = c.get("text", "")
                    break
            break

    if not assistant_text:
        return labels

    parsed = parse_json_tolerant(assistant_text)
    if not parsed:
        return labels

    per_criterion = parsed.get("per_criterion")
    if not isinstance(per_criterion, list):
        return labels

    for item in per_criterion:
        if not isinstance(item, dict):
            continue
        name = str(item.get("criterion", "")).strip().lower()
        idx = CRITERION_ALIASES.get(name)
        if idx is None:
            continue
        score = item.get("aigc score")
        if isinstance(score, bool):
            labels[idx] = int(score)
        elif isinstance(score, (int, float)):
            if int(score) in (0, 1):
                labels[idx] = int(score)
        elif isinstance(score, str):
            s = score.strip()
            if s in ("0", "1"):
                labels[idx] = int(s)

    return labels


def get_image_path(entry):
    """Return the image path from a Stage 1 entry, or None."""
    for msg in entry.get("messages", []):
        if msg.get("role") == "user":
            for c in msg.get("content", []):
                if c.get("type") == "image":
                    return c.get("image")
    return None


def build_image_to_labels(progress_path):
    """Scan progress.jsonl and build {image_path: criterion_labels} mapping.

    Uses the s2 JSON in each paired entry to derive the labels.
    """
    image_to_labels = {}
    total = 0
    parsed_ok = 0
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            s1 = pair.get("s1")
            s2 = pair.get("s2")
            if not s1 or not s2:
                continue
            image = get_image_path(s1)
            if not image:
                continue
            labels = extract_labels_from_stage2(s2)
            total += 1
            if any(v != -100 for v in labels):
                parsed_ok += 1
                image_to_labels[image] = labels
    return image_to_labels, total, parsed_ok


def attach_labels(dataset, image_to_labels):
    """Attach criterion_labels to each Stage 1 entry in place.

    Returns (num_attached, num_stage1_total).
    """
    attached = 0
    stage1_total = 0
    for entry in dataset:
        if entry.get("stage") != 1:
            continue
        stage1_total += 1
        image = get_image_path(entry)
        if not image:
            continue
        labels = image_to_labels.get(image)
        if labels is not None:
            entry["criterion_labels"] = labels
            attached += 1
        else:
            entry["criterion_labels"] = [-100] * 8
    return attached, stage1_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory (expects train.json, val.json, progress.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir (default: overwrites dataset_dir)",
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default="progress.jsonl",
        help="Progress file with paired s1/s2 entries",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = dataset_dir / args.progress_file
    if not progress_path.exists():
        raise FileNotFoundError(f"progress.jsonl not found: {progress_path}")

    print(f"Building image->labels map from {progress_path}")
    image_to_labels, total, parsed_ok = build_image_to_labels(progress_path)
    print(f"  Paired s1/s2 entries: {total}")
    print(f"  Successfully parsed:  {parsed_ok}")
    print(f"  Unique images:        {len(image_to_labels)}")

    # Attach to train.json and val.json
    for split in ("train.json", "val.json"):
        src = dataset_dir / split
        if not src.exists():
            print(f"  Skipping {split} (not found)")
            continue

        with open(src) as f:
            data = json.load(f)

        attached, total_s1 = attach_labels(data, image_to_labels)

        dst = output_dir / split
        with open(dst, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also compute label distribution stats
        label_counts = [0] * 8  # how many entries have valid labels for each criterion
        fake_counts = [0] * 8   # how many fake samples per criterion
        for entry in data:
            if entry.get("stage") != 1:
                continue
            labels = entry.get("criterion_labels")
            if not labels:
                continue
            for i, v in enumerate(labels):
                if v != -100:
                    label_counts[i] += 1
                    if v == 1:
                        fake_counts[i] += 1

        print(f"\n  {split}: {attached}/{total_s1} Stage 1 entries labeled")
        print(f"  Criterion distribution (valid / of which fake=1):")
        for i, name in enumerate(CRITERION_ORDER):
            valid = label_counts[i]
            fake = fake_counts[i]
            pct = (100 * fake / valid) if valid else 0
            print(f"    {i}. {name:42s}  valid={valid:6d}  fake={fake:6d}  ({pct:5.1f}%)")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
