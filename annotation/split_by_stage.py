"""
Split a merged train/val dataset into stage1-only and stage2-only files.

Usage:
    python annotation/split_by_stage.py --input_dir dataset/finetune_qwen35_v2
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    for base in ["train", "val"]:
        src = input_dir / f"{base}.json"
        if not src.exists():
            print(f"  Skipping {src} (not found)")
            continue

        with open(src) as f:
            data = json.load(f)

        s1 = [d for d in data if d.get("stage") == 1]
        s2 = [d for d in data if d.get("stage") == 2]

        s1_path = input_dir / f"{base}_stage1.json"
        s2_path = input_dir / f"{base}_stage2.json"

        with open(s1_path, "w") as f:
            json.dump(s1, f, indent=2, ensure_ascii=False)
        with open(s2_path, "w") as f:
            json.dump(s2, f, indent=2, ensure_ascii=False)

        s1_real = sum(1 for d in s1 if d.get("is_fake") == 0)
        s1_fake = sum(1 for d in s1 if d.get("is_fake") == 1)
        s2_real = sum(1 for d in s2 if d.get("is_fake") == 0)
        s2_fake = sum(1 for d in s2 if d.get("is_fake") == 1)

        print(f"{base}.json: {len(data)} total")
        print(f"  → {s1_path.name}: {len(s1)} (real={s1_real}, fake={s1_fake})")
        print(f"  → {s2_path.name}: {len(s2)} (real={s2_real}, fake={s2_fake})")


if __name__ == "__main__":
    main()
