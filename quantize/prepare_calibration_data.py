"""
Prepare mixed calibration data for quantization.
Mixes real (COCO) + fake (aigen) images for better quantization on AI detection task.

Output images are normalized to `.jpg` so the Qualcomm VEG calibration step sees a
consistent file format.

Usage:
    python quantize/prepare_calibration_data.py --real_n 50 --fake_n 50 --output_dir dataset/calibration
    python quantize/prepare_calibration_data.py --real_n 20 --fake_n 30 --output_dir dataset/calibration
"""

import argparse
import csv
import glob
import os
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def collect_images(directory: str) -> list[str]:
    images = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(images)


def save_as_jpg(src_path: str, dst_path: Path, quality: int = 95) -> None:
    with Image.open(src_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(dst_path, format="JPEG", quality=quality)


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed calibration data")
    parser.add_argument("--coco_dir", type=str, default="/dataset/coco/train2017",
                        help="Path to COCO train2017 images")
    parser.add_argument("--aigen_dir", type=str, default="dataset/aigen/train",
                        help="Path to aigen dataset")
    parser.add_argument("--sample_dir", type=str, default="dataset/sample_dataset/Fake",
                        help="Path to sample dataset fake images")
    parser.add_argument("--real_n", type=int, default=50, help="Number of real images")
    parser.add_argument("--fake_n", type=int, default=50, help="Number of fake images")
    parser.add_argument("--output_dir", type=str, default="dataset/calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previously prepared image files so the output folder is JPG-only.
    for pattern in IMAGE_EXTENSIONS:
        for old_file in output_dir.glob(pattern):
            old_file.unlink()

    # Collect real images from COCO
    print(f"Collecting real images from {args.coco_dir}...")
    real_images = sorted(glob.glob(os.path.join(args.coco_dir, "*.jpg")))
    if not real_images:
        print(f"  WARNING: No images found in {args.coco_dir}")
        print("  Trying alternative paths...")
        for alt in ["/dataset/coco/train2017", "dataset/coco/train2017", "/work/coco/train2017"]:
            real_images = sorted(glob.glob(os.path.join(alt, "*.jpg")))
            if real_images:
                print(f"  Found {len(real_images)} at {alt}")
                break
    print(f"  Available: {len(real_images)} real images")

    # Collect fake images from aigen + sample dataset
    print("Collecting fake images...")
    fake_images = []

    # From aigen shards
    aigen_dir = Path(args.aigen_dir)
    if aigen_dir.exists():
        for shard_dir in sorted(aigen_dir.glob("shard_*")):
            labels_file = shard_dir / "labels.csv"
            images_dir = shard_dir / "images"
            if not labels_file.exists():
                continue
            with open(labels_file, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["label"]) == 1:  # fake
                        img_path = images_dir / row["image_name"]
                        if img_path.exists():
                            fake_images.append(str(img_path))

    # From sample dataset
    fake_images.extend(collect_images(args.sample_dir))
    print(f"  Available: {len(fake_images)} fake images")

    # Sample
    sampled_real = random.sample(real_images, min(args.real_n, len(real_images)))
    sampled_fake = random.sample(fake_images, min(args.fake_n, len(fake_images)))

    # Convert all outputs to JPG so VEG calibration sees a consistent file type.
    count = 0
    for img_path in sampled_real:
        dst = output_dir / f"real_{count:04d}.jpg"
        save_as_jpg(img_path, dst)
        count += 1

    for img_path in sampled_fake:
        dst = output_dir / f"fake_{count:04d}.jpg"
        save_as_jpg(img_path, dst)
        count += 1

    print("Done!")
    print(f"  Real: {len(sampled_real)}")
    print(f"  Fake: {len(sampled_fake)}")
    print(f"  Total: {count} images")
    print(f"  Output: {output_dir}/")
    print("  Format: JPG only")
    print("Update veg_config.json:")
    print(f'  "calibration_images": "{output_dir}"')
    print(f'  "num_calibration_samples": {count}')


if __name__ == "__main__":
    main()
