"""
Package submission files into the required zip format for LPCVC 2026 Track 3.

Usage:
    python prepare_submission.py --team_name YOUR_TEAM --tutorial_dir /path/to/Tutorial_for_Qwen2_VL_2b_IoT

If you don't have the tutorial outputs yet, use --from_sample to package the sample solution instead:
    python prepare_submission.py --team_name YOUR_TEAM --from_sample
"""

import argparse
import json
import os
import shutil
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LPCVC Track 3 submission")
    parser.add_argument("--team_name", type=str, required=True, help="Your team name")
    parser.add_argument("--tutorial_dir", type=str, default=None,
                        help="Path to Tutorial_for_Qwen2_VL_2b_IoT directory")
    parser.add_argument("--from_sample", action="store_true",
                        help="Use pre-built sample_solution/ instead of tutorial outputs")
    parser.add_argument("--inputs_json", type=str, default=None,
                        help="Path to custom inputs.json (optional)")
    parser.add_argument("--output_dir", type=str, default="output/submission",
                        help="Output directory for the submission zip")
    return parser.parse_args()


def validate_submission(submission_dir):
    """Check that all required files are present."""
    required_patterns = {
        "ar*-ar*-cl*/weight_sharing_model_*.serialized.bin": "Quantized LLM binary",
        "serialized_binaries/veg.serialized.bin": "Vision encoder binary",
        "embedding_weights*.raw": "Embedding weights",
        "inputs.json": "Model configuration",
        "tokenizer.json": "Tokenizer",
        "position_ids_cos.raw": "Position IDs (cos)",
        "position_ids_sin.raw": "Position IDs (sin)",
    }
    # mask.raw OR (full_attention_mask.raw + window_attention_mask.raw)
    mask_patterns = {
        "mask.raw": "Attention mask (Qwen2-VL)",
        "full_attention_mask.raw": "Full attention mask (Qwen2.5-VL)",
    }

    all_ok = True
    for pattern, desc in required_patterns.items():
        matches = glob.glob(str(submission_dir / pattern))
        if not matches:
            print(f"  MISSING: {pattern} ({desc})")
            all_ok = False
        else:
            for m in matches:
                size_mb = os.path.getsize(m) / (1024 * 1024)
                print(f"  OK: {os.path.relpath(m, submission_dir)} ({size_mb:.1f} MB)")

    # Check mask
    mask_found = False
    for pattern, desc in mask_patterns.items():
        matches = glob.glob(str(submission_dir / pattern))
        if matches:
            mask_found = True
            for m in matches:
                size_mb = os.path.getsize(m) / (1024 * 1024)
                print(f"  OK: {os.path.relpath(m, submission_dir)} ({size_mb:.1f} MB)")
    if not mask_found:
        print(f"  MISSING: mask.raw or full_attention_mask.raw + window_attention_mask.raw")
        all_ok = False

    # Count totals
    subdirs = [d for d in submission_dir.iterdir() if d.is_dir()]
    json_files = list(submission_dir.glob("*.json"))
    raw_files = list(submission_dir.glob("*.raw"))
    print(f"\n  Subdirectories: {len(subdirs)} (need 2)")
    print(f"  JSON files: {len(json_files)} (need 2)")
    print(f"  RAW files: {len(raw_files)} (need 4+)")

    return all_ok


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    submission_dir = output_dir / args.team_name
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True)

    if args.from_sample:
        # Use pre-built sample solution
        sample_dir = Path("sample_solution")
        if not sample_dir.exists():
            print("ERROR: sample_solution/ directory not found.")
            print("Download it from the link in README.md and unzip here.")
            return

        print(f"Copying from sample_solution/ ...")
        for item in sample_dir.iterdir():
            dest = submission_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    elif args.tutorial_dir:
        tutorial = Path(args.tutorial_dir)
        if not tutorial.exists():
            print(f"ERROR: Tutorial directory not found: {tutorial}")
            return

        print(f"Collecting files from tutorial: {tutorial}")

        # Example1A outputs
        e1a_veg = tutorial / "example1/Example1A/output_dir/veg_exports"
        for f in ["mask.raw", "position_ids_cos.raw", "position_ids_sin.raw"]:
            src = e1a_veg / f
            if src.exists():
                shutil.copy2(src, submission_dir / f)
                print(f"  Copied: {f}")
            else:
                print(f"  WARNING: {src} not found")

        # Example1B outputs
        e1b_out = tutorial / "example1/Example1B/output_dir"
        embed_files = glob.glob(str(e1b_out / "embedding_weights*.raw"))
        if embed_files:
            for ef in embed_files:
                shutil.copy2(ef, submission_dir / os.path.basename(ef))
                print(f"  Copied: {os.path.basename(ef)}")
        else:
            print(f"  WARNING: embedding_weights*.raw not found in {e1b_out}")

        tokenizer_src = e1b_out / "tokenizer/tokenizer.json"
        if tokenizer_src.exists():
            shutil.copy2(tokenizer_src, submission_dir / "tokenizer.json")
            print(f"  Copied: tokenizer.json")

        # Example2A outputs — VEG serialized binary
        veg_dir = tutorial / "example2/Example2A/host_linux/exports/serialized_binaries"
        dest_veg_dir = submission_dir / "serialized_binaries"
        if veg_dir.exists():
            shutil.copytree(veg_dir, dest_veg_dir)
            print(f"  Copied: serialized_binaries/")
        else:
            print(f"  WARNING: {veg_dir} not found")

        # Example2B outputs — LLM serialized binary
        artifacts_dir = tutorial / "example2/Example2B/host_linux/assets/artifacts"
        if artifacts_dir.exists():
            ar_dirs = glob.glob(str(artifacts_dir / "ar*-ar*-cl*"))
            for ar_dir in ar_dirs:
                dest_ar = submission_dir / os.path.basename(ar_dir)
                shutil.copytree(ar_dir, dest_ar)
                print(f"  Copied: {os.path.basename(ar_dir)}/")
        else:
            print(f"  WARNING: {artifacts_dir} not found")
    else:
        print("ERROR: Provide either --tutorial_dir or --from_sample")
        return

    # Copy or create inputs.json
    if args.inputs_json:
        shutil.copy2(args.inputs_json, submission_dir / "inputs.json")
        print(f"  Copied custom inputs.json")
    elif not (submission_dir / "inputs.json").exists():
        # Copy from sample
        sample_inputs = Path("26LPCVC_Track3_Sample_Solution/contestant_uploads/inputs.json")
        if sample_inputs.exists():
            shutil.copy2(sample_inputs, submission_dir / "inputs.json")
            print(f"  Copied inputs.json from sample solution")
        else:
            print("  WARNING: No inputs.json found! You need to create one.")

    # Validate
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")
    all_ok = validate_submission(submission_dir)

    if all_ok:
        # Create zip
        zip_path = output_dir / f"{args.team_name}.zip"
        shutil.make_archive(str(output_dir / args.team_name), 'zip', output_dir, args.team_name)
        zip_size = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"\n  Submission ready: {zip_path} ({zip_size:.0f} MB)")
    else:
        print(f"\n  VALIDATION FAILED — fix missing files before submitting")


if __name__ == "__main__":
    main()
