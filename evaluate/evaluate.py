"""
Evaluate model predictions against ground truth annotations.

Usage:
    python evaluate.py [--results_dir DIR] [--annotation_path PATH]
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


CRITERIA = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common-Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

# Alternative names the model might use (map to canonical)
CRITERIA_ALIASES = {
    "Physical & Common Sense Logic": "Physical & Common-Sense Logic",
    "Lighting & Shadow Consistency": "Lighting & Shadows Consistency",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AIGC detection results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing per-image JSON results")
    parser.add_argument("--annotation_path", type=str,
                        default="dataset/sample dataset/annotation.json",
                        help="Path to ground truth annotation file")
    parser.add_argument("--postprocess", action="store_true",
                        help="Override overall_likelihood using per-criterion scores "
                             "(fixes model not following instructions)")
    return parser.parse_args()


def normalize_criterion(name):
    """Normalize criterion name to canonical form."""
    name = name.strip()
    return CRITERIA_ALIASES.get(name, name)


def binarize_gt_score(score):
    """Map ground truth score (0/1/2) to binary (0 or 1). Score >= 1 means AI artifact detected."""
    return 0 if score == 0 else 1


def evaluate():
    args = parse_args()

    # Load ground truth (only Fake images have annotations)
    with open(args.annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} ground truth annotations (Fake images)")

    # Load prediction results
    results_dir = Path(args.results_dir)
    each_image_dir = results_dir / "each_image_result"
    full_results_path = each_image_dir / "_full_results.json"

    predictions = {}
    if full_results_path.exists():
        # New format: each_image_result/_full_results.json
        with open(full_results_path, "r", encoding="utf-8") as f:
            predictions = json.load(f).get("results", {})
    elif each_image_dir.exists():
        # New format: load from individual files
        for f in each_image_dir.glob("*.json"):
            if f.name.startswith("_"):
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            key = data.get("annotation_key", f.stem)
            predictions[key] = data
    elif (results_dir / "summary.json").exists():
        # Old format: summary.json with embedded results
        with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        predictions = summary.get("results", {})
    else:
        # Old format: individual files in root
        for f in results_dir.glob("*.json"):
            if f.name in ("summary.json", "eval_results.json"):
                continue
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            key = data.get("annotation_key", f.stem)
            predictions[key] = data

    print(f"Loaded {len(predictions)} predictions")

    # === Evaluate Overall Likelihood (Real vs Fake classification) ===
    overall_correct = 0
    overall_total = 0
    tp = fp = tn = fn = 0

    for key, pred in predictions.items():
        gt_label = pred.get("ground_truth_label", "Unknown")
        parsed = pred.get("parsed_json")
        if parsed is None:
            continue

        pred_likelihood = parsed.get("overall_likelihood", "").strip()

        # Post-processing: override overall_likelihood based on per-criterion scores
        # (the model often ignores the stage2 instruction to mark AI-Generated when any score=1)
        if args.postprocess and "per_criterion" in parsed:
            aigc_count = sum(
                1 for c in parsed["per_criterion"]
                if c.get("aigc score", c.get("score", 0)) == 1
            )
            if aigc_count >= 1:
                pred_likelihood = "AI-Generated"
            else:
                pred_likelihood = "Real"

        overall_total += 1

        # Ground truth: Real images are "Real", Fake images are "AI-Generated"
        gt_is_fake = (gt_label == "Fake")
        pred_is_fake = (pred_likelihood == "AI-Generated")

        if gt_is_fake == pred_is_fake:
            overall_correct += 1

        if gt_is_fake and pred_is_fake:
            tp += 1
        elif not gt_is_fake and pred_is_fake:
            fp += 1
        elif not gt_is_fake and not pred_is_fake:
            tn += 1
        elif gt_is_fake and not pred_is_fake:
            fn += 1

    print(f"\n{'='*60}")
    print("OVERALL LIKELIHOOD ACCURACY")
    print(f"{'='*60}")
    if overall_total > 0:
        acc = overall_correct / overall_total * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  Accuracy:  {overall_correct}/{overall_total} = {acc:.1f}%")
        print(f"  Precision: {precision:.1f}%  (TP={tp}, FP={fp})")
        print(f"  Recall:    {recall:.1f}%  (TP={tp}, FN={fn})")
        print(f"  F1 Score:  {f1:.1f}%")
        print(f"  TN={tn} (Real correctly classified)")
    else:
        print("  No valid predictions found!")

    # === Evaluate Per-Criterion Scores (Fake images only) ===
    print(f"\n{'='*60}")
    print("PER-CRITERION ACCURACY (Fake images with annotations)")
    print(f"{'='*60}")

    criterion_stats = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for key in annotations:
        if key not in predictions:
            continue
        pred = predictions[key]
        parsed = pred.get("parsed_json")
        if parsed is None:
            continue

        gt_criteria = {c["criterion"]: c["score"] for c in annotations[key]["per_criterion"]}
        pred_criteria = {}
        for c in parsed.get("per_criterion", []):
            name = normalize_criterion(c.get("criterion", ""))
            # Handle both "score" and "aigc score" keys
            score = c.get("aigc score", c.get("score", None))
            if score is not None:
                pred_criteria[name] = int(score)

        for criterion in CRITERIA:
            if criterion in gt_criteria and criterion in pred_criteria:
                gt_bin = binarize_gt_score(gt_criteria[criterion])
                pred_score = pred_criteria[criterion]
                stats = criterion_stats[criterion]
                stats["total"] += 1
                if gt_bin == pred_score:
                    stats["correct"] += 1
                if gt_bin == 1 and pred_score == 1:
                    stats["tp"] += 1
                elif gt_bin == 0 and pred_score == 1:
                    stats["fp"] += 1
                elif gt_bin == 0 and pred_score == 0:
                    stats["tn"] += 1
                elif gt_bin == 1 and pred_score == 0:
                    stats["fn"] += 1

    total_criterion_correct = 0
    total_criterion_total = 0
    for criterion in CRITERIA:
        stats = criterion_stats[criterion]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            total_criterion_correct += stats["correct"]
            total_criterion_total += stats["total"]
            print(f"  {criterion:<42s} {stats['correct']}/{stats['total']} = {acc:.0f}%"
                  f"  (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']})")
        else:
            print(f"  {criterion:<42s} No data")

    if total_criterion_total > 0:
        overall_crit_acc = total_criterion_correct / total_criterion_total * 100
        print(f"\n  Overall criterion accuracy: {total_criterion_correct}/{total_criterion_total} = {overall_crit_acc:.1f}%")

    # === Per-image breakdown (with postprocessing applied) ===
    print(f"\n{'='*60}")
    print("PER-IMAGE RESULTS")
    print(f"{'='*60}")
    per_image_results = []
    img_correct_count = 0
    img_total_count = 0
    for key, pred in sorted(predictions.items()):
        parsed = pred.get("parsed_json")
        if parsed is None:
            continue
        gt_label = pred.get("ground_truth_label", "?")
        pred_likelihood = parsed.get("overall_likelihood", "?")

        # Apply same postprocessing as the metrics above
        if args.postprocess and "per_criterion" in parsed:
            aigc_count = sum(
                1 for c in parsed["per_criterion"]
                if c.get("aigc score", c.get("score", 0)) == 1
            )
            if aigc_count >= 1:
                pred_likelihood = "AI-Generated"
            else:
                pred_likelihood = "Real"

        is_correct = (gt_label == "Fake" and pred_likelihood == "AI-Generated") or \
                     (gt_label == "Real" and pred_likelihood in ("Real", "Uncertain"))
        correct_str = "OK" if is_correct else "WRONG"
        img_total_count += 1
        if is_correct:
            img_correct_count += 1

        time_s = pred.get("inference_time_sec", "?")
        print(f"  {key:<45s} GT={gt_label:<5s} Pred={pred_likelihood:<14s} {correct_str}  ({time_s}s)")
        per_image_results.append({
            "image": key, "gt": gt_label, "pred": pred_likelihood, "correct": is_correct
        })

    real_acc = round(img_correct_count / img_total_count * 100, 2) if img_total_count > 0 else 0

    # === Save evaluation results ===
    eval_results = {
        "overall": {
            "accuracy": real_acc,
            "precision": round(precision, 2) if overall_total > 0 else 0,
            "recall": round(recall, 2) if overall_total > 0 else 0,
            "f1": round(f1, 2) if overall_total > 0 else 0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total": overall_total,
        },
        "per_criterion": {
            c: dict(criterion_stats[c]) for c in CRITERIA
        },
        "criterion_accuracy": round(overall_crit_acc, 2) if total_criterion_total > 0 else 0,
        "per_image": per_image_results,
        "postprocess": args.postprocess,
    }

    results_dir = Path(args.results_dir)
    eval_path = results_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Evaluation saved to: {eval_path}")


if __name__ == "__main__":
    evaluate()
