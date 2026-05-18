"""
Estimate the LPCVC 2026 Track 3 server score locally.

Server score formula:
    Real images:       score = DetectionScore           (1 if overall_likelihood == "Real" else 0)
    AI-generated:      score = 0.5 * DetectionScore
                             + 0.25 * CriterionScore     (per-criterion exact match)
                             + 0.25 * EvidenceScore      (semantic similarity to GT evidence)
    Final score:       average across all images

Local evaluate.py only measures Detection + Criterion. This adds EvidenceScore
via sentence-transformers cosine similarity, giving a direct server-score
estimator.

Usage:
    pip install sentence-transformers   # one-time install (~80MB model)

    python evaluate/evaluate_server_score.py \
        --results_dir output/eval/my_run \
        --annotation_path dataset/sample_dataset/annotation.json

    # Compare two runs side by side:
    python evaluate/evaluate_server_score.py \
        --results_dir output/eval/my_run \
        --baseline_dir output/eval/baseline_run
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

CRITERIA_ALIASES = {
    "Physical & Common Sense Logic": "Physical & Common-Sense Logic",
    "Lighting & Shadow Consistency": "Lighting & Shadows Consistency",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate LPCVC server score")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with each_image_result/ subfolder")
    parser.add_argument("--annotation_path", type=str,
                        default="dataset/sample_dataset/annotation.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for evidence similarity")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Optional: baseline run for side-by-side comparison")
    parser.add_argument("--postprocess", action="store_true",
                        help="Override overall_likelihood from per-criterion (matches evaluate.py)")
    return parser.parse_args()


def normalize_criterion(name):
    name = (name or "").strip()
    return CRITERIA_ALIASES.get(name, name)


def binarize_gt_score(score):
    """GT score 0/1/2 → binary (0 or 1, where 1 means AI artifact present)."""
    return 0 if score == 0 else 1


def load_predictions(results_dir):
    """Load per-image prediction JSONs."""
    each_image_dir = Path(results_dir) / "each_image_result"
    if not each_image_dir.exists():
        raise FileNotFoundError(f"each_image_result/ not found in {results_dir}")

    predictions = {}
    for f in each_image_dir.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            key = data.get("annotation_key", f.stem)
            predictions[key] = data
        except Exception as e:
            print(f"  [warn] Failed to load {f.name}: {e}")
    return predictions


def get_pred_likelihood(pred, postprocess=False):
    """Extract predicted overall_likelihood, optionally overriding from per-criterion."""
    parsed = pred.get("parsed_json")
    if not parsed:
        return None
    label = (parsed.get("overall_likelihood") or "").strip()
    if postprocess and "per_criterion" in parsed:
        aigc_count = sum(
            1 for c in parsed["per_criterion"]
            if c.get("aigc score", c.get("score", 0)) == 1
        )
        label = "AI-Generated" if aigc_count >= 1 else "Real"
    return label


def compute_per_criterion_match(gt_per_criterion, pred_per_criterion):
    """Per-image criterion exact match accuracy (8 binary scores).

    Returns: (match_score in [0,1], num_criteria_compared)
    """
    gt_map = {c["criterion"]: binarize_gt_score(c["score"]) for c in gt_per_criterion}
    pred_map = {}
    for c in pred_per_criterion:
        name = normalize_criterion(c.get("criterion", ""))
        score = c.get("aigc score", c.get("score"))
        if score is not None:
            try:
                pred_map[name] = int(score)
            except (TypeError, ValueError):
                continue

    matches = 0
    total = 0
    for crit in CRITERIA:
        if crit in gt_map and crit in pred_map:
            total += 1
            if gt_map[crit] == pred_map[crit]:
                matches += 1
    return (matches / total if total > 0 else 0.0), total


def compute_evidence_similarity(model, gt_per_criterion, pred_per_criterion):
    """Per-image evidence semantic similarity averaged across criteria.

    Returns: (avg_similarity in [0,1], num_criteria_compared)
    """
    gt_map = {c["criterion"]: c.get("evidence", "") for c in gt_per_criterion}
    pred_map = {}
    for c in pred_per_criterion:
        name = normalize_criterion(c.get("criterion", ""))
        ev = c.get("evidence", "")
        if isinstance(ev, str):
            pred_map[name] = ev.strip().rstrip('"').strip()

    pairs = []
    for crit in CRITERIA:
        if crit in gt_map and crit in pred_map:
            gt = gt_map[crit].strip()
            pr = pred_map[crit].strip()
            if gt and pr:
                pairs.append((gt, pr))

    if not pairs:
        return 0.0, 0

    gt_texts = [p[0] for p in pairs]
    pred_texts = [p[1] for p in pairs]
    gt_emb = model.encode(gt_texts, convert_to_tensor=True, show_progress_bar=False)
    pred_emb = model.encode(pred_texts, convert_to_tensor=True, show_progress_bar=False)

    from sentence_transformers import util
    cos = util.cos_sim(gt_emb, pred_emb)
    # Diagonal = pairwise similarity for each criterion
    sims = [float(cos[i][i]) for i in range(len(pairs))]
    # Clamp to [0, 1] (cosine can be negative for unrelated text)
    sims = [max(0.0, min(1.0, s)) for s in sims]
    return sum(sims) / len(sims), len(sims)


def evaluate(results_dir, annotations, model, postprocess=False):
    """Run full server-score estimation for one results dir.

    Returns: dict with per-image scores and aggregates
    """
    predictions = load_predictions(results_dir)

    per_image = []
    real_correct = 0
    real_total = 0
    fake_total = 0
    sum_real = 0.0
    sum_fake_det = 0.0
    sum_fake_cri = 0.0
    sum_fake_evi = 0.0
    fake_with_cri = 0
    fake_with_evi = 0

    for key, pred in predictions.items():
        gt_label = pred.get("ground_truth_label", "Unknown")
        pred_likelihood = get_pred_likelihood(pred, postprocess=postprocess)
        if pred_likelihood is None:
            # Invalid JSON → score = 0 (per server format constraint)
            per_image.append({
                "image": key, "gt": gt_label, "pred": None,
                "detection": 0.0, "criterion": 0.0, "evidence": 0.0,
                "image_score": 0.0, "note": "invalid_json"
            })
            continue

        is_fake_gt = (gt_label == "Fake")
        is_fake_pred = (pred_likelihood == "AI-Generated")
        detection = 1.0 if is_fake_gt == is_fake_pred else 0.0

        if not is_fake_gt:
            # Real image: score = DetectionScore only
            real_total += 1
            real_correct += int(detection)
            sum_real += detection
            per_image.append({
                "image": key, "gt": "Real", "pred": pred_likelihood,
                "detection": detection, "criterion": None, "evidence": None,
                "image_score": detection
            })
        else:
            # AI-Generated: 0.5*Det + 0.25*Cri + 0.25*Evi
            fake_total += 1
            sum_fake_det += detection

            criterion_score = 0.0
            evidence_score = 0.0
            ann = annotations.get(key)
            if ann and ann.get("per_criterion"):
                pred_per_criterion = pred.get("parsed_json", {}).get("per_criterion", [])
                criterion_score, n_cri = compute_per_criterion_match(
                    ann["per_criterion"], pred_per_criterion
                )
                evidence_score, n_evi = compute_evidence_similarity(
                    model, ann["per_criterion"], pred_per_criterion
                )
                if n_cri > 0:
                    sum_fake_cri += criterion_score
                    fake_with_cri += 1
                if n_evi > 0:
                    sum_fake_evi += evidence_score
                    fake_with_evi += 1

            image_score = 0.5 * detection + 0.25 * criterion_score + 0.25 * evidence_score
            per_image.append({
                "image": key, "gt": "Fake", "pred": pred_likelihood,
                "detection": detection,
                "criterion": criterion_score,
                "evidence": evidence_score,
                "image_score": image_score
            })

    total_images = real_total + fake_total
    sum_total = sum(p["image_score"] for p in per_image)
    final_score = sum_total / total_images if total_images > 0 else 0.0

    return {
        "per_image": per_image,
        "real_total": real_total,
        "real_correct": real_correct,
        "real_avg_detection": sum_real / real_total if real_total else 0,
        "fake_total": fake_total,
        "fake_avg_detection": sum_fake_det / fake_total if fake_total else 0,
        "fake_avg_criterion": sum_fake_cri / fake_with_cri if fake_with_cri else 0,
        "fake_avg_evidence": sum_fake_evi / fake_with_evi if fake_with_evi else 0,
        "total_images": total_images,
        "estimated_server_score": final_score,
    }


def print_report(label, result):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total images:     {result['total_images']}")
    print(f"")
    print(f"  Real images ({result['real_total']}):")
    print(f"    Detection avg:    {result['real_avg_detection']*100:5.1f}%  "
          f"(correct {result['real_correct']}/{result['real_total']})")
    print(f"")
    print(f"  AI-Generated images ({result['fake_total']}):")
    print(f"    Detection avg:    {result['fake_avg_detection']*100:5.1f}%")
    print(f"    Criterion avg:    {result['fake_avg_criterion']*100:5.1f}%")
    print(f"    Evidence avg:     {result['fake_avg_evidence']*100:5.1f}%")
    print(f"")
    print(f"  Per-image AI score = 0.5*Det + 0.25*Cri + 0.25*Evi")
    print(f"                     = 0.5*{result['fake_avg_detection']:.3f} + "
          f"0.25*{result['fake_avg_criterion']:.3f} + "
          f"0.25*{result['fake_avg_evidence']:.3f}")
    print(f"                     = {0.5*result['fake_avg_detection'] + 0.25*result['fake_avg_criterion'] + 0.25*result['fake_avg_evidence']:.3f}")
    print(f"")
    print(f"  ESTIMATED SERVER SCORE: {result['estimated_server_score']:.3f}")
    print(f"  {'='*68}")


def main():
    args = parse_args()

    print(f"Loading sentence-transformers model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"  Model loaded.")

    with open(args.annotation_path) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} ground-truth annotations\n")

    main_result = evaluate(args.results_dir, annotations, model, postprocess=args.postprocess)
    print_report(args.results_dir, main_result)

    if args.baseline_dir:
        baseline_result = evaluate(args.baseline_dir, annotations, model, postprocess=args.postprocess)
        print_report(f"BASELINE: {args.baseline_dir}", baseline_result)

        diff = main_result["estimated_server_score"] - baseline_result["estimated_server_score"]
        print(f"\n{'='*70}")
        print(f"  DIFF: {args.results_dir} vs baseline")
        print(f"{'='*70}")
        print(f"  Detection (real):  {(main_result['real_avg_detection'] - baseline_result['real_avg_detection'])*100:+.1f}%")
        print(f"  Detection (fake):  {(main_result['fake_avg_detection'] - baseline_result['fake_avg_detection'])*100:+.1f}%")
        print(f"  Criterion:         {(main_result['fake_avg_criterion'] - baseline_result['fake_avg_criterion'])*100:+.1f}%")
        print(f"  Evidence:          {(main_result['fake_avg_evidence'] - baseline_result['fake_avg_evidence'])*100:+.1f}%")
        print(f"  Server score:      {diff:+.3f}")

    # Save full breakdown
    out_path = Path(args.results_dir) / "server_score_estimate.json"
    with open(out_path, "w") as f:
        json.dump(main_result, f, indent=2)
    print(f"\n  Saved per-image breakdown to: {out_path}")


if __name__ == "__main__":
    main()
