"""
Add generated Infinity and Janus images to an existing finetune dataset.

All generated images are AI-generated (is_fake=1).
Creates both stage-1 (analysis) and stage-2 (JSON synthesis) entries.

Usage:
    python annotation/add_generated_images_to_finetune.py \
        --finetune_dir dataset/finetune_qwen35_18k \
        --generated_dirs dataset/generated_infinity dataset/generated_janus
"""

import argparse
import json
import glob
import random
from pathlib import Path


STAGE1_PROMPT = (
    '"Is this image real or fake? Think step-by-step before giving a conclusion. '
    'Please analyze based on the following three aspects: Edge & Boundary Integrity, '
    'Texture & Resolution Coherence, Material & Object Detail Fidelity.\n'
    '"Is this image real or fake? Think step-by-step before giving a conclusion. '
    'Please analyze based on the following three aspects: Physical & Common Sense Logic, '
    'Text & Symbol Authenticity, Human & Biological Structure Integrity.\n'
    '"Is this image real or fake? Think step-by-step before giving a conclusion. '
    'Please analyze based on the following two aspects: Lighting & Shadow Consistency, '
    'Perspective & Spatial Accuracy.'
)

STAGE2_PROMPT_TEMPLATE = """*** INSTRUCTIONS ***
TASK:
You are given multiple analytical answer excerpts about the input image. Your task is to synthesize ALL provided answers and produce a single JSON object in the EXACT schema below—no extra text, no comments.

OUTPUT SCHEMA (must match structure EXACTLY):
{{
  "per_criterion": [
    {{
      "criterion": "Lighting & Shadows Consistency",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Edges & Boundaries",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Texture & Resolution",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Perspective & Spatial Relationships",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Physical & Common Sense Logic",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Text & Symbols",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Human & Biological Structure Integrity",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }},
    {{
      "criterion": "Material & Object Details",
      "evidence": your text here,
      "aigc score": 0 or 1,
    }}
  ],
  "overall_likelihood": "Real" | "Uncertain" | "AI-Generated",
}}

STRICT PER Criterion SCORING GUIDELINES:
- aigc score=0, evidence mentions realism for this criterion, no AIGC artifacts mentioned for score=0.
- aigc score=1, evidence MUST explicitly describe AIGC artifacts or contradictions for this criterion (e.g., "gibberish text," "extra finger," "inconsistent shadows," "warped perspective"). Only assign 1 when you can quote or paraphrase a specific artifact claim.

GUIDELINES for overall_likelihood:
Determine the nature of the image based on whether each criterion has AIGC evidence.
If ONE or MORE THAN ONE criterion has an AIGC score=1, then this image is AI-generated.

STRICT OUTPUT RULES:
- Output **ONLY** the JSON object in the exact schema above.
- Preserve the criterion names **exactly**

*** ANALYSIS DATA TO PROCESS ***
{analysis}
"""

# Category-specific artifact descriptions for more realistic training data
CATEGORY_ARTIFACTS = {
    "Animal": {
        "stage1": (
            "- Edge & Boundary Integrity: Fur edges appear artificially smooth with unnatural blending into background.\n"
            "- Texture & Resolution Coherence: Animal fur texture is overly uniform, lacking natural variation in strand direction.\n"
            "- Material & Object Detail Fidelity: Eyes have an unnatural glassy quality; whiskers lack individual strand definition.\n"
            "- Physical & Common Sense Logic: Animal posture appears plausible but proportions are slightly off.\n"
            "- Text & Symbol Authenticity: No text present in the image.\n"
            "- Human & Biological Structure Integrity: Animal anatomy shows subtle distortions in limb joints.\n"
            "- Lighting & Shadow Consistency: Shadow direction is inconsistent between the animal and surrounding elements.\n"
            "- Perspective & Spatial Accuracy: Depth of field appears artificially rendered with unnatural bokeh patterns.\n\n"
            "The image is AI-GENERATED."
        ),
        "evidence": {
            "Lighting & Shadows Consistency": "Shadow direction is inconsistent between the animal and surrounding elements; light source appears artificially placed.",
            "Edges & Boundaries": "Fur edges appear artificially smooth with unnatural blending into background; boundary artifacts visible.",
            "Texture & Resolution": "Animal fur texture is overly uniform, lacking natural variation in strand direction and density.",
            "Perspective & Spatial Relationships": "Depth of field appears artificially rendered with unnatural bokeh patterns.",
            "Physical & Common Sense Logic": "Animal proportions are slightly distorted; posture appears subtly unnatural.",
            "Text & Symbols": "No text present in the image.",
            "Human & Biological Structure Integrity": "Animal anatomy shows subtle distortions in limb joints and body proportions.",
            "Material & Object Details": "Eyes have an unnatural glassy quality; whiskers and fine details lack individual definition.",
        },
    },
    "Food": {
        "stage1": (
            "- Edge & Boundary Integrity: Food edges blend unnaturally into the plate; garnish boundaries are smeared.\n"
            "- Texture & Resolution Coherence: Surface textures are overly smooth; food lacks natural imperfections.\n"
            "- Material & Object Detail Fidelity: Liquid surfaces show unrealistic reflections; sauce appears painted on.\n"
            "- Physical & Common Sense Logic: Food arrangement is plausible but portions appear unnaturally perfect.\n"
            "- Text & Symbol Authenticity: No text or symbols present.\n"
            "- Human & Biological Structure Integrity: No humans present in the scene.\n"
            "- Lighting & Shadow Consistency: Highlights on food surfaces are inconsistent with ambient lighting direction.\n"
            "- Perspective & Spatial Accuracy: Plate edges and table perspective show minor warping artifacts.\n\n"
            "The image is AI-GENERATED."
        ),
        "evidence": {
            "Lighting & Shadows Consistency": "Highlights on food surfaces are inconsistent with ambient lighting direction; specular reflections appear artificial.",
            "Edges & Boundaries": "Food edges blend unnaturally into the plate; garnish boundaries are smeared and lack definition.",
            "Texture & Resolution": "Surface textures are overly smooth; food lacks natural imperfections and grain.",
            "Perspective & Spatial Relationships": "Plate edges and table perspective show minor warping artifacts.",
            "Physical & Common Sense Logic": "Food arrangement is overly perfect; portions lack natural imperfections.",
            "Text & Symbols": "No text or symbols present in the image.",
            "Human & Biological Structure Integrity": "No humans present in the scene.",
            "Material & Object Details": "Liquid surfaces show unrealistic reflections; sauce and glazing appear artificially painted.",
        },
    },
    "CityLandscape": {
        "stage1": (
            "- Edge & Boundary Integrity: Building edges show warping and inconsistent lines; window frames are distorted.\n"
            "- Texture & Resolution Coherence: Building facades have repetitive patterns with unnatural uniformity.\n"
            "- Material & Object Detail Fidelity: Glass reflections are inconsistent; architectural details are blurred.\n"
            "- Physical & Common Sense Logic: Building scale and placement appear plausible but windows are irregular.\n"
            "- Text & Symbol Authenticity: Signage text is garbled or illegible upon close inspection.\n"
            "- Human & Biological Structure Integrity: Distant pedestrians appear as formless blobs without structure.\n"
            "- Lighting & Shadow Consistency: Shadows from buildings point in conflicting directions.\n"
            "- Perspective & Spatial Accuracy: Vanishing points do not converge correctly; buildings lean unnaturally.\n\n"
            "The image is AI-GENERATED."
        ),
        "evidence": {
            "Lighting & Shadows Consistency": "Shadows from buildings point in conflicting directions; ambient lighting is inconsistent.",
            "Edges & Boundaries": "Building edges show warping and inconsistent lines; window frames are distorted.",
            "Texture & Resolution": "Building facades have repetitive patterns with unnatural uniformity across surfaces.",
            "Perspective & Spatial Relationships": "Vanishing points do not converge correctly; buildings lean unnaturally.",
            "Physical & Common Sense Logic": "Window placement is irregular; building scale relationships appear inconsistent.",
            "Text & Symbols": "Signage text is garbled or illegible upon close inspection; characters appear nonsensical.",
            "Human & Biological Structure Integrity": "Distant pedestrians appear as formless blobs without anatomical structure.",
            "Material & Object Details": "Glass reflections are inconsistent with surroundings; architectural details are blurred.",
        },
    },
    "Merchandise": {
        "stage1": (
            "- Edge & Boundary Integrity: Product edges are unnaturally sharp or blurred; boundaries with background are inconsistent.\n"
            "- Texture & Resolution Coherence: Material textures appear synthetically smooth; leather or fabric lacks natural grain.\n"
            "- Material & Object Detail Fidelity: Stitching patterns are irregular; brand markings appear fabricated.\n"
            "- Physical & Common Sense Logic: Object proportions are subtly wrong; functional elements are non-functional.\n"
            "- Text & Symbol Authenticity: Brand text or labels show character-level distortions.\n"
            "- Human & Biological Structure Integrity: No humans present in the scene.\n"
            "- Lighting & Shadow Consistency: Specular highlights do not match the apparent light source position.\n"
            "- Perspective & Spatial Accuracy: Object surfaces show slight curvature inconsistencies.\n\n"
            "The image is AI-GENERATED."
        ),
        "evidence": {
            "Lighting & Shadows Consistency": "Specular highlights do not match the apparent light source position; shadows are inconsistent.",
            "Edges & Boundaries": "Product edges are unnaturally sharp or blurred; boundaries with background are inconsistent.",
            "Texture & Resolution": "Material textures appear synthetically smooth; leather or fabric lacks natural grain and wear.",
            "Perspective & Spatial Relationships": "Object surfaces show slight curvature inconsistencies and warping.",
            "Physical & Common Sense Logic": "Object proportions are subtly wrong; functional elements appear non-functional.",
            "Text & Symbols": "Brand text or labels show character-level distortions; text appears fabricated.",
            "Human & Biological Structure Integrity": "No humans present in the scene.",
            "Material & Object Details": "Stitching patterns are irregular; material transitions appear artificially blended.",
        },
    },
    "NaturalScenery": {
        "stage1": (
            "- Edge & Boundary Integrity: Treeline and horizon edges are unnaturally smooth; foliage boundaries lack definition.\n"
            "- Texture & Resolution Coherence: Ground and vegetation textures are repetitive with visible tiling artifacts.\n"
            "- Material & Object Detail Fidelity: Water surfaces lack natural caustic patterns; rock faces appear plasticky.\n"
            "- Physical & Common Sense Logic: Scene composition is plausible but weather elements are inconsistent.\n"
            "- Text & Symbol Authenticity: No text present in the scene.\n"
            "- Human & Biological Structure Integrity: No humans or animals present.\n"
            "- Lighting & Shadow Consistency: Cloud lighting does not match ground shadow patterns; golden hour effect is artificial.\n"
            "- Perspective & Spatial Accuracy: Atmospheric perspective is missing; distant objects lack expected haze.\n\n"
            "The image is AI-GENERATED."
        ),
        "evidence": {
            "Lighting & Shadows Consistency": "Cloud lighting does not match ground shadow patterns; golden hour effect appears artificially applied.",
            "Edges & Boundaries": "Treeline and horizon edges are unnaturally smooth; foliage boundaries lack natural definition.",
            "Texture & Resolution": "Ground and vegetation textures are repetitive with visible tiling artifacts.",
            "Perspective & Spatial Relationships": "Atmospheric perspective is missing; distant objects lack expected haze and depth.",
            "Physical & Common Sense Logic": "Weather elements are inconsistent with scene composition; natural phenomena appear off.",
            "Text & Symbols": "No text present in the scene.",
            "Human & Biological Structure Integrity": "No humans or animals present in the scene.",
            "Material & Object Details": "Water surfaces lack natural caustic patterns; rock faces appear plasticky and uniform.",
        },
    },
}

# Default fallback
DEFAULT_ARTIFACTS = CATEGORY_ARTIFACTS["NaturalScenery"]


def make_stage1_entry(image_path, category):
    """Create a stage-1 training entry for an AI-generated image."""
    artifacts = CATEGORY_ARTIFACTS.get(category, DEFAULT_ARTIFACTS)
    return {
        "stage": 1,
        "is_fake": 1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": STAGE1_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": artifacts["stage1"]},
                ],
            },
        ],
    }


def make_stage2_entry(category):
    """Create a stage-2 training entry for an AI-generated image."""
    artifacts = CATEGORY_ARTIFACTS.get(category, DEFAULT_ARTIFACTS)

    analysis = artifacts["stage1"]
    stage2_prompt = STAGE2_PROMPT_TEMPLATE.format(analysis=analysis)

    result = {
        "per_criterion": [],
        "overall_likelihood": "AI-Generated",
    }
    for criterion, evidence in artifacts["evidence"].items():
        result["per_criterion"].append({
            "criterion": criterion,
            "evidence": evidence,
            "aigc score": 1,
        })

    return {
        "stage": 2,
        "is_fake": 1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": stage2_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)},
                ],
            },
        ],
    }


def collect_generated_images(generated_dir):
    """Collect all images from a generated directory organized by category."""
    gen_path = Path(generated_dir)
    images_by_category = {}

    for category_dir in sorted(gen_path.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        imgs = sorted(
            glob.glob(str(category_dir / "*.png"))
            + glob.glob(str(category_dir / "*.jpg"))
            + glob.glob(str(category_dir / "*.jpeg"))
        )
        if imgs:
            images_by_category[category] = imgs

    return images_by_category


def main():
    parser = argparse.ArgumentParser(
        description="Add generated AI images to existing finetune dataset"
    )
    parser.add_argument(
        "--finetune_dir",
        type=str,
        default="dataset/finetune_qwen35_18k",
        help="Path to existing finetune dataset directory",
    )
    parser.add_argument(
        "--generated_dirs",
        nargs="+",
        default=["dataset/generated_infinity", "dataset/generated_janus"],
        help="Paths to generated image directories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: overwrites finetune_dir)",
    )
    args = parser.parse_args()

    finetune_dir = Path(args.finetune_dir)
    output_dir = Path(args.output_dir) if args.output_dir else finetune_dir

    # Load existing data
    train_path = finetune_dir / "train.json"
    val_path = finetune_dir / "val.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    print(f"Existing train: {len(train_data)}, val: {len(val_data)}")

    # Collect new entries
    new_entries = []
    for gen_dir in args.generated_dirs:
        print(f"\nProcessing: {gen_dir}")
        images_by_cat = collect_generated_images(gen_dir)

        for category, imgs in images_by_cat.items():
            print(f"  {category}: {len(imgs)} images")
            for img_path in imgs:
                # Stage 1: image + analysis
                entry1 = make_stage1_entry(img_path, category)
                new_entries.append(entry1)

                # Stage 2: text-only synthesis (one per category, not per image)
            # Add one stage-2 entry per category
            entry2 = make_stage2_entry(category)
            new_entries.append(entry2)

    print(f"\nNew entries: {len(new_entries)}")

    # Split new entries: 90% train, 10% val
    random.seed(42)
    random.shuffle(new_entries)
    val_count = max(1, int(len(new_entries) * 0.1))
    new_val = new_entries[:val_count]
    new_train = new_entries[val_count:]

    # Append to existing data
    train_data.extend(new_train)
    val_data.extend(new_val)

    # Shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_train = output_dir / "train.json"
    out_val = output_dir / "val.json"

    with open(out_train, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(out_val, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    # Stats
    train_fake = sum(1 for c in train_data if c.get("is_fake") == 1)
    val_fake = sum(1 for c in val_data if c.get("is_fake") == 1)
    print(f"\nUpdated dataset:")
    print(f"  Train: {len(train_data)} ({train_fake} fake, {len(train_data)-train_fake} real)")
    print(f"  Val:   {len(val_data)} ({val_fake} fake, {len(val_data)-val_fake} real)")
    print(f"  Saved: {out_train}, {out_val}")

if __name__ == "__main__":
    main()