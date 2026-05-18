"""
Generate AI images using FoundationVision/Infinity-8B for training data.

Infinity is a bitwise visual autoregressive model (CVPR 2025 Oral).
Generates 512x512 images by default (matching sample_dataset Infinity8B images).

Setup:
    # 1. Clone the Infinity repo
    git clone https://github.com/FoundationVision/Infinity.git
    cd Infinity
    pip install -e .

    # 2. Download weights (run from Infinity repo root)
    # Option A: Use huggingface-cli
    huggingface-cli download FoundationVision/Infinity \
        --include "infinity_2b_reg.pth" "infinity_vae_d32reg.pth" \
        --local-dir weights

    # Option B: For 8B model (needs ~33GB + 10GB for flan-t5-xl)
    huggingface-cli download FoundationVision/Infinity \
        --include "infinity_8b_512x512_weights/*" "infinity_vae_d56_f8_14_patchify.pth" \
        --local-dir weights
    huggingface-cli download google/flan-t5-xl --local-dir weights/flan-t5-xl

Usage:
    # From the Infinity repo root directory:

    # Quick test with 2B model (less VRAM, ~8GB)
    python /path/to/generate_infinity_images.py --model_size 2b --num_per_category 1

    # Full generation with 2B model
    python /path/to/generate_infinity_images.py --model_size 2b --num_per_category 30

    # 8B model (needs ~40GB VRAM)
    python /path/to/generate_infinity_images.py --model_size 8b --num_per_category 30

    # Specify GPU
    CUDA_VISIBLE_DEVICES=0 python /path/to/generate_infinity_images.py --model_size 2b

Requirements:
    torch>=2.5.1, transformers, flash_attn, timm==0.9.6, einops, opencv-python, kornia
"""

import argparse
import random
import sys
import numpy as np
import torch
import cv2
import os
from pathlib import Path

# Add Infinity repo to sys.path so `tools.run_infinity` is importable
_infinity_dir = str(Path(__file__).resolve().parent.parent / "Infinity")
if _infinity_dir not in sys.path:
    sys.path.insert(0, _infinity_dir)


# Same categories as JanusPro generation (matching sample_dataset)
PROMPTS = {
    "Animal": [
        "A golden retriever playing fetch in a sunny park",
        "A cat sleeping on a windowsill with sunlight streaming in",
        "A parrot perched on a branch in a tropical forest",
        "Two kittens playing with a ball of yarn on a carpet",
        "A deer standing in a misty forest at dawn",
        "A butterfly landing on a colorful flower in a garden",
        "A horse galloping across an open grassland",
        "An owl sitting on a tree branch at dusk",
        "A rabbit eating clover in a green meadow",
        "A dog sitting on a beach watching the sunset",
        "A squirrel holding an acorn on a tree trunk",
        "A flamingo standing in shallow water",
        "Three ducks swimming in a calm pond",
        "A tiger resting under a tree in the shade",
        "A panda eating bamboo in a forest",
        "A fox walking through autumn leaves in a forest",
        "A hummingbird hovering near a red flower",
        "A sea turtle swimming in clear blue ocean water",
        "A peacock displaying its colorful tail feathers",
        "A koala bear hugging a eucalyptus tree",
        "A polar bear walking on arctic ice",
        "Two dolphins jumping out of the ocean at sunset",
        "A chameleon sitting on a green branch",
        "A hedgehog curled up on fallen leaves",
        "An eagle soaring above a mountain valley",
        "A group of penguins standing on an ice shelf",
        "A red panda sleeping on a tree branch",
        "A jellyfish glowing in dark ocean water",
        "A white swan floating on a misty lake",
        "A baby elephant playing in a mud puddle",
        "A pack of wolves howling at the moon",
        "A tortoise slowly crossing a sandy path",
        "A macaw flying through a tropical canopy",
        "A school of clownfish swimming near an anemone",
        "A mountain goat standing on a rocky ledge",
        "A barn owl in flight over a wheat field at night",
        "A group of meerkats standing alert in the desert",
        "A lynx stalking through deep snow in a forest",
        "A stingray gliding over a sandy ocean floor",
        "A ladybug resting on a green leaf with dew drops",
        "A mother duck leading ducklings across a pond",
        "A leopard resting on a tree branch in the savanna",
        "A frog sitting on a lily pad in a calm pond",
        "A bison grazing on a wide open prairie",
        "A kingfisher diving into a stream to catch fish",
        "A sloth hanging from a tree branch in a rainforest",
        "An octopus camouflaging against ocean rocks",
        "A pair of swans forming a heart shape on a lake",
        "A cheetah running at full speed across the plains",
        "A snowy owl perched on a fence post in winter",
    ],
    "Food": [
        "A plate of sushi arranged beautifully on a wooden board",
        "A freshly baked pizza with melted cheese and basil",
        "A bowl of ramen with egg, nori, and green onions",
        "A colorful fruit salad in a glass bowl",
        "A cup of latte art coffee on a cafe table",
        "A stack of pancakes with maple syrup and berries",
        "A cheeseburger with fries on a restaurant table",
        "A slice of chocolate cake with raspberry garnish",
        "A bowl of pho with herbs and lime wedges",
        "An ice cream sundae with whipped cream and cherry",
        "A plate of tacos with fresh salsa and guacamole",
        "A croissant and espresso on a marble countertop",
        "Grilled salmon with asparagus on a white plate",
        "A colorful smoothie bowl with granola and fruit toppings",
        "A box of assorted macarons in pastel colors",
        "A wood-fired steak with roasted vegetables",
        "A bowl of tom yum soup with shrimp and mushrooms",
        "Fresh oysters on a bed of ice with lemon wedges",
        "A platter of cheese and grapes on a rustic board",
        "A glass of red wine next to a bruschetta appetizer",
        "Homemade cookies cooling on a baking rack",
        "A colorful poke bowl with salmon and avocado",
        "A plate of dumplings with dipping sauce",
        "A layered tiramisu in a glass dessert cup",
        "A basket of fresh bread rolls on a tablecloth",
        "A tray of baklava with pistachios and honey",
        "A bowl of bibimbap with vegetables and fried egg",
        "A plate of fish and chips with tartar sauce",
        "A matcha green tea latte with foam art",
        "A platter of fresh spring rolls with peanut sauce",
        "A wood board with charcuterie and olives",
        "A steaming bowl of miso soup with tofu and seaweed",
        "A plate of pad thai with peanuts and lime",
        "A glass of fresh orange juice on a breakfast table",
        "A tray of freshly baked cinnamon rolls with icing",
        "A plate of eggs benedict with hollandaise sauce",
        "A bowl of acai topped with banana and coconut flakes",
        "A plate of paella with shrimp and saffron rice",
        "A stack of waffles with strawberries and cream",
        "A bowl of green curry with vegetables and rice",
        "A platter of grilled vegetables with balsamic glaze",
        "A cup of hot chocolate with marshmallows on top",
        "A plate of lamb chops with mint sauce and potatoes",
        "A tray of freshly made sushi rolls being sliced",
        "A bowl of clam chowder in a sourdough bread bowl",
        "A glass jar of overnight oats with berries and nuts",
        "A plate of chicken tikka masala with naan bread",
        "A slice of key lime pie with whipped cream",
        "A platter of fried calamari with marinara sauce",
        "A bowl of Vietnamese bun cha with herbs and noodles",
    ],
    "CityLandscape": [
        "A city skyline at sunset with orange and purple clouds",
        "A busy street in Tokyo with neon signs at night",
        "A European cobblestone street with cafes and flowers",
        "A modern city reflected in a glass building facade",
        "An aerial view of a city park surrounded by skyscrapers",
        "A bridge over a river with city lights at twilight",
        "A rainy city street with reflections on wet pavement",
        "A historic cathedral in a European city square",
        "A waterfront promenade with boats and city buildings",
        "A narrow alley in an old Mediterranean town",
        "A subway station platform with arriving train",
        "A rooftop view of a cityscape with mountains behind",
        "A street market with colorful stalls and crowds",
        "A modern glass office building against a blue sky",
        "A park bench under autumn trees in an urban park",
        "A row of Victorian houses on a hilly street",
        "A canal in Amsterdam with bicycles and houseboats",
        "A futuristic city with flying cars and neon lights",
        "An old train station with ornate architecture",
        "A foggy London street with red telephone booth",
        "A busy intersection in New York City with yellow taxis",
        "A quiet residential street lined with cherry blossom trees",
        "A harbor at dawn with fishing boats and seagulls",
        "A castle on a hilltop overlooking a medieval town",
        "A modern shopping district with glass storefronts",
        "A street performer playing guitar in a city square",
        "An ancient Roman ruins with columns and arches",
        "A night market in Bangkok with food stalls and lights",
        "A snowy city street with holiday decorations",
        "A coastal town with colorful houses on a cliff",
        "A grand mosque illuminated at night in a city square",
        "A tram running along a tree-lined boulevard in autumn",
        "A floating market with boats loaded with produce",
        "A glass skyscraper reflecting clouds and surrounding buildings",
        "A pedestrian bridge lit with colorful LED lights at night",
        "An outdoor cafe with umbrella tables on a sunny plaza",
        "A graffiti-covered alley in a vibrant arts district",
        "A lighthouse on a rocky coast with waves below",
        "A ferris wheel glowing at night on a seaside pier",
        "A traditional Japanese temple gate surrounded by maple trees",
        "A flower market with rows of colorful bouquets",
        "A cable car climbing a steep city hill with bay views",
        "A winding mountain road through a small alpine village",
        "A university campus with ivy-covered brick buildings",
        "A fountain plaza in front of a baroque palace",
        "A riverboat passing under a stone bridge at sunset",
        "A neon-lit street in Seoul with K-pop billboards",
        "A gothic cathedral with stained glass windows at dusk",
        "A farmers market with fresh produce under white tents",
        "A monorail train passing through a futuristic downtown",
    ],
    "Merchandise": [
        "A pair of white sneakers on a wooden floor",
        "A luxury watch displayed on a velvet cushion",
        "A leather handbag on a marble surface",
        "Stacked books on a wooden shelf",
        "A pair of sunglasses on a beach towel",
        "A ceramic coffee mug with a minimalist design",
        "A vintage camera on a wooden desk",
        "A bottle of perfume with flower petals around it",
        "A set of colorful candles on a tray",
        "A woven basket with dried flowers",
        "A pair of earrings on a jewelry display",
        "A notebook and pen on a clean desk",
        "A potted succulent plant on a window ledge",
        "A glass vase with fresh tulips on a table",
        "A backpack leaning against a brick wall",
        "A pair of wireless earbuds in an open case",
        "A silk scarf draped over a wooden chair",
        "A set of paintbrushes in a ceramic holder",
        "A leather wallet on a dark wooden table",
        "A bicycle parked against a colorful mural",
        "A stack of vinyl records next to a turntable",
        "A yoga mat rolled up with a water bottle nearby",
        "A pair of hiking boots on a rocky surface",
        "A chess set made of marble on a table",
        "A decorative globe on a dark wood bookshelf",
        "A set of kitchen knives on a magnetic strip",
        "A telescope pointing at a starry sky",
        "A handmade ceramic bowl on a linen cloth",
        "A pair of running shoes on a track",
        "A leather journal with a brass clasp",
        "A standing desk lamp with a brass finish",
        "A set of colored pencils fanned out on paper",
        "A mechanical keyboard with RGB backlighting",
        "A pair of binoculars on a wooden windowsill",
        "A wristband fitness tracker on a gym towel",
        "A handwoven rug on a hardwood floor",
        "A crystal perfume bottle on a vanity mirror tray",
        "A vintage typewriter on an antique desk",
        "A stainless steel water bottle on a hiking trail",
        "A leather messenger bag on a park bench",
        "A set of watercolor paints with a brushed canvas",
        "A decorative table clock on a fireplace mantel",
        "A pair of gardening gloves beside potted herbs",
        "A drone sitting on a grassy field ready for flight",
        "A stack of board games on a living room shelf",
        "A hand-carved wooden sculpture on a pedestal",
        "A pair of ice skates hanging on a wooden wall",
        "A digital tablet with a stylus on a drawing desk",
        "A hammock tied between two trees in a backyard",
        "A ceramic teapot and cups on a bamboo tray",
    ],
    "NaturalScenery": [
        "A mountain lake reflecting snow-capped peaks at sunrise",
        "A waterfall cascading into a turquoise pool in a forest",
        "A field of lavender under a clear blue sky",
        "A desert landscape with sand dunes at golden hour",
        "A tropical beach with palm trees and crystal clear water",
        "A dense foggy forest with sunlight filtering through trees",
        "A snowy mountain peak against a clear winter sky",
        "A river winding through a green valley",
        "An aurora borealis over a frozen lake",
        "A meadow of wildflowers with mountains in the background",
        "A rocky coastline with waves crashing on cliffs",
        "A bamboo forest path with dappled sunlight",
        "A calm lake at sunset with a silhouette of trees",
        "A volcanic landscape with steam rising from the ground",
        "A rolling hillside with vineyards in autumn colors",
        "A frozen waterfall surrounded by icy rocks",
        "A misty morning over a rice paddy field",
        "A coral reef with colorful fish underwater",
        "A canyon with red rock formations and blue sky",
        "A cherry blossom grove beside a tranquil stream",
        "A lightning storm over an open prairie at night",
        "A glacier calving into an icy fjord",
        "A sunflower field stretching to the horizon",
        "A mangrove forest reflected in still water",
        "A starry night sky over a desert mesa",
        "A hot spring pool with steam and surrounding snow",
        "A rainbow over a lush green valley after rain",
        "A dense jungle canopy seen from above",
        "A tide pool with starfish and sea anemones",
        "A morning mist rising from a tropical rainforest",
        "A field of poppies waving in a gentle breeze",
        "A crystal-clear mountain stream with smooth pebbles",
        "A dormant volcano with a crater lake at the top",
        "A savanna landscape with acacia trees at golden hour",
        "A cave entrance with stalactites and green moss",
        "A frozen tundra stretching under a pale winter sun",
        "A bioluminescent bay glowing blue at night",
        "A terraced hillside with lush green rice paddies",
        "A forest trail covered in autumn leaves",
        "A dramatic sea stack rising from the ocean surf",
        "A wildflower meadow with butterflies and bees",
        "A snow-covered pine forest under a starry sky",
        "A salt flat reflecting a perfect mirror of clouds",
        "A waterfall hidden behind a curtain of green vines",
        "A coastal marsh with tall grasses at low tide",
        "A redwood forest with towering ancient trees",
        "A turquoise glacial lake surrounded by rocky peaks",
        "A misty cliffside overlooking a vast ocean",
        "A blooming cherry orchard in spring sunlight",
        "A sand dune casting long shadows at sunset",
    ],
}


def load_models(args):
    """Load Infinity models (text encoder, VAE, transformer)."""
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer

    # Model configs
    if args.model_size == "2b":
        model_args = argparse.Namespace(
            pn="0.25M",  # 512x512
            model_path=args.model_path or "weights/infinity_2b_reg.pth",
            vae_path=args.vae_path or "weights/infinity_vae_d32reg.pth",
            vae_type=32,
            apply_spatial_patchify=0,
            model_type="infinity_2b",
            checkpoint_type="torch",
            text_encoder_ckpt=args.text_encoder_path or "weights/flan-t5-xl",
            text_channels=2048,
            cfg_insertion_layer=0,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/dev/shm",
            enable_model_cache=False,
            seed=0,
            bf16=1,
            save_file="tmp.jpg",
        )
    else:  # 8b
        model_args = argparse.Namespace(
            pn="0.25M",  # 512x512 (use "1M" for 1024x1024)
            model_path=args.model_path or "weights/infinity_8b_512x512_weights",
            vae_path=args.vae_path or "weights/infinity_vae_d56_f8_14_patchify.pth",
            vae_type=14,
            apply_spatial_patchify=1,
            model_type="infinity_8b",
            checkpoint_type="torch_shard",
            text_encoder_ckpt=args.text_encoder_path or "weights/flan-t5-xl",
            text_channels=2048,
            cfg_insertion_layer=0,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/dev/shm",
            enable_model_cache=False,
            seed=0,
            bf16=1,
            save_file="tmp.jpg",
        )

    print(f"Loading text encoder from: {model_args.text_encoder_ckpt}")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=model_args.text_encoder_ckpt)

    print(f"Loading VAE from: {model_args.vae_path}")
    vae = load_visual_tokenizer(model_args)

    print(f"Loading Infinity {args.model_size.upper()} from: {model_args.model_path}")
    infinity = load_transformer(vae, model_args)

    return infinity, vae, text_tokenizer, text_encoder, model_args


@torch.no_grad()
def generate_one_image(infinity, vae, text_tokenizer, text_encoder, model_args, prompt, seed=None):
    """Generate a single image from a text prompt."""
    from tools.run_infinity import gen_one_img
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

    if seed is None:
        seed = random.randint(0, 100000)

    # Get scale schedule for square images
    h_div_w = 1.0
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][model_args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    generated = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=3.0,       # classifier-free guidance scale
        tau_list=1.0,        # temperature
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[model_args.cfg_insertion_layer],
        vae_type=model_args.vae_type,
        sampling_per_bits=model_args.sampling_per_bits,
        enable_positive_prompt=0,
    )

    # gen_one_img returns a numpy array (BGR, uint8) ready for cv2.imwrite
    # Convert BGR to RGB for PIL
    if isinstance(generated, torch.Tensor):
        img_np = generated.cpu().numpy()
    else:
        img_np = np.array(generated)

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    from PIL import Image
    return Image.fromarray(img_np)


def generate_images(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Infinity {args.model_size.upper()} models...")
    infinity, vae, text_tokenizer, text_encoder, model_args = load_models(args)
    print("Models loaded!")

    total_generated = 0
    total_failed = 0

    for category, prompts in PROMPTS.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)

        num_to_generate = min(args.num_per_category, len(prompts))
        print(f"\n{'='*60}")
        print(f"Category: {category} ({num_to_generate} images)")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts[:num_to_generate]):
            seed = random.randint(0, 100000)
            filename = f"Infinity_{category}_{i+1}_seed{seed}.png"
            filepath = cat_dir / filename

            if filepath.exists() and not args.overwrite:
                print(f"  [{i+1}/{num_to_generate}] SKIP (exact seed exists): {filename}")
                total_generated += 1
                continue

            print(f"  [{i+1}/{num_to_generate}] Generating (seed={seed}): {prompt[:60]}...")

            try:
                image = generate_one_image(
                    infinity, vae, text_tokenizer, text_encoder,
                    model_args, prompt, seed=seed
                )
                image.save(str(filepath))
                print(f"           Saved: {filepath} ({image.size[0]}x{image.size[1]})")
                total_generated += 1

            except Exception as e:
                print(f"           ERROR: {e}")
                total_failed += 1

    print(f"\n{'='*60}")
    print(f"Done! Generated: {total_generated}, Failed: {total_failed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate images with Infinity")
    parser.add_argument(
        "--model_size", type=str, default="2b", choices=["2b", "8b"],
        help="Model size: 2b (~8GB VRAM) or 8b (~40GB VRAM)",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to model weights (auto-detected from model_size)",
    )
    parser.add_argument(
        "--vae_path", type=str, default=None,
        help="Path to VAE weights (auto-detected from model_size)",
    )
    parser.add_argument(
        "--text_encoder_path", type=str, default=None,
        help="Path to flan-t5-xl (default: weights/flan-t5-xl)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/generated_infinity",
        help="Output directory",
    )
    parser.add_argument(
        "--num_per_category", type=int, default=10,
        help="Number of images per category (max 30)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing images",
    )
    args = parser.parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
