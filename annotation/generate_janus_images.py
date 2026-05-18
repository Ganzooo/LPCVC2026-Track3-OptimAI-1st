"""
Generate AI images using JanusPro-7B for training data.

No extra packages needed — uses transformers with trust_remote_code=True.

Usage:
    # Generate 1 image per category (quick test)
    python annotation/generate_janus_images.py --num_per_category 1

    # Generate 10 images per category (50 total)
    python annotation/generate_janus_images.py --num_per_category 10

    # Use smaller model (less VRAM)
    python annotation/generate_janus_images.py --model deepseek-ai/Janus-Pro-1B

    # Specify GPU
    CUDA_VISIBLE_DEVICES=0 python annotation/generate_janus_images.py

Requirements:
    pip install torch transformers pillow
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# Prompts organized by category (matching sample_dataset categories)
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


def load_model(model_id):
    """Load JanusPro using HuggingFace-native classes."""
    from transformers import JanusForConditionalGeneration, JanusProcessor

    # Use the HF-native version (no trust_remote_code needed)
    hf_model_id = model_id.replace("deepseek-ai/", "deepseek-community/")
    print(f"  Using HF-native model: {hf_model_id}")

    processor = JanusProcessor.from_pretrained(hf_model_id)
    model = JanusForConditionalGeneration.from_pretrained(
        hf_model_id, torch_dtype=torch.bfloat16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor


@torch.no_grad()
def generate_one_image(model, processor, prompt, seed=None):
    """Generate a single image from a text prompt."""
    from PIL import Image

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text, generation_mode="image", return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    outputs = model.generate(
        **inputs,
        generation_mode="image",
        do_sample=True,
        use_cache=True,
    )

    decoded = model.decode_image_tokens(outputs)
    from PIL import Image
    # decoded shape: (1, 384, 384, 3), range [-1, 1]
    arr = decoded.cpu().float().numpy()[0]
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_images(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    print(f"Model loaded on {model.device}")

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
            filename = f"JanusPro_{category}_{i+1}_seed{seed}.png"
            filepath = cat_dir / filename

            if filepath.exists() and not args.overwrite:
                print(f"  [{i+1}/{num_to_generate}] SKIP (exact seed exists): {filename}")
                total_generated += 1
                continue

            print(f"  [{i+1}/{num_to_generate}] Generating (seed={seed}): {prompt[:60]}...")

            try:
                image = generate_one_image(model, tokenizer, prompt, seed=seed)
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
    parser = argparse.ArgumentParser(description="Generate images with JanusPro")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/Janus-Pro-7B",
        help="Model ID (default: 7B, use deepseek-ai/Janus-Pro-1B for less VRAM)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/generated_janus",
        help="Output directory",
    )
    parser.add_argument(
        "--num_per_category",
        type=int,
        default=10,
        help="Number of images per category (max 15)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images",
    )
    args = parser.parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
