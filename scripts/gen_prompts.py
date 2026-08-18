#!/usr/bin/env python3
"""Generate a deterministic, web-realistic prompt list for the AI-image class.

The prompt list is the seed of the benchmark's AI side: 300 prompts covering
ordinary web content (people, places, products, food, pets, events) plus a
minority of clearly synthetic styles (illustration, 3D render), which is what
AI content actually looks like in the wild.

Every choice is driven by a fixed RNG seed so the list is reproducible.
Output: data/manifests/ai_prompts.csv  (id,prompt,aspect,seed)
"""
import csv
import random
from pathlib import Path

SEED = 20260813
N = 300
ASPECTS = ["landscape_4_3"] * 60 + ["portrait_4_3"] * 40  # web-mix: more landscape

# (template, slots) — slot values are joined with the template's {n} placeholders
TEMPLATES = [
    # --- people / social (the most common web-AI content) ---
    ("portrait of a {0} in {1}, {2}", ["young woman", "middle-aged man", "elderly woman", "teenage boy", "woman in her 30s", "man with a beard", "young couple", "smiling woman", "man in a suit", "child"], ["a sunlit cafe", "an urban street", "a park", "a home living room", "a rooftop", "a city sidewalk", "a garden", "an office", "a studio with soft light", "a market"], ["natural light", "candid look", "soft bokeh background", "golden hour", "casual clothing", "slight smile", "looking at camera", "window light", "outdoor light", "warm tones"]),
    ("group photo of {0} {1}, {2}", ["friends", "a family", "coworkers", "a sports team", "classmates"], ["at a barbecue", "on a hiking trip", "at a wedding", "in an office", "at a birthday party", "on vacation", "at a restaurant table", "in front of a landmark"], ["everyone smiling", "candid moment", "daylight", "smartphone photo style", "natural poses"]),
    ("{0} taking a {1} at {2}", ["a woman", "a man", "a teenager", "a couple"], ["selfie", "group selfie", "mirror selfie"], ["the gym", "a concert", "a beach", "a festival", "a theme park", "a city street at night"]),
    ("candid photo of {0} {1}", ["children", "a toddler", "kids", "a baby"], ["playing in a yard", "at a playground", "eating ice cream", "running on a beach", "at a birthday party", "with a puppy"]),
    ("{0} at {1}, event photography", ["people", "a crowd", "guests", "attendees"], ["a wedding reception", "a conference", "a graduation ceremony", "a food festival", "a sports game", "a street parade"]),
    # --- places / travel ---
    ("{0} street scene in {1}, {2}", ["a busy", "a quiet", "a rainy", "a sunny"], ["a european old town", "tokyo", "new york", "barcelona", "lisbon", "paris", "a mexican town", "chicago"], ["candid pedestrians", "shop fronts", "cafes and awnings", "evening lights", "daytime", "morning light"]),
    ("{0} {1} at {2}", ["a scenic", "a misty", "a dramatic"], ["mountain range", "coastal cliff", "lake shore", "desert vista", "forest trail", "waterfall"], ["sunset", "sunrise", "midday", "golden hour"]),
    ("{0} interior photo, {1}", ["a cozy", "a modern", "a rustic", "a bright"], ["living room", "bedroom", "kitchen", "home office", "hotel room", "apartment", "cafe"], ["real estate listing style", "natural window light", "warm inviting feel", "styled but lived-in"]),
    ("{0} in {1}", ["a hotel pool", "a resort lobby", "a boutique shop", "a brewery taproom", "a museum hall", "a spa"], ["miami", "bali", "amsterdam", "austin", "santorini", "prague"], ["travel photography", "bright daylight", "wide shot"]),
    # --- food ---
    ("{0} on a {1}, {2}", ["a burger and fries", "a bowl of ramen", "pancakes with syrup", "a gourmet pizza", "a poke bowl", "a brunch spread", "tacos", "a steak dinner", "sushi platter", "a salad bowl"], ["wooden table", "marble counter", "cafe table", "restaurant table"], ["restaurant food photography", "overhead shot", "natural light", "appetizing", "close up"]),
    ("{0} {1} photography", ["artisanal", "freshly baked", "colorful"], ["pastry", "bread", "donuts", "cupcakes", "coffee with latte art"], ["bakery display", "overhead", "warm lighting", "macro detail"]),
    # --- products / e-commerce ---
    ("product photo of {0} on {1}, {2}", ["wireless headphones", "a leather handbag", "sneakers", "a smartwatch", "a coffee maker", "a skincare set", "a mechanical keyboard", "a backpack", "a candle", "sunglasses"], ["a clean white background", "a neutral gray background", "a stone surface", "a pastel background"], ["studio lighting", "e-commerce style", "high detail", "soft shadows"]),
    ("{0} {1} for sale, {2}", ["a vintage", "a modern", "a handcrafted"], ["armchair", "desk lamp", "ceramic vase", "wall art print", "record player", "plant stand"], ["marketplace listing photo", "home setting", "natural light", "clear product focus"]),
    # --- pets / animals ---
    ("photo of a {0} {1}", ["golden retriever", "labrador puppy", "tabby cat", "french bulldog", "border collie", "kitten", "parrot", "hamster"], ["in a park", "on a couch", "at the beach", "in a backyard", "in a car", "next to a window"], ["candid pet photo", "soft daylight", "looking at camera", "playful"]),
    # --- fitness / lifestyle ---
    ("{0} {1}", ["a woman", "a man", "a trainer"], ["lifting weights at a gym", "doing yoga in a studio", "running on a trail", "stretching on a mat", "climbing at an indoor gym", "cycling on a road"]),
    ("{0} of a {1} at {2}", ["lifestyle photo", "casual snapshot"], ["person", "couple"], ["a coffee shop", "a farmers market", "a book store", "a concert venue", "a dog park"]),
    # --- clearly synthetic styles (minority, but real on the web) ---
    ("{0} of {1}, digital art style", ["vibrant illustration", "concept art", "flat vector art", "fantasy painting"], ["a city skyline", "a forest spirit", "a space scene", "an underwater world", "a cyberpunk street"]),
    ("3d render of {0}, {1}", ["a cozy room", "a cute character", "an isometric scene", "a product mockup", "a tiny house"], ["blender style", "pixar style", "clay render", "soft studio lighting"]),
    ("{0} {1} poster art", ["minimalist", "bold typography", "retro"], ["concert", "movie", "travel", "motivational"]),
]

# extra standalone prompts to round out diversity
EXTRAS = [
    "an infographic about coffee brewing methods, clean modern design",
    "a birthday cake with candles being lit, party snapshot",
    "a messy desk with a laptop and notes, work from home",
    "a city skyline at dusk seen from a rooftop bar",
    "a backyard barbecue with people around a grill, evening",
    "a snowy mountain village with warm lit windows",
    "a tropical beach with turquoise water and palm trees",
    "a farmer's market stall with fresh vegetables, morning light",
    "a gym interior with rows of treadmills, no people",
    "a cozy reading nook with a bookshelf and armchair",
    "a vintage car parked on a suburban driveway",
    "a conference hall during a tech talk, audience silhouettes",
    "a hospital waiting room, realistic photography",
    "an airport terminal with travelers and luggage",
    "a classroom with students at desks, candid",
    "a christmas tree in a living room corner, warm lights",
    "a stack of pancakes with berries at a diner",
    "a garage workshop with tools on a pegboard",
    "a grocery store aisle with products on shelves",
    "a small dog in a sweater on a city sidewalk",
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--out", type=str, default="ai_prompts.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(__file__).resolve().parent.parent / "data" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    out = out / args.out

    rows = []
    used: set[str] = set()
    i = 0
    pool: list[str] = []
    # one pass: templates, then extras, both sampled with replacement until N
    for tmpl, *slots in TEMPLATES:
        for _ in range(120):
            filled = tmpl.format(*[rng.choice(s) for s in slots])
            pool.append(filled)
    pool = pool[: 4 * args.n] + list(EXTRAS)
    rng.shuffle(pool)

    while len(rows) < args.n:
        prompt = pool[i % len(pool)]
        i += 1
        if prompt in used:
            continue
        used.add(prompt)
        seed = rng.randrange(2**31)
        aspect = rng.choice(ASPECTS)
        rows.append([len(rows) + 1, prompt, aspect, seed])

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "aspect", "seed"])
        w.writerows(rows)
    print(f"wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
