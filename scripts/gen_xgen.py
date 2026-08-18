#!/usr/bin/env python3
"""Generate cross-generator AI images for TRAINING EXPANSION or HOLDOUT.

  --dest train   -> data/xgen/train/<model>/*.jpg   (joins training)
  --dest holdout -> data/holdout/ai/<model>/*.jpg   (NEVER trained on; probe only)

Each destination keeps its own manifest. Prompts are drawn deterministically
from the original 300-prompt list with a configurable step (no overlap with
the existing 24-per-model probe set when step differs).

Usage: FAL_KEY=... python3 scripts/gen_xgen.py --dest train --per-model 100
       FAL_KEY=... python3 scripts/gen_xgen.py --dest holdout --per-model 60
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_probe_images import MODELS, generate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "manifests" / "ai_prompts.csv"

EXTRA_MODELS = {
    "flux_pro": {
        "endpoint": "fal-ai/flux-pro/v1.1",
        "payload": lambda p, seed: {"prompt": p, "image_size": "landscape_4_3", "num_images": 1, "seed": seed},
    },
    "nano_banana": {
        "endpoint": "fal-ai/nano-banana",
        "payload": lambda p, seed: {"prompt": p, "image_size": "landscape_4_3", "num_images": 1, "seed": seed},
    },
    "ideogram_v3": {
        "endpoint": "fal-ai/ideogram/v3",
        "payload": lambda p, seed: {"prompt": p, "aspect_ratio": "16:10", "num_images": 1, "seed": seed},
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", choices=["train", "holdout", "train2", "holdout2"], required=True)
    ap.add_argument("--per-model", type=int, default=60)
    ap.add_argument("--offset", type=int, default=2,
                    help="prompt list offset so different dests pick different prompts")
    ap.add_argument("--extra-models", action="store_true",
                    help="also generate flux_pro / nano_banana / ideogram_v3")
    args = ap.parse_args()

    key = os.environ.get("FAL_KEY")
    if not key:
        sys.exit("FAL_KEY env var required")

    dest_dir = {"train": "data/xgen/train", "holdout": "data/holdout/ai",
                "train2": "data/xgen2/train", "holdout2": "data/holdout2/ai"}[args.dest]
    manifest_name = {"train": "xgen_train_manifest.csv", "holdout": "holdout_ai_manifest.csv",
                     "train2": "xgen2_train_manifest.csv", "holdout2": "holdout2_ai_manifest.csv"}[args.dest]
    out_root = ROOT / dest_dir
    manifest = ROOT / "data" / "manifests" / manifest_name
    out_root.mkdir(parents=True, exist_ok=True)

    header_written = manifest.exists() and manifest.stat().st_size > 0
    done = set()
    if header_written:
        with open(manifest) as f:
            done = {(r["model"], r["prompt_id"]) for r in csv.DictReader(f)}

    with open(PROMPTS) as f:
        rows = [r for r in csv.DictReader(f)]
    step = 300 // args.per_model
    picked = [rows[(args.offset + i * step) % 300] for i in range(args.per_model)]

    model_set = dict(MODELS)
    if args.extra_models:
        model_set.update(EXTRA_MODELS)

    jobs = []
    for model, cfg in model_set.items():
        for pr in picked:
            if (model, pr["id"]) in done:
                continue
            jobs.append((model, cfg, pr))
    if not jobs:
        print("all requested images already generated")
        return
    print(f"generating {len(jobs)} images -> {out_root} ...", flush=True)

    t0 = time.time()
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=len(model_set)) as ex:
        futs = {
            ex.submit(generate, key, cfg["endpoint"], cfg["payload"](pr["prompt"], int(pr["seed"]) % (2**31))): (model, pr)
            for model, cfg, pr in jobs
        }
        for fut in as_completed(futs):
            model, pr = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL {model} prompt {pr['id']}: {e}", flush=True)
                continue
            d = out_root / model
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{model}_{pr['id']}.jpg").write_bytes(r["data"])
            with open(manifest, "a", newline="") as f:
                w = csv.writer(f)
                if not header_written:
                    w.writerow(["model", "prompt_id", "prompt", "width", "height", "url"])
                    header_written = True
                w.writerow([model, pr["id"], pr["prompt"], r["width"], r["height"], r["url"]])
            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
