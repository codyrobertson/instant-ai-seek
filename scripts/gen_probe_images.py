#!/usr/bin/env python3
"""Probe-set generator: AI images from fal models OTHER than flux/schnell.

Purpose: measure cross-generator generalization of the detector. Each model
gets a deterministic subset of the original 300 prompts (every Nth), so probe
content distribution mirrors the benchmark's AI class.

Probe sets are diagnostics — they are NOT part of the harness eval and never
touch data/eval. (They may be merged into data/train for training robustness.)

Usage: FAL_KEY=... python3 scripts/gen_probe_images.py [--smoke]
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "manifests" / "ai_prompts.csv"
OUT = ROOT / "data" / "probe" / "ai"
IMAGES_PER_MODEL = 24
STEP = 300 // IMAGES_PER_MODEL  # 12 -> every 12th prompt

# model id -> payload builder (fal queue schema varies per model)
MODELS = {
    "flux_dev": {
        "endpoint": "fal-ai/flux/dev",
        "payload": lambda p, seed: {"prompt": p, "image_size": "landscape_4_3", "num_images": 1, "seed": seed},
    },
    "sdxl": {
        "endpoint": "fal-ai/fast-sdxl",
        "payload": lambda p, seed: {"prompt": p, "image_size": "landscape_4_3", "num_images": 1, "seed": seed},
    },
    "sd35": {
        "endpoint": "fal-ai/stable-diffusion-v35-large",
        "payload": lambda p, seed: {"prompt": p, "image_size": "landscape_4_3", "num_images": 1, "seed": seed},
    },
    "ideogram": {
        "endpoint": "fal-ai/ideogram/v2",
        "payload": lambda p, seed: {"prompt": p, "aspect_ratio": "16:10", "num_images": 1, "seed": seed},
    },
    "recraft": {
        "endpoint": "fal-ai/recraft-v3",
        "payload": lambda p, seed: {"prompt": p, "size": "1024x1024", "num_images": 1, "seed": seed, "style": "realistic_image"},
    },
}
def generate(key: str, endpoint: str, payload: dict, timeout: int = 300) -> dict:
    hdrs = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    # POST with retries on transient non-JSON/5xx responses
    for attempt in range(4):
        try:
            r = requests.post(f"https://queue.fal.run/{endpoint}", json=payload, headers=hdrs, timeout=timeout)
            if r.status_code != 200 or not r.text.strip():
                raise RuntimeError(f"HTTP {r.status_code} empty")
            meta = r.json()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    rid = meta["request_id"]
    status_url = meta["status_url"]
    response_url = meta["response_url"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = requests.get(status_url, headers=hdrs, timeout=timeout)
            if s.status_code != 200 or not s.text.strip():
                raise RuntimeError(f"status HTTP {s.status_code}")
            st = s.json()
        except Exception:  # noqa: BLE001
            time.sleep(2)
            continue
        if st["status"] == "COMPLETED":
            res = requests.get(response_url, headers=hdrs, timeout=timeout).json()
            img = res["images"][0]
            data = requests.get(img["url"], timeout=timeout).content
            return {
                "request_id": rid, "url": img["url"],
                "width": img.get("width"), "height": img.get("height"),
                "seed": res.get("seed"), "bytes": len(data), "data": data,
            }
        if st["status"] in ("FAILED", "CANCELED"):
            raise RuntimeError(f"{endpoint} {rid} {st['status']}: {st.get('error')}")
        time.sleep(3)
    raise TimeoutError(f"{endpoint} {rid} timed out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one image per model only")
    args = ap.parse_args()

    key = os.environ.get("FAL_KEY")
    if not key:
        sys.exit("FAL_KEY env var required")

    with open(PROMPTS) as f:
        rows = [r for r in csv.DictReader(f)]
    picked = [rows[i] for i in range(0, len(rows), STEP)][:IMAGES_PER_MODEL]
    if args.smoke:
        picked = picked[:1]

    manifest = ROOT / "data" / "manifests" / "probe_ai_manifest.csv"
    header_written = manifest.exists() and manifest.stat().st_size > 0
    done = set()
    if header_written:
        with open(manifest) as f:
            done = {(r["model"], r["prompt_id"]) for r in csv.DictReader(f)}

    jobs = []
    for model, cfg in MODELS.items():
        for pr in picked:
            if (model, pr["id"]) in done:
                continue
            jobs.append((model, cfg, pr))
    if not jobs:
        print("all probe images already generated")
        return
    print(f"generating {len(jobs)} probe images ({'smoke' if args.smoke else 'full'}) ...", flush=True)

    t0 = time.time()
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
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
            d = OUT / model
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{model}_{pr['id']}.jpg").write_bytes(r["data"])
            with open(manifest, "a", newline="") as f:
                w = csv.writer(f)
                if not header_written:
                    w.writerow(["model", "prompt_id", "prompt", "width", "height", "url"])
                    header_written = True
                w.writerow([model, pr["id"], pr["prompt"], r["width"], r["height"], r["url"]])
            ok += 1
    print(f"done: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
