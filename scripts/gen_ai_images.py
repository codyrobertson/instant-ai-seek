#!/usr/bin/env python3
"""Generate the AI-image class via fal.ai flux/schnell.

Reads data/manifests/ai_prompts.csv, generates each image, saves
data/raw/ai/<id>.jpg and appends to data/manifests/ai_manifest.csv.

Resumable: images already present in the manifest are skipped.
Requires FAL_KEY env var ("key_id:key_secret").

Usage: FAL_KEY=... python3 scripts/gen_ai_images.py [--workers N] [--limit M]
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://queue.fal.run/fal-ai/flux/schnell"
ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "manifests" / "ai_prompts.csv"
MANIFEST = ROOT / "data" / "manifests" / "ai_manifest.csv"
OUT = ROOT / "data" / "raw" / "ai"


def generate(key: str, prompt: str, aspect: str, seed: int, timeout: int = 120) -> dict:
    payload = {"prompt": prompt, "image_size": aspect, "num_images": 1, "seed": seed}
    hdrs = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    r = requests.post(API, json=payload, headers=hdrs, timeout=timeout)
    r.raise_for_status()
    rid = r.json()["request_id"]
    # poll
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = requests.get(
            f"https://queue.fal.run/fal-ai/flux/requests/{rid}/status",
            headers=hdrs, timeout=timeout,
        ).json()
        if s["status"] == "COMPLETED":
            res = requests.get(
                f"https://queue.fal.run/fal-ai/flux/requests/{rid}",
                headers=hdrs, timeout=timeout,
            ).json()
            img = res["images"][0]
            data = requests.get(img["url"], timeout=timeout).content
            return {
                "request_id": rid, "url": img["url"],
                "width": img["width"], "height": img["height"],
                "seed": res.get("seed", seed), "bytes": len(data), "data": data,
            }
        if s["status"] in ("FAILED", "CANCELED"):
            raise RuntimeError(f"request {rid} {s['status']}: {s}")
        time.sleep(0.8)
    raise TimeoutError(f"request {rid} timed out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompts", type=str, default=None,
                    help="prompt csv (default data/manifests/ai_prompts.csv)")
    args = ap.parse_args()

    prompts_path = PROMPTS if args.prompts is None else ROOT / "data" / "manifests" / args.prompts
    key = os.environ.get("FAL_KEY")
    if not key:
        sys.exit("FAL_KEY env var required (key_id:key_secret)")

    OUT.mkdir(parents=True, exist_ok=True)
    header_written = MANIFEST.exists() and MANIFEST.stat().st_size > 0
    done = set()
    if header_written:
        with open(MANIFEST) as f:
            done = {r["id"] for r in csv.DictReader(f)}
    with open(prompts_path) as f:
        jobs = [r for r in csv.DictReader(f) if r["id"] not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("all prompts already generated")
        return

    print(f"generating {len(jobs)} images with {args.workers} workers ...", flush=True)
    t0 = time.time()
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(generate, key, j["prompt"], j["aspect"], int(j["seed"])): j
            for j in jobs
        }
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL {j['id']} {j['prompt'][:60]!r}: {e}", flush=True)
                continue
            (OUT / f"{j['id']}.jpg").write_bytes(r["data"])
            with open(MANIFEST, "a", newline="") as f:
                w = csv.writer(f)
                if not header_written:
                    w.writerow(["id", "prompt", "aspect", "seed", "request_id", "url", "width", "height", "gen_seed"])
                    header_written = True
                w.writerow([j["id"], j["prompt"], j["aspect"], j["seed"], r["request_id"], r["url"], r["width"], r["height"], r["seed"]])
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
