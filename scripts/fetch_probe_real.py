#!/usr/bin/env python3
"""Fetch independent real photos for the generalization probe.

Source: picsum.photos (Unsplash-hosted real photos, free license), served
deterministically via fixed seeds. 640x480 (landscape) / 480x640 (portrait)
mix mirrors typical web images.

Probe images are diagnostics — they never enter data/eval.

Usage: python3 scripts/fetch_probe_real.py [--n 120]
"""
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probe" / "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i in range(args.n):
        p = OUT / f"picsum_{i:04d}.jpg"
        if p.exists():
            ok += 1
            continue
        seed = 1000 + i
        w, h = (640, 480) if i % 3 else (480, 640)
        r = requests.get(f"https://picsum.photos/seed/{seed}/{w}/{h}", timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
        ok += 1
        if ok % 25 == 0:
            print(f"  {ok}/{args.n}", flush=True)
    print(f"done: {ok} real probe images in {OUT}")


if __name__ == "__main__":
    main()
