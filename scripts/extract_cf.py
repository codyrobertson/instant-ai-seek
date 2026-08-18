#!/usr/bin/env python3
"""Extract a balanced training subset from a CommunityForensics systematic
parquet into data/train_extra/{ai_cf,real_cf} (run on the Spark).

CF fakes span GAN and diffusion generators plus manipulated images — the
private-benchmark risk family our diffusion-only training missed.
The CF-Eval parquets in data/bench_pub/cf/ are NEVER used here (sealed).

Usage: python3 scripts/extract_cf.py <parquet> [--per-class 3000]
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_AI = ROOT / "data" / "train_extra" / "ai_cf"
OUT_REAL = ROOT / "data" / "train_extra" / "real_cf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=str)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--tag", type=str, default="cf")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    t = pq.read_table(args.parquet)
    cols = t.column_names
    img_col = next((c for c in ("image", "image_data", "bytes") if c in cols), cols[0])
    label_col = next((c for c in ("label", "is_fake", "fake", "class") if c in cols), None)
    rows = t.column(img_col).to_pylist()
    labels = t.column(label_col).to_pylist() if label_col else [1] * len(rows)

    ai, real = [], []
    for i, raw in enumerate(rows):
        raw = raw["bytes"] if isinstance(raw, dict) else raw
        if not isinstance(raw, (bytes, bytearray)):
            continue
        v = labels[i]
        if isinstance(v, str):
            is_ai = v.lower() in ("ai", "fake", "1", "true", "synthetic")
        else:
            is_ai = v in (1, True)
        (ai if is_ai else real).append(bytes(raw))
    print(f"parquet: {len(ai)} ai-ish, {len(real)} real-ish")

    rng = random.Random(7)
    OUT_AI2 = ROOT / "data" / "train_extra" / f"ai_{args.tag}"
    for out_dir, pool, name in ((OUT_AI2, ai, "ai"),):
        out_dir.mkdir(parents=True, exist_ok=True)
        rng.shuffle(pool)
        for i, raw in enumerate(pool[: args.per_class]):
            (out_dir / f"{name}_{i:05d}.jpg").write_bytes(raw)
        print(f"{name}: wrote {len(list(out_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
