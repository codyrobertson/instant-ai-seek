#!/usr/bin/env python3
"""Extract JPEG/PNG images from DiffusionDB parquet shards into
data/train_extra/ai_sd_db (training subset) — run on the Spark.

DiffusionDB stores images as struct{bytes, path} in the `image` column.
Deterministic (fixed order, seeded skip), resumable.

Usage: python3 scripts/extract_sd_db.py <shard_glob> [--limit N]
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "train_extra" / "ai_sd_db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="parquet shard paths")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in OUT.glob("*.jpg")}
    n = 0
    for shard in args.shards:
        if args.limit and n >= args.limit:
            break
        table = pq.read_table(shard, columns=["image"])
        rows = table.column("image").to_pylist()
        for row in rows:
            if args.limit and n >= args.limit:
                break
            raw = row["bytes"] if isinstance(row, dict) else row
            name = f"sd_{n:06d}"
            if name not in existing:
                (OUT / f"{name}.jpg").write_bytes(raw)
            n += 1
            if n % 1000 == 0:
                print(f"  {n} images", flush=True)
    print(f"done: {n} total (dir has {len(list(OUT.glob('*.jpg')))} files)")


if __name__ == "__main__":
    main()
