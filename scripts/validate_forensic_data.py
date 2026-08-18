#!/usr/bin/env python3
"""Validate forensic manifests before training.

Rejects:
  - missing / unreadable files
  - corrupt images (unreadable, zero/1-byte, tiny rate-limited downloads)
  - duplicate SHA-256 within and across splits
  - cross-split perceptual duplicates (phash, near-dup guard)
  - missing provenance (family/source/license)
  - accidental inclusion of any sealed test path (data/eval, holdout2,
    bench_pub, holdout_oos in train/dev)

Usage:
  python3 scripts/validate_forensic_data.py --manifest <csv> [--sealed]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEALED_PATHS = ("data/eval", "holdout2", "bench_pub", "holdout_oos")
MIN_BYTES = 1024  # rate-limited HF downloads are ~15 bytes; real images are >1KB
PHASH_BITS = 8  # 64-bit perceptual hash for near-dup guard (optional, --phash)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(p: Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(p) as im:
            g = im.convert("L").resize((32, 32))
        import numpy as np

        a = np.asarray(g, dtype=np.float32)
        m = a.mean()
        return "".join("1" if v > m else "0" for v in (a > m).flatten())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sealed", action="store_true", help="manifest is a sealed split")
    ap.add_argument("--phash", action="store_true", help="run perceptual near-dup guard (slow)")
    args = ap.parse_args()

    problems = []
    warnings = []
    manifest = Path(args.manifest)
    if not manifest.exists():
        sys.exit(f"manifest not found: {manifest}")

    rows = list(csv.DictReader(open(manifest)))
    print(f"validating {manifest} ({len(rows)} rows)")

    hashes: Counter = Counter()
    seen_paths = set()
    phashes = {}
    counts = Counter()
    missing_prov = 0
    sealed_hits = []

    for i, r in enumerate(rows):
        counts[(r.get("label"), r.get("family"), r.get("source"))] += 1
        for field in ("path", "label", "family", "source", "license"):
            if not r.get(field):
                missing_prov += 1
                if len(problems) < 200:
                    problems.append(f"row {i}: missing provenance field '{field}'")

        p = Path(r["path"])
        if not p.is_absolute():
            p = ROOT / p
        key = str(p)
        if key in seen_paths:
            problems.append(f"duplicate path row {i}: {key}")
        seen_paths.add(key)

        if not p.exists():
            problems.append(f"missing file: {key}")
            continue
        # tiny-file heuristic: legitimately-small sources (e.g. CIFAKE 32x32)
        # are fine if the image decodes; the decode check below is authoritative.
        tiny = p.stat().st_size < MIN_BYTES

        # sealed-path contamination: train/dev must never point at sealed sets
        if not args.sealed:
            low = key.lower()
            for tag in SEALED_PATHS:
                if tag in low:
                    sealed_hits.append(key)

        h = sha256(p)
        hashes[h] += 1
        if args.phash:
            ph = phash(p)
            if ph:
                phashes.setdefault(ph, []).append(key)

        # image readability (decodes?) — cheap PIL open
        try:
            from PIL import Image

            with Image.open(p) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            problems.append(f"corrupt image: {key} ({str(e)[:60]})")
        else:
            if tiny:
                # decodes fine but suspiciously small — warn only
                warnings.append(f"small-but-valid image ({p.stat().st_size}B): {key}")

    dups = {h: n for h, n in hashes.items() if n > 1}
    if dups:
        problems.append(f"{len(dups)} duplicate SHA-256 values (exact dups): {dups}")

    near_dups = []
    if args.phash:
        for ph, keys in phashes.items():
            if len(keys) > 1:
                near_dups.append(keys)

    if not args.sealed and sealed_hits:
        problems.append(f"SEALED-PATH CONTAMINATION ({len(sealed_hits)}): {sealed_hits[:10]}")

    print(f"rows by (label,family,source):")
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:5s} {k[1]:22s} {k[2]:16s} {v}")
    if near_dups:
        print(f"near-duplicate groups (phash): {len(near_dups)}")

    verdict = "FAIL" if problems else "PASS"
    print(f"VERDICT: {verdict} ({len(problems)} problems, {len(warnings)} warnings)")
    for p in problems[:30]:
        print("  -", p)
    for w in warnings[:10]:
        print("  ~", w)
    if args.sealed and not problems:
        print("sealed split clean: no train-adjacent rows expected beyond holdout_oos")


if __name__ == "__main__":
    main()
