#!/usr/bin/env python3
"""Mine hard examples from the train/dev pools only (never sealed sets).

Collects:
  - highest-confidence real-as-AI images (real scored >= 0.65)
  - lowest-confidence AI images (ai scored < 0.65)

Adjudicates each with automatic heuristics (JPEG blockiness, grain/noise,
matched-pair filename hints) and writes
data/manifests/hard_negative_registry.csv with a "reason" column:
  jpeg_artifact | camera_noise | matched_pair | generator | ambiguous | review

Usage:
  python3 scripts/mine_hard_examples.py --manifest data/manifests/dev_forensic.csv \
      [--max-real 200] [--max-ai 200] [--out data/manifests/hard_negative_registry.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def adjudicate(p: Path) -> str:
    """Heuristic reason codes for hard examples (fast, honest, conservative)."""
    import io

    from PIL import Image, ImageFilter

    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((256, 256))
            g = np.asarray(im, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return "review"
    # grain: Laplacian energy
    lap = np.abs(g[1:, 1:] - 2 * g[1:, :-1] + g[:-1, 1:]).mean()
    # JPEG blockiness: boundary vs interior difference at 8px grid
    h, w = g.shape
    g8 = g[: h // 8 * 8, : w // 8 * 8].reshape(-1, 8, 8)
    dh = np.abs(np.diff(g8, axis=2)).reshape(-1, 8, 7).mean(axis=0)
    edge = np.abs(np.diff(g8, axis=1)).mean()
    block = float(edge - dh[1:].mean())
    name = p.name.lower()
    if any(k in name for k in ("pair", "matched", "real")):
        return "matched_pair"
    if block > 6.0:
        return "jpeg_artifact"
    if lap > 26.0:
        return "camera_noise"
    if lap < 8.0:
        return "generator"
    return "review"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="dev or train forensic csv (never sealed)")
    ap.add_argument("--max-real", type=int, default=200)
    ap.add_argument("--max-ai", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "data/manifests/hard_negative_registry.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    real = [r for r in rows if r["label"] == "real"]
    ai = [r for r in rows if r["label"] == "ai"]
    print(f"pool: {len(rows)} ({len(real)} real, {len(ai)} ai)")

    def resolve(r):
        p = Path(r["path"])
        return p if p.is_absolute() else (ROOT / p)

    print("scoring pool with the current ensemble ...", flush=True)
    confs = np.array(predict([str(resolve(r)) for r in rows]))
    for r, c in zip(rows, confs):
        r["conf"] = c

    hard_real = sorted((r for r in real if r["conf"] >= THRESHOLD), key=lambda r: -r["conf"])[: args.max_real]
    hard_ai = sorted((r for r in ai if r["conf"] < THRESHOLD), key=lambda r: r["conf"])[: args.max_ai]

    out_rows = []
    for r in hard_real:
        p = resolve(r)
        out_rows.append({
            "path": str(p), "label": "real", "conf": f"{r['conf']:.4f}",
            "reason": adjudicate(p), "source": r.get("source", ""), "split": r.get("split", "dev"),
        })
    for r in hard_ai:
        p = resolve(r)
        out_rows.append({
            "path": str(p), "label": "ai", "conf": f"{r['conf']:.4f}",
            "reason": adjudicate(p), "source": r.get("source", ""), "split": r.get("split", "dev"),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    from collections import Counter

    print(f"wrote {args.out}: {len(out_rows)} hard examples")
    print("reason histogram:", dict(Counter(r["reason"] for r in out_rows)))
    print("by source:", dict(Counter(r["source"] for r in out_rows)))


if __name__ == "__main__":
    main()
