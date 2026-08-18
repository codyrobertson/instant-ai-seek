#!/usr/bin/env python3
"""Evaluate the detector on public AI-detection benchmarks (the LAW).

Benchmarks:
  - Synthbuster+   (data/bench_pub/synthbuster/*.parquet)  — fake-only (DALL-E 2/3 etc.)
  - Community Forensics Eval (data/bench_pub/cf/*.parquet) — real+AI

Images are extracted from parquet (raw bytes in image/image_data/bytes),
deduplicated, sampled deterministically, scored with detector.detect (the
same inference path the extension ships), and reported as balanced accuracy
@0.65 (when both classes exist) or per-class recall (fake-only sets).

These benchmark images are EVALUATION-ONLY — never training data.

Usage: python3 scripts/bench_pub.py [--per-class 400]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def load_parquet_images(parquet_path: Path, limit: int = 0) -> list[tuple[bytes, str]]:
    import pyarrow.parquet as pq

    if not parquet_path.exists():
        print(f"  [skip] missing {parquet_path.name}", flush=True)
        return []
    try:
        table = pq.read_table(parquet_path)
    except Exception as e:
        print(f"  [skip] unreadable {parquet_path.name}: {str(e)[:60]}", flush=True)
        return []
    cols = table.column_names
    img_col = next((c for c in ("image", "image_data", "bytes") if c in cols), cols[0])
    label_col = next((c for c in ("label", "target", "ground_truth", "class", "is_fake", "fake") if c in cols), None)
    rows = table.column(img_col).to_pylist()
    labels = table.column(label_col).to_pylist() if label_col else None

    out = []
    for i, row in enumerate(rows):
        raw = row["bytes"] if isinstance(row, dict) else row
        if not isinstance(raw, (bytes, bytearray)):
            continue
        lab = None
        if labels is not None:
            v = labels[i]
            if isinstance(v, str):
                lab = "ai" if v.lower() in ("ai", "fake", "1", "true", "synthetic", "gen") else "real"
            else:
                lab = "ai" if v in (1, True) else "real"
        out.append((bytes(raw), lab))
        if limit and len(out) >= limit:
            break
    return out


def _dump(items, name) -> list[str]:
    tmp = ROOT / "data" / "bench_pub" / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (raw, _) in enumerate(items):
        p = tmp / f"{name}_{i:05d}.jpg"
        p.write_bytes(raw)
        paths.append(str(p))
    return paths


def score_set(items: list[tuple[bytes, str]], name: str) -> None:
    paths = _dump(items, name.replace("/", "_"))
    confs = np.array(predict(paths))
    labels = [l for _, l in items]
    n_ai = sum(1 for l in labels if l == "ai")
    n_real = sum(1 for l in labels if l == "real")
    y = np.array([1.0 if l == "ai" else 0.0 for l in labels])
    # single measurement layer (bench/scorecard.metrics) for the report line
    from bench.scorecard import metrics as scorecard_metrics

    m = scorecard_metrics(y, confs, THRESHOLD)
    bal = m["balanced_accuracy"] if m["balanced_accuracy"] is not None else (m["ai_recall"] if n_ai else m["real_recall"])
    print(f"{name:28s} n={len(items):4d} (ai={n_ai}, real={n_real})  bal_acc={bal:.4f}  "
          f"ai_recall={m['ai_recall']:.4f}  real_recall={m['real_recall']:.4f}  "
          f"brier={m['brier']:.4f} ece10={m['ece10']:.4f}  mean_conf={confs.mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    args = ap.parse_args()

    rng = random.Random(0)
    for bench, glob in (("synthbuster+", "synthbuster/*.parquet"), ("cf-eval", "cf/*.parquet"),
                     ("cf-commercial(test)", "cf_commercial_test.parquet")):
        shards = sorted((ROOT / "data" / "bench_pub").glob(glob))
        if not shards:
            print(f"{bench:28s} no parquet shards (download first)")
            continue
        all_items = []
        for sh in shards:
            all_items.extend(load_parquet_images(sh))
        seen = set()
        uniq = []
        for raw, lab in all_items:
            h = hash(raw[:256])
            if h in seen:
                continue
            seen.add(h)
            uniq.append((raw, lab))
        rng.shuffle(uniq)
        ai = [x for x in uniq if x[1] == "ai"][: args.per_class]
        real = [x for x in uniq if x[1] == "real"][: args.per_class]
        n_ai_tot = sum(1 for x in uniq if x[1] == "ai")
        n_real_tot = sum(1 for x in uniq if x[1] == "real")
        print(f"--- {bench}: {len(uniq)} unique (ai={n_ai_tot}, real={n_real_tot}) — scoring "
              f"{len(ai)} ai + {len(real)} real ---")
        if ai and not real:
            score_set(ai, f"{bench} (fake-only)")
        elif real and not ai:
            score_set(real, f"{bench} (real-only)")
        elif ai and real:
            score_set(ai + real, f"{bench}")
        else:
            print(f"{bench:28s} label parse issue — check parquet schema")


if __name__ == "__main__":
    main()
