#!/usr/bin/env python3
"""Calibration shift report: Brier/ECE/false-AI per stratum.

Strata:
  - clean eval (committed harness set)
  - in-the-wild reals (data/probe/real)
  - matched real cells (CF-Eval parquets, sampled; run on the DGX Spark)
  - generator fakes (Synthbuster+, CF fakes)

A low ECE on data/eval alone is not sufficient — this report shows where
calibration degrades under shift.

Usage:
  python3 bench/shift_report.py [--cf-csv ...] [--json out.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.scorecard import metrics as scorecard_metrics  # noqa: E402
from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def report(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    m = scorecard_metrics(y, p, THRESHOLD)
    out = {
        "name": name,
        "n": int(len(y)),
        "brier": m["brier"],
        "ece10": m["ece10"],
        "false_ai_rate": m["false_ai_rate"],
        "ai_recall": m["ai_recall"],
        "real_recall": m["real_recall"],
        "mean_conf": float(p.mean()),
    }
    print(f"{name:24s} n={out['n']:5d} brier={out['brier']:.4f} ece10={out['ece10']:.4f} "
          f"false_ai={out['false_ai_rate']:.4f} ai_rec={out['ai_recall']:.3f} real_rec={out['real_recall']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-csv", default=None, help="CSV of CF matched-real image paths (label real)")
    ap.add_argument("--json", help="write json report")
    args = ap.parse_args()

    reports = []

    # 1. clean eval (both classes)
    rows = []
    with open(ROOT / "data/manifests/dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "eval":
                rows.append(r)
    y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])
    p = np.array(predict([str(ROOT / r["path"]) for r in rows]))
    reports.append(report("eval (clean)", y, p))

    # 2. in-the-wild reals
    files = sorted((ROOT / "data/probe/real").glob("*.jpg"))
    if files:
        y = np.zeros(len(files))
        p = np.array(predict([str(f) for f in files]))
        reports.append(report("probe_real (in-the-wild)", y, p))

    # 3. matched real cells (CF-Eval, sampled via --cf-csv of paths)
    if args.cf_csv:
        paths = [l.strip() for l in Path(args.cf_csv).read_text().splitlines() if l.strip()]
        if paths:
            y = np.zeros(len(paths))
            p = np.array(predict(paths))
            reports.append(report("cf matched reals", y, p))

    # 4. generator fakes (Synthbuster+ local shards)
    shards = sorted((ROOT / "data/bench_pub/synthbuster").glob("*.parquet"))
    if shards:
        import pyarrow.parquet as pq

        tmp = ROOT / "data/bench_pub/_tmp_shift"
        tmp.mkdir(parents=True, exist_ok=True)
        paths = []
        for sh in shards:
            t = pq.read_table(str(sh))
            cols = t.column_names
            img_col = next((c for c in ("image", "image_data", "bytes") if c in cols), cols[0])
            for i, raw in enumerate(t.column(img_col).to_pylist()[:200]):
                b = raw["bytes"] if isinstance(raw, dict) else raw
                pth = tmp / f"sb_{sh.stem}_{i}.jpg"
                pth.write_bytes(bytes(b))
                paths.append(str(pth))
        if paths:
            y = np.ones(len(paths))
            p = np.array(predict(paths))
            reports.append(report("synthbuster fakes", y, p))

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
