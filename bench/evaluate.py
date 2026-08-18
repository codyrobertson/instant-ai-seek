#!/usr/bin/env python3
"""Benchmark: evaluate the detector on the held-out eval set.

Computes the bounty metric — balanced accuracy at the 65% confidence
threshold — plus sensitivity/specificity and AUROC, and prints them as
METRIC lines consumed by autoresearch.sh.

Output (stdout):
    METRIC balanced_accuracy=0.XXXX
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def load_eval_rows() -> list[dict]:
    rows = []
    with open(ROOT / "data" / "manifests" / "dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "eval":
                rows.append(r)
    return rows


def main() -> None:
    rows = load_eval_rows()
    if not rows:
        sys.exit("no eval rows in data/manifests/dataset.csv — run scripts/build_dataset.py first")

    paths = [str(ROOT / r["path"]) for r in rows]
    confs = np.array(predict(paths))

    y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])

    # single measurement layer: bench/scorecard.metrics (frozen contract)
    from bench.scorecard import metrics as scorecard_metrics

    m = scorecard_metrics(y, confs, THRESHOLD)
    print(f"METRIC balanced_accuracy={m['balanced_accuracy']:.4f}")
    print(f"METRIC ai_recall={m['ai_recall']:.4f}")
    print(f"METRIC real_recall={m['real_recall']:.4f}")
    print(f"METRIC auroc={m['auroc']:.4f}")
    print(f"# n={len(rows)} threshold={THRESHOLD} tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}", file=sys.stderr)


if __name__ == "__main__":
    main()
