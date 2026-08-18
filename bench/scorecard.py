#!/usr/bin/env python3
"""SOTA scorecard: one measurement layer for every candidate detector.

Emits JSON + a compact table with:
  - balanced accuracy, AUROC, AUPRC
  - AI recall, real recall, false-AI rate at the production threshold
  - AI recall at fixed real-FPR targets (0.5%, 1%, 2%)
  - Brier score and 10-bin ECE
  - per-generator and per-real-source results
  - bootstrap 95% confidence intervals

Threshold/calibration policy is read from the benchmark registry. The
scorecard NEVER selects a threshold — that is bench/calibrate.py's job on a
dev/calibration split only.

Usage:
  python3 bench/scorecard.py --model current \
      --registry data/manifests/benchmark_registry.json [--json out.json]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65
N_BOOT = 2000
SEED = 0


# ---------------- metrics ----------------

def metrics(y: np.ndarray, p: np.ndarray, thr: float = THRESHOLD) -> dict:
    y = y.astype(float)
    pred = p >= thr
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    ai_recall = tp / max(1, (y == 1).sum())
    real_recall = tn / max(1, (y == 0).sum())
    bal = (ai_recall + real_recall) / 2 if ((y == 1).sum() and (y == 0).sum()) else None
    false_ai = fp / max(1, (y == 0).sum())

    # AUROC / AUPRC
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, len(y) + 1)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    auroc = None
    auprc = None
    if n_pos and n_neg:
        u = ranks[y[order] == 1].sum() - n_pos * (n_pos + 1) / 2
        auroc = u / (n_pos * n_neg)
        # AUPRC via sorted precision-recall (step integration)
        order_desc = np.argsort(-p, kind="stable")
        ys = y[order_desc]
        prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
        rec = np.cumsum(ys) / max(1, n_pos)
        auprc = float(np.sum(prec * np.concatenate([[rec[0]], np.diff(rec)])))

    # Brier + ECE10
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    if len(y) >= 10:
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        for b in range(10):
            m = idx == b
            if m.sum() == 0:
                continue
            conf = p[m].mean()
            acc = y[m].mean()
            ece += (m.sum() / len(y)) * abs(conf - acc)

    # AI recall at fixed real-FPR targets
    rec_at_fpr = {}
    for tgt in (0.005, 0.01, 0.02):
        if n_pos and n_neg:
            cand = sorted(p[y == 0])  # real confidences
            cut = cand[max(0, int(np.ceil((1 - tgt) * len(cand))) - 1)] if len(cand) else 1.0
            rec_at_fpr[tgt] = float(((p >= cut) & (y == 1)).sum() / n_pos)
        else:
            rec_at_fpr[tgt] = None

    return {
        "n": int(len(y)),
        "n_ai": int(n_pos),
        "n_real": int(n_neg),
        "balanced_accuracy": None if bal is None else float(bal),
        "auroc": None if auroc is None else float(auroc),
        "auprc": None if auprc is None else float(auprc),
        "ai_recall": float(ai_recall),
        "real_recall": float(real_recall),
        "false_ai_rate": float(false_ai),
        "ai_recall_at_fpr": {str(k): (None if v is None else float(v)) for k, v in rec_at_fpr.items()},
        "brier": brier,
        "ece10": float(ece),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float = THRESHOLD,
                 n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    idx = list(range(len(y)))
    fields = ("balanced_accuracy", "ai_recall", "real_recall", "false_ai_rate", "auroc", "brier")
    samples = {f: [] for f in fields}
    for _ in range(n_boot):
        b = [rng.choice(idx) for _ in idx]
        m = metrics(y[b], p[b], thr)
        for f in fields:
            v = m.get(f)
            if v is not None:
                samples[f].append(v)
    out = {}
    for f, vals in samples.items():
        if not vals:
            out[f] = None
            continue
        a = np.array(vals)
        out[f] = {"mean": float(a.mean()), "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    return out


# ---------------- data loading ----------------

def load_eval_rows() -> list[dict]:
    rows = []
    with open(ROOT / "data" / "manifests" / "dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "eval":
                rows.append(r)
    return rows


def row_source(row: dict) -> str:
    """Per-generator / per-real-source label from the path."""
    p = Path(row["path"])
    parts = p.parts
    if "real" in parts:
        return "real:" + (parts[parts.index("real") - 1] if parts.index("real") > 0 else "unknown")
    if "ai" in parts:
        return "ai:" + (parts[parts.index("ai") - 1] if parts.index("ai") > 0 else "unknown")
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="current", help="model id (for reporting)")
    ap.add_argument("--registry", default=str(ROOT / "data/manifests/benchmark_registry.json"))
    ap.add_argument("--json", help="write JSON report to this path")
    ap.add_argument("--sources", default="eval", help="comma-separated suite names (eval, holdout2, probe_real)")
    args = ap.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    thr = registry["policy"]["threshold"]

    reports = {}
    for suite in args.sources.split(","):
        suite = suite.strip()
        if suite == "eval":
            rows = load_eval_rows()
            y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])
            p = np.array(predict([str(ROOT / r["path"]) for r in rows]))
        elif suite == "holdout2":
            rows = []
            for f in sorted((ROOT / "data/holdout2/ai").glob("*/*.jpg")):
                rows.append({"path": str(f.relative_to(ROOT)), "label": "ai"})
            y = np.ones(len(rows))
            p = np.array(predict([str(ROOT / r["path"]) for r in rows]))
        elif suite == "probe_real":
            rows = []
            for f in sorted((ROOT / "data/probe/real").glob("*.jpg")):
                rows.append({"path": str(f.relative_to(ROOT)), "label": "real"})
            y = np.zeros(len(rows))
            p = np.array(predict([str(ROOT / r["path"]) for r in rows]))
        else:
            print(f"unknown suite: {suite}", file=sys.stderr)
            sys.exit(1)

        m = metrics(y, p, thr)
        ci = bootstrap_ci(y, p, thr)
        per = {}
        src_of = [row_source(r) for r in rows]
        for s in sorted(set(src_of)):
            idx = [i for i, x in enumerate(src_of) if x == s]
            if len(idx) >= 10:
                per[s] = metrics(y[idx], p[idx], thr)
        reports[suite] = {"n": len(rows), "metrics": m, "ci95": ci, "per_source": per,
                          "model": args.model, "threshold": thr}

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"wrote {args.json}")

    print(f"== scorecard: model={args.model} threshold={thr} ==")
    for suite, rep in reports.items():
        m = rep["metrics"]
        ci = rep["ci95"]
        bal = m["balanced_accuracy"]
        bal_ci = ci["balanced_accuracy"]["ci95"] if ci.get("balanced_accuracy") else None
        print(f"--- {suite} (n={rep['n']}) ---")
        if bal is not None:
            print(f"  balanced_acc = {bal:.4f}  [95% CI {bal_ci[0]:.4f}..{bal_ci[1]:.4f}]")
        else:
            print("  balanced_acc = n/a (single-class)")
        print(f"  auroc={m['auroc']:.4f}  auprc={m['auprc']:.4f}  brier={m['brier']:.4f}  ece10={m['ece10']:.4f}")
        print(f"  ai_recall={m['ai_recall']:.4f}  real_recall={m['real_recall']:.4f}  false_ai={m['false_ai_rate']:.4f}")
        print(f"  ai_recall@fpr .5%/1%/2%: {m['ai_recall_at_fpr']['0.005']:.4f}/{m['ai_recall_at_fpr']['0.01']:.4f}/{m['ai_recall_at_fpr']['0.02']:.4f}")
        if rep["per_source"]:
            for s, sm in sorted(rep["per_source"].items()):
                bal_s = sm["balanced_accuracy"]
                bal_s = f"{bal_s:.4f}" if bal_s is not None else "n/a"
                print(f"    {s:28s} {bal_s}  ai={sm['ai_recall']:.3f} real={sm['real_recall']:.3f} n={sm['n']}")


if __name__ == "__main__":
    main()
