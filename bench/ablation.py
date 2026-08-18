#!/usr/bin/env python3
"""Single-variable ablation / tournament comparison between two detectors.

Scores two model bundles on the same suites (eval, holdout2, probe_real or
custom dirs) with the frozen scorecard and prints a side-by-side delta.

Usage:
  python3 bench/ablation.py --base current --candidate runs/t1_seed0/model_hybrid.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.scorecard import bootstrap_ci, metrics  # noqa: E402
from detector.detect import preprocess_cnn  # noqa: E402

THRESHOLD = 0.65


def predict_onnx(sess, paths: list[str], batch: int = 32) -> np.ndarray:
    import onnxruntime as ort

    out = []
    for i in range(0, len(paths), batch):
        x = np.stack([preprocess_cnn(p) for p in paths[i : i + batch]])
        logits = sess.run(None, {"image": x.astype(np.float32)})[0]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        out.append((e[..., 1] / e.sum(axis=-1)).reshape(-1))
    return np.concatenate(out)


def load_suite(name: str) -> tuple[np.ndarray, list[str]]:
    import csv

    if name == "eval":
        rows = []
        with open(ROOT / "data/manifests/dataset.csv") as f:
            for r in csv.DictReader(f):
                if r["split"] == "eval":
                    rows.append(r)
        y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])
        return y, [str(ROOT / r["path"]) for r in rows]
    if name == "holdout2":
        paths = sorted((ROOT / "data/holdout2/ai").glob("*/*.jpg"))
        return np.ones(len(paths)), [str(p) for p in paths]
    if name == "probe_real":
        paths = sorted((ROOT / "data/probe/real").glob("*.jpg"))
        return np.zeros(len(paths)), [str(p) for p in paths]
    sys.exit(f"unknown suite {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="current", help="base model id (current = detector bundle) or onnx path")
    ap.add_argument("--candidate", required=True, help="candidate onnx path")
    ap.add_argument("--suites", default="eval", help="comma-separated suites")
    ap.add_argument("--json", help="write comparison json")
    args = ap.parse_args()

    import onnxruntime as ort

    def get_sess(spec: str):
        if spec == "current":
            return None, "current-bundle"
        if not Path(spec).exists():
            sys.exit(f"not found: {spec}")
        return ort.InferenceSession(spec, providers=["CPUExecutionProvider"]), spec

    base_sess, base_name = get_sess(args.base)
    cand_sess, cand_name = get_sess(args.candidate)

    def run(sess, paths):
        if sess is None:
            from detector.detect import predict

            return np.array(predict(paths))
        return predict_onnx(sess, paths)

    print(f"ablation: {base_name} vs {cand_name}")
    rows_out = {}
    for suite in args.suites.split(","):
        y, paths = load_suite(suite)
        pb = run(base_sess, paths)
        pc = run(cand_sess, paths)
        mb = metrics(y, pb, THRESHOLD)
        mc = metrics(y, pc, THRESHOLD)
        cib = bootstrap_ci(y, pb, THRESHOLD, n_boot=1000)
        cic = bootstrap_ci(y, pc, THRESHOLD, n_boot=1000)
        print(f"--- {suite} (n={len(y)}) ---")
        for k in ("balanced_accuracy", "ai_recall", "real_recall", "false_ai_rate", "brier", "ece10"):
            vb = mb.get(k)
            vc = mc.get(k)
            if vb is None or vc is None:
                print(f"  {k:18s} n/a")
                continue
            if isinstance(vb, dict):
                continue
            cb = cib.get(k, {})
            cc = cic.get(k, {})
            ci_b = f" [{cb['ci95'][0]:.4f}..{cb['ci95'][1]:.4f}]" if cb.get("ci95") else ""
            ci_c = f" [{cc['ci95'][0]:.4f}..{cc['ci95'][1]:.4f}]" if cc.get("ci95") else ""
            print(f"  {k:18s} {vb:.4f}{ci_b}  ->  {vc:.4f}{ci_c}   (Δ {vc - vb:+.4f})")
        rows_out[suite] = {"base": mb, "candidate": mc}
    if args.json:
        Path(args.json).write_text(json.dumps(rows_out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
