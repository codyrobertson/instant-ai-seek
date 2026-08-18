#!/usr/bin/env python3
"""Evaluate the current detector on generalization probe sets.

Sources:
  data/holdout{,2}/ai/<model>/*.jpg  (fresh generations, NEVER trained on)
  data/probe/ai/<model>/*.jpg        (earlier probe set, partially in training)
  data/probe/real/*.jpg              (independent real photos)

Prints per-source: mean confidence, fraction called AI (>=0.65), and for the
real source the false-AI rate. Also runs within-source controls on data/train.

This is a diagnostic — it does NOT touch data/eval and does not run the
benchmark.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def score_dir(d: Path, label: str) -> None:
    files = sorted(d.glob("*.jpg"))
    if not files:
        print(f"{label:28s} no images")
        return
    confs = np.array(predict([str(p) for p in files]))
    called_ai = (confs >= THRESHOLD).mean()
    print(f"{label:28s} n={len(files):3d}  mean_conf={confs.mean():.3f}  pct_ai={called_ai*100:5.1f}%  "
          f"min={confs.min():.3f} max={confs.max():.3f}")


def main() -> None:
    print(f"threshold={THRESHOLD}")
    print("--- HOLDOUT AI probes (never trained on; honest out-of-sample) ---")
    for ho_name in ("holdout", "holdout2"):
        ho_root = ROOT / "data" / ho_name / "ai"
        for model_dir in sorted(ho_root.iterdir()) if ho_root.exists() else []:
            if model_dir.is_dir():
                score_dir(model_dir, f"{ho_name}/{model_dir.name}")
    print("--- training-time probe sources (partially in-sample) ---")
    ai_root = ROOT / "data" / "probe" / "ai"
    for model_dir in sorted(ai_root.iterdir()) if ai_root.exists() else []:
        if model_dir.is_dir():
            score_dir(model_dir, f"ai/{model_dir.name}")
    print("--- real probe (lower pct_ai = better) ---")
    score_dir(ROOT / "data" / "probe" / "real", "real/picsum")
    # scorecard line: false-AI rate + CI on the in-the-wild real set
    files = sorted((ROOT / "data" / "probe" / "real").glob("*.jpg"))
    if files:
        from bench.scorecard import metrics as scorecard_metrics, bootstrap_ci
        import numpy as _np
        _p = _np.array(predict([str(p) for p in files]))
        _y = _np.zeros(len(files))
        _m = scorecard_metrics(_y, _p, THRESHOLD)
        _ci = bootstrap_ci(_y, _p, THRESHOLD, n_boot=1000)
        _f = _ci["false_ai_rate"]["ci95"]
        print(f"real-probe scorecard: n={len(files)} false_ai={_m['false_ai_rate']:.4f} "
              f"[95% CI {_f[0]:.4f}..{_f[1]:.4f}] real_recall={_m['real_recall']:.4f} brier={_m['brier']:.4f}")
    print("--- within-source controls ---")
    score_dir(ROOT / "data" / "train" / "ai", "train/ai (flux schnell)")
    score_dir(ROOT / "data" / "train" / "real", "train/real (imagenette)")


if __name__ == "__main__":
    main()
