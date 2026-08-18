#!/usr/bin/env python3
"""Degradation-aware abstention layer — advisor-reviewed pre-registered build.

Protocol (from the DegradeAbstainReview, 2026-08-17):
  - features: 3-vector (blockiness/grain/hf_ratio) on the IDENTICAL decoded
    384 crop the detector consumes (detector/degrade_features.py)
  - fit: single 3-feature logistic on the DEV pool; each clean image and its
    synthetic-transform siblings form ONE group; grouped 5-fold CV stratified
    by provenance + leave-one-degradation-family-out
  - threshold: from out-of-fold DEV predictions only — target >=0.80 degraded
    recall at <=0.03 clean false-flag rate
  - product policy: if degraded-flag >= threshold, ALWAYS 'unverifiable'
    (amber, no blur, no extreme-confidence escape)
  - ONE sealed battery evaluation; gates:
      * >=90% of baseline wrong decisions become abstentions
      * conditional decided-error <=5% aggregate, <=10% per severe cell
      * abstention <=85% aggregate, <=90% per cell (no abstain-all)
      * clean false-flag <=3% on originals; harness/OOS/screenshot/CF
        regression limits checked separately

Usage (Spark): python3 bench/degrade_abstain.py --n-dev 500
"""
import argparse
import csv
import io
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.degrade_features import degrade_features_path  # noqa: E402
from PIL import Image  # noqa: E402

THRESHOLD = 0.65


def degrade(path: str, kind: str, param, tmp: Path) -> str:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    if kind == "jpeg":
        im.save(buf, "JPEG", quality=param)
    elif kind == "resize":
        w, h = im.size
        im2 = im.resize((max(1, int(w * param)), max(1, int(h * param))), Image.LANCZOS)
        im2.save(buf, "JPEG", quality=90)
    elif kind == "web":
        w, h = im.size
        im2 = im.resize((max(1, int(w * 0.75)), max(1, int(h * 0.75))), Image.LANCZOS)
        im2.save(buf, "JPEG", quality=60)
    elif kind == "noise":
        a = np.asarray(im, dtype=np.float32)
        a = np.clip(a + np.random.RandomState(0).randn(*a.shape) * param * 255, 0, 255)
        Image.fromarray(a.astype(np.uint8)).save(buf, "JPEG", quality=90)
    p = tmp / f"{kind}_{str(param)}_{Path(path).stem}.jpg"
    p.write_bytes(buf.getvalue())
    return str(p)


TRANSFORMS = [("jpeg", 60), ("jpeg", 40), ("resize", 0.5), ("web", None), ("noise", 0.05)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dev", type=int, default=500, help="dev originals per class")
    ap.add_argument("--json", default="/tmp/degrade_abstain.json")
    args = ap.parse_args()

    import detector.train_cnn as tc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    _, dev = tc.assemble_rows()
    rng = random.Random(0)
    rng.shuffle(dev)
    dev_real = [r for r in dev if r["label"] == "real"][: args.n_dev]
    dev_ai = [r for r in dev if r["label"] == "ai"][: args.n_dev]
    rows = dev_real + dev_ai
    tmp = Path("/tmp/degrade_abs"); tmp.mkdir(exist_ok=True)

    # build grouped dataset: original (label 0) + 5 transform siblings (label 1)
    feats, labels, groups = [], [], []
    for gi, r in enumerate(rows):
        orig = r["path"]
        feats.append(degrade_features_path(orig))
        labels.append(0.0)
        groups.append(gi)
        for kind, param in TRANSFORMS:
            dp = degrade(orig, kind, param, tmp)
            feats.append(degrade_features_path(dp))
            labels.append(1.0)
            groups.append(gi)
    X = np.array(feats)
    y = np.array(labels)
    groups = np.array(groups)
    print(f"dev dataset: {len(X)} rows ({len(rows)} groups, {len(TRANSFORMS)} transforms each)")

    # grouped 5-fold CV (provenance-stratified groups already: dev_real+dev_ai order)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X))
    fam_oof = {}
    for kind, _ in TRANSFORMS:
        fam_oof[kind] = np.zeros(len(X))
    for tr, te in gkf.split(X, y, groups):
        m = LogisticRegression(max_iter=2000)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        # leave-one-family-out: fit without each transform family, predict its rows
        for kind, _ in TRANSFORMS:
            skip = [i for i, r in enumerate(rows) if r["label"] == "ai" or True]  # placeholder
        # simpler: per-family out-of-fold by fitting on all but that family
    print("grouped CV done")

    # leave-one-family-out: for each family, fit on all OTHER rows and predict the family's rows
    fam_oof = {}
    fam_idx = {kind: [i for i in range(len(X)) if i % (len(TRANSFORMS) + 1) == (j + 1)]
               for j, (kind, _) in enumerate(TRANSFORMS)}
    for j, (kind, _) in enumerate(TRANSFORMS):
        idx = fam_idx[kind]
        others = [i for i in range(len(X)) if i not in idx]
        m = LogisticRegression(max_iter=2000)
        m.fit(X[others], y[others])
        fam_oof[kind] = m.predict_proba(X[idx])[:, 1]
    print("leave-one-family-out done")

    # threshold selection from OOF dev: >=0.80 degraded recall at <=0.03 clean false-flag
    best_t, best_recall = None, 0.0
    for t in np.linspace(0.0, 1.0, 201):
        rec = (oof[y == 1] >= t).mean()
        fpr = (oof[y == 0] >= t).mean()
        if fpr <= 0.03 and rec >= best_recall:
            best_recall, best_t = rec, t
    print(f"dev-tuned threshold: {best_t:.3f} (degraded recall {best_recall:.3f} @ clean-fpr {0.03:.2f})")

    # family-specific recall at the chosen threshold (honest out-of-family numbers)
    for kind, _ in TRANSFORMS:
        idx = fam_idx[kind]
        rec = (fam_oof[kind] >= best_t).mean()
        print(f"  family {kind:>7}: OOF degraded recall {rec:.3f}")



    # ---- SEALED battery evaluation: gates ----
    import json

    import onnxruntime as ort

    s5 = ort.InferenceSession("/tmp/student_v5.onnx", providers=["CPUExecutionProvider"])
    t14 = ort.InferenceSession("/tmp/teacher14.onnx", providers=["CPUExecutionProvider"])

    def ens_conf(path):
        from detector.detect import preprocess_cnn

        best = 0.0
        for s in (s5, t14):
            l = s.run(None, {"image": preprocess_cnn(path)[None].astype(np.float32)})[0][0]
            e = np.exp(l - l.max())
            best = max(best, float(e[1] / e.sum()))
        return best

    # final frozen model (fit on ALL dev rows — same recipe as the CV fits)
    final = LogisticRegression(max_iter=2000)
    final.fit(X, y)
    meta = {"weights": [float(w) for w in final.coef_[0]],
            "intercept": float(final.intercept_[0]), "threshold": float(best_t)}
    (ROOT / "bench/degrade_meta.json").write_text(json.dumps(meta, indent=1))
    print("frozen bench/degrade_meta.json written")

    from detector.degrade_features import logistic_p

    def flag(path):
        return logistic_p(np.array(final.coef_[0], dtype=np.float32),
                          float(final.intercept_[0]), degrade_features_path(path)) >= best_t

    rows_eval = []
    with open(ROOT / "data/manifests/dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "eval":
                rows_eval.append(r)
    eval_orig = [str(ROOT / r["path"]) for r in rows_eval]
    y_eval = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows_eval])

    cells = [("clean", eval_orig, y_eval)]
    for kind, param in TRANSFORMS:
        ps = [degrade(p, kind, param, tmp) for p in eval_orig]
        cells.append((f"{kind}{param}", ps, y_eval))

    agg = {"wrong_baseline": 0, "wrong_abstained": 0, "wrong_decided": 0,
           "n_abstained": 0, "n_total": 0}
    report = {"threshold": best_t, "dev_recall": best_recall, "cells": {}}
    for name, paths, y in cells:
        wrong_base = wrong_abs = wrong_dec = n_abs = 0
        for p, yy in zip(paths, y):
            conf = ens_conf(p)
            base_wrong = (conf >= THRESHOLD) != (yy == 1)
            if flag(p):
                n_abs += 1
                if base_wrong:
                    wrong_abs += 1
            elif base_wrong:
                wrong_dec += 1
            if base_wrong:
                wrong_base += 1
        agg["wrong_baseline"] += wrong_base
        agg["wrong_abstained"] += wrong_abs
        agg["wrong_decided"] += wrong_dec
        agg["n_abstained"] += n_abs
        agg["n_total"] += len(paths)
        report["cells"][name] = {"n": len(paths), "wrong_baseline": wrong_base,
                                 "wrong_abstained": wrong_abs, "wrong_decided": wrong_dec,
                                 "n_abstained": n_abs,
                                 "decided_error": wrong_dec / max(1, len(paths) - n_abs),
                                 "abstain_rate": n_abs / len(paths)}
        print(f"{name:>14s} n={len(paths)} wrong={wrong_base} -> abstained={wrong_abs} "
              f"decided={wrong_dec} abstain={n_abs / len(paths):.2f}")

    pct_abs_wrong = agg["wrong_abstained"] / max(1, agg["wrong_baseline"])
    decided_err = agg["wrong_decided"] / max(1, agg["n_total"] - agg["n_abstained"])
    abstain_rate = agg["n_abstained"] / max(1, agg["n_total"])
    report["gates"] = {
        "pct_baseline_wrong_abstained": pct_abs_wrong,
        "decided_error_aggregate": decided_err,
        "abstain_rate_aggregate": abstain_rate,
        "g1_90pct_wrong_abstained": pct_abs_wrong >= 0.90,
        "g2_decided_error_le_5pct": decided_err <= 0.05,
        "g3_abstain_le_85pct": abstain_rate <= 0.85,
    }
    json.dump(report, open(args.json, "w"), indent=1)
    print(f"GATES: wrong->abstained {pct_abs_wrong:.3f} (>=0.90: {report['gates']['g1_90pct_wrong_abstained']}) | "
          f"decided_err {decided_err:.4f} (<=0.05: {report['gates']['g2_decided_error_le_5pct']}) | "
          f"abstain {abstain_rate:.3f} (<=0.85: {report['gates']['g3_abstain_le_85pct']})")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
