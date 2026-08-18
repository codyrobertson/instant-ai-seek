#!/usr/bin/env python3
"""Train the Phase-1 baseline detector: logistic regression on forensic features.

Deterministic (fixed seeds). Reads data/manifests/dataset.csv (split=train),
extracts features, standardizes, fits sklearn LogisticRegression, and writes
detector/model.json with the scaler and coefficients. The harness only
evaluates; this script is a setup step.

Usage: python3 detector/train.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.features import FEATURE_NAMES, extract_features  # noqa: E402

SEED = 0


def load_rows(split: str) -> list[dict]:
    rows = []
    with open(ROOT / "data" / "manifests" / "dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append(r)
    return rows


def main() -> None:
    train = load_rows("train")
    print(f"train rows: {len(train)}")
    X = np.array([extract_features(ROOT / r["path"]) for r in train])
    y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in train])

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    print(f"train balanced acc (0.65 thr): {balanced_accuracy_score(y, probs >= 0.65):.4f}")
    print(f"train auroc: {roc_auc_score(y, probs):.4f}")

    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    payload = {
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "version": 1,
    }
    out = ROOT / "detector" / "model.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
