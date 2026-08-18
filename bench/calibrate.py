#!/usr/bin/env python3
"""Calibration-policy selection on a DEV/CALIBRATION split only.

Never touches sealed test sets. Fits and compares:
  - identity (no-op)
  - temperature scaling (scipy optimize on logits)
  - isotonic regression (scikit-learn)
  - two-threshold policy (AI likely / uncertain / likely real)

Threshold stays fixed at 0.65 (product contract). The calibrated confidence
is what the product reports; the binary decision at 0.65 remains.

Usage:
  python3 bench/calibrate.py --manifest data/manifests/calibration_split.csv \
      [--out bench/calibration.json] [--cv-folds 5]

The manifest is a CSV: label,path (label = ai|real). Images must be reachable
from the repo root (the calibration split lives on the DGX Spark with the
training pool).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65


def load_manifest(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])
    def resolve(rel):
        q = Path(rel)
        return q if q.is_absolute() else (ROOT / q)
    p = np.array(predict([str(resolve(r["path"])) for r in rows]))
    return y, p, [r["path"] for r in rows]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
    e = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimize NLL over a single temperature on raw logits."""
    from scipy.optimize import minimize

    def nll(T: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-6)))
        eps = 1e-9
        return -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    res = minimize(nll, x0=1.0, method="Nelder-Mead")
    return float(res.x[0])


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    logit = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    return 1.0 / (1.0 + np.exp(-logit / max(T, 1e-6)))


def fit_isotonic(y: np.ndarray, p: np.ndarray, min_fold: int = 100) -> tuple[object | None, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return iso, iso.predict(p)


def two_threshold(p: np.ndarray, y: np.ndarray, low: float = 0.35, high: float = 0.65) -> dict:
    """Abstention policy: below low => likely real, above high => AI likely,
    between => uncertain (abstain). Reports abstention rate + calibrated
    errors within the decided regions."""
    decided = (p >= high) | (p <= low)
    if decided.sum() == 0:
        return {"low": low, "high": high, "abstain_rate": 1.0, "error_within_decided": None}
    err = float(np.mean((p[decided] >= 0.65) != (y[decided] == 1)))
    return {"low": low, "high": high, "abstain_rate": float(1 - decided.mean()),
            "error_within_decided": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=str(ROOT / "bench/calibration.json"))
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    y, p, paths = load_manifest(Path(args.manifest))
    print(f"calibration split: n={len(y)} ai={(y == 1).sum()} real={(y == 0).sum()}")
    if (y == 1).sum() < 100 or (y == 0).sum() < 100:
        print("WARNING: small calibration split; CIs will be wide", file=sys.stderr)

    logits = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))

    # cross-validated policy comparison (keep threshold fixed)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(len(y)), args.cv_folds)
    cv = {"identity": [], "temperature": [], "isotonic": [], "two_threshold": []}
    for k in range(args.cv_folds):
        test = folds[k]
        tr = np.concatenate([folds[j] for j in range(args.cv_folds) if j != k])
        # temperature fit on train logits
        T = fit_temperature(logits[tr], y[tr])
        pT = apply_temperature(p[test], T)
        iso = fit_isotonic(y[tr], p[tr])[0]
        pI = iso.predict(p[test]) if iso is not None else p[test]
        tt = two_threshold(p[test], y[test])
        cv["identity"].append(brier(y[test], p[test]))
        cv["temperature"].append(brier(y[test], pT))
        cv["isotonic"].append(brier(y[test], pI))
        cv["two_threshold"].append(tt)

    # final fits on the whole calibration split (parameters for the product)
    T_final = fit_temperature(logits, y)
    iso_final = fit_isotonic(y, p)[0]

    report = {
        "threshold": THRESHOLD,
        "n": int(len(y)),
        "policy_cv": {k: ({"brier_mean": float(np.mean(v)) if v and isinstance(v[0], float) else None,
                           "brier_folds": [float(x) if isinstance(x, float) else None for x in v]}) for k, v in cv.items()},
        "two_threshold_cv": cv["two_threshold"],
        "final_temperature": T_final,
        "final_isotonic_fitted": bool(iso_final is not None),
        "identity": {"brier": brier(y, p), "ece10": ece10(y, p)},
        "temperature": {"brier": brier(y, apply_temperature(p, T_final)),
                        "ece10": ece10(y, apply_temperature(p, T_final))},
        "isotonic": {"brier": brier(y, iso_final.predict(p)) if iso_final else None,
                     "ece10": ece10(y, iso_final.predict(p)) if iso_final else None},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
