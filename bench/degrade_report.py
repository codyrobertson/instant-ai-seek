#!/usr/bin/env python3
"""Degradation-robustness battery (AIGIBench-style protocol).

Synthesizes post-processing degradations of the sealed eval set and measures
the bundle's drop per degradation:
  - JPEG recompress q80/q60/q40
  - resize 0.75x / 0.5x (and back to 384 pipeline)
  - Gaussian blur sigma 1/2
  - Gaussian noise sigma 0.02/0.05
  - mixed "web" chain: resize 0.75 + JPEG q60

A detector that holds under degradation has a real-world robustness claim;
a big drop names the next training augmentation. Evaluation-only — the
transformed images are never training data.

Usage: python3 bench/degrade_report.py [--json out.json]
"""
import argparse
import csv
import json
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.detect import predict  # noqa: E402

THRESHOLD = 0.65
TMP = ROOT / "data/bench_pub/_tmp_degrade"


def degrade(img_bytes: bytes, kind: str, param) -> bytes:
    from PIL import Image, ImageFilter

    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if kind == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=param)
        return buf.getvalue()
    if kind == "resize":
        scale = param
        w, h = im.size
        im2 = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im2.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    if kind == "blur":
        im2 = im.filter(ImageFilter.GaussianBlur(param))
        buf = io.BytesIO()
        im2.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    if kind == "noise":
        a = np.asarray(im, dtype=np.float32)
        a = np.clip(a + np.random.RandomState(0).randn(*a.shape) * param * 255, 0, 255)
        im2 = Image.fromarray(a.astype(np.uint8))
        buf = io.BytesIO()
        im2.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    if kind == "web":
        scale, q = param
        w, h = im.size
        im2 = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im2.save(buf, "JPEG", quality=q)
        return buf.getvalue()
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    with open(ROOT / "data/manifests/dataset.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "eval":
                rows.append(r)
    y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in rows])
    base_paths = [str(ROOT / r["path"]) for r in rows]

    # baseline
    p0 = np.array(predict(base_paths))

    def report(name, p):
        pred = p >= THRESHOLD
        tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
        tnr = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
        bal = (tpr + tnr) / 2
        drop = bal - baseline_bal
        print(f"{name:28s} bal={bal:.4f} (Δ {drop:+.4f})  ai={tpr:.3f} real={tnr:.3f}")
        return {"name": name, "bal": bal, "delta": drop, "ai": tpr, "real": tnr}

    pred = p0 >= THRESHOLD
    tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
    baseline_bal = (tpr + tnr) / 2
    print(f"{'baseline (clean)':28s} bal={baseline_bal:.4f}  ai={tpr:.3f} real={tnr:.3f}")

    TMP.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for kind, param, name in [
        ("jpeg", 80, "jpeg q80"), ("jpeg", 60, "jpeg q60"), ("jpeg", 40, "jpeg q40"),
        ("resize", 0.75, "resize 0.75x"), ("resize", 0.5, "resize 0.5x"),
        ("blur", 1, "blur sigma1"), ("blur", 2, "blur sigma2"),
        ("noise", 0.02, "noise 0.02"), ("noise", 0.05, "noise 0.05"),
        ("web", (0.75, 60), "web chain (0.75x+q60)"),
    ]:
        paths = []
        for i, r in enumerate(rows):
            raw = Path(base_paths[i]).read_bytes()
            out = TMP / f"{kind}_{param}_{i:03d}.jpg"
            out.write_bytes(degrade(raw, kind, param))
            paths.append(str(out))
        p = np.array(predict(paths))
        report_rows.append(report(name, p))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "baseline_bal": baseline_bal, "degradations": report_rows,
        }, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
