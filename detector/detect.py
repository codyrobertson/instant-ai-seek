#!/usr/bin/env python3
"""Detector inference CLI.

Modes:
  detector/detect.py <image>          -> prints P(AI) as a float
  detector/detect.py --batch          -> reads image paths from stdin, prints one
                                         P(AI) per line (batched, order preserved)

Backend: detector/model_cnn.onnx (convnext_tiny fine-tuned, ONNX Runtime) if
present, else the Phase-1 logistic-regression baseline (model.json). Both
paths share the same CLI contract: P(ai) in [0,1], benchmark threshold 0.65.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

INPUT_SIZE = 384
RESIZE_TO = 416  # resize shortest side (matches training val), then center crop

# three product states (calibrated; threshold fixed at 0.65):
#   AI likely:   p >= 0.65
#   likely real: p < 0.35
#   uncertain:   otherwise (product should abstain from blurring)
LOW_CONF = 0.35
HIGH_CONF = 0.65


def product_state(p_ai: float) -> str:
    if p_ai >= HIGH_CONF:
        return "ai"
    if p_ai < LOW_CONF:
        return "real"
    return "uncertain"


# ---------------- CNN backend ----------------

def _load_cnn() -> tuple:
    """Returns ((student_sess, teacher_sess_or_None), meta)."""
    onnx_path = ROOT / "detector/model_cnn.onnx"
    meta_path = ROOT / "detector/model_cnn.json"
    if not (onnx_path.exists() and meta_path.exists() and ort is not None):
        return None, None
    meta = json.loads(meta_path.read_text())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # teacher: prefer fp32, fall back to fp16 then q8 (public repo ships fp16+q8)
    tsess = None
    for tname in ("model_cnn_teacher.onnx", "model_cnn_teacher_fp16.onnx", "model_cnn_teacher_q.onnx"):
        tpath = ROOT / f"detector/{tname}"
        if not tpath.exists():
            continue
        try:
            tsess = ort.InferenceSession(str(tpath), providers=["CPUExecutionProvider"])
            break
        except Exception:
            continue
    return (sess, tsess), meta


def preprocess_cnn(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = RESIZE_TO / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        left = (im.width - INPUT_SIZE) // 2
        top = (im.height - INPUT_SIZE) // 2
        im = im.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (arr - mean) / std


def predict_cnn_batch(paths: list[str], sess_pair, batch: int = 32) -> list[float]:
    """Ensemble of student + teacher (mean of P(AI)) when both are present."""
    sess, tsess = sess_pair
    confs: list[float] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        x = np.stack([preprocess_cnn(p) for p in chunk])
        probs = []
        for s in (sess, tsess):
            if s is None:
                continue
            logits = s.run(None, {"image": x})[0]
            if logits.shape[-1] == 2:  # 2-class head: P(ai) = softmax[:,1]
                e = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs.append(e[..., 1] / e.sum(axis=-1))
            else:  # 1-class head: sigmoid
                probs.append(1.0 / (1.0 + np.exp(-logits.reshape(-1))))
        p = np.max(probs, axis=0)  # max-fusion: any alarm is an alarm
        confs.extend(p.reshape(-1).tolist())
    return confs


# ---------------- LR backend (Phase-1 baseline) ----------------

def _load_lr() -> dict:
    with open(ROOT / "detector" / "model.json") as f:
        return json.load(f)


def predict_lr(features: list[float], model: dict) -> float:
    x = (np.asarray(features, dtype=np.float64) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = float(np.dot(x, model["coef"]) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def _lr_worker(path: str) -> float:
    from detector.features import extract_features

    return predict_lr(extract_features(path), _load_lr())


# ---------------- public API ----------------

def predict(paths: str | list[str]) -> list[float]:
    """Predict P(AI) for one path or a list (CNN batched when available)."""
    if isinstance(paths, str):
        paths = [paths]
    sess, meta = _load_cnn()
    if sess is not None:
        return predict_cnn_batch(paths, sess)
    return [_lr_worker(p) for p in paths]


def main() -> None:
    sess, meta = _load_cnn()
    backend = "cnn" if sess is not None else "lr"
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        paths = [p.strip() for p in sys.stdin if p.strip()]
        if sess is not None:
            for c in predict_cnn_batch(paths, sess):
                print(f"{c:.6f}")
        else:
            with mp.Pool(mp.cpu_count()) as pool:
                for c in pool.imap(_lr_worker, paths, chunksize=8):
                    print(f"{c:.6f}")
    elif len(sys.argv) == 2:
        print(f"{predict(sys.argv[1])[0]:.6f}")
    else:
        sys.exit("usage: detector/detect.py <image> | detector/detect.py --batch")
    sys.stderr.write(f"# backend={backend}\n")


if __name__ == "__main__":
    main()
