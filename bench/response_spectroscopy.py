#!/usr/bin/env python3
"""Forensic response spectroscopy (advisor plan item 1 + 2).

The frozen ensemble scores ONE image once. This measures how its evidence
BEHAVES under a controlled stress family and builds a response vector:

  [p_clean,
   d_jpeg95, d_jpeg80, d_jpeg60, d_jpeg40,     # file-level recompression deltas
   d_resize075, d_resize05,                     # downscale deltas
   d_blur2, d_noise02,                          # smoothing/noise deltas
   crop_spread, max_crop_delta]                 # 5-position crop grid (idea 2)

Question (no retraining): do REAL and AI images have different response
curves even when their clean scores overlap? If yes, a tiny logistic
meta-classifier on the DEV split can sharpen the boundary.

Fit: dev-pool sample. Test: sealed eval + screenshot stratum + CF matched
reals. Report AUROC clean-only vs meta on the overlap zone (0.3..0.8).

Usage (DGX Spark, GPU idle): python3 bench/response_spectroscopy.py
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

from detector.detect import preprocess_cnn  # noqa: E402
from PIL import Image  # noqa: E402

SIZE = 384


def _conf1(path, sess_pair):
    x = preprocess_cnn(path)[None].astype(np.float32)
    best = 0.0
    for sess in sess_pair:
        if sess is None:
            continue
        l = sess.run(None, {"image": x})[0][0]
        e = np.exp(l - l.max())
        best = max(best, float(e[1] / e.sum()))
    return best


def _conf_batch(paths, sess_pair):
    x = np.stack([preprocess_cnn(p) for p in paths]).astype(np.float32)
    best = np.zeros(len(paths))
    for sess in sess_pair:
        if sess is None:
            continue
        l = sess.run(None, {"image": x})[0]
        e = np.exp(l - l.max(axis=-1, keepdims=True))
        best = np.maximum(best, e[..., 1] / e.sum(axis=-1))
    return best


def file_degrade(path, kind, param, tmp):
    im = Image.open(path).convert("RGB")
    if max(im.size) > 768:
        s = 768 / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    buf = io.BytesIO()
    if kind == "jpeg":
        im.save(buf, "JPEG", quality=param)
    elif kind == "resize":
        w, h = im.size
        im2 = im.resize((max(1, int(w * param)), max(1, int(h * param))), Image.LANCZOS)
        im2.save(buf, "JPEG", quality=90)
    elif kind == "blur":
        im2 = im.filter(ImageFilter := __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(param))
        im2.save(buf, "JPEG", quality=90)
    elif kind == "noise":
        a = np.asarray(im, dtype=np.float32)
        rng = np.random.RandomState(0)
        a = np.clip(a + rng.randn(*a.shape) * param * 255, 0, 255)
        Image.fromarray(a.astype(np.uint8)).save(buf, "JPEG", quality=90)
    p = tmp / f"{kind}_{str(param)}_{Path(path).stem}.jpg"
    p.write_bytes(buf.getvalue())
    return str(p)


def crop_confs(path, sess_pair, tmp):
    """5-position crop grid: center + 4 corners of the 384 window on the
    416-resize. Returns the per-crop confidences."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = 416 / min(w, h)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    W, H = im.size
    positions = [(0, 0), (W - SIZE, 0), (0, H - SIZE), (W - SIZE, H - SIZE),
                 ((W - SIZE) // 2, (H - SIZE) // 2)]
    crops = []
    for j, (lx, ty) in enumerate(positions):
        p = tmp / f"crop_{Path(path).stem}_{j}.jpg"
        im.crop((lx, ty, lx + SIZE, ty + SIZE)).save(p, "JPEG", quality=92)
        crops.append(str(p))
    return _conf_batch(crops, sess_pair)


def response_vector(path, sess_pair, tmp) -> np.ndarray:
    p_clean = _conf1(path, sess_pair)
    feats = [p_clean]
    degrades = []
    for kind, param in (("jpeg", 95), ("jpeg", 80), ("jpeg", 60), ("jpeg", 40),
                        ("resize", 0.75), ("resize", 0.5), ("blur", 2), ("noise", 0.02)):
        degrades.append(file_degrade(path, kind, param, tmp))
    dconfs = _conf_batch(degrades, sess_pair)
    for dp in dconfs:
        feats.append(float(dp) - p_clean)
    cc = crop_confs(path, sess_pair, tmp)
    feats.append(float(cc.std()))
    feats.append(float(np.abs(cc - cc[-1]).max()))
    return np.array(feats, dtype=np.float32)


def auroc(y, p):
    """Mann-Whitney U with AVERAGE ranks (tie-correct) — the previous rank
    assignment mishandled ties."""
    n_pos = (y == 1).sum()
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    # average ranks for tied values
    ranks = np.empty(len(ps))
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1  # 1-indexed average rank
        i = j + 1
    u = ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-sample", type=int, default=1200, help="dev pool images per class")
    ap.add_argument("--out", default="/tmp/spectroscopy.json")
    args = ap.parse_args()

    import json
    import onnxruntime as ort

    s5 = ort.InferenceSession("/tmp/student_v5.onnx", providers=["CPUExecutionProvider"])
    t14 = ort.InferenceSession("/tmp/teacher14.onnx", providers=["CPUExecutionProvider"])
    pair = (s5, t14)
    tmp = Path("/tmp/spectro"); tmp.mkdir(exist_ok=True)

    def feats_for(paths):
        return np.array([response_vector(p, pair, tmp) for p in paths])

    # dev-pool sample (fit the meta-classifier honestly)
    import detector.train_cnn as tc

    _, dev = tc.assemble_rows()
    rng = random.Random(0)
    rng.shuffle(dev)
    dev_real = [r for r in dev if r["label"] == "real"][: args.dev_sample]
    dev_ai = [r for r in dev if r["label"] == "ai"][: args.dev_sample]
    dev_paths = [r["path"] for r in dev_real + dev_ai]
    dev_y = np.array([0.0] * len(dev_real) + [1.0] * len(dev_ai))
    print(f"dev sample: {len(dev_paths)} ({len(dev_real)} real, {len(dev_ai)} ai)")
    F_dev = feats_for(dev_paths)
    print("response features done (dev)")

    # sealed eval test
    rows = list(csv.DictReader(open(ROOT / "data/manifests/dataset.csv")))
    ev = [r for r in rows if r["split"] == "eval"]
    ev_paths = [str(ROOT / r["path"]) for r in ev]
    ev_y = np.array([1.0 if r["label"] == "ai" else 0.0 for r in ev])
    F_ev = feats_for(ev_paths)
    print("response features done (eval)")

    # screenshot stratum (sealed)
    ss_dir = ROOT / "data/probe2/screenshots"
    ss_real = sorted(ss_dir.glob("real_shot_*"))[:40]
    ss_ai = sorted(ss_dir.glob("ai_shot_*"))[:40]
    ss_paths = [str(p) for p in ss_real + ss_ai]
    ss_y = np.array([0.0] * len(ss_real) + [1.0] * len(ss_ai))
    F_ss = feats_for(ss_paths)
    print("response features done (screenshots)")

    # CF matched reals + GAN fakes (the fuzzy population — eval is saturated)
    import pyarrow.parquet as pq

    rng = random.Random(0)
    cf_paths, cf_y = [], []
    tmpcf = Path("/tmp/spectro_cf"); tmpcf.mkdir(exist_ok=True)
    for shard, want, tag in (("CompEval-00000-of-00413.parquet", 0, "cfr"),
                             ("CompEval-00010-of-00413.parquet", 0, "cfr10"),
                             ("CompEval-00000-of-00413.parquet", 1, "cff")):
        t = pq.read_table(str(ROOT / "data/bench_pub/cf" / shard), columns=["image_data", "label"])
        rows = t.column("image_data").to_pylist()
        labs = t.column("label").to_pylist()
        idx = [i for i, l in enumerate(labs) if l == want]
        rng.shuffle(idx)
        for j, i in enumerate(idx[:40]):
            raw = rows[i]["bytes"] if isinstance(rows[i], dict) else rows[i]
            fp = tmpcf / f"{tag}_{j}.jpg"
            fp.write_bytes(bytes(raw))
            cf_paths.append(str(fp))
            cf_y.append(float(want))
    cf_y = np.array(cf_y)
    F_cf = feats_for(cf_paths)
    print("response features done (cf)")

    # fit tiny logistic meta on the dev response vectors (clean-only baseline vs full)
    from sklearn.linear_model import LogisticRegression

    def eval_meta(F_tr, y_tr, F_te, y_te, cols):
        m = LogisticRegression(max_iter=2000)
        m.fit(F_tr[:, cols], y_tr)
        return m.predict_proba(F_te[:, cols])[:, 1]

    clean_col = [0]
    full_col = list(range(F_dev.shape[1]))
    res = {}
    for name, F_te, y_te in (("eval", F_ev, ev_y), ("screenshots", F_ss, ss_y), ("cf-matched", F_cf, cf_y)):
        pc = eval_meta(F_dev, dev_y, F_te, y_te, clean_col)
        pf = eval_meta(F_dev, dev_y, F_te, y_te, full_col)
        # overlap zone: images whose CLEAN confidence sits in 0.3..0.8
        p_clean_te = F_te[:, 0]
        mask = (p_clean_te >= 0.3) & (p_clean_te <= 0.8)
        res[name] = {
            "auroc_clean_all": auroc(y_te, pc),
            "auroc_meta_all": auroc(y_te, pf),
            "n_overlap": int(mask.sum()),
            "auroc_clean_overlap": auroc(y_te[mask], pc[mask]),
            "auroc_meta_overlap": auroc(y_te[mask], pf[mask]),
        }
        print(f"--- {name} ---")
        for k, v in res[name].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # save artifacts: features + fitted meta (for the decision-layer export)
    np.savez("/tmp/spectro_features.npz", F_dev=F_dev, dev_y=dev_y,
             F_ev=F_ev, ev_y=ev_y, F_ss=F_ss, ss_y=ss_y, F_cf=F_cf, cf_y=cf_y)
    m_full = LogisticRegression(max_iter=2000).fit(F_dev[:, full_col], dev_y)
    import json as _json

    meta_export = {
        "weights": [float(w) for w in m_full.coef_[0]],
        "intercept": float(m_full.intercept_[0]),
        "n_features": int(F_dev.shape[1]),
    }
    Path("/tmp/spectro_meta.json").write_text(_json.dumps(meta_export, indent=1))
    print("saved /tmp/spectro_features.npz + /tmp/spectro_meta.json")

    # product decision metric: BAcc@0.65 clean vs meta on cf-matched
    for tag, F_te, y_te in (("clean", F_cf[:, [0]], cf_y), ("meta", F_cf, cf_y)):
        pp = (m_full.predict_proba(F_te[:, [0]])[:, 1] if tag == "clean"
              else m_full.predict_proba(F_te[:, full_col])[:, 1])
        pred = pp >= 0.65
        tpr = ((pred == 1) & (y_te == 1)).sum() / max(1, (y_te == 1).sum())
        tnr = ((pred == 0) & (y_te == 0)).sum() / max(1, (y_te == 0).sum())
        print(f"cf-matched BAcc@0.65 {tag}: {(tpr + tnr) / 2:.4f} (ai={tpr:.3f} real={tnr:.3f})")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
