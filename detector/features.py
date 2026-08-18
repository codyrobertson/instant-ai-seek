#!/usr/bin/env python3
"""Handcrafted forensic features for AI-image detection (Phase-1 baseline).

All features are deterministic: no RNG, no external state. They are chosen to
be cheap enough to run in a browser later (numpy-free ports are feasible), and
to capture the artifacts that separate camera/processed photos from modern
generative images:

  - sensor noise (AI images are denoised-smooth)
  - JPEG re-encode fingerprints (double compression, PNG-sourced AI images)
  - frequency spectrum shape
  - texture / sharpness statistics
  - color statistics (colorfulness, saturation, banding)
  - EXIF presence

Feature vector order is fixed by FEATURE_NAMES; the trained model and any
harness logic must use the same ordering.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

FEATURE_NAMES = [
    "noise_sigma",        # MAD-based sensor noise estimate (smoothest tiles)
    "ela_q90",            # mean abs diff on q=90 re-encode
    "blockiness",         # JPEG 8x8 grid boundary artifact ratio
    "roundtrip_q55",      # mean abs diff re-encode q=55
    "roundtrip_q75",      # mean abs diff re-encode q=75
    "roundtrip_q95",      # mean abs diff re-encode q=95
    "hf_energy_ratio",    # high-frequency energy fraction (FFT, 256px)
    "spectral_flatness",  # geomean/arithmean of power spectrum
    "grad_mean",          # mean gradient magnitude (256px)
    "sharp_edges",        # fraction of pixels with |grad| > 60
    "flatness",           # fraction of pixels in near-uniform neighborhoods
    "colorfulness",       # Hasler-Suesstrunk colorfulness
    "sat_mean",           # mean HSV saturation
    "sat_std",            # std of HSV saturation
    "unique_colors",      # unique colors in 128px image (fraction of px)
    "exif_present",       # 1 if EXIF data present
]

_Q_LIST = (55, 75, 95)


def _to_gray_256(im: Image.Image) -> np.ndarray:
    g = np.asarray(ImageOps.grayscale(im).resize((256, 256), Image.BILINEAR), dtype=np.float64)
    return g


def _reencode_diff(im: Image.Image, quality: int) -> float:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    re = Image.open(buf)
    re.load()
    a = np.asarray(im.convert("RGB"), dtype=np.float64)
    b = np.asarray(re.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        b = np.asarray(re.convert("RGB").resize(a.shape[1::-1]), dtype=np.float64)
    return float(np.abs(a - b).mean())


def _blockiness(im: Image.Image) -> float:
    """Ratio of across-block vs within-block pixel differences (8x8 JPEG grid)."""
    g = np.asarray(ImageOps.grayscale(im).resize((256, 256), Image.BILINEAR), dtype=np.float64)
    d = np.abs(np.diff(g, axis=1))  # horizontal neighbor diffs
    within_h = d[:, 1::8].mean()
    across_h = d[:, ::8].mean()
    dv = np.abs(np.diff(g, axis=0))  # vertical neighbor diffs
    within = (within_h + dv[1::8, :].mean()) / 2
    across = (across_h + dv[::8, :].mean()) / 2
    return float(across / (within + 1e-9))

def _noise_sigma(im: Image.Image) -> float:
    g = np.asarray(ImageOps.grayscale(im).resize((512, 512), Image.BILINEAR), dtype=np.float64)
    resid = g - ndimage.gaussian_filter(g, sigma=1.0)
    # 64x64 tiles from the 512x512 residual
    tiles = resid[:512, :512].reshape(8, 64, 8, 64).transpose(0, 2, 1, 3).reshape(64, 64 * 64)
    var = tiles.var(axis=1)
    smooth = np.sort(var)[:6]  # smoothest 10% of tiles
    return float(np.sqrt(smooth.mean()))


def _fft_features(g: np.ndarray) -> tuple[float, float]:
    f = np.fft.rfft2(g - g.mean())
    ps = np.abs(f) ** 2
    ps[0, 0] = 0.0
    total = ps.sum()
    h, w = ps.shape
    cy, cx = 0, 0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    hf = ps[r > 0.5 * r.max()].sum()
    hf_ratio = float(hf / (total + 1e-9))
    psnz = ps[ps > 0]
    flatness = float(np.exp(np.log(psnz).mean()) / (psnz.mean() + 1e-12))
    return hf_ratio, flatness


def _colorfulness(im: Image.Image) -> float:
    small = im.convert("RGB").resize((256, 256), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float64)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _saturation_stats(im: Image.Image) -> tuple[float, float]:
    hsv = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR).convert("HSV"), dtype=np.float64)
    s = hsv[..., 1] / 255.0
    return float(s.mean()), float(s.std())


def _unique_colors(im: Image.Image) -> float:
    small = im.convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = np.asarray(small).reshape(-1, 3)
    n = len(np.unique(arr, axis=0))
    return float(n / len(arr))


def _grad_stats(g: np.ndarray) -> tuple[float, float, float]:
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    sharp = float((mag > 60).mean())
    # local std via E[x^2]-E[x]^2 with uniform filters (fast, deterministic)
    mean = ndimage.uniform_filter(g, size=3, mode="nearest")
    mean_sq = ndimage.uniform_filter(g**2, size=3, mode="nearest")
    local_std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    flat = float((local_std < 1.0).mean())
    return float(mag.mean()), sharp, flat


def extract_features(path: str | Path) -> list[float]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        exif = 1.0 if (im.getexif() or im.info.get("exif")) else 0.0
        g256 = _to_gray_256(im)
        g512 = np.asarray(ImageOps.grayscale(im).resize((512, 512), Image.BILINEAR), dtype=np.float64)

        feats = [
            _noise_sigma(im),
            _reencode_diff(im, 90),
            _blockiness(im),
        ]
        for q in _Q_LIST:
            feats.append(_reencode_diff(im, q))
        hf, flat = _fft_features(g256)
        feats.append(hf)
        feats.append(flat)
        gm, sharp, flatpix = _grad_stats(g512)
        feats.extend([gm, sharp, flatpix])
        feats.append(_colorfulness(im))
        sm, ss = _saturation_stats(im)
        feats.extend([sm, ss])
        feats.append(_unique_colors(im))
        feats.append(exif)
        return feats


if __name__ == "__main__":
    import json
    import sys

    for p in sys.argv[1:]:
        f = extract_features(p)
        print(json.dumps(dict(zip(FEATURE_NAMES, f))))
