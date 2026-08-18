#!/usr/bin/env python3
"""Degradation-evidence features (advisor-reviewed spec).

Three hand-crafted spectral/statistical features computed on the IDENTICAL
decoded 384x384 center-crop that the detector consumes (both in Python and
in the extension's canvas path — same pixels in, same features out):

  1. blockiness  — JPEG 8x8-grid boundary-vs-interior gradient, PHASE-INVARIANT
                   over all 8 grid offsets, normalized by interior gradient.
  2. grain       — true two-axis Laplacian energy, normalized by total
                   gradient + variance (sensor-grain proxy, robust to scale).
  3. hf_ratio    — Hann-windowed radial power-spectrum tail (32..64 cyc/384)
                   vs low band, DC-excluded, correct FFT frequency coords.

Forbidden inputs: MIME/codec, EXIF, dimensions, file size, filename, source
identifiers. The features are computed on decoded pixels only.

Output: a 3-vector per image; plus the tiny logistic used to flag
"degraded/unverifiable" (frozen weights live in bench/degrade_meta.json).
"""
import numpy as np
from PIL import Image

SIZE = 384


def _luma384(img: Image.Image) -> np.ndarray:
    """Resize shortest side to 416, center-crop 384, return float luma."""
    w, h = img.size
    scale = 416 / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    left = (img.width - SIZE) // 2
    top = (img.height - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    a = np.asarray(img.convert("L"), dtype=np.float32)
    return a


def blockiness(g: np.ndarray) -> float:
    """Phase-invariant JPEG blockiness: mean over all 8 grid offsets of
    (edge gradient - interior gradient) at the 8px grid, normalized by the
    interior gradient so flat images don't dominate."""
    h, w = g.shape
    h, w = h - h % 8, w - w % 8
    g = g[:h, :w]
    vals = []
    for ox in range(8):
        for oy in range(8):
            gs = g[oy:, ox:]
            gh, gw = gs.shape
            gh, gw = gh - gh % 8, gw - gw % 8
            gs = gs[:gh, :gw]
            blocks = gs.reshape(gh // 8, 8, gw // 8, 8)
            # vertical edge (between columns 7 and 8 of each block)
            edge_v = np.abs(np.diff(blocks, axis=3)[:, :, :, 6]).mean()
            interior_v = np.abs(np.diff(blocks, axis=3)[:, :, :, :6]).mean()
            # horizontal edge
            edge_h = np.abs(np.diff(blocks, axis=1)[:, 6, :, :]).mean()
            interior_h = np.abs(np.diff(blocks, axis=1)[:, :6, :, :]).mean()
            edge = (edge_v + edge_h) / 2
            interior = (interior_v + interior_h) / 2
            vals.append(edge - interior)
    return float(np.mean(vals))


def grain(g: np.ndarray) -> float:
    """Two-axis Laplacian energy normalized by total gradient + variance."""
    dxx = np.abs(g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2])
    dyy = np.abs(g[2:, :] - 2 * g[1:-1, :] + g[:-2, :])
    lap = (dxx[:-2] + dyy[:, :-2]).mean()
    gx = np.abs(np.diff(g, axis=1))[1:-1, :-1]  # (H-2, W-2)
    gy = np.abs(np.diff(g, axis=0))[:-1, 1:-1]  # (H-2, W-2)
    grad = (gx + gy).mean() / 2
    var = g.var()
    return float(lap / (grad + var + 1e-6))


def hf_ratio(g: np.ndarray) -> float:
    """Hann-windowed radial power spectrum: high band (32..64 cyc) vs
    low band (1..16 cyc) at 384px, DC excluded."""
    win = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    gw = (g - g.mean()) * win
    f = np.fft.fftshift(np.fft.fft2(gw))
    ps = np.abs(f) ** 2
    ny = ps.shape[0] // 2
    yy, xx = np.mgrid[-ny:ny, -ny:ny]
    r = np.sqrt(xx**2 + yy**2).astype(int)
    radial = np.bincount(r.ravel(), weights=ps.ravel())[1:65] / np.bincount(r.ravel())[1:65]
    low = radial[1:16].mean()  # 2..16 cyc
    high = radial[32:64].mean()  # 33..64 cyc
    return float(high / (low + 1e-9))


def degrade_features(img: Image.Image) -> np.ndarray:
    g = _luma384(img)
    return np.array([blockiness(g), grain(g), hf_ratio(g)], dtype=np.float32)


def degrade_features_path(path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
    return degrade_features(im)


def logistic_p(weights: np.ndarray, intercept: float, feats: np.ndarray) -> float:
    """P(degraded) from the frozen 3-feature logistic."""
    z = float(intercept + np.dot(weights, feats))
    return 1.0 / (1.0 + np.exp(-z))
