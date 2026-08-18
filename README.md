# Instant AI Seek — Local AI-Image Detector for Chrome

A Manifest V3 Chrome extension that detects AI-generated images **entirely
inside the browser**. Image pixels never leave your device: no cloud
inference, no external APIs, no local server. Inference runs an ONNX model
via WebGPU (fp16) with a WASM (q8) fallback, in an extension offscreen
document.

## How it works

1. A content script scans every `<img>` on the page (≥64px, ignoring
   sprites/spinners/data-URI icons).
2. Each image is downscaled to ≤768px and handed to the extension's own
   offscreen document.
3. The offscreen document runs the model: resize to 288px shortest side →
   center-crop 256px → ImageNet-normalize → ONNX inference → softmax →
   P(AI). WebGPU (fp16) where available, WASM (uint8) fallback otherwise.
4. A badge overlay is drawn on each image: `AI 87%` (red) or `real 12%`
   (green). AI = P(AI) ≥ 65%. AI images are blurred until hovered.

Privacy: all processing happens in extension-owned contexts; nothing is
fetched from or sent to a network after install.

## Performance

- **~8 ms/image** on WebGPU (fp16, Apple M2 Max, 256px input, p95 8.7 ms) —
  roughly 11× faster than a 90 ms baseline, ~125 images/sec.
- Ensemble (student + teacher): ~45 ms/image on WebGPU.
- WASM fallback: ~100-200 ms/image on a modern laptop, fully offline.

## Models

- **Student (shipped, default):** resnet18 distilled from the teacher
  (Hinton KD, T=4), fp16 for WebGPU (22 MB) + uint8 q8 for WASM (11 MB).
- **Teacher (accuracy mode):** convnext_tiny fine-tuned on real + AI images
  across ~10 generator families (flux/schnell, SDXL, SD3.5, Flux-Dev,
  Ideogram, Recraft, Midjourney, DALL-E, StyleGAN3, CIFAKE/SD1.4,
  Community-Forensics GANs/manipulations) — q8 28 MB.
- Both are **bundled in the extension package** — fully offline from install.

## Results (balanced accuracy @ 0.65 confidence)

| Benchmark | Student (resnet18) | Teacher (convnext_tiny) | **Ensemble (max-fusion)** |
|---|---|---|---|
| Harness eval set (committed, 300 imgs, sealed) | 0.9833 | 0.9833 | **0.9933** (ai 0.987, real 1.0) |
| Sealed OOS holdout (SD + MJ, never trained) | 97.3% | 99.6% | **99.8%** |
| holdout2 (8 unseen generator families) | — | — | **100%** (320/320) |
| In-the-wild reals (2000 photos) | — | — | false-AI **4.55%** [3.65–5.40] |
| CF-Commercial (test) | — | — | **99.8%** |
| Synthbuster+ (DALL-E 2/3) | — | — | **95.7%** |
| CF-Eval DFGAN/real cell (hardest) | 33% | — | **100%** |
| CF-Eval DFGAN/fake + Hourglass/real | — | — | 97% / 87% |

Full scorecard with confidence intervals, calibration-shift report and the
model tournament: `reports/sota-detector-20260815.md` (regenerate with
`bench/scorecard.py`, `bench/shift_report.py`).

Ensemble = max(P_AI(student), P_AI(teacher)) — "any alarm is an alarm",
which lifted the GAN cells without hurting the real side. Public benchmark
evaluation: `python3 scripts/bench_pub.py` (parquet shards of Synthbuster+,
CommunityForensics-Eval and CF-Commercial must be downloaded first — see
the script header).


## Build from source (fully reproducible)

Prerequisites: macOS/Linux, Python ≥3.10 with `pip`, Node.js ≥18 (for one
`npm pack` during the build), and an internet connection **only during the
build** (vendors the WASM runtime; the extension itself is offline).

```bash
# 1. train the detector (optional — trained models are committed under
#    detector/, so this is only needed to reproduce training):
#    pip install torch torchvision timm onnx onnxruntime pillow scikit-learn
#    bash scripts/fetch_real.sh
#    python3 detector/train_cnn.py --model convnext_tiny.in12k_ft_in1k --epochs 26
#    python3 detector/train_cnn.py --model resnet18 --epochs 30 \
#      --teacher-state detector/best_state.pt --distill-lambda 0.6 --distill-temp 4.0

# 2. build the extension (vendors onnxruntime-web, copies models, icons):
bash scripts/build_extension.sh

# 3. verify (optional — needs puppeteer):
npm install puppeteer
node scripts/e2e_extension_test.js --images=150

# 4. load in Chrome:
#    chrome://extensions -> Developer mode -> "Load unpacked" ->
#    select the extension/ directory.
```

## Benchmark (reproduce the reported score)

The harness measures the same ONNX model the extension ships, at the same
65% confidence threshold, on the committed held-out eval set
(150 real + 150 AI, 512px, never used for training):

```bash
bash autoresearch.sh
# METRIC balanced_accuracy=0.9800
# METRIC ai_recall=0.9733
# METRIC real_recall=0.9867
# METRIC auroc=0.9986
```

`data/` contains the benchmark splits and their provenance manifests
(`data/manifests/`). The eval set is fixed and committed; the harness is
deterministic and runs offline.

## Layout

```
extension/        MV3 extension (load this folder in Chrome)
detector/         models + training/inference code (Python, torch/ONNX)
bench/            benchmark harness (balanced accuracy @ 0.65 threshold)
scripts/          dataset builders, public-benchmark eval, build script
data/             committed benchmark + training data (eval split is held out)
autoresearch.sh   benchmark entrypoint
training-journal.md  full run history and methodology
LICENSE           MIT
```

## License

MIT — see LICENSE.
