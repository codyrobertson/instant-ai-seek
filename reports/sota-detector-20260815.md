# SOTA Detector Report — 2026-08-15

## Frozen bundle

- **Commit**: `9ca5815c` + measurement layer (`bfeb50dc`, `799a4745`) — model files
  hash-pinned in `detector/model_bundle_manifest.json`.
- **Models**: student `model_cnn.onnx` (resnet18 @384) + teacher
  `model_cnn_teacher.onnx` (convnext_tiny @384), fused by **max-fusion**
  at threshold 0.65. fp16/q8 variants verified byte-hash consistent
  (`scripts/verify_model_bundle.py --strict` passes).
- **Preprocessing contract**: resize shortest side 416 → center-crop 384 →
  ImageNet normalize (authoritative manifest; Python and extension read it).
- **Data registry**: `data/manifests/benchmark_registry.json` +
  `forensic_registry.json` (train 78,725 / dev 19,681 / sealed 1,069;
  dedup 835, no corruption, no sealed-path contamination — validator PASS).

## Scorecard (threshold 0.65, isotonic-calibrated confidence)

| Suite | Metric | Value | 95% CI |
|---|---|---|---|
| Harness eval (300, sealed) | balanced acc | **0.9933** | [0.9833, 1.0000] |
|  | ai_recall / real_recall | 0.9867 / 1.0000 | |
|  | auroc / brier / ece10 | 1.0000 / 0.0053 / 0.0099 | |
| holdout2 (320 AI, 8 families, sealed) | ai recall | 320/320 (1.000) | |
| probe/real (2000 in-the-wild, sealed) | false-AI rate | **0.0455** | [0.0365, 0.0540] |
| Sealed OOS (1069 SD+MJ, never trained) | ai recall | 0.9981 | |
| Synthbuster+ (600) | ai recall | 0.9567 | |
| CF-Eval (41 shards) | ai / real aggregate | 0.9850 / 0.3717 | |
|  — DFGAN/real cell (hardest) | real recall | **1.0000** (was 0.33 @256) | |
|  — DFGAN/fake, Hourglass/real | 0.97 / 0.87 | | |
| CF-Commercial test (600) | ai recall | 0.9983 | |

External baseline: CLIP ViT-B/32 zero-shot (naive softmax@100) = 0.5133 bal
on the harness eval — our bundle is +48pp on the same images. No SOTA claim
beyond the measured suites; dataset-specific ranks only.

## Calibration under shift (bench/shift_report.py)

| Stratum | Brier | ECE10 | false-AI |
|---|---|---|---|
| eval (clean) | 0.0053 | 0.0099 | 0.000 |
| in-the-wild reals | 0.0493 | 0.0843 | 0.0455 |
| CF matched reals | 0.467 | 0.507 | 0.484 |
| Synthbuster+ fakes | 0.0365 | 0.0510 | 0.000 |

Degradation is concentrated in the matched-real cells — the product responds
with the **abstention state** (Task 6): scores in [0.35, 0.65) are shown as
uncertain and never blurred (the matched-real mean confidence 0.507 sits in
that band).

## Tournament (Task 4) — T0 retained

| Family | dev val_bal | sealed harness | verdict |
|---|---|---|---|
| T0 convnext_tiny @384 (teacher14) | 0.9828 | 0.9933 | **retained** |
| T1 multi-view (full+hp+lp experts) | 0.9310 | 0.8400 | loss |
| T2 + patch-shuffle | 0.9477 | 0.8600 | loss |
| Hard-negative round (t15) | 0.9003 | sanity-ai 0.78 | loss (COCO lesson) |
| Feature-KD student (v6) | 0.9690 | 0.9833 | loss (v5 kept) |

## Deployment gates (Task 2/7/8)

- Both forced backends: **E2E PASS** — wasm: 48/48 eval + 71/72 incl.
  hard-real (98.6%, mean delta 0.008); webgpu: same parity.
- Model bundle: 6 artifacts @384, hashes match, output shapes consistent.
- Runtime (measured this session): WASM q8 ensemble p50 1.1s / p95 1.4s
  (headless, this machine — offline fallback); WebGPU fp16 student ~8 ms
  @256 measured previously (M2 Max) ≈ 18 ms @384 by kernel scaling; the
  fp16 teacher adds ~50 ms. WebGPU is the primary path.
- Abstention shipped: uncertain state amber `?` badge, no blur.
- Leakage: registry + validator PASS; every sealed suite untouched by
  training (train_cnn guards + forensic manifests).

## Remaining risks

- In-the-wild false-AI 4.55% vs the plan's ≤1% budget — mitigated by
  abstention framing, not solved; the boundary is the fuzzy matched-real
  cell (3+ recorded negative attempts).
- WASM ensemble latency exceeds the 200 ms target at 384; acceptable as an
  offline fallback, documented.
- CF-Eval aggregate real recall 37% — concentrated in matched-pair cells.
