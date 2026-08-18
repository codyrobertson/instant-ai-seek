# F01D SOTA Detector Implementation Plan

> **Required execution skill:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current teacher14/student v5 detector into a genuinely robust, browser-deployable detector that is competitive across unseen generators, scenes, real-image sources, and post-processing—not merely excellent on the 300-image local proxy.

**Architecture:** Train a large offline teacher with complementary semantic and forensic views: a full-image representation branch plus high/low-frequency local-patch experts, with optional generator-aware prototype learning. Fuse the views with a calibrated score rather than unconditional max-fusion, then distill the winning teacher into a small 384px/256px browser student and verify fp32, fp16/WebGPU, and q8/WASM as separate runtime products. Keep an explicit abstain/uncertain state in the product even if benchmark exports remain binary.

**Tech Stack:** PyTorch, torchvision, timm, ONNX Runtime, ONNX Runtime Web, Chrome MV3, Puppeteer, DGX Spark/CUDA, deterministic CSV/JSON manifests, and the existing `bench/`/`scripts/` harnesses.

---

## Definition of SOTA for this project

Do not define SOTA as “the highest `bench/evaluate.py` score.” The current baseline already scores `0.9933` on that proxy while reaching only `95.5%` real recall on 2,000 independent real photos and `50.17%` real recall on the sampled CF-Eval set.

The promotion objective is a constrained scorecard:

1. **Generalization:** macro-average performance across GenImage, Community Forensics, Chameleon/AIDE, AIGIBench, Synthbuster+, CF-Eval, and the existing local holdouts.
2. **Specificity:** target ≤1% false-AI rate on a large, source-stratified in-the-wild real set at the production threshold. Report the 95% confidence interval, not only the point estimate.
3. **Unknown-generator recall:** target ≥95% overall and ≥90% in every sufficiently sampled generator family on sealed holdouts.
4. **Hard-real balance:** target at least +10 percentage points CF-Eval balanced accuracy over the current sampled baseline, with a stretch target of ≥80% matched-real recall while retaining ≥95% fake recall.
5. **Robustness:** no more than a 5-point drop under JPEG/WebP recompression, resize, blur, screenshot/UI embedding, crop, and camera-like noise.
6. **Deployment:** identical model contract across Python, WebGPU/fp16, and WASM/q8; WebGPU p95 under 45 ms/image for the shipped student and WASM under 200 ms/image on a modern laptop.

If a detector cannot meet a binary decision safely on the matched-real corner, the product should abstain rather than pretend that a 0.99 score is a calibrated probability.

## Research constraints

- Never train on public benchmark test files, `data/holdout_oos`, Chameleon test data, AIGIBench test data, or any sealed project holdout.
- Split by generator identity, prompt/scene identity, near-duplicate group, real-image source, and transformation chain. A random image split is not sufficient.
- Keep training, calibration, development, and sealed test manifests separate and hash-pinned.
- Every comparison uses the same preprocessing, threshold-selection policy, and metric code.
- One hypothesis per run family. Record failed and reverted runs in `training-journal.md` rather than silently replacing them.
- Use public datasets only within their licenses. Do not copy multi-million-image datasets into git; commit manifests, provenance, hashes, and reproducible acquisition instructions.

## Current baseline packet

- Commit: `cd242fe0`, teacher14 + student v5, 384px.
- Local proxy: balanced accuracy `0.9933`, AUROC `1.0000`, Brier `0.005252`, ECE10 `0.009861`.
- `data/holdout2`: `320/320` AI images detected.
- `data/probe/real`: `90/2000` false AI (`95.5%` real recall).
- Sampled public results: Synthbuster+ AI `95.0%`; CF-Eval AI `97.75%`, real `50.17%`, balanced `73.96%`; CF-Commercial AI `99.75%`.
- Known release defect: `extension/offscreen.js` sends 384×384 tensors while `detector/model_cnn_q.onnx` and `detector/model_cnn.json` still describe 256×256.

---

### Task 1: Freeze the benchmark contract and baseline scorecard

**Files:**
- Create: `bench/scorecard.py`
- Create: `bench/calibrate.py`
- Create: `data/manifests/benchmark_registry.json`
- Modify: `bench/evaluate.py`
- Modify: `scripts/run_probe.py`
- Modify: `scripts/bench_pub.py`
- Modify: `training-journal.md`

**Step 1: Write the benchmark registry.**

Record, for every suite and split: source URL or acquisition command, license, SHA-256 manifest, label semantics, generator family, scene/prompt group, real-image source, compression/transformation tags, and whether the split is train/dev/calibration/sealed-test.

**Step 2: Add scorecard metrics.**

`bench/scorecard.py` must emit JSON and a compact table containing:

- balanced accuracy, AUROC, AUPRC;
- AI recall and real recall;
- false-AI rate at the selected threshold;
- AI recall at fixed real FPR targets of 0.5%, 1%, and 2%;
- Brier score and 10-bin ECE;
- per-generator and per-real-source results;
- bootstrap 95% confidence intervals.

**Step 3: Separate calibration from test.**

Use only a development/calibration split to select the threshold or temperature. The sealed test score must be produced once per candidate and must not influence threshold selection.

**Step 4: Run the frozen baseline.**

Run:

```bash
python3 bench/evaluate.py
python3 scripts/run_probe.py
python3 scripts/bench_pub.py --per-class 400
python3 bench/scorecard.py --model current --registry data/manifests/benchmark_registry.json
```

Expected result: a committed baseline packet with model hashes, exact sample counts, per-source results, and no “SOTA” claim yet.

**Step 5: Commit the measurement layer.**

```bash
git add bench data/manifests/benchmark_registry.json
git commit -m "test: freeze detector SOTA scorecard"
```

---

### Task 2: Repair and lock the model/export/runtime contract

**Files:**
- Create: `scripts/verify_model_bundle.py`
- Modify: `detector/model_cnn.json`
- Modify: `detector/detect.py`
- Modify: `detector/train_cnn.py`
- Modify: `scripts/build_extension.sh`
- Modify: `extension/offscreen.js`
- Modify: `scripts/e2e_extension_test.js`
- Create: `scripts/model_bundle_manifest.json`

**Step 1: Make metadata authoritative.**

Put `input_size`, `resize_to`, mean/std, backbone, student/teacher IDs, output semantics, quantization type, and SHA-256 hashes in one manifest. Python and JavaScript must read or be generated from the same values; remove the stale hard-coded 256-vs-384 split.

**Step 2: Regenerate every artifact at the same input size.**

Export fp32, fp16, student q8, and teacher q8 from the same checkpoint and assert that all four graphs accept the same `[N, 3, H, W]` shape.

**Step 3: Add static bundle checks.**

`verify_model_bundle.py --strict` must fail on any of:

- input-size mismatch;
- missing or stale metadata;
- missing model hash;
- fp32/fp16/q8 output-shape mismatch;
- preprocessing resize/crop mismatch;
- q8 graph unsupported by the selected ORT Web backend.

**Step 4: Add forced-backend E2E.**

Extend `scripts/e2e_extension_test.js` with `--ep=webgpu` and `--ep=wasm`. Run both against at least 24 eval images plus a small hard-real/holdout packet. Require ≥98% classification agreement and mean score delta <0.05; record max delta, p50, p95, and errors by backend.

**Step 5: Commit the contract fix separately.**

```bash
python3 scripts/verify_model_bundle.py --strict
node scripts/e2e_extension_test.js --ep=webgpu --images=12
node scripts/e2e_extension_test.js --ep=wasm --images=12
git add detector scripts extension
git commit -m "fix: make 384px model bundle backend-consistent"
```

No training result is promotion-eligible until both backends pass.

---

### Task 3: Build a provenance-safe, hard-real training corpus

**Files:**
- Create: `scripts/build_forensic_manifest.py`
- Create: `data/manifests/forensic_registry.schema.json`
- Create: `scripts/validate_forensic_data.py`
- Modify: `detector/train_cnn.py`
- Create: `detector/data_manifest.py`
- Create: `data/manifests/train_forensic.csv`
- Create: `data/manifests/dev_forensic.csv`
- Create: `data/manifests/sealed_forensic.csv`

**Step 1: Define real-image strata.**

Build explicit quotas for:

- ordinary camera photos and smartphone recompressions;
- web/social-media JPEGs and screenshots;
- FFHQ-like portraits;
- LAION/Community-Forensics matched source reals;
- scene-matched real counterparts for generated images;
- illustrations/art and UI-embedded images.

The real set must be much larger than the current 2,000-photo probe and must preserve source identity in the manifest.

**Step 2: Define fake-image strata by family, not product name.**

Cover latent diffusion, pixel diffusion, GAN/StyleGAN/DFGAN, autoregressive/transformer generators, commercial/API generators, image-to-image/inpainting, and post-processed or screenshot-embedded outputs. Keep prompt IDs and generator checkpoints where known.

**Step 3: Add matched-pair construction.**

For each scene/prompt bucket, pair generated images with real source images without making the pair itself a training leakage path. Preserve an unseen generator and unseen scene split for every family.

**Step 4: Validate data before training.**

`validate_forensic_data.py` must reject corrupt files, tiny/rate-limited downloads, duplicate hashes, cross-split perceptual duplicates, missing provenance, and accidental inclusion of any sealed test path.

Run:

```bash
python3 scripts/validate_forensic_data.py --manifest data/manifests/train_forensic.csv
python3 scripts/validate_forensic_data.py --manifest data/manifests/dev_forensic.csv
python3 scripts/validate_forensic_data.py --manifest data/manifests/sealed_forensic.csv --sealed
```

Expected result: source counts by family, duplicate counts, transformation counts, and a machine-checkable “no leakage” verdict.

**Step 5: Start with a curated 100k–200k image training set.**

Do not immediately ingest millions of mostly redundant diffusion samples. Increase generator-family diversity first; scale volume only when the scorecard shows a positive marginal gain.

---

### Task 4: Establish the teacher model tournament

**Files:**
- Create: `detector/model_hybrid.py`
- Create: `detector/train_hybrid.py`
- Create: `detector/export_hybrid.py`
- Create: `bench/ablation.py`
- Modify: `detector/train_cnn.py`
- Modify: `training-journal.md`

Run one controlled baseline and three candidate families:

1. **T0 — Current control:** ConvNeXt Tiny 384px, current augmentation and max-fusion.
2. **T1 — AIDE-style multi-view teacher:** a semantic foundation branch plus high-frequency and low-frequency local patch experts. Use full-image features for context and patch experts for noise, aliasing, resampling, and texture artifacts.
3. **T2 — Semantic-artifact-resistant teacher:** patch-shuffle/scene-randomization training so the classifier cannot win by memorizing scene semantics or dataset identity.
4. **T3 — Generator-aware representation teacher:** add a compact prototype/contrastive head over the forensic representation, with optional LoRA adaptation of the foundation encoder. Keep the head auxiliary at first; do not make generator classification the product output.

For each candidate:

- train seeds 0, 1, and 2;
- keep the same data, split, budget, and calibration policy;
- compare the full scorecard, not only local balanced accuracy;
- run a single-variable ablation for patch views, prototype loss, hard-negative weighting, and augmentation policy;
- retain the best checkpoint by dev macro score subject to the real-FPR gate.

Run on DGX Spark with an explicit log path, for example:

```bash
python3 detector/train_hybrid.py \
  --config configs/teacher_t1_384.yaml \
  --seed 0 \
  --output runs/teacher_t1_seed0
```

Expected result: a tournament table with val/test AUC, Brier, ECE, per-family AI recall, per-source real recall, runtime, and model size. A candidate that improves fake recall while worsening real FPR is a regression, not a win.

---

### Task 5: Add hard-example mining without contaminating the test

**Files:**
- Create: `scripts/mine_hard_examples.py`
- Create: `data/manifests/hard_negative_registry.csv`
- Modify: `detector/data_manifest.py`
- Modify: `detector/train_hybrid.py`
- Modify: `training-journal.md`

**Step 1: Mine only train/dev pools.**

Collect the highest-confidence real-as-AI images and lowest-confidence AI images from a large development pool. Never mine from Chameleon, AIGIBench test, Community Forensics Eval, or the sealed holdout.

**Step 2: Adjudicate the hard set.**

Assign each hard example a reason: JPEG/storage artifact, matched semantic pair, screenshot/UI, heavy crop, camera noise, generator family, ambiguity, or bad label. Remove corrupt and genuinely ambiguous samples from the binary training target or label them as abstention candidates.

**Step 3: Train a hard-negative round.**

Add capped per-source hard-negative weight, not unrestricted oversampling. The first round should cap any source at 2× its ordinary contribution so one failure mode cannot collapse the classifier as COCO did in the prior experiment.

**Step 4: Re-score the untouched sealed sets.**

Require improvement in real recall and no more than 2 points loss in unknown-generator fake recall before keeping the round.

---

### Task 6: Calibrate the decision policy and add abstention

**Files:**
- Modify: `bench/calibrate.py`
- Modify: `bench/scorecard.py`
- Modify: `detector/detect.py`
- Modify: `extension/offscreen.js`
- Modify: `extension/popup.js`
- Modify: `extension/content.js`

**Step 1: Fit calibration only on a held-out calibration split.**

Compare temperature scaling, isotonic regression, and a two-threshold policy. Keep the calibrated threshold fixed before sealed evaluation.

**Step 2: Define three product states.**

- `AI likely`: score above the high threshold and stable across views;
- `likely real`: score below the low threshold and stable across views;
- `uncertain`: disagreement between views, low margin, or known hard-real ambiguity.

The benchmark can still expose the raw continuous score and binary result, but the extension should not blur uncertain cases as if they were certain.

**Step 3: Test calibration under shift.**

Report reliability diagrams, ECE, Brier, and false-AI rate separately for camera photos, social JPEGs, matched reals, screenshots, and each generator family. A low ECE on `data/eval` alone is not sufficient.

---

### Task 7: Distill the winning teacher into a browser student

**Files:**
- Modify: `detector/train_cnn.py`
- Create: `detector/distill_multiview.py`
- Modify: `detector/export_hybrid.py`
- Modify: `scripts/build_extension.sh`
- Modify: `scripts/verify_model_bundle.py`
- Modify: `scripts/e2e_extension_test.js`

Distill the winning teacher using:

- calibrated teacher logits;
- intermediate representation/moment matching;
- patch/view consistency loss;
- hard-negative examples;
- explicit real-FPR-aware validation.

Train and compare 256px and 384px students. Keep 384px if the product gate requires it; keep 256px only if it matches the teacher on the scorecard and survives hard-real tests.

Required student artifacts:

```text
student_fp32.onnx
student_fp16.onnx
student_q8.onnx
teacher_fp16.onnx
teacher_q8.onnx
model_manifest.json
```

All artifacts must share the same preprocessing contract. Verify Python, WebGPU, and WASM parity on clean, compressed, hard-real, and unseen-generator packets—not only six easy eval images.

---

### Task 8: Run the final SOTA tournament and promotion gate

**Files:**
- Modify: `bench/scorecard.py`
- Modify: `training-journal.md`
- Modify: `README.md`
- Create: `reports/sota-detector-YYYYMMDD.md`

**Step 1: Score every surviving teacher/student pair.**

Produce one immutable report with model hashes, data-registry hash, split hashes, commands, metrics, confidence intervals, calibration plots, and runtime measurements.

**Step 2: Apply the hard gates.**

Reject the model if any of the following is true:

- leakage or split provenance is unresolved;
- real FPR exceeds the product budget on the large in-the-wild set;
- any required backend is unverified or shape-incompatible;
- unknown-generator recall is driven by a single generator or prompt family;
- the candidate improves a proxy while regressing the hard-real scorecard;
- the student drops materially from the teacher after quantization/distillation.

**Step 3: Compare to external baselines.**

Run reproducible open-source baselines where licenses and compute permit. Report rank and dataset-specific performance; do not claim absolute SOTA from a private benchmark.

**Step 4: Promote only one frozen bundle.**

Freeze model IDs, hashes, metadata, threshold/calibration parameters, and extension artifacts. Update README claims to match measured evidence. Append the final selection entry to `training-journal.md` with val/test AUC, Brier, ECE, OOD deltas, runtime, and unresolved risks.

---

## Execution order and stop conditions

1. **Unblock deployment contract first:** repair 384/q8/metadata and force both E2E backends.
2. **Freeze scorecard second:** no training run is comparable until the benchmark registry and calibration policy are fixed.
3. **Build hard-real data third:** the next gain is more likely to come from source-matched real data and semantic-artifact resistance than from another small batch of familiar generators.
4. **Run teacher ablations fourth:** T0 → T1 → T2 → T3, with three seeds and journal writeback after each family.
5. **Mine hard examples fifth:** only from train/dev, then re-run sealed tests.
6. **Distill and ship last:** the browser student is a deployment target, not the experiment-selection oracle.

Stop and reframe the product as “AI-likelihood with abstention” if the large, independently sourced real set cannot meet the false-positive budget without unacceptable unknown-generator recall. That is a more credible product than a binary detector with saturated confidence and hidden hard-real failures.

## Research references

- [AIGIBench / NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/fb693c67f61e5321746ffce8b6fdd2d0-Abstract-Datasets_and_Benchmarks_Track.html): evaluates multi-source generalization, degradation, augmentation, and test-time preprocessing, and reports large real-world drops.
- [Chameleon and AIDE / ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0303773962ea1b5394c3a83cc7dd066-Abstract-Conference.html): motivates semantic plus high/low-frequency patch experts for hard unseen images.
- [Community Forensics / CVPR 2025](https://arxiv.org/abs/2411.04125): supports generator diversity and provenance-rich training, including thousands of generator models.
- [Breaking Semantic Artifacts / NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html): motivates patch-shuffle and cross-scene robustness against semantic shortcuts.
- [Generator-Aware Prototypes / 2025 preprint](https://arxiv.org/abs/2512.12982): motivates testing prototype learning and warns that indiscriminate source aggregation can eventually hurt.
- [Open-source detector benchmark / 2026 preprint](https://arxiv.org/abs/2602.07814): motivates reporting dataset-specific rankings instead of claiming a universal winner.

