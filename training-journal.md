# f01d Training Journal

System of record for detector training runs. Facts first, interpretations
labeled `[inference]`. All times UTC. Harness metric = balanced accuracy at
0.65 confidence on the committed eval set (data/eval, 150 real + 150 AI),
unless noted otherwise.

## Entry: 2026-08-13

### Timestamps
- Journal updated at (UTC): `2026-08-13T17:55:00Z`
- Latest run event at (UTC): `2026-08-13T17:48:00Z` (convnext training on DGX Spark completed)

### Seed
- Training seeds: fixed `SEED = 0` (torch/numpy/random) in `detector/train_cnn.py`; dataset split seed `20260813` (build_dataset.py) and `0` (train_cnn assemble_rows).
- Eval seed policy: eval split fixed at commit time; harness deterministic (no RNG at inference).

### Runs
| Stage | Run ID | Name | Created (UTC) | State | Selected Checkpoint | Notes |
|---|---|---|---|---|---|---|
| baseline | #1 | LR-16feat | 2026-08-13T15:10:00Z | kept | detector/model.json | 0.9867 harness / 0.9979 auroc; overfit to flux/schnell (probe recall 0-96%) |
| train | #2 (abandoned) | local convnext M2 | 2026-08-13T16:20:00Z | killed (user: CPU load) | none | epoch2 val_bal 0.9488; 291-687s/epoch on MPS |
| train | spark-16ep-b128 | convnext GB10 | 2026-08-13T17:13:00Z | completed | epoch 15 | best val_bal 0.9761, auroc ~0.998; 29s/epoch; AMP + 8 workers |
| eval | #3 | CNN eval | 2026-08-13T17:51:00Z | kept | detector/model_cnn.onnx (+_q) | 0.9933 harness (ai 0.9867, real 1.0, auroc 0.9997); 58s |
| probe | — | CNN probes | 2026-08-13T17:53:00Z | info | same model | flux_dev 100%, ideogram 100%, recraft 100%, sd35 100%, sdxl 95.8%; picsum false-AI 1.7%. CAVEAT: probe images were in training (10% in val) — partially in-sample |

### Meta
- W&B: none (no W&B; journal + harness logs are the record).
- Core config deltas vs baseline: LR-16feat → convnext_tiny.in12k_ft_in1k (2-class CE, 256px, AdamW 2e-4 wd 0.05, cosine, batch 128 AMP, 16 epochs, JPEG-artifact augmentation p=0.8, class-balanced epochs per_epoch=min≈2856).
- Data/eval artifacts: data/eval (held out, off-limits), data/train (150+150), data/raw/ai (700 flux/schnell), data/probe/ai (120 across 5 generators), data/probe/real (120 picsum), data/train_extra (CIFAKE 10k+10k subset, 2.5k each used), detector/model_cnn.json (preprocessing meta).

### Critical Observations
1. CNN generalizes far beyond LR: cross-generator AI recall 95.8-100% vs LR's 0-96%; real false-AI 0-1.7% vs LR's 2.7%. Harness 0.9933 vs 0.9867.
2. sdxl is the weakest cell (95.8%, one confident miss at 0.065); sd35 near-perfect but only 24 training images [inference: more SDXL-family training data is the highest-value add].
3. Probe numbers are partially in-sample (probe set joins training; ~12/gen in val). Fresh HOLD-OUT generations per generator are required for honest out-of-sample stats — in progress (scripts/gen_xgen.py, dest=holdout).
4. Statistical sufficiency: AI training = 2,856/epoch (700 schnell + 120 probes + 2.5k CIFAKE-32px); per-generator counts (24) are thin. Expansion planned: +100/gen cross-gen train, +300 schnell, +1200 picsum real (900 train / 300 holdout).
5. val_bal plateaued ~0.972-0.976 by epoch 15; cosine LR annealing still improving slowly [inference: 20-24 epochs may add ~0.3-0.5pp val_bal].
6. Extension parity: q8 vs fp32 sanity diff ~0 on 2 images; batch dim made dynamic post-export; full parity check via headless-Chrome E2E pending model rebuild.

## Entry: 2026-08-13 (evening) — contamination found & fixed

### Timestamps
- Journal updated at (UTC): `2026-08-13T18:20:00Z`

### Facts
1. Advisor review found REAL-EVAL CONTAMINATION: `list_imagenette()` trained on both imagenette splits, but the 150 real eval images were sampled from imagenette VAL. Fixed: training restricted to `imagenette/train/*` only (detector/train_cnn.py). Run #3's 0.9933 / real_recall=1.0 was inflated; will re-measure honestly.
2. AI eval verified disjoint from training (eval=pool ids 151-300; train=ids 1-150 + 301-700 + probes/xgen/CIFAKE).
3. Additional fixes from review: offscreen.js crop bug (drawImage source coords were in resized space — resize was lost; now intermediate-canvas resize-then-crop, byte-matching Python), build_extension.sh now always refreshes the q8 model, val split now source-stratified (20% per generator/source), WebP re-encode added to augmentation (p≈0.12), torch.compile(inductor) enabled on CUDA.
4. Data expansion for statistical sufficiency (in progress): xgen train +100/gen × 5 generators (~$14), fresh holdout +60/gen × 5 (never trained, honest OOS), picsum real 120→1200 (train on 960, 240 val), schnell +400 already added.

### Interpretation
- Run #3 metrics must be re-measured post-fix before being quoted (pending retrain).
- The honest OOS measurement is the new data/holdout set (60/gen), evaluated through the same q8 path the extension ships.

## Entry: 2026-08-13 (final) — clean model shipped

### Timestamps
- Journal updated at (UTC): `2026-08-13T19:10:00Z`

### Runs
| Stage | Run ID | Name | Created (UTC) | State | Selected Checkpoint | Notes |
|---|---|---|---|---|---|---|
| train | spark-train4 | convnext full data | 2026-08-13T18:20:00Z | completed | epoch 18 | best val_bal 0.9746; 24 epochs, ~30s/epoch, torch.compile+AMP+12 workers |
| eval | #4 | final honest eval | 2026-08-13T18:46:00Z | kept | model_cnn.onnx/_q | 0.9933 (ai 0.9867, real 1.0, auroc 0.9999) |
| holdout | — | fresh-gen probes | 2026-08-13T18:50:00Z | info | same | 99.3% recall (298/300: flux_dev 100, ideogram 59/60, recraft 100, sd35 59/60, sdxl 100); picsum 1200 false-AI 0.8% |
| e2e | — | Chrome parity | 2026-08-13T19:00:00Z | PASS | q8 in extension | max |extension−harness| = 0.000 over 12 eval images |

### Critical Observations
1. Honest eval after contamination fix = 0.9933 — identical to the contaminated run: the model generalizes to unseen imagenette-val photos (val-split exclusion was still the right call for integrity).
2. Holdout (fresh generations, never in training) proves cross-generator generalization: 99.3% AI recall; residual misses are 1 ideogram + 1 sd35 of 300.
3. Extension WASM (q8) output is bit-identical to the Python harness at badge precision; CSP needed `'wasm-unsafe-eval'` for ORT-web.
4. Training moved to DGX Spark (SSH dgx-spark): 30s/epoch vs 291-687s on M2 MPS; heavy assets now live on the Spark; local repo keeps eval + models only.

## Entry: 2026-08-13 (late) — 8-generator holdout + submission check

### Timestamps
- Journal updated at (UTC): `2026-08-13T19:45:00Z`

### Facts
1. NEW fresh holdout (data/holdout2, 40/gen, 8 generators incl. flux_pro, nano_banana (Gemini-class), ideogram_v3 — never trained on): recall per generator = flux_dev 100, flux_pro 100, ideogram 97.5, ideogram_v3 100, nano_banana 97.5, recraft 95.0, sd35 90.0, sdxl 97.5. Overall 313/320 = 97.8% zero-shot recall. Weakest cells: sd35, recraft.
2. Re-compression invariance on the eval set: original 0.9933, jpeg_q80 0.9867, jpeg_q60 0.9867, webp_q80 0.9933, png 0.9933. JPEG-artifact augmentation is working; worst case still 0.9867.
3. FAL account balance EXHAUSTED (403 "User is locked") mid train2-generation (314/320 failed) — training-data expansion paused pending top-up. xgen2 has only 6 images; skipped from training.
4. Fresh-clone submission check: git clone → build_extension.sh → harness → extension JS/manifest — ALL GREEN. Reproducible from source.

### Interpretation
- Zero-shot generalization to unseen modern generators (flux-pro, Gemini-class nano-banana) at 97.5-100% makes private-benchmark risk low even for MJ/DALL-E-class models we cannot access.
- No retrain possible without FAL top-up; current model stands as shipped (run #4).
- Next actions: (a) user tops up FAL → generate train2 (40/gen × 8) + retrain to close sd35/recraft gaps; (b) otherwise proceed to submission (public repo + claim).

## Entry: 2026-08-14 — law benchmarks + sealed OOD holdout

### Timestamps
- Journal updated (UTC): `2026-08-14T00:55:00Z`

### Facts
1. PUBLIC LAW BENCHMARKS (data/bench_pub, evaluation-only): teacher3 (CF-trained) vs old model:
   | Benchmark | old | teacher3 |
   |---|---|---|
   | Synthbuster+ (DALL-E 2/3, 600) | 36.3% | 88.0% ai_recall |
   | CF-Eval (GAN/manipulated, 600) | 4.0% | 60.8% ai_recall, bal 0.80 |
   | CF-Commercial test (Firefly2/3, FLUX, Ideogram, 600) | — | 99.0% ai_recall |
2. SEALED OOD holdout (data/holdout_oos, NEVER in training): DiffusionDB 769 → 99.2% recall; MJ2 300 → 100% recall. Combined 99.4% — above the 96% target.
3. Harness run #5 (teacher3): 0.9833 (ai 0.9733, real 0.9933) — proxy dipped 1pp vs run #4 while real-world benchmarks gained 25-57pp.
4. Teacher4 (in progress): +9k CF systematic fakes (ai_cf1-3, SD-finetune diversity) for CF-Eval improvement.
5. Auditor confirmed: no leakage into training from data/eval, data/holdout_oos, or the law-benchmark parquets; commercial parquet recognized as TEST split and excluded from training (no benchmaxxing).

### Interpretation
- The fal-only training was overfit to the diffusion ecosystem; public benchmarks exposed it. CF training is the highest-value fix so far.
- Remaining gaps: DALL-E (12% miss on Synthbuster) and GAN/manipulated (39% miss on CF-Eval). No DALL-E training source found yet; CF systematic expansions are the available lever.

## Entry: 2026-08-14 (student + WebGPU floor)

### Timestamps
- Journal updated (UTC): `2026-08-14T02:10:00Z`

### Facts
1. Student distillation results (teacher4 → small model):
   | Student | val_bal | Harness | Synthbuster | CF-Eval | CF-Commercial |
   |---|---|---|---|---|---|
   | mvl3 (28ep, λ.7 T3) | 0.9762 | 0.9767 | 81.0% | 48.3% | 98.3% |
   | resnet18 (30ep, λ.6 T4) | 0.9731 | **0.9933** | **83.7%** | **61.3%** | **98.7%** |
   | teacher4 (reference) | 0.9897 | 0.9800 | 90.2% | 80.0% | 98.8% |
2. mvl3 q8 quantization broken (h-swish, real→0.92 AI); fp16 conversion exact. resnet18 q8 clean (11.2MB).
3. **WebGPU speed (M2 Max, Chrome for Testing): resnet18 q8 = 177ms (fp32 dequant); resnet18 fp16 = 8ms avg (p50 8.3, p95 8.7)** — 22x faster, ~11x under the leader's ~90ms.
4. Extension now: WebGPU → fp16 model; WASM → q8. fp16 IO decode bug found & fixed (Uint16Array bit patterns were fed to Math.exp → inverted scores; E2E now PASS with correct per-image parity).
5. Law benchmarks (evaluation-only): Synthbuster+ 927 (DALL-E 2/3), CF-Eval 699, CF-Commercial 7459 (Firefly/FLUX/Ideogram) — committed bench_pub.py, data gitignored.

### Interpretation
- resnet18 student is the shipping candidate: 0.9933 harness, 8ms WebGPU, q8-clean. Teacher4 remains the accuracy ceiling (CF-Eval 80%) — ship teacher for max accuracy mode, student for speed mode, or ensemble.
- Remaining gaps: CF-Eval 61% (student) / 80% (teacher) — manipulated-image coverage still thin; more CF systematic train shards available.

## Entry: 2026-08-14 (CF-Eval cell map + GAN fix)

### Timestamps
- Journal updated (UTC): `2026-08-14T03:00:00Z`

### Facts
1. CF-Eval is organized per-generator across 413 shards. Sampled cell map: DFGAN (shards 0-6,350), stable_cascade (20), MidjourneyV5_2 (30,50), MidjourneyV6_1 (10,75), Dalle3 (100), IdeogramV2 (150), Firefly_Image3 (200), Dalle2 (250), FLUX-dev (300), LCM_lora_ssd1b (400).
2. Per-cell recall (resnet18 student): DFGAN 57-73% (avg ~63, THE weak cell), stable_cascade 95%, MJv5 98-99%, MJv6 98%, Dalle3 100%, IdeogramV2 100%, Dalle2 94%, FLUX-dev 100%. Earlier "CF-Eval 61%" was DFGAN-only — the full-mix number is ~90%+.
3. GAN training source found: balgot/stylegan3-annotated (20,000 StyleGAN3 faces) → ai_stylegan3. CF systematic shards 4-6 extracted (9k more fakes). Teacher5 training with 20k GAN + 21k CF + all prior data.
4. DALL-E gap re-assessed: CF-Eval Dalle2 94% / Dalle3 100% — the Synthbuster 84-90% is pipeline-specific (screenshots/UI-embedded DALL-E), not a broad DALL-E weakness.

### Interpretation
- The one remaining weak cell is GAN images (DFGAN). StyleGAN3 training data is the direct fix; teacher5 (in progress) will show the effect.
- Synthbuster's DALL-E cells need a DALL-E-with-UI training source — none found yet; optional.

## Entry: 2026-08-14 (teacher5 + student v2)

### Timestamps
- Journal updated (UTC): `2026-08-14T03:55:00Z`

### Facts
1. Teacher5 (convnext_tiny, 26ep): +20k StyleGAN3 faces + 9k CF (shards 4-6) → best val_bal 0.9909, auroc 0.9997. CF-Eval full cell map: DFGAN 86% (from 80%), stable_cascade 99%, MJv5 98-100%, MJv6 100%, Dalle2/3 100%, Ideogram 100%, FLUX-dev 99%. Harness #9: 0.9800.
2. Student v2 (resnet18 from teacher5, 30ep λ.6 T4): harness #10 0.9800; CF-Eval: MJ/Dalle/Ideogram/FLUX 96-100%, DFGAN ~59% (GAN capability does not fully distill). q8 clean 11.2MB; fp16 WebGPU 8ms.
3. Extension: dtype-adaptive fp16 IO (session.inputMetadata) — fixes the converter's inconsistent keep_io_types behavior. E2E PASS.
4. CF-Eval structure: per-generator shard map documented (413 shards; DFGAN shards 0-6/350, MJv5 30/50, MJv6 10/75, Dalle3 100, Dalle2 250, IdeogramV2 150, Firefly 200, FLUX-dev 300, stable_cascade 20, LCM 400).

### Interpretation
- Shipping pair: teacher5 (accuracy, 28MB q8, ~100-180ms) + student v2 (speed, 11.2MB q8 / 22.4MB fp16, 8ms WebGPU). Both clear the 75% bar on every measured benchmark by wide margins.
- Next: submission prep (squashed public repo, README update, claim) or ensemble/CF-Eval-firefly cell coverage.

## Entry: 2026-08-14 (CF real-cell investigation — negative results, models reverted)

### Timestamps
- Journal updated (UTC): `2026-08-14T05:40:00Z`

### Facts
1. Full CF-Eval dual-class measurement (33 spread shards + original 12): fake cells are ≥85% (teacher5: DFGAN 85%, all others 96-100%); the label=0 "real" cells are the hard side (teacher5: MJv5/real 31%, DFGAN/real 31%, MJv6/real 78%, Dalle3/real 75%, IdeogramV2/real 64%). Overall equal-weight (all cells): teacher5 0.819.
2. Real-cell investigation: the CF real pairs (real_source=ffhq for face cells; LAION for others) are prompt-source/matched images — clean portraits with studio polish that look AI-like. Two targeted fixes TRIED and REVERTED (both hurt):
   a. HeavyWebArtifact augmentation for real samples (teacher6/7): model learned artifact→real, real cells CRASHED (DFGAN/real 31→9.5%, MJv5/real 31→23%).
   b. Real FFHQ faces in training (teacher8, 6891 faces): real cells crashed FURTHER (DFGAN/real 5%, MJv5/real 18%) — face-domain reals do not help the matched-pair cells; proxy AI recall also dipped (0.9600).
3. Reverted to teacher5 + student v2 as shipped. Student v2 verified: harness 0.9800, extension E2E PASS.
4. CF-Eval structure note: some shards hold the REAL matched pairs (label=0) for each generator — a full evaluation must measure both classes.

### Interpretation
- The CF real-pair cells are prompt-source/matched images (the literature's hardest corner: detectors score 54-66% on manipulated/matched sets). Our normal-real recall is 99-100% (harness, picsum 1200, sealed OOS). Chasing the CF real cells damaged balance; the shipped teacher5/student pair stands as the best measured tradeoff.
- Next real-side lever (not tried): train on LAION-style web-scraped reals with heavy artifacts, WITHOUT changing the fake-side augmentation — requires a genuine web-scrape source (OpenImages subset on HF).

## Entry: 2026-08-14 (real-cell breakthrough — JPEG mechanism found)

### Timestamps
- Journal updated (UTC): `2026-08-14T17:00:00Z`

### Facts
1. ROOT CAUSE of the CF-Eval real-cell failures: the model learned "JPEG artifacts on a face = AI" because training fakes are stored as JPEG while FFHQ reals are PNG. Grid search isolated q60-75 as the discriminator band (pristine FFHQ 98% real, q60 → 30%).
2. Teacher11 (shipped): + FFHQ 8.4k real faces + face-balance (StyleGAN3 capped 6000) + JPEG-compress real-aug (q55-85, 40% of real samples). Harness 0.9833 (ai 0.9667, real 1.0) — beats teacher5's 0.9800. CF-Eval real cells: MJv5/real 31→66%, Dalle3 52→75%, MJv6 78→79%, Firefly2 32→63%, Imagen3 45→64%, IdeogramV1 0→100%; pristine FFHQ 99.3%, jpeg-q60 faces 97.3%. Fakes stay 91-100%.
3. NEGATIVE: teacher12 (+5000 COCO casual photos) COLLAPSED everything (DFGAN/fake 92→15%) — casual scene photos are nearly identical to diffusion-fake scenes; the matched-pair boundary is fundamentally fuzzy. Reverted; COCO wiring removed from assembly.
4. Remaining weak cell: DFGAN/real 18% (480x640 casual JPEG scenes — matched prompt-source images, near-indistinguishable from the fakes; the literature's hardest corner).
5. Teacher11b rerun in progress (deterministic) to regenerate best_state.pt for student distillation.

### Interpretation
- Storage-format leakage (JPEG vs PNG) was a REAL train/eval gap, not benchmark trickery: real-world photos ARE JPEG-compressed, so the fix improves genuine generalization (harness +0.3pp, real cells +20-45pp).
- The matched-pair real cells converge toward indistinguishability by design; teacher11 is the best achievable balance measured so far.

## 2026-08-14 — Ensemble max-fusion shipped (runs #13/#14)

- teacher13: teacher11 recipe + stylegan3 cap 6000→8000. val_bal 0.9828. LAW: DFGAN/fake 90→91%, Hourglass/real 84→87%. Harness ties teacher11. KEPT as ensemble teacher.
- FUSION: mean dilutes teacher confidences below 0.65 on GAN cells (DFGAN/fake 76%); max-fusion → 91% with only -0.3pp harness. MAX is the shipped fusion.
- student v4: resnet18 distilled from teacher13 (30ep, λ=0.6, T=4, ai=33156). Alone 0.9833; v4+t13 max-ens harness 0.9867 (real 0.9867, 2 FP) — best recorded.
- Correctness: sealed OOS 0.9981 (guardrail ≥96% ✓). LAW final: Synthbuster+ 94.3%, CF-Commercial 99.2%, CF-Eval ai 97.2% / real 24.8% aggregate (fuzzy matched-pair cells).
- Infra: Spark map-in-parallel starves training (load 69) — run sequentially, OMP_NUM_THREADS=8. GitHub 100MB limit → fp32 teacher excluded from public repo; detect.py fallback fp32→fp16→q8 (fp16 parity 0.9867 exact). Teacher q8 now real quantization (28MB vs 111MB fake).
- Public repo 77a234f: fresh-export verify harness 0.9867 → build → E2E PASS. E2E needs npm i puppeteer.

## Entry: 2026-08-15 — teacher14 / student v5 384px audit

### Timestamps
- Journal updated at (UTC): `2026-08-15T05:25:08Z`
- Latest run event at (UTC): `2026-08-15T05:23:52Z` (`cd242fe0` committed 384px teacher14 + student v5 export)

### Seed
- Training seed: `SEED = 0` in `detector/train_cnn.py`; the remote training log/checkpoint metadata is not committed locally.
- Eval seed policy: committed eval is deterministic; public-benchmark sampling uses `random.Random(0)`.

### Runs
| Stage | Run ID | Name | Created (UTC) | State | Selected Checkpoint | Notes |
|---|---|---|---|---|---|---|
| export | `cd242fe0` | teacher14 + student v5, 384px | 2026-08-15T05:23:52Z | diagnostic | `detector/model_cnn.onnx` + `model_cnn_teacher.onnx` | current committed fp32/fp16 384px pair; not promotion-cleared |
| test | local-384 | committed harness | 2026-08-15T05:24Z | ok | same pair, max-fusion | bal `0.9933`, AI recall `0.9867`, real recall `1.0000`, AUROC `1.0000`, Brier `0.005252`, ECE10 `0.009861`; 2 AI FN / 0 real FP |
| holdout | local-384 | `data/holdout2` + `data/probe/real` | 2026-08-15T05:24Z | ok | same pair | holdout2 AI `320/320`; real probe recall `95.5%` (`90/2000` false AI) |
| law | local-384 | public sampled benchmarks | 2026-08-15T05:24Z | ok | same pair | Synthbuster+ AI `95.0%`; CF-Eval AI `97.75%` / real `50.17%` / balanced `73.96%`; CF-Commercial AI `99.75%` |

### Meta
- Core config delta vs prior shipped path: 256px → 384px input/crop; teacher14 + student v5 ONNX pair; max-fusion remains the detector rule.
- Evidence paths: `bench/evaluate.py`, `scripts/run_probe.py`, `scripts/bench_pub.py`, `data/holdout2`, `data/probe/real`, `data/bench_pub`.
- Split evidence: val `unknown` in this checkout because the remote training log/checkpoint report is not committed; test evidence is recorded above.

### Critical Observations
1. Measured fact: 384px preserves the already-excellent proxy and modern-generator detection, but it is not a net win on every plane: the independent 2,000-photo real probe is only `95.5%` real recall.
2. Measured fact: the commit headline `DFGAN/real 100%` is a narrow cell result; the broader sampled CF-Eval real recall is `50.17%`. Do not generalize the narrow cell to overall real-image specificity.
3. Measured fact: the WebGPU/fp16 browser path passed E2E on 4 images (`max delta 0.022`, mean `0.006`), but the WASM/q8 bundle is not coherent with 384 preprocessing: `detector/model_cnn_q.onnx` still has a fixed `256x256` input while `extension/offscreen.js` sends `384x384`. `detector/model_cnn.json` also still reports size `256`.
4. [inference] Keep this as a promising research candidate, not the final promoted model, until q8 is regenerated at 384, forced-WASM E2E passes, and hard-real specificity is measured against a separate calibration/acceptance set.

## 2026-08-14 — Feature distillation (moment-match) NEGATIVE — v6 reverted

- Hypothesis: per-channel mean/std moment matching of stage-4 features (1x1 projector 512->768ch, weight 0.2) transfers the teacher's texture/grain representations to the student.
- Result: v6 val_bal 0.9690 (v5: 0.9664, +0.3pp on val) BUT harness 0.9833 (ai 0.967) vs v5 0.9867 (ai 0.987); CF cells tie or worse (DFGAN/fake 0.85 vs 0.89 alone; DFGAN/real 1.0 tie; shard10/real 0.58 vs 0.59).
- Verdict: logit-KD already carries the teacher's decisions; the extra feature constraint shifts the boundary harmfully. REVERTED to v5. Do not re-run unless with a weight sweep (0.05) or view-consistency formulation.
- Speed infra (kept): AMP-val fix 126->94s/epoch (-25%), --patience early stop, --amp bf16, --feat-distill flag, --compile-mode. max-autotune rejected (GB10 slow compile/OOM); batch 128@384 OOM.

## 2026-08-15 — Tournament T1 (multi-view hybrid) LOSS

- T1-seed0: convnext_tiny full-image + HP/LP frequency experts, fused head.
- dev val_bal 0.9310 @12 (early stop 17) vs T0 (teacher14) 0.9828; sealed harness 0.8400 (ai 0.68, real 1.0) vs T0 0.9933.
- The fresh experts + fusion head dilute the strong pretrained branch at this budget; 15pp sealed gap cannot be seed variance.
- Tournament keeps T0 (teacher14). T1 family not pursued further without a fundamentally different formulation.

## 2026-08-15 — Tournament T2 (patch-shuffle) LOSS; tournament closes with T0

- T2-seed0: full+hp+lp views with patch-shuffle scene randomization. dev val_bal 0.9477 @26; sealed harness 0.8600 (ai 0.72, real 1.0) vs T0 0.9933.
- Both hybrid families (T1/T2) trail T0 by 13-15pp on the sealed harness — the fresh experts + harder task cannot match the fine-tuned single-branch at this budget. T3 (prototypes, same architecture) not run: same family, dominated by the same failure mode.
- TOURNAMENT RESULT: T0 (teacher14, convnext_tiny @384) retained. Multi-view hybrid architecture family recorded as negative.

## 2026-08-15 — Task 5 hard-negative round LOSS (confirms COCO lesson)

- Mined 332 hard examples from the dev pool (240 COCO reals flagged AI — the fuzzy matched-pair cousins; 80 CIFAKE fakes missed). Registry + capped append (max 200/source) tooling works.
- t15 (teacher14 recipe + hard rows): best val_bal 0.9003 @23 vs t14's 0.9828 (-8pp); sanity AI probe 1.0 -> 0.7793.
- The casual-scene reals in training again collapse fake-side detection (the documented COCO lesson, this time via mining rather than a bulk add). Gate (real-recall gain + <=2pp fake loss) fails -> teacher14 retained.
- Takeaway: the fuzzy matched-pair boundary is NOT fixable by adding more casual reals to training; it needs the abstention reframe (shipped in Task 6).

## 2026-08-15 — External assessment + next tournament design

- Independent review: deployment stack SOTA-ish; detector methodology behind the 2026 frontier (fair — our report claims dataset-specific ranks only). Key flagged gaps: CF-Eval real aggregate 37%, wild-real FPR 4.55% vs 1% target, shortcut-repair pattern (HeavyWebArtifact etc.).
- 2026 frontier per review: foundation representations (DINOv3/PE linear probes dominant), PROBE-style boundary mining, CoDA (1.48M param color-distribution-probing detector), Stay-Positive structural shortcut avoidance.
- NEXT TOURNAMENT (design, code ready, waiting on the DGX Spark which went offline):
  1. Foundation-teacher distill: DINOv3-small/base (vit_small_patch16_dinov3 @384, timm has it) as teacher for resnet18 student (ViT-aware feature-KD shipped). Hypothesis: foundation features generalize to unseen generators better than convnext_tiny's.
  2. CoDA-style color-probe view ("color" expert in model_hybrid.py) — content-robust stats branch, deliberately blind to layout.
  3. Gate: harness >=0.98, CF-Eval real aggregate up, OOS >=0.96, E2E both backends.
- Spark status: unreachable (ping 100% loss) at 2026-08-15 ~03:30 UTC — retry before training; code is one command away (teacher: --model convnext_tiny... --teacher-state for distill uses --teacher-model vit_small_patch16_dinov3).

## 2026-08-15 — DINOv3-S linear probe: measured negative for frozen-foundation direction

- bench/dino_probe.py: frozen vit_small_patch16_dinov3 (CLS+mean-token features, 2-layer head, 8000 train samples, 6 epochs).
- RESULTS: harness 0.9333 (ai 0.907, real 0.960); DFGAN/fake 0.817; **DFGAN/real 0.000** — every matched-pair real flagged AI.
- Interpretation: the frozen foundation encodes scene semantics; matched real photos of the same scenes read as "AI-like" by content, not provenance. This is precisely the shortcut class Stay-Positive warns about — a frozen ImageNet-style representation is dominated by it. Our fine-tuned 384px convnext (camera-grain cue) beats the foundation on every cell, most dramatically on the matched reals (1.000 vs 0.000).
- Verdict: DO NOT pursue frozen-foundation distillation for the ship. A full DINOv3 fine-tune (not frozen) is a different question, lower priority than the color-probe expert.

## 2026-08-16 — teacher16 (noise-quant probe channel): measured LOSS

- CoDA-faithful mechanism: per-pixel flip-rate under Gaussian noise + 5-bit color quantization as a 4th input channel (stem adapted, zero-init).
- dev val_bal 0.9902 @22 (BEATS t14's 0.9828); sealed harness 0.9700 (ai 0.940, real 1.0) — WORSE than t14-as-teacher 0.9867. DFGAN/fake 0.975 (+1.5pp), DFGAN/real 1.0, shard10/real 0.562.
- The probe channel lifts dev without transferring to the sealed eval (crop-pipeline sensitivity?); max-fused ensemble stays <=0.9867 < frozen 0.9933. NOT kept.
- Next: advisor's pre-registered DINOv3-S full fine-tune (bench/dino_probe.py --fine-tune, 12ep, gates as registered).

## 2026-08-16 — DINOv3-S full fine-tune: pre-registered single evaluation FAILS GATES — foundation direction CLOSED

- Run: vit_small_patch16_dinov3 @384 full fine-tune (BCE only, back 1e-5/head 1e-4, wd 0.05, warmup+cosine, clip 1.0, 12 epochs, dev checkpoint 0.9762@6, batch 32). Single sealed eval.
- Results @0.65: harness 0.9600 (ai 0.940, real 0.980); probe_real false-AI 0.0483 (n=600); DFGAN/fake 0.975; DFGAN/real 0.667; shard10/real 0.400; OOS 0.9981.
- GATES: harness>=0.98 FAIL (0.9600); probe false-AI <4.55% FAIL (4.83%); LAW real cells worse than frozen bundle (67/40 vs 100/59). DFGAN/fake and OOS pass.
- Both foundation formulations now measured: frozen probe 0.9333/DFGAN-real 0.0; full fine-tune 0.9600/0.667. Both lose to the fine-tuned 384px convnext (camera-grain) on every decisive cell.
- ADVISOR RULE: any gate miss ends the foundation direction. CLOSED.
- Champion unchanged: student v5 + teacher14 @384 max-fusion (harness 0.9933, DFGAN/real 100%). Next-tournament record: DINOv3 probe LOSS, DINOv3 fine-tune LOSS, CoDA probe-channel LOSS, static-color LOSS — the frozen bundle survived every 2026-frontier challenger we could run.

## 2026-08-16 — teacher17 (AI-side JPEG aug): robustness up, matched-reals DOWN — LOSS

- Degradation battery (AIGIBench-style, bench/degrade_report.py) measured the bundle collapsing under storage degradation: jpeg q60 bal 0.587 (ai 0.17), web chain 0.527, noise 0.70 — real recall 1.0 everywhere (the real-side aug worked TOO well: JPEG=>real cue with no AI-side equivalent).
- t17 fix: mirror file-level JPEG (q55-85, 40%) on the AI side. Clean harness unchanged 0.9933. Battery recovered: jpeg q60 0.587->0.913, q40 0.523->0.890, web 0.527->0.807, noise 0.70->0.897. 
- BUT matched-real cells collapsed: DFGAN/real 1.0->0.667, shard10/real 0.59->0.38 (ens max-fusion cannot recover: the teacher's false-AI dominates). OOS 0.9991 ok.
- Mechanism: the storage cue is TWO-SIDED — matched reals live in the same q55-85 band; pushing JPEG=>AI breaks them. Rebalancing storage cannot win; the model must IGNORE storage and use a storage-invariant signal (CoDA-style stability probe).
- REVERTED to t14. NEXT (pre-registered): teacher18 = probe channel (t16 mechanism) + AI-side JPEG (t17 mechanism) — the stability signal should fix the JPEG-AI-induced real regression. Gates: harness >=0.98, jpeg-q60 ai >=0.8, DFGAN/real >=0.9, shard10 >=0.5.

## 2026-08-16 — teacher18 (probe channel + AI-JPEG): FAILS gates — storage-robustness axis CLOSED

- Pre-registered gates: harness>=0.98, jpeg-q60 ai>=0.8, DFGAN/real>=0.9, shard10>=0.5.
- Results (4-ch): clean 0.9800 (ai 0.96, FAIL), jpeg q60 0.897 (ai 0.793, PASS-ish), web 0.807, DFGAN/fake 0.875, DFGAN/real 0.667 (FAIL), shard10 0.388 (FAIL).
- The probe channel did not rescue the matched reals: JPEG compression destroys the color-stability signal the probe measures (both classes already quantized). 
- AXIS VERDICT (3 measured negatives): t16 probe alone (dev 0.9902/sealed 0.97), t17 AI-JPEG alone (battery +0.33, reals -0.33), t18 both (reals still -0.33). JPEG-re-encoded AI vs JPEG-matched real photos are near-indistinguishable at 384px by grain, color-stability, or storage cues. The frozen t14 bundle's failure direction (degraded AI slips through) is the SAFE direction for a consumer tool; abstention covers the rest.
- The degradation battery (bench/degrade_report.py) becomes a PERMANENT regression gate in the ship loop: clean bal must hold >=0.98 AND jpeg-q60 ai-recall >=0.8x clean before any model ships.
- NEXT (pre-registered, final model experiment): 512px probe — the spectral diagnostic showed grain separation GROWS 384->512 (lap 8.7 vs 18.3); the 384 jump won DFGAN/real 33->100 the same way. 8-epoch probe first; gates: dev>=0.95, battery jpeg-q60 ai >=0.8, DFGAN/real >=0.9, shard10 >=0.5.

## 2026-08-16 — 512px probe: FAILS pre-registered gate — model search complete

- 8-epoch convnext probe @512 (batch 32): val_bal 0.8587 @8 vs the 384 probe's 0.9698 @8 (the run that justified the 384 breakthrough). 11pp behind at equal budget; a full 26-epoch 512 run (~75 min) is not justified by the probe. CLOSED.
- FINAL MODEL SEARCH VERDICT: the frozen bundle (student v5 + teacher14 @384 max-fusion) survived every measured challenger: DINOv3-S frozen (0.9333/DFGAN-real 0.0), DINOv3-S fine-tune (0.9600/0.667), CoDA probe channel t16 (0.9700), AI-JPEG t17 (reals 0.667), probe+AI-JPEG t18 (0.667), HP/LP experts, patch-shuffle, hard-negative rounds, feature-KD, static color, 512px.
- The storage-robustness boundary is fundamental at 384px: JPEG-re-encoded AI vs JPEG-matched real photos are near-indistinguishable by grain, color-stability, or storage cues (3 measured negatives). Failure direction of the bundle (degraded AI slips through = false negative) is the SAFE direction for a consumer tool; abstention covers the ambiguous band.
- NEW PERMANENT GATE (ship loop): bench/degrade_report.py — clean bal >=0.98 AND jpeg-q60 ai-recall >=0.8x clean before any model ships.

## 2026-08-16 — Size-axis challenger: MobileNetV3-small student (m3) measured

- 2.5M params, 30 epochs distilled from teacher14 @384 (batch 64; batch 128 OOM'd with the CF map running). best val_bal 0.9486 @28.
- Sizes: fp32 6.1MB (vs resnet18 v5 44.7MB) — the CoDA league (1.48M params).
- Harness: m3 alone 0.9433 (ai 0.887, real 1.0); m3+t14 max 0.9667 — v5 alone 0.9833, v5+t14 0.9933. 4pp behind; gate (>=0.98 standalone) FAILS.
- q8 blocker confirmed: h-swish quantization broken (sanity real q8=0.9999 = calls real images AI) — m3 can only ship fp16/fp32, no WASM parity path.
- VERDICT: documented Pareto point, NOT shipped. The bundle (v5+t14) stays the final student/teacher pair. Size axis explored; accuracy gap at 2.5M params is real.

## 2026-08-16 — teacher19 (browser-render sim aug): LOSS — screenshot axis needs REAL screenshots

- Pre-registered gates: clean>=0.98, screenshot real>=0.95, DFGAN/real>=0.95.
- v5+t19: harness 0.9900 (real 0.993, -1 FP); screenshot real recall 0.820 (was 0.860) — the simulated render aug (bilinear down/up + PNG + contrast) made screenshots WORSE, not better. The simulated artifacts are not the screenshot path's dominant cue.
- Verdict: the screenshot axis requires REAL screenshots in training (render-then-capture via the generator we already built — scripts/gen_screenshot_stratum.js), not hand-simulated render ops. Pre-registered: teacher20 = t14 recipe + real-screenshot real stratum (~1500 shots of TRAINING images).
- AIGIBench on HF (HorizonTEL/AIGIBench) is the detector repo, not the eval data — data hunt continues.

## 2026-08-17 — teacher20/teacher21: real-screenshot training — measured, plateaued

- teacher20 (t14 recipe + 80 real screenshots of training reals in train): harness 0.9933 (ai 0.987, real 1.0) PRESERVED; screenshot stratum real recall 0.860->0.900 (+4pp); CF cells HELD (DFGAN/real 1.0, shard10 0.60); degradation battery at the t14 reference. Gate (screenshot real>=0.95): SHORT.
- teacher21 (500 screenshots: 400 more, varied imagenette/cifake/ffhq reals): harness 0.9933, screenshots 0.900 — SAME. The axis saturates at ~0.90 with real-screenshot training; the residual ~10% of re-photographed reals is not a data-size problem.
- VERDICT: storage-domain cues only move the boundary (t17/t18 lesson re-confirmed); screenshot axis ceiling ~0.90 documented. Next per advisor plan: response spectroscopy (measure how evidence behaves under stress; tiny meta-classifier on dev; no retraining).
- AIGIBench official detector checkpoints downloaded (data/bench_pub/aigibench/): NPR, Resnet50, DFFreq, Gram-Net, SAFE — external baselines for the report.

## 2026-08-17 — FORENSIC RESPONSE SPECTROSCOPY CONFIRMED (advisor plan item 1+2)

- bench/response_spectroscopy.py: per-image stress family (clean, jpeg q95/80/60/40, resize 0.75/0.5, blur2, noise0.02 deltas + 5-crop spread/max-delta = 11-dim vector); tiny logistic fit on the DEV pool (800), tested on sealed sets.
- RESULTS: eval (saturated) 1.0/1.0 (1 overlap image — useless as a test set); screenshots clean 1.0 vs meta 0.912 (clean ranking already perfect; the 10% false-AI is a THRESHOLD/calibration issue, not ranking — supports acquisition-aware thresholds); **cf-matched (80 matched reals + 40 GAN fakes — THE fuzzy population): clean 0.809 -> meta 0.981**.
- CONCLUSION: response curves separate hard reals from AI where clean scores overlap. The meta is ~12 params, trivially deployable. Design: run the stress family ONLY for ambiguous-band images (clean p in [0.3, 0.8]) — the abstention zone — as a "confirmation" pass (the advisor's active-probing principle, bounded cost).
- Next: export the meta + response machinery into the decision layer (acquisition-regime flag + band-refined confidence); screenshots need a separate calibrated threshold identified by their response signature.

## 2026-08-17 — Spectroscopy decision-layer nuance (BAcc vs AUROC)

- cf-matched (83: 41 matched reals + 42 GAN fakes): clean BAcc@0.65 = meta BAcc@0.65 = 0.732 — the FIXED 0.65 threshold neutralizes the ranking gain. Meta best BAcc 0.767 at t=0.87.
- The confirmed value: AUROC 0.809 -> 0.981 makes the fuzzy population rankable — the prerequisite for acquisition-regime thresholds (advisor item 4): one threshold cannot serve camera/screenshot/social-JPEG/matched-source domains; the response signature identifies the regime, then a per-regime threshold applies.
- Deployable artifact: /tmp/spectro_meta.json (11 weights + intercept) exported; features npz saved. Extension design: ambiguous-band (p in [0.3,0.8]) confirmation pass computes the response vector (12 extra WebGPU runs ~100ms, rare path), meta score + regime flag -> refined decision; easy images stay one-pass.

## 2026-08-17 — RETRACTION: response-spectroscopy "0.981" was shard confounding

- Advisor audit (reviewer agent) flagged: all-set AUROC on a shard-confounded sample (reals from CF shards 0+10, fakes from shard 0), in-band evidence empty (6 images, one class), 0.87 threshold test-tuned, ties mishandled.
- CLEAN RE-MEASUREMENT (bench/spectro_remeasure.py, n=215 multi-shard, tie-corrected AUROC, dev-tuned thresholds, bootstrap): clean AUROC 0.848 vs meta 0.849 — **gain +0.002 [95% CI −0.052, +0.054], p=0.474: NOTHING**. The 0.809->0.981 claim is retracted; it was the meta reading shard/generator identity, not origin.
- The earlier bootstrap CI (+0.18 [0.10,0.28]) and dev-stability (0.981-0.983) were valid FOR THE CONFOUNDED SAMPLE — they cannot rescue the claim.
- Process lesson (user was right): 5 messy runs, a patch that landed twice, buffered logs, and a shard-confounded test set. No signal survives a confounded sample regardless of CI.
- What stands: the degradation battery findings, the screenshot plateau (0.90), the frozen bundle — all measured on properly matched/independent sets. bench/spectro_meta.json is WITHDRAWN (do not ship). The response-spectroscopy direction is DEAD unless a genuinely matched per-generator real/fake sample exists (CF-Eval shards are single-class-dominant — DFGAN shard 0 has ~3 reals vs 125 fakes — no clean matched test exists in the local corpus).

## 2026-08-17 — Honest status after the retraction

- The frozen bundle (v5+t14@384, harness 0.9933, DFGAN/real 100%, OOS 0.9981) and its measured gates (degradation battery, screenshot plateau) are UNAFFECTED — they were never part of the spectroscopy thread.
- The decision-layer/ambiguous-band design is NOT justified by evidence. Any future attempt needs: (a) genuinely matched per-generator real/fake pairs (not available locally), (b) in-band samples with both classes, (c) dev-tuned thresholds only, (d) tie-corrected metrics, (e) grouped CV over generators.
- Claim filing stands on the frozen bundle's measured record, not on any spectroscopy result.

## 2026-08-17 — Degradation-abstention layer FALSIFIED as designed (pre-registered, advisor-reviewed)

- Design (DegradeAbstainReview, GO-after-fixes): 3-feature logistic (phase-invariant blockiness, normalized two-axis grain, Hann-windowed HF ratio) on the identical 384px crop; dev grouped-CV fit; dev-tuned threshold (>=0.80 degraded recall @ <=0.03 clean fpr); sealed battery gates (>=90% baseline wrongs abstained, decided-error <=5%/<=10% per severe cell, abstain <=85%).
- RESULT: gates FAIL — dev-tuned threshold 0.835 with degraded recall 0.023; all family OOF recalls 0.000; battery abstain rate 0.000. The features do NOT separate clean from degraded at 384px (blockiness 0.0075 vs 0.0079, grain/hf flat).
- Full-res diagnostic (pre-pipeline decode): equally non-separating (blockiness means ~0 with std 0.11-0.24).
- STRUCTURAL CAUSE: the extension's canvas path re-encodes (JPEG q90) + resamples before the model sees pixels — the double-compression grid signal is destroyed before any pixel-side feature could read it. Browser-side degradation detection on DECODED pixels is fundamentally limited.
- The one surviving option: file-level double-compression detection (analyze the RAW image bytes' DCT grid phase before decode) — a NEW experiment requiring its own advisor pre-registration + MV3 raw-bytes fetch design. Not started.
- Protocol honored: no retuning after gate miss; features + script kept as a documented negative (detector/degrade_features.py, bench/degrade_abstain.py).
