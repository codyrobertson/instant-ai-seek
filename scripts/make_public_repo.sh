#!/usr/bin/env bash
# Export a clean public submission repo (single squashed commit) from the
# working tree. The working repo keeps its full history; the export carries
# only the current state: harness, eval set, models, scripts, extension.
#
# Usage: bash scripts/make_public_repo.sh [dest_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="$PWD"
DEST="${1:-/tmp/f01d-public}"

rm -rf "$DEST"
mkdir -p "$DEST"
git ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    data/holdout/*|data/holdout2/*|data/probe/*|data/xgen/*|data/xgen2/*)
      continue;;  # fal probe diagnostics — not needed for build/harness
    detector/model_cnn_teacher.onnx)
      continue;;  # 106MB fp32 > GitHub 100MB limit; detect.py falls back to fp16/q8
  esac
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$f" "$DEST/$f"
done

cd "$DEST"
git init -q -b main
git -c user.name="f01d" -c user.email="f01d@local" add -A
git -c user.name="f01d" -c user.email="f01d@local" commit -q -m "f01d: in-browser AI image detector (MV3 Chrome extension)

- convnext_tiny teacher (accuracy) + resnet18 student (8ms WebGPU fp16)
- fully offline after install; WASM q8 fallback
- benchmark: bash autoresearch.sh (balanced accuracy @ 0.65 threshold)
- public law benchmarks: Synthbuster+ / CF-Eval / CF-Commercial
  (see scripts/bench_pub.py; data downloads documented in README)"
echo "public repo ready: $DEST ($(du -sh . | cut -f1))"
