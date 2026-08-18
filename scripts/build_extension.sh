#!/usr/bin/env bash
# Build the f01d Chrome extension (fully reproducible, offline-capable).
#
# Steps:
#   1. Vendor onnxruntime-web (WASM runtime) into extension/lib/onnxruntime-web
#   2. Copy the quantized ONNX detector into extension/lib/model_cnn_q.onnx
#   3. Generate the extension icons
#
# After this script, extension/ is a complete unpacked MV3 extension that runs
# fully offline (no downloads needed at install time).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
EXT="$ROOT/extension"
LIB="$EXT/lib"

mkdir -p "$LIB"

# --- 1. onnxruntime-web WASM runtime ---------------------------------------
if [[ ! -f "$LIB/onnxruntime-web/ort.min.js" ]]; then
  echo "vendoring onnxruntime-web ..."
  TMP="$(mktemp -d)"
  cd "$TMP"
  npm pack onnxruntime-web --silent >/dev/null 2>&1 || { echo "npm pack onnxruntime-web failed (needs network once)" >&2; exit 1; }
  tar -xzf onnxruntime-web-*.tgz
  mkdir -p "$LIB/onnxruntime-web"
  # WASM-only runtime: the JS loader + threaded/simd wasm variants
  cp package/dist/ort.min.js "$LIB/onnxruntime-web/"
  cp package/dist/ort-wasm-simd-threaded.* "$LIB/onnxruntime-web/"
  cd "$ROOT"
  rm -rf "$TMP"
fi
echo "onnxruntime-web: $(ls "$LIB/onnxruntime-web" | tr '\n' ' ')"

# --- 2. detector model bundle (verify + copy — no stale artifacts) ---------
python3 scripts/verify_model_bundle.py --strict >/dev/null
for f in model_cnn_q.onnx model_cnn_fp16.onnx model_cnn_teacher_fp16.onnx model_cnn_teacher_q.onnx \
         model_bundle_manifest.json; do
  if [[ -f "detector/$f" ]]; then
    cp -f "detector/$f" "$LIB/$f"
  fi
done
echo "models: q8 $(du -h "$LIB/model_cnn_q.onnx" | cut -f1), fp16 $(du -h "$LIB/model_cnn_fp16.onnx" | cut -f1), teacher-fp16 $(du -h "$LIB/model_cnn_teacher_fp16.onnx" | cut -f1), teacher-q8 $(du -h "$LIB/model_cnn_teacher_q.onnx" | cut -f1)"
# --- 3. icons ---------------------------------------------------------------
python3 - "$EXT" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ext = Path(sys.argv[1])
icons = ext / "icons"
icons.mkdir(exist_ok=True)
for size in (16, 48, 128):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # rounded square, red-ish (AI) gradient-less flat
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 5, fill=(185, 42, 42, 255))
    # white "f"
    fw = size * 0.42
    d.rectangle([size * 0.36, size * 0.20, size * 0.62, size * 0.80], fill=(255, 255, 255, 255))
    d.rectangle([size * 0.36, size * 0.20, size * 0.78, size * 0.38], fill=(255, 255, 255, 255))
    im.save(icons / f"icon{size}.png")
    print("icon", size)
PY

echo "extension build complete: $EXT"
