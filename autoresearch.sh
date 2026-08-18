#!/usr/bin/env bash
# f01d benchmark: AI-image detector balanced accuracy at the 0.65 confidence
# threshold, measured on the committed held-out eval set (data/eval/).
#
# Deterministic: the eval images, feature extractor, and model are fixed and
# committed; no network, no RNG, no time-of-day dependencies.
#
# Prints the primary metric and secondaries as `METRIC <name>=<value>` lines.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f data/manifests/dataset.csv ]]; then
  echo "dataset manifest missing; run scripts/build_dataset.py (see scripts/fetch_real.sh)" >&2
  exit 1
fi
if [[ ! -f detector/model.json ]]; then
  echo "detector/model.json missing; training baseline (deterministic)..." >&2
  python3 detector/train.py
fi

exec python3 bench/evaluate.py
