#!/usr/bin/env bash
# Fetch the real-photo source dataset (imagenette2-320) used to build the
# benchmark splits. Run once before scripts/build_dataset.py on a fresh clone.
set -euo pipefail
mkdir -p /tmp/f01d-datasets
cd /tmp/f01d-datasets
if [[ ! -d imagenette2-320 ]]; then
  curl -sL -o imagenette2-320.tgz "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
  tar -xzf imagenette2-320.tgz
fi
echo "imagenette2-320 ready at /tmp/f01d-datasets/imagenette2-320"
