#!/bin/bash
set -euo pipefail
campaign=$1
n=$2
dim=$3
image=/data/personal/andrinr/runner/artifacts/mosaic/phiflow-ssprk3-2895304.sqsh
for variant in baseline candidate; do
  mkdir -p "$campaign/results/${variant}-${dim}d-${n}"
  srun --container-image="$image" --container-mounts="$campaign:/validation" --container-workdir=/validation --export=ALL,PYTHONPATH=/tesseract,XLA_PYTHON_CLIENT_PREALLOCATE=false /python-env/bin/python /validation/validate.py --adapter "/validation/$variant.py" --resolution "$n" --ndim "$dim" --dt .0025 --out "/validation/results/${variant}-${dim}d-${n}"
done
