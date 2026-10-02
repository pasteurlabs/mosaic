#!/usr/bin/env bash
# Metadata comparison and numerical differences run only in this CPU allocation.
set -euo pipefail
campaign=$1
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/local/bin/uv run --no-project --with numpy==2.4.6 python "$campaign/dataset_audit.py" \
  --campaign /data/personal/andrinr/runner/results/mosaic/pr116-control-pilot-20261002 \
  --out "$campaign/report"
