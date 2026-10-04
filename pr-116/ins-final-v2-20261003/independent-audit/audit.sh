#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/local/bin/uv run --no-project --with numpy==2.4.6 python /data/personal/andrinr/runner/results/mosaic/pr116-ins-independent-audit-20261004/audit.py --campaign /data/personal/andrinr/runner/results/mosaic/pr116-ins-final-v2-20261003 --out /data/personal/andrinr/runner/results/mosaic/pr116-ins-independent-audit-20261004/independent-audit.json
