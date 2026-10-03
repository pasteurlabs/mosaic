#!/usr/bin/env bash
# CPU report only; all solver optimization remains in GPU allocations.
set -euo pipefail
campaign=$1
work="$WORKDIR/report-repo"
mkdir -p "$work"
tar -xf "$campaign/source.tar" -C "$work"
cd "$work"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/report-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/usr/local/bin/uv sync --frozen --extra dev
"$UV_PROJECT_ENVIRONMENT/bin/python" experiments/flow_control/direct_report.py --campaign "$campaign" --out "$campaign/direct-report"
