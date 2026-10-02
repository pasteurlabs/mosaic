#!/usr/bin/env bash
# Lightweight report orchestration; invoke through slurm-runner on a CPU node.
set -euo pipefail
campaign=$1
mkdir -p "$WORKDIR/report-repo"
tar -xf "$campaign/source.tar" -C "$WORKDIR/report-repo"
cd "$WORKDIR/report-repo"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/report-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/usr/local/bin/uv sync --frozen --extra dev
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/pilot_report.py" \
  --campaign "$campaign" --out "$campaign/pilot-report"
