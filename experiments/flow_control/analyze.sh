#!/usr/bin/env bash
set -euo pipefail
campaign=$1
work="$WORKDIR/mosaic"
mkdir -p "$work"
tar -xf "$campaign/source.tar" -C "$work"
cd "$work"
cp production.uv.lock uv.lock
cp "$campaign/report.py" experiments/flow_control/report.py
cp "$campaign/plots.py" experiments/flow_control/plots.py
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/mosaic-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/local/bin/uv sync --frozen --extra dev
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH" PYTHONPATH="$work"
ruff check experiments/flow_control/report.py experiments/flow_control/plots.py
python experiments/flow_control/report.py --campaign "$campaign" --out "$campaign/report"
