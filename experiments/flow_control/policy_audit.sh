#!/usr/bin/env bash
set -euo pipefail
audit=$1
campaign=$2
work="$WORKDIR/mosaic"
mkdir -p "$work"
tar -xf "$campaign/source.tar" -C "$work"
cd "$work"
cp production.uv.lock uv.lock
cp "$audit/policy_audit.py" experiments/flow_control/policy_audit.py
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/mosaic-venv" UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/local/bin/uv sync --frozen --extra dev
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH" PYTHONPATH="$work"
python experiments/flow_control/policy_audit.py --campaign "$campaign" --out "$audit/audit.json"
