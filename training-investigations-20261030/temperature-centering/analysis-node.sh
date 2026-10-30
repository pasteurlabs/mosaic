#!/bin/bash
set -euo pipefail
campaign=$1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$WORKDIR/source"
tar -xf "$campaign/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/analyze.py"
