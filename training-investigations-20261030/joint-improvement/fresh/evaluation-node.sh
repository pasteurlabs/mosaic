#!/bin/bash
set -euo pipefail
campaign=$1
selection=$2
selection_sha=$3
mkdir -p "$WORKDIR/source"
(cd "$campaign" && sha256sum -c source.sha256 && sha256sum -c evaluation-scripts.sha256)
tar -xf "$campaign/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" 'jax[cuda12]==0.10.0' 'equinox==0.13.8' 'jax-cfd==0.2.1' 'phiflow==3.4.0' 'phiml==1.16.1' . ./mosaic/mosaic_shared
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$campaign/evaluation-environment.txt"
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_COMPILATION_CACHE=false
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/evaluate_frozen.py" --selection "$selection" --selection-sha256 "$selection_sha"
