#!/bin/bash
set -euo pipefail
campaign=$1
solver=$2
output="$campaign/$solver"
mkdir -p "$output" "$WORKDIR/source"
finish() { code=$?; trap - EXIT; printf '{"script_completed":%s,"exit_code":%s}\n' "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$output/job-outcome.json"; exit "$code"; }
trap finish EXIT
(cd "$campaign" && sha256sum -c source.sha256 && sha256sum -c scripts.sha256)
tar -xf "$campaign/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" 'jax[cuda12]==0.10.0' 'equinox==0.13.8' 'jax-cfd==0.2.1' 'phiflow==3.4.0' 'phiml==1.16.1' . ./mosaic/mosaic_shared
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$output/environment.txt"
export JAX_ENABLE_COMPILATION_CACHE=false JAX_COMPILATION_CACHE_DIR="$WORKDIR/isolated-fit-cache"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/driver.py" --solver "$solver" --output "$output" 2>&1 | tee "$output/client.log"
