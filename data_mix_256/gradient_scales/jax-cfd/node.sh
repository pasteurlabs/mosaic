#!/bin/bash
set -euo pipefail
campaign=$1
root=/data/personal/andrinr/runner/results/mosaic/buoyancy-data-mix-20261013/source-v1
trap 'code=$?; printf "{\"script_completed\":%s,\"exit_code\":%s}\n" "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$campaign/outcome.json"' EXIT
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_CPU_TARGET=generic JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
mkdir -p "$WORKDIR/probe"
(cd "$root" && sha256sum -c source.sha256)
tar -xf "$root/source.tar" -C "$WORKDIR/probe"
cd "$WORKDIR/probe"
uv venv "$WORKDIR/python-env"
uv pip install --python "$WORKDIR/python-env/bin/python" 'jax==0.10.0' 'equinox==0.13.8' 'scipy==1.17.0' 'jax-cfd==0.2.1' pytest . ./mosaic/mosaic_shared
export ANALYSIS_OUTPUT_DIR="$campaign"
export SOLVER_NAME=jax-cfd MOSAIC_RESULTS_DIR="$campaign/results"
mkdir -p "$MOSAIC_RESULTS_DIR"
uv pip freeze --python "$WORKDIR/python-env/bin/python" > "$campaign/python-environment.txt"
export SAVED_RESULT_DIR=/data/personal/andrinr/runner/results/mosaic/buoyancy-data-mix-20261013/jax-cfd/results/buoyancy-grid/optimization/solver_in_loop
(cd "/data/personal/andrinr/runner/results/mosaic/buoyancy-gradient-scales-20261013/script-v1" && sha256sum -c analyze.sha256)
cp "/data/personal/andrinr/runner/results/mosaic/buoyancy-gradient-scales-20261013/script-v1/analyze.py" "$WORKDIR/probe/analyze.py"
"$WORKDIR/python-env/bin/python" analyze.py 2>&1 | tee "$campaign/analysis.txt"
