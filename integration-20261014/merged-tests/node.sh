#!/bin/bash
set -euo pipefail
campaign=$1
root=/data/personal/andrinr/runner/results/mosaic/buoyancy-integration-20261014/source-merged
trap 'code=$?; printf "{\"script_completed\":%s,\"exit_code\":%s}\n" "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$campaign/outcome.json"' EXIT
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_CPU_TARGET=generic JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
mkdir -p "$WORKDIR/probe"
(cd "$root" && sha256sum -c source.sha256)
tar -xf "$root/source.tar" -C "$WORKDIR/probe"
cd "$WORKDIR/probe"
uv venv "$WORKDIR/python-env"
uv pip install --python "$WORKDIR/python-env/bin/python" 'jax==0.10.0' 'equinox==0.13.8' 'jax-cfd==0.2.1' pytest . ./mosaic/mosaic_shared
export SOLVER_NAME=jax-cfd MOSAIC_RESULTS_DIR="$campaign/results"
mkdir -p "$MOSAIC_RESULTS_DIR"
uv pip freeze --python "$WORKDIR/python-env/bin/python" > "$campaign/python-environment.txt"
"$WORKDIR/python-env/bin/python" -m pytest -q tests/test_buoyancy_schema.py tests/test_buoyancy_training.py tests/test_buoyancy_status.py tests/test_buoyancy_plots.py tests/test_solver_alias.py tests/test_charts.py tests/core/test_plot_views.py tests/test_problem_configs.py --junitxml="$campaign/junit.xml" 2>&1 | tee "$campaign/tests.txt"

export CAMPAIGN="$campaign"
"$WORKDIR/python-env/bin/python" /data/personal/andrinr/runner/results/mosaic/buoyancy-integration-20261014/render_saved.py 2>&1 | tee "$campaign/render.txt"
