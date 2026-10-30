#!/bin/bash
set -euo pipefail
campaign=$1
base=/data/personal/andrinr/runner/results/mosaic/buoyancy-joint-improvement-20261030
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$WORKDIR/source"
(cd "$base" && sha256sum -c source.sha256)
tar -xf "$base/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
git apply "$campaign/changes.patch"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$campaign/environment.txt"
uvx --from ruff==0.17.2 ruff check mosaic/benchmarks/problems/buoyancy_grid/corrector.py mosaic/benchmarks/problems/buoyancy_grid/training.py tests/test_buoyancy_training.py
uvx --from ruff==0.17.2 ruff format --check mosaic/benchmarks/problems/buoyancy_grid/corrector.py mosaic/benchmarks/problems/buoyancy_grid/training.py tests/test_buoyancy_training.py
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest -q tests/test_buoyancy_training.py tests/test_buoyancy_training_jit.py tests/test_buoyancy_plots.py tests/test_buoyancy_status.py --junitxml="$campaign/junit.xml" 2>&1 | tee "$campaign/tests.txt"
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/parity.py"
