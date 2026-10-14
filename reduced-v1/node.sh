#!/bin/bash
set -euo pipefail
campaign=$1
trap 'code=$?; printf "{\"script_completed\":%s,\"exit_code\":%s}\n" "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$campaign/outcome.json"' EXIT
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$WORKDIR/probe"
(cd /data/personal/andrinr/runner/results/mosaic/fd-path-average-evaluation-20261014/source-main && sha256sum -c source.sha256)
tar -xf /data/personal/andrinr/runner/results/mosaic/fd-path-average-evaluation-20261014/source-main/source.tar -C "$WORKDIR/probe"
(cd /data/personal/andrinr/runner/results/mosaic/fd-path-average-evaluation-20261014/script-v1 && sha256sum -c scripts.sha256)
cp /data/personal/andrinr/runner/results/mosaic/fd-path-average-evaluation-20261014/script-v1/evaluate.py "$WORKDIR/probe/"
cd "$WORKDIR/probe"
uv venv "$WORKDIR/python-env"
uv pip install --python "$WORKDIR/python-env/bin/python" 'jax==0.10.0' 'equinox==0.13.8' 'jax-cfd==0.2.1' 'scipy==1.17.0' . ./mosaic/mosaic_shared
uv pip freeze --python "$WORKDIR/python-env/bin/python" > "$campaign/python-environment.txt"
"$WORKDIR/python-env/bin/python" evaluate.py --mode reduced --output "$campaign/results" 2>&1 | tee "$campaign/evaluation.txt"
