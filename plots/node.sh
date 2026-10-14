#!/bin/bash
set -euo pipefail
campaign=$1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv venv "$WORKDIR/plot-env"
uv pip install --python "$WORKDIR/plot-env/bin/python" numpy==2.4.4 matplotlib==3.10.8
"$WORKDIR/plot-env/bin/python" "$campaign/analyze.py" /data/personal/andrinr/runner/results/mosaic/fd-path-average-evaluation-20261014/full-v2/results/evaluation.json --output "$campaign/artifacts" 2>&1 | tee "$campaign/analysis.txt"
