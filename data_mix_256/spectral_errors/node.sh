#!/bin/bash
set -euo pipefail
campaign=$1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv venv "$WORKDIR/spectral-env"
uv pip install --python "$WORKDIR/spectral-env/bin/python" numpy==2.4.4 scipy==1.17.0 matplotlib==3.10.8
"$WORKDIR/spectral-env/bin/python" "$campaign/analyze.py" 2>&1 | tee "$campaign/analysis.txt"
