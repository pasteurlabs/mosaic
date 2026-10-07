#!/bin/bash
set -euo pipefail
c=$1
srun --container-image=/data/personal/andrinr/runner/artifacts/mosaic/phiflow-ssprk3-2895304.sqsh --container-mounts="$c:/validation,/home/andrinr/mosaic/.venv/lib/python3.12/site-packages:/hostsite" --container-workdir=/validation --export=ALL,PYTHONPATH=/tesseract,XLA_PYTHON_CLIENT_PREALLOCATE=false /python-env/bin/python /validation/pytest_runner.py
