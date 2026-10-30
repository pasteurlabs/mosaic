#!/bin/bash
set -euo pipefail
campaign=$1
mode=${2:-full}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$WORKDIR/source"
(cd "$campaign" && sha256sum -c source.sha256)
(cd "$campaign/render" && sha256sum -c scripts.sha256)
tar -xf "$campaign/source.tar" -C "$WORKDIR/source"
cd "$WORKDIR/source"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$WORKDIR/python-env"
uv sync --frozen --extra dev --python 3.12
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$campaign/render/environment-$mode.txt"
args=()
if [[ "$mode" == preview ]]; then args+=(--allow-subset); fi
"$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/render/render.py" --campaign "$campaign" "${args[@]}"
