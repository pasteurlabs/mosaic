#!/bin/bash
set -euo pipefail
campaign=$1
solver=$2
mode=$3
export MODEL_SEED=$4
export EXPERIMENT_BUDGET=$5
export PILOT_ARM="$mode"
output="$campaign/$solver/$EXPERIMENT_BUDGET/seed-$MODEL_SEED/$mode"
mkdir -p "$output" "$WORKDIR/source"
server_pid=''
finish() {
 code=$?
 trap - EXIT
 if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
 if [[ -f "$WORKDIR/server.log" ]]; then cp "$WORKDIR/server.log" "$output/server.log"; fi
 printf '{"script_completed":%s,"exit_code":%s}\n' "$([[ $code == 0 ]] && echo true || echo false)" "$code" > "$output/job-outcome.json"
 exit "$code"
}
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
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TESSERACT_API_PATH="$PWD/mosaic/tesseracts/buoyancy-grid/$solver/tesseract_api.py"
port=$((20000 + SLURM_JOB_ID % 20000))
JAX_PLATFORMS=cuda,cpu "$UV_PROJECT_ENVIRONMENT/bin/tesseract-runtime" serve --host 127.0.0.1 --port "$port" > "$WORKDIR/server.log" 2>&1 &
server_pid=$!
export PROFILE_PORT=$port
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import os,socket,time
for _ in range(120):
 try:
  with socket.create_connection(('127.0.0.1',int(os.environ['PROFILE_PORT'])),timeout=1):break
 except OSError:time.sleep(1)
else:raise RuntimeError('HTTP server did not start')
PY
args=()
if [[ "$mode" == full ]]; then args+=(--full); fi
JAX_PLATFORMS=cuda,cpu "$UV_PROJECT_ENVIRONMENT/bin/python" "$campaign/${EXPERIMENT_BUDGET}_driver.py" --solver "$solver" --url "http://127.0.0.1:$port" --output "$output" "${args[@]}" 2>&1 | tee "$output/client.log"
