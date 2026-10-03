#!/usr/bin/env bash
# Execute inside a slurm-runner allocation. All compute and churn stay on the node.
set -euo pipefail
campaign=$1
mode=$2
cell=${3:-}
work="$WORKDIR/mosaic"
mkdir -p "$work"
tar -xf "$campaign/source.tar" -C "$work"
cd "$work"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/mosaic-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$SCRATCH_DIR/jax-cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
extras=(--extra dev)
if [[ "$mode" == train ]]; then extras+=(--extra gpu); fi
/usr/local/bin/uv sync --frozen "${extras[@]}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export PYTHONPATH="$work${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$mode" == validate ]]; then
  export JAX_PLATFORMS=cpu
  ruff check experiments/flow_control tests/test_flow_control*.py
  ruff format --check experiments/flow_control tests/test_flow_control*.py
  pytest -q tests/test_flow_control*.py tests/test_solver_rpc_jit.py
  python .github/scripts/validate-problem-configs.py
  python .github/scripts/validate-tesseract-configs.py
  exit
fi
if [[ "$mode" == assemble ]]; then
  export JAX_PLATFORMS=cpu
  python -m experiments.flow_control.dataset --campaign "$campaign"
  exit
fi
if [[ "$mode" == build ]]; then
  export MOSAIC_TESSERACT_SLUG="$cell" MOSAIC_JAX_BUNDLED_CUDA=1
  bash /data/personal/andrinr/tools/mosaic-tesseract-buildah-roundtrip.sh
  exit
fi
config="$campaign/configs/$cell.json"
image=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$config")
port=$((10000 + SLURM_JOB_ID % 50000))
out="$WORKDIR/results"
mkdir -p "$out"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
service_pid=""
publish() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  mkdir -p "$campaign/results/$cell"
  tar -cf "$campaign/results/$cell/results.tar.part" -C "$out" .
  mv "$campaign/results/$cell/results.tar.part" "$campaign/results/$cell/results.tar"
}
trap publish EXIT
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 \
  --container-image="$image" \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,XLA_PYTHON_CLIENT_PREALLOCATE \
  /tesseract/entrypoint.sh /python-env/bin/tesseract-runtime \
  --output-format json+base64 serve --host 0.0.0.0 --port "$port" \
  >"$out/service.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 300); do
  if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then break; fi
  if ! kill -0 "$service_pid" 2>/dev/null; then cat "$out/service.log"; exit 1; fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$port/health"
python experiments/flow_control/run.py --config "$config" \
  --url "http://127.0.0.1:$port" --out "$out"
