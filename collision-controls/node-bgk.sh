#!/usr/bin/env bash
# Execute inside a slurm-runner allocation. All compute and churn stay on the node.
set -euo pipefail
campaign=$1
mode=$2
cell=${3:-}
work="$WORKDIR/mosaic"
mkdir -p "$work"
(cd "$campaign" && sha256sum --check source.sha256)
tar -xf "$campaign/source.tar" -C "$work"
cp "$campaign/diagnose_xlb_corrector.py" "$work/experiments/solver_in_loop/diagnose_xlb_corrector.py"
cd "$work"
cp production.uv.lock uv.lock
export UV_PROJECT_ENVIRONMENT="$SCRATCH_DIR/mosaic-venv"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
export XLB_SUB_K_DISABLE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$SCRATCH_DIR/jax-cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
extras=(--extra dev)
if [[ "$mode" == prepare || "$mode" == train || "$mode" == evaluate ]]; then extras+=(--extra gpu); fi
/usr/local/bin/uv sync --frozen "${extras[@]}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export PYTHONPATH="$work${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$mode" == validate ]]; then
  export JAX_PLATFORMS=cpu
  ruff check experiments/solver_in_loop/final_*.py tests/test_solver_in_loop_final*.py tests/test_correction_final_report*.py
  ruff format --check experiments/solver_in_loop/final_*.py tests/test_solver_in_loop_final*.py tests/test_correction_final_report*.py
  pytest -q tests/test_solver_in_loop_final*.py tests/test_correction_final_report*.py
  exit
fi
if [[ "$mode" == assemble ]]; then
  export JAX_PLATFORMS=cpu
  python experiments/solver_in_loop/final_run.py --phase assemble --config "$campaign/configs/$cell.json" --out "$campaign/results/$cell"
  exit
fi
if [[ "$mode" == select_extension || "$mode" == select_final ]]; then
  export JAX_PLATFORMS=cpu
  python experiments/solver_in_loop/final_select.py --campaign "$campaign" --stage "$mode"
  exit
fi
if [[ "$mode" == report ]]; then
  export JAX_PLATFORMS=cpu
  python experiments/solver_in_loop/final_report.py --campaign "$campaign"
  exit
fi
config="$campaign/configs/$cell.json"
image=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$config")
port=$((10000 + SLURM_JOB_ID % 50000))
out="$WORKDIR/results"
mkdir -p "$out"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 --container-image="$image" cat /tesseract/tesseract_api.py >"$out/original_adapter.py"
python "$campaign/build_xlb_collision_candidate.py" --base "$out/original_adapter.py" --out "$out/candidate_adapter.py" --velocity-bound 8 --collision bgk
service_pid=""
publish() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
  local staged="$campaign/results/.${cell}.stage-${SLURM_JOB_ID}"
  mkdir -p "$staged"
  cp -a "$out/." "$staged/"
  tar -cf "$staged/results.tar" -C "$out" .
  if [[ -e "$campaign/results/$cell" ]]; then
    echo "Refusing to replace existing result directory: $cell" >&2
    return 1
  fi
  mv "$staged" "$campaign/results/$cell"
}
trap publish EXIT
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 \
  --container-image="$image" \
  --container-mounts="$out/candidate_adapter.py:/tesseract/tesseract_api.py:ro" \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,XLA_PYTHON_CLIENT_PREALLOCATE,XLB_SUB_K_DISABLE \
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
python experiments/solver_in_loop/diagnose_xlb_corrector.py --model-dir "/data/personal/andrinr/runner/results/mosaic/pr116-xlb-reference-refinement-20261005/results/gradient-runtime-xlb-t192" --config "$config" \
  --url "http://127.0.0.1:$port" --out "$out"
