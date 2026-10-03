#!/usr/bin/env bash
set -euo pipefail
job_root=/data/personal/andrinr/runner/surrogate/standalone-20261003
source_root="$job_root/source"
export MOSAIC_RESULTS_DIR="$job_root/results-${SLURM_JOB_ID}"
mkdir -p "$MOSAIC_RESULTS_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_TF32_OVERRIDE=0
export MLFLOW_DISABLE_AGENT_HINT=1
port_base="$((10000 + SLURM_JOB_ID % 20000 * 2))"
export MATCHED_URLS="{\"XLB\":\"http://127.0.0.1:${port_base}\",\"XLB 3D surrogate\":\"http://127.0.0.1:$((port_base+1))\"}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$MOSAIC_RESULTS_DIR/gpu.txt"
for slug in xlb xlb-3d-surrogate; do
  if [[ "$slug" == xlb ]]; then
    image=/data/personal/andrinr/runner/artifacts/mosaic/xlb-2853707.sqsh
    port="$port_base"
    mounts="$source_root:/workspace"
  else
    image=/data/personal/andrinr/runner/artifacts/mosaic/xlb-3d-surrogate-1696556.sqsh
    port="$((port_base+1))"
    mounts="$source_root:/workspace,$source_root/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz:/tesseract/weights.npz"
  fi
  mounts="$mounts,$source_root/mosaic/tesseracts/navier-stokes-grid/$slug/tesseract_api.py:/tesseract/tesseract_api.py"
  if [[ "$slug" == xlb-3d-surrogate ]]; then
    mounts="$mounts,$source_root/mosaic/tesseracts/navier-stokes-grid/$slug/surrogate_model.py:/tesseract/surrogate_model.py"
  fi
  env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 \
    --container-image="$image" --container-mounts="$mounts" \
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_TF32_OVERRIDE \
    /tesseract/entrypoint.sh env PYTHONPATH=/workspace/mosaic XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /python-env/bin/tesseract-runtime \
    --output-format json+base64 serve --host 0.0.0.0 --port "$port" \
    > "$MOSAIC_RESULTS_DIR/$slug-service.log" 2>&1 &
  pids+=("$!")
  ready=0
  for attempt in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "${pids[-1]}" 2>/dev/null; then cat "$MOSAIC_RESULTS_DIR/$slug-service.log"; exit 1; fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then cat "$MOSAIC_RESULTS_DIR/$slug-service.log"; exit 1; fi
  echo "READY $slug"
done
python "$job_root/client.py" 2>&1 | tee "$MOSAIC_RESULTS_DIR/benchmark.log"
