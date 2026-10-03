#!/usr/bin/env bash
set -euo pipefail
root=/data/personal/andrinr/runner/surrogate/gradient-pilot-20261003
code="$root/source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_TF32_OVERRIDE=0
mkdir -p "$root/results"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$root/results/gpu-${SLURM_JOB_ID}.txt"
container() {
  local image="$1"
  shift
  env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=8 \
    --container-image="$image" \
    --container-mounts="$root:$root,/data/personal/andrinr/runner/surrogate/mosaic-3d:/teacher-data" \
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_TF32_OVERRIDE \
    /tesseract/entrypoint.sh env PYTHONPATH="$code:$root/source/mosaic" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 /python-env/bin/python "$@"
}
container /data/personal/andrinr/runner/artifacts/mosaic/xlb-2853707.sqsh \
  "$root/evaluate.py" --root "$root" 2>&1 | tee "$root/results/evaluate.log"
