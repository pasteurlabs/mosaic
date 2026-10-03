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
if [[ ! -f "$root/results/labels.json" ]]; then
  container /data/personal/andrinr/runner/artifacts/mosaic/xlb-2853707.sqsh \
    "$code/sobolev.py" generate \
    --teacher-api "$root/source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py" \
    --dataset /teacher-data/recovery_3d_xlb_trajectories_16k.npy \
    --output "$root/results/labels.npz" --train-samples 128 --validation-samples 32 \
    --secant-step 0.01 2>&1 | tee "$root/results/labels.log"
fi
for arm in field vjp001 vjp01 vjp1 secant01 linear_vjp01; do
  if [[ -f "$root/results/$arm.metrics.json" ]]; then continue; fi
  method=vjp
  weight=0
  extra=()
  case "$arm" in
    vjp001) weight=0.01;;
    vjp01) weight=0.1;;
    vjp1) weight=1;;
    secant01) method=secant; weight=0.1;;
    linear_vjp01) weight=0.1; extra=(--linear-correction);;
  esac
  echo "START $arm $(date -u +%FT%TZ)"
  container /data/personal/andrinr/runner/artifacts/mosaic/xlb-3d-surrogate-1696556.sqsh \
    "$code/sobolev.py" train --labels "$root/results/labels.npz" \
    --init-weights "$code/weights.npz" --output "$root/results/$arm.npz" \
    --method "$method" --weight "$weight" --updates 500 --validation-interval 100 \
    "${extra[@]}" 2>&1 | tee "$root/results/$arm.log"
  echo "DONE $arm $(date -u +%FT%TZ)"
done
