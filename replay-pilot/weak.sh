#!/usr/bin/env bash
set -euo pipefail
root=/data/personal/andrinr/runner/surrogate/replay-pilot-20261003
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
    --container-mounts="$root:$root,/data/personal/andrinr/runner/surrogate/mosaic-3d:/teacher-data,/data/personal/andrinr/runner/surrogate/gradient-pilot-20261003/results:/pilot-labels" \
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_TF32_OVERRIDE \
    /tesseract/entrypoint.sh env PYTHONPATH="$code:$root/source/mosaic" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 /python-env/bin/python "$@"
}
for arm in replay_vjp0001 replay_path0001; do
 if [[ -f "$root/results/$arm.metrics.json" ]]; then continue; fi
 weight=0
 labels=/pilot-labels/labels.npz
 case "$arm" in
  replay_vjp0001) weight=0.001;;
  replay_path0001) weight=0.001; labels=/pilot-labels/path-labels.npz;;
  replay_path01) weight=0.1; labels=/pilot-labels/path-labels.npz;;
 esac
 echo "START $arm $(date -u +%FT%TZ)"
 container /data/personal/andrinr/runner/artifacts/mosaic/xlb-3d-surrogate-1696556.sqsh \
  "$code/sobolev.py" train --labels "$labels" \
  --init-weights "$code/weights.npz" --output "$root/results/$arm.npz" \
  --method vjp --weight "$weight" --updates 1000 --validation-interval 200 \
  --replay-dataset /teacher-data/recovery_3d_xlb_trajectories_16k.npy \
  --replay-normalization /teacher-data/recovery_3d_autoregressive_weights_16k.metrics.json \
  --replay-validation-samples 32 2>&1 | tee "$root/results/$arm.log"
 echo "DONE $arm $(date -u +%FT%TZ)"
done
