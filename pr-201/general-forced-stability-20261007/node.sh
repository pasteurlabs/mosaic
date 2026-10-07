#!/usr/bin/env bash
set -euo pipefail
campaign=$1
variant=$2
n=$3
factor=$4
ndim=${5:-3}
cell="$variant-${ndim}d-$n"
mkdir -p "$campaign/results/$cell"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
nvidia-smi > "$campaign/results/$cell/hardware.txt"
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 --container-image=/data/personal/andrinr/runner/artifacts/mosaic/warp-ns-2856606.sqsh --container-mounts="$campaign:/audit" --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES /python-env/bin/python /audit/check_warp_stability.py --api "/audit/$variant.py" --out "/audit/results/$cell" --n "$n" --factor "$factor" --ndim "$ndim" --burn-time "${6:-75}"
