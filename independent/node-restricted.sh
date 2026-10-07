#!/usr/bin/env bash
set -euo pipefail
campaign=$1
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
nvidia-smi > "$campaign/hardware.txt"
env TMPDIR=/tmp srun --overlap --exact --ntasks=1 --cpus-per-task=4 --container-image=/data/personal/andrinr/runner/artifacts/mosaic/xlb-2853707.sqsh --container-mounts="$campaign:/audit,/data/personal/andrinr/runner/results/mosaic/pr116-xlb-reference-refinement-20261005/results/reference-xlb-t192-audit384-timeout-retry1:/reference:ro" --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES /python-env/bin/python /audit/restricted.py
