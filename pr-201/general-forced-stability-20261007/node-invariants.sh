#!/usr/bin/env bash
set -euo pipefail
campaign=$1
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
srun --overlap --exact --ntasks=1 --cpus-per-task=4 --container-image=/data/personal/andrinr/runner/artifacts/mosaic/warp-ns-2856606.sqsh --container-mounts="$campaign:/audit" --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES /bin/bash -c '/python-env/bin/python /audit/check_warp_fft.py --api /audit/candidate.py && /python-env/bin/python /audit/check_warp_invariants.py --api /audit/candidate.py --out /audit/invariants.json'
