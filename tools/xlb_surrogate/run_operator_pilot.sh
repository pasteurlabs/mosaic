#!/usr/bin/env bash
#SBATCH --job-name=m197-local-training
#SBATCH --partition=dev
#SBATCH --qos=b200-pool
#SBATCH --account=research
#SBATCH --gres=gpu:b200:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%j.out
set -euo pipefail
. /etc/kander/job-env.sh
root="${OPERATOR_RUN_ROOT:?Set the shared archive root containing source.tar}"
teacher_image="${OPERATOR_TEACHER_IMAGE:?Set the immutable teacher squashfs path}"
job_scratch="${SCRATCH_DIR:?}/m197"
mkdir -p "$job_scratch" "${SCRATCH_DIR}/tmp"
# Source/manifest are a single shared read, and imports thereafter are local.
cp "$root/source.tar" "$job_scratch/source.tar"
tar -xf "$job_scratch/source.tar" -C "$job_scratch"
python3 "$job_scratch/source/operator_storage.py" check "$job_scratch"
image_cache="/cache/users/$(id -u)/mosaic-images"
mkdir -p "$image_cache"
python3 "$job_scratch/source/operator_storage.py" check "$image_cache"
image="$image_cache/$(basename "$teacher_image")"
(
  flock 9
  if [[ ! -s "$image" || ! -s "$image.sha256" ]]; then
    cp "$teacher_image" "$image.partial"
    mv "$image.partial" "$image"
    sha256sum "$image" > "$image.sha256"
  fi
) 9> "$image_cache/.stage.lock"
read -r image_digest image_cached_name < "$image.sha256"
export MOSAIC_TEACHER_IMAGE_ID="$image_digest"
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_TF32_OVERRIDE=0
mkdir -p "$job_scratch/data" "$job_scratch/output" "$job_scratch/cache" "$job_scratch/tmp"
mkdir -p "$root/artifacts"
data_root="${OPERATOR_LOCAL_DATA:-$job_scratch/data}"
python3 "$job_scratch/source/operator_storage.py" check "$data_root"
train_pids=()
archive_pid=""
stop_requested=0
signal_trainers() {
  stop_requested=1
  scancel --signal=USR1 "$SLURM_JOB_ID" || true
}
trap signal_trainers USR1 TERM
run_container() {
  srun --exclusive --exact --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 \
    --container-image="$image" \
    --container-mounts="$job_scratch:/work,$data_root:/work/data,$job_scratch/source/teacher_api.py:/tesseract/tesseract_api.py" \
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NVIDIA_TF32_OVERRIDE,MOSAIC_TEACHER_IMAGE_ID \
    /tesseract/entrypoint.sh env PYTHONPATH=/work/source PYTHONDONTWRITEBYTECODE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=1 \
    XDG_CACHE_HOME=/work/cache/xdg CUDA_CACHE_PATH=/work/cache/cuda TMPDIR=/work/tmp \
    /python-env/bin/python "$@"
}
# The only persistent log is this small phase record. GPU/program logs are local.
printf 'LOCAL_ROOT=%s\n' "$job_scratch"
if [[ -z "${OPERATOR_LOCAL_DATA:-}" ]]; then
run_container /work/source/generate_operator_data.py --manifest /work/manifest.json \
  --output /work/data --batch-size 4 --shard-size 16 --cache-dir /work/cache/generation \
  > "$job_scratch/output/generation.log" 2>&1 &
generation_pid=$!
set +e
wait "$generation_pid"
generation_status=$?
set -e
if [[ "$generation_status" != 0 || "$stop_requested" != 0 ]]; then
  wait "$generation_pid" || true
  python3 "$job_scratch/source/operator_storage.py" publish "$job_scratch/output" "$root/artifacts/$SLURM_JOB_ID-failure.tar"
  exit 1
fi
printf 'GENERATION_COMPLETE\n'
# Data are now immutable. One sequential archive transfer runs in the background
# while training reads only the local copy. No shared destination tree scans.
python3 "$job_scratch/source/operator_storage.py" publish "$data_root" "$root/artifacts/$SLURM_JOB_ID-data.tar" \
  > "$job_scratch/output/data-export.json" 2>&1 &
archive_pid=$!
else
  printf 'REUSING_LOCAL_DATA=%s\n' "$data_root"
fi
for variant in unprojected projected; do
  extra=()
  if [[ "$variant" == projected ]]; then extra+=(--project-output); fi
  run_container /work/source/train_operator.py --dataset /work/data --manifest /work/manifest.json --data-cache /work/mmap-cache \
    --output "/work/output/$variant" --conserve-energy --updates "${OPERATOR_UPDATES:-10000}" --validation-interval 1000 --eval-stride 5 --validation-samples "${OPERATOR_VALIDATION_SAMPLES:-24}" \
    --batch-size 4 --width 24 --modes 4 --layers 4 "${extra[@]}" \
    > "$job_scratch/output/$variant.log" 2>&1 &
  train_pids+=("$!")
done
status=0
for task_pid in "${train_pids[@]}"; do
  set +e
  wait "$task_pid"
  task_status=$?
  if [[ "$task_status" -gt 128 && "$stop_requested" == 1 ]]; then
    wait "$task_pid"
    task_status=$?
  fi
  set -e
  if [[ "$task_status" != 0 ]]; then status=1; fi
done
if [[ -n "$archive_pid" ]]; then wait "$archive_pid" || status=1; fi
# One final checkpoint/report/log bundle, excluding disposable compilation caches.
python3 "$job_scratch/source/operator_storage.py" publish "$job_scratch/output" "$root/artifacts/$SLURM_JOB_ID-training.tar"
printf 'TRAINING_COMPLETE status=%s interrupted=%s\n' "$status" "$stop_requested"
exit "$status"
