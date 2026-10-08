# Training the periodic 3D operator

The active pipeline generates continuous XLB trajectories, trains a conditioned
Fourier operator, and exports a selected checkpoint for the Tesseract runtime.
See [model card](../../mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/MODEL.md) for the packaged model's results and limitations. The
retired fixed-resolution model and its training experiments are available in
[Git history](https://github.com/pasteurlabs/mosaic/tree/3a8c4cc/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate).

## Generate data

Prepare the manifests from the repository root in the Mosaic Python environment:

```bash
PYTHONPATH=. python tools/xlb_surrogate/operator_dataset.py \
  --output /local/coverage.json --training /local/manifest.json
```

Coverage expands the actual 3D registrations through the XLB input factory,
including resolution-dependent timestep/horizon scaling. The training plan
contains 384 parents per N=8/16/32 case and 128 per N=20/48/64 holdout case:
5,376 trajectories across 16 cases. Parents share their split across physics and
resolutions. Families include randomized TGV, ABC, solenoidal waves, mean flows,
near-zero fields, and small longitudinal perturbations. Component wave numbers
are limited to three; this distribution does not cover arbitrary spectra.

Stage source, manifest, and teacher image on **node-local storage**. Mount local
scratch at `/local`, and run inside the XLB image with the staged operator source
on `PYTHONPATH`:

```bash
python generate_operator_data.py --manifest /local/manifest.json \
  --output /local/data --teacher-api /tesseract/tesseract_api.py \
  --batch-size 4 --shard-size 16 --cache-dir /local/cache/generation
python summarize_operator_data.py --manifest /local/manifest.json \
  --results /local/data --output /local/data-report.json
```

Generation preserves native lattice populations between snapshots and matches
the teacher's collision/substep policy. Early/interior/final snapshots are checked
against the public forward implementation. Float64 teacher computation is the
default; decoded fields are stored as float32. Float32 teacher generation failed
parity on sampled stiff and recovery cases, so it is only an experimental option.

Training horizons reach 100 steps; held-out horizons reach 200/300/320 steps.
Early and late 20-step dense windows support four updates at span five; sparse
intermediate snapshots keep storage near 55.4 GiB. Adjacent saved snapshots are
not necessarily adjacent timesteps. Invalid trajectories retain their parent IDs
and are counted explicitly. The report identifies missing and partial cases.

Use repeated `--case ID` arguments to distribute cases. Within a node, workers
can share local output with `--workers K --worker I`: shard indices determine
ownership and advisory locks protect publication. Across nodes, use disjoint
shards and separate local output roots. Preserve the manifest, seed, precision,
source, teacher environment, and shard size on resume. Completed shard hashes
are verified; batch size and worker count may change. Keep shards immutable after
generation. The loader rejects mixed generation signatures and incomplete cases.

## Train and export

Run from `tools/xlb_surrogate/` (or a staged source bundle) with local data, caches, logs, and outputs:

```bash
python train_operator.py --dataset /local/data --manifest /local/manifest.json \
  --data-cache /local/mmap-cache --output /local/output/projected \
  --updates 10000 --batch-size 4 --width 24 --modes 4 --layers 4 \
  --validation-interval 1000 --validation-samples 24 --eval-stride 5 \
  --project-output --conserve-energy
python export_operator.py /local/output/projected/best.npz /local/operator_weights.npz
```

These are the packaged model's training settings. Omit `--project-output` for
the unprojected comparison. Each independent run uses one GPU; run ablations on
separate GPUs sharing the immutable node-local dataset.

The loader indexes uncompressed NPZ headers once and maps their velocity arrays
directly, without unpacking a second dataset copy. A locked metadata index is
shared by training processes. One batch is prefetched ahead; updates read only
local mapped data. Training uses shape buckets, a 1/2/4-update curriculum, mixed
spans of one and five teacher steps, and normalized field/spectral losses.

Validation uses deterministic, family-balanced parents at each case's full saved
horizon. Resolution holdouts do not participate in training or checkpoint
selection. Export `best.npz`, selected by validation; the exporter strips Adam
state and retains architecture, source hash, selected step, and training identity.
Changing model source requires a compatible artifact. The packaged numerical
model and its source hash are preserved exactly during code cleanup.

## Archive and resume

The generator and trainer inspect Linux mount information and reject network
source/data/output/cache paths. Keep imports, temporary files, JAX/CUDA caches,
program logs, and checkpoints local. Publish only quiescent outputs or immutable
data:

```bash
python operator_storage.py publish /local/data /shared/run-data.tar
python operator_storage.py publish /local/output /shared/run-training.tar
```

Publication streams one buffered tar to shared storage while computing its hash,
then atomically renames it and writes a completion record. It avoids an additional
local tar and shared checksum reread. Disposable caches are excluded. An immutable
data archive can run in the background while training reads the local copy.

Restore into new local directories:

```bash
python operator_storage.py restore /shared/run-data.tar /local/data
python operator_storage.py restore /shared/run-training.tar /local/output
```

Restore reads the shared archive once and verifies its hash locally before
extraction. Resume with the same training command and settings, adding
`--resume /local/output/projected/latest.npz`. Restore the full output bundle:
`best.npz` must accompany `latest.npz`. Optimizer state, sampling RNG, and saved
validation history are retained. A completed resume preserves its report.
SIGUSR1/TERM requests a checkpoint at the next safe update boundary. Local scratch
is disposable; abrupt node loss can lose progress since the last publication.

Use the frozen source matching the checkpoint's trainer/model hashes. The
packaged checkpoint predates subsequent utility fixes; exact resumption of that
run requires its archived source, not today's trainer.

`run_operator_pilot.sh` automates staging, generation, two concurrent training
variants, and archival on Kander Slurm. Set `OPERATOR_RUN_ROOT` to a shared folder
containing `source.tar` and `OPERATOR_TEACHER_IMAGE` to an immutable squashfs. The
bundle contains `manifest.json` and `source/` with the Python files from this
directory, the runtime `operator_model.py` and `tesseract_api.py`,
`teacher_api.py` (XLB), and `mosaic_shared/`. Keep image basenames unique: the node
cache uses them as keys. The wrapper defaults to two B200s; adjust Slurm resource
requests for other GPUs. `OPERATOR_UPDATES` and `OPERATOR_VALIDATION_SAMPLES`
override budgets. To reuse existing local data, set `OPERATOR_LOCAL_DATA` and
schedule on its node; the manifest must match the cached cases. Only phase
messages are written to the shared Slurm log.

## Evaluate a frozen checkpoint

Stage the benchmark IC module (`navier_stokes_3d_grid/ics.py`) and coverage manifest
locally. Inside the teacher image, with the operator source on `PYTHONPATH`, run:

```bash
python evaluate_operator.py --coverage /local/coverage.json \
  --ic-module /local/ics.py --weights /local/operator_weights.npz \
  --teacher-api /tesseract/tesseract_api.py --output /local/evaluation.json
```

Evaluation constructs exact benchmark ICs in float32 and compares against the
float64 teacher. It records finite status, field error, energy, timing, and VJP
checks across all registered 3D cases, including the 10,240-step horizon. Test
fields are never added to training. This direct evaluator complements the full
Mosaic benchmark harness; it does not establish complete cost, Jacobian-SVD, or
optimization results. See [operator_validation.json](operator_validation.json)
for the packaged model's measured evidence. Long-horizon accuracy, 2D/obstacle
support, and a fresh declared-runtime image build remain outstanding.
