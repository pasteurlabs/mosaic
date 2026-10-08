# Offline training recipes

Run these commands from this Tesseract directory, with Python, JAX, NumPy and
SciPy available. Data generation additionally needs the XLB teacher environment
and its `mosaic_shared` dependency. Mount this directory into the appropriate
solver image and put it on `PYTHONPATH`; training scripts are not packaged into
the inference image. GPU allocation/container launch is site-specific (the
recorded pilots used the preemptible `nice` queue).

Use a writable output directory, and keep candidate checkpoints separate from
`weights.npz`. In the examples, `/surrogate-output` stores data and results and
`/tesseract/tesseract_api.py` refers to the **XLB teacher** when generating data.

## Trajectory training

```bash
python generate_trajectories.py --teacher-api /tesseract/tesseract_api.py \
  --output /surrogate-output/trajectories.npy --samples 16384
python train.py --dataset /surrogate-output/trajectories.npy \
  --weights /surrogate-output/curriculum.npz
```

The generator evolves native KBC D3Q27 lattice populations continuously for
100 steps, decoding velocity every five steps. It writes train/validation/test
split metadata beside the dataset and excludes benchmark IC seeds 0/1/2.
`training_data.py` samples divergence-free fields with varied spectra and
amplitudes from zero to 1.25. Velocity alone does not encode XLB's hidden lattice
state; do not restart the teacher from decoded velocities between snapshots.

`train.py --help` exposes the unroll curriculum, learning rate, batch size,
validation cadence and initialization. For additional full-horizon training,
use `--init-weights /surrogate-output/curriculum.npz --curriculum 20:4800`
and a distinct `--weights` path. This is a recipe, not a claim of bitwise
reproduction of the shipped checkpoint. Preserve the generated metrics JSON,
which records normalization, split/data hashes, settings and validation results.
The original dataset SHA-256 is
`4836fba4e6a8524af7a552c5977721118e726afa21db9a9f4d0b612a879a0005`.

## Derivative labels

```bash
python sobolev.py generate --teacher-api /tesseract/tesseract_api.py \
  --dataset /surrogate-output/trajectories.npy \
  --output /surrogate-output/labels.npz --secant-step 0.01
python generate_path_labels.py --teacher-api /tesseract/tesseract_api.py \
  --dataset /surrogate-output/trajectories.npy --init-weights weights.npz \
  --output /surrogate-output/path-labels.npz
```

Random labels use 128 training and 32 validation ICs, amplitude factors
0/0.1/0.5/1, and one fixed random unit output cotangent per state. Secant labels
use projected random input directions and perturbation RMS
`0.01 * max(input RMS, 0.01)`; sweep `--secant-step` into separate output files.
Path labels sample baseline L-BFGS-B iterates for eight training and two validation
ICs, alternating random and normalized teacher-residual cotangents. Splits follow
the source IC, and test trajectories are excluded. Teacher computations use
float64; saved labels use float32.

VJP supervision matches the student's and teacher's derivatives using the same
output cotangent. Training requires mixed input/parameter second derivatives
through the native JAX student, but only cached **first derivatives** from XLB.
Secant supervision needs only first-order parameter gradients. The original
spectral trajectory loss weights spatial frequencies; it is distinct from this
input-Jacobian supervision.

## Terminal-only pilot

```bash
for weight in 0 0.01 0.1 1; do
  python sobolev.py train --labels /surrogate-output/labels.npz \
    --init-weights weights.npz --output "/surrogate-output/vjp-$weight.npz" \
    --weight "$weight" --updates 500 --validation-interval 100
done
```

The eight recorded arms use the four weights above, plus these variations of the
same command, each with a distinct output path:

| Labels            | Weight | Extra option          |
| ----------------- | -----: | --------------------- |
| `labels.npz`      |    0.1 | `--method secant`     |
| `labels.npz`      |    0.1 | `--linear-correction` |
| `path-labels.npz` |      0 | —                     |
| `path-labels.npz` |    0.1 | —                     |

All arms start from the packaged checkpoint, using Adam at 1e-5, batch size one,
seed 20261003 and 500 updates. Weight zero is a **terminal-field** control, not
the original trajectory objective; it still evaluates VJPs. The optional
zero-initialized linear branch adds real 3×3 channel mixing per squared-frequency
shell (`w_linear`) and permits a learned derivative at zero. Original checkpoints
retain their behavior.

## Trajectory-replay pilot

```bash
python sobolev.py train --labels /surrogate-output/labels.npz \
  --init-weights weights.npz --output /surrogate-output/replay-vjp-0.001.npz \
  --replay-dataset /surrogate-output/trajectories.npy \
  --replay-normalization /surrogate-output/original-training.metrics.json \
  --weight 0.001 --updates 1000 --validation-interval 200
```

Use the metrics JSON from the **original training run** for output normalization;
retain the checkpoint's input/correction normalization. The six recorded arms
use random labels at weights 0, 0.001 and 0.1, and path labels at 0.001, 0.01 and
0.1, all with 1,000 updates and otherwise identical settings.

Each update samples an independent full training trajectory. The shared
`train.py:trajectory_loss` retains the time-weighted field, 0.02 spectral and
0.25 terminal terms, plus 1e-8 parameter L2 in the training objective. Cached-label
terminal field loss is diagnostic only in replay mode. All arms share the replay
RNG stream; checkpoint selection uses 32 fixed validation trajectories and the
cached derivative validation split. This preserves the original loss, not its
optimizer schedule or curriculum.

## Results

Checkpoint selection within an arm uses validation field/trajectory loss plus
weighted derivative loss. Scores across different weights are not comparable.
The recorded candidate comparison uses three common validation cases, fresh
cotangents, and 100-iteration native SciPy L-BFGS-B recovery from zero. Final
selection uses mean full-space XLB-target IC error. These are separate from the
registered self-target benchmark; test cases were not used for selection.

None of the eight terminal-only or six replay arms beat the original checkpoint.
The closest replay candidate had 4.24% forward error and 35.12% XLB-target IC
error versus 4.22% and 35.03% originally. One training seed and three cases do not
support a significance claim for that near tie. The original checkpoint remains
packaged. Replay controls reused identical-model validation results; the final
benchmark disposition verified weights/model/API hashes and reused the completed
archived benchmark run rather than claiming a new measurement. Its temporary
forward/gradient/cost and unconstrained recovery registrations are not part of
the final solver integration; only existing projected recovery is admitted.

![Eight terminal-only training arms: forward, VJP and recovery errors](https://raw.githubusercontent.com/pasteurlabs/mosaic/37c9d6b/gradient-pilot/pilot-comparison.png)

![Six trajectory-replay training arms: forward, VJP and recovery errors](https://raw.githubusercontent.com/pasteurlabs/mosaic/37c9d6b/replay-pilot/pilot-comparison.png)

Fourier-restricted recovery reduced the baseline surrogate's validation IC error
from 35.03% to 26.40%, but increased XLB's from 6.76% to 21.66%. A separate audit
of 32 orthonormal low-frequency directions at one IC found better condition
numbers could coexist with large teacher-Jacobian errors. These are restricted
probes, not full Jacobian spectra. Neither conditioning nor training loss alone
is an adequate criterion for replacing the checkpoint.

[Terminal-pilot results, evaluation/plot scripts and provenance](https://github.com/pasteurlabs/mosaic/tree/37c9d6b/gradient-pilot)
and [replay-pilot artifacts](https://github.com/pasteurlabs/mosaic/tree/37c9d6b/replay-pilot)
contain per-case metrics, optimizer termination details, training curves, and job
records. Large datasets/checkpoints and site-specific submission wrappers remain
external. To reproduce the figures, run the archived `summarize.py` for the
matched benchmark or each pilot's `plot.py` against its archived JSON results.

## General operator data generation

The fixed-task recipes above reproduce the original checkpoint. The following
pipeline prepares **new periodic 3D operator data** across resolutions and physics;
it does not change the packaged model, its supported inputs, or benchmark eligibility.

Prepare a coverage manifest from the repository root, in the normal Mosaic
Python environment:

```bash
PYTHONPATH=. python mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/operator_dataset.py \
  --output /surrogate-output/coverage.json --pilot /surrogate-output/pilot.json
```

The coverage manifest expands the existing registrations and calls the actual
XLB input factory, including `lbm_N_base` time-step and horizon scaling. The pilot
selects all six grid sizes (8/16/20/32/48/64), the stiff finite-difference setting,
and the recovery setting, with 24 trajectories per case and at most 100 effective
teacher steps. The cap is explicit: this pilot does not validate the 10,240-step
horizon-limit experiment or every coverage combination.

Inside the XLB image, with this source directory on `PYTHONPATH`:

```bash
python generate_operator_data.py --manifest /surrogate-output/pilot.json \
  --output /surrogate-output/operator-data --teacher-api /tesseract/tesseract_api.py \
  --batch-size 4 --shard-size 8 --cache-dir /surrogate-output/jax-cache
```

Use repeated `--case CASE_ID` to partition cases across GPU jobs. Alternatively,
launch workers with `--workers K --worker I` for `I=0..K-1`; shards are assigned
by index modulo K. Workers can share the output directory. Keep the manifest,
seed, teacher/source/environment, precision, and shard size fixed when resuming.
Batch size and worker count may change. Each worker holds advisory locks until
its writes finish, and publishes the completion JSON only after the NPZ is
flushed, atomically renamed, and hashed. Completed shards are hash-checked on
resume; source mismatches and corruption fail explicitly. Output directories
must support advisory file locking and atomic rename.

Generation uses the API's collision/substep policy and preserves the native
lattice populations across every snapshot. Early and late dense windows support
short-rollout training; sparse intermediate snapshots support longer targets.
Always use the saved `snapshot_steps` to construct training windows: adjacent
stored snapshots are not necessarily adjacent teacher steps. Only use rows
where `valid` is true; nonfinite trajectories are retained and counted, with
parent IDs, rather than silently disappearing from the dataset.

Parent IDs and train/validation/test splits are independent of resolution,
physics, batching, and worker count. A parent is a continuous, low-frequency
field sampled on each grid, not a new random array at every resolution. Families
include randomized TGV, ABC, random solenoidal waves, mean flows, near-zero
fields, and small longitudinal perturbations. All amplitude/physics/resolution
variants of a parent must retain its split. The pilot's random fields have
component wave numbers at most three; a production distribution needs additional
spectral bandwidth and independent held-out checks before claiming generality.

Each case checks early/interior/final snapshots against the same `xlb_fwd`
function used by public `apply`, at matched precision and solver policy. It also
compares the first padded batch's terminal fields against float64 XLB. This is a
small precision probe, not a certification of all ICs or long horizons. Float64 is the default teacher precision. Use
`--precision float32` only for an explicitly validated precision experiment in
a separate subdirectory; saved velocity arrays are float32 in either case.
Set `MOSAIC_TEACHER_IMAGE_ID` to the immutable image identifier when launching;
metadata also records source hashes, JAX/XLB versions, and teacher policy flags.

Summarize metadata in an environment with NumPy:

```bash
python summarize_operator_data.py --manifest /surrogate-output/pilot.json \
  --results /surrogate-output/operator-data --output /surrogate-output/report.json \
  --target-samples 1024
```

The report retains missing/partial cases and nonfinite precision references. It
separates compilation, native execution, IC preparation/transfers, and disk writes.
GPU memory peaks are process-wide JAX allocator peaks (including parity checks),
not isolated solver working-set measurements. Dataset projections assume the
same horizon/snapshot policy; they exclude queueing, startup, compilation and
parity validation. The reported figures describe data generation, not model
training throughput. Runtime images are reused with explicitly mounted source;
this procedure does not claim a fresh image build.

### Measured pilot (2026-10-07)

Slurm jobs `2911595_0` and `2911601_1` through `2911601_7` completed the eight
float64 cases: 192/192 finite trajectories, with maximum native/API snapshot
absolute difference `7.45e-9`. These checks establish generator parity on the
sampled cases, not full benchmark coverage or trained-model accuracy.

Float32 was rejected as the default. Native/API float32 parity failed on the
N=8 stiff case and the N=16 recovery case (jobs `2911594_0`, `2911602_7`). Even
the N=16, nu=0.01, dt=0.01, 50-step case that passed float32 parity differed from
the float64 teacher by 4.27% median / 6.01% maximum terminal relative L2 across
four sampled ICs (`2911602_1`). The generator therefore defaults to float64
teacher computation while storing decoded fields as float32.

For the N=64, 100-step case, native generation of 24 trajectories took 5.21 s on
an RTX 5090 versus 1.01 s on a B200 (`2911601_5`, `2911603_5`), both at batch 4.
This approximately 5.2x ratio excludes compilation, validation, transfers, and
I/O. Shared-storage shard writing/hashing/fsync took 25.56 s and 20.60 s,
respectively, so the GPU speedup did not translate into the same end-to-end gain.
These are small single-run measurements under concurrent cluster load.

The follow-up B200 job `2911646` generated 32 trajectories on node-local scratch:

| Batch | Native compute | Shard writes/hash/fsync | Generator invocation | Peak JAX allocation |
| ----: | -------------: | ---------------------: | -------------------: | ------------------: |
| 4     | 1.34 s         | 5.06 s                 | 11.63 s              | 2.06 GiB            |
| 16    | 1.37 s         | 5.02 s                 | 12.98 s              | 8.34 GiB            |

Re-running batch 4 verified all shard hashes and reused every shard without
compilation. Archiving both datasets (about 5.44 GiB together) to shared storage
took another 38 s. Thus prefer batch 4, generate/read training windows on local
scratch, and archive completed shards/cases separately. Local scratch is not
durable: retain deterministic manifests and copy completed data before releasing
the allocation; interruption can require regenerating unarchived shards.

With the pilot's 29-snapshot policy, 1,024 N=64 trajectories occupy approximately
87 GiB. This storage projection is a reason to expand high-resolution data
selectively and shorten retained windows where appropriate. It does not include
future training checkpoints or derivative labels. No generalized model training
was performed by this data-generation pilot.

### Node-local execution and generalized operator pilot

`generate_operator_data.py` and `train_operator.py` now reject network-backed
source, data, output, and cache paths by inspecting Linux mount information.
Stage source, the manifest, and the teacher image onto node-local storage before
launching either command. `/surrogate-output` in the examples above must be a
mount of local scratch. Multi-worker shard sharing is intended within one node;
use separate local output roots and disjoint shards across nodes.

`run_operator_pilot.sh` is a Kander Slurm wrapper for two GPUs on one node
(default B200; override the Slurm GPU/QoS requests for RTX 5090s). Set
`OPERATOR_RUN_ROOT` to a shared directory containing `source.tar`, and
`OPERATOR_TEACHER_IMAGE` to the immutable teacher squashfs. The source bundle
contains `manifest.json` and `source/`, with the operator Python modules,
`teacher_api.py` (the XLB API), and `mosaic_shared/`. Submit with `sbatch`; keep
image basenames unique across image versions because the node cache uses them
as keys. The wrapper verifies local scratch/cache mounts, stages source once,
and caches the image under an advisory lock. Only local paths are mounted into
the GPU containers. Imports, temporary files, CUDA/JAX caches, program logs,
data generation, sampled training reads, and checkpoints stay local.

The loader reads each uncompressed NPZ member header once and memory maps its
velocity payload directly. It does not unpack a second dataset copy. The small
metadata index is locked across training processes; batches are prefetched one
ahead. No ZIP parsing or shared archive access occurs per update. Dataset shards
must remain immutable and available for the life of the mapped dataset. Compressed
NPZ files are rejected; generator hashes and bundle verification establish integrity. Training uses resolution buckets, shared Fourier
mode weights, physical-wave-number diffusion, and conditioning on viscosity,
time interval, grid spacing, and domain length. Nyquist handling avoids mode
aliasing on small grids. A 1/2/4-step rollout curriculum mixes step spans 1 and 5.
The optional output projection is an ablation, not a requirement of the model.

Once generation is complete, the wrapper publishes one immutable data tar while
training reads the local copy. After training stops, it publishes one model,
optimizer, report, and log tar. Disposable caches are excluded. Publishing streams
immutable local files directly into one buffered temporary shared tar, computing
SHA-256 without writing a second local archive or rereading the shared destination,
then atomically renames it and writes a small completion record. The Slurm log contains only phase messages. Restore
on another node with:

```bash
python operator_storage.py restore /shared/JOB-data.tar /local/data
python operator_storage.py restore /shared/JOB-training.tar /local/output
python train_operator.py --dataset /local/data --data-cache /local/mmap-cache \
  --output /local/output/unprojected \
  --resume /local/output/unprojected/latest.npz
```

Use the same frozen source and settings as the checkpoint; the example assumes
default training settings. Repository utilities may have changed since a saved
run; resume with the archived source matching its trainer/model hashes. Restore the full output bundle so `best.npz` accompanies
`latest.npz`. Checksums are verified locally before extraction. A completed
resume preserves the report. SIGUSR1/TERM requests a local checkpoint at the next
safe update boundary; the wrapper then archives results. Abrupt node loss can
still lose progress since the last published archive. Scratch is disposable.

Job `2911778` completed the first generalized pilot in 2m55s, including source /
image staging, generation, two concurrent training runs, final tests, and archive
publication. It generated 1,344 finite float64-teacher trajectories in 84 shards
across 16 physics/grid cases, retaining every step through step 40. Native teacher
execution summed to 2.25s; that excludes compilation, checks, transfers and I/O.
Training used N=8/16/32; N=20/48/64 were held out entirely from updates and
checkpoint selection. Each model trained for 1,000 updates on one B200.

| Variant | Training + validation time* | Validation normalized RMS | Test normalized RMS | N=20 / 48 / 64 test |
| --- | ---: | ---: | ---: | --- |
| Unprojected | 57.2s | 0.04205 | 0.1317 | 0.2872 / 0.0331 / 0.0263 |
| Projected | 60.9s | 0.04216 | 0.1336 | 0.2730 / 0.0290 / 0.0229 |

*Trainer timer excludes loader setup and final test evaluation. Error aggregates
are equal-weight means over cases, with an RMS normalization floor of 0.01.
All evaluated predictions were finite. Both variants selected update 1,000.
The diffusion-only validation baseline was 0.06619. However, this smoke run used
only two validation and two test parents per case: its validation parents were
near-zero flows. Consequently these numbers establish execution and a small
resolution-transfer check, not broad accuracy. The default validation sample cap
is now 16; expand parent/family coverage before selecting a production model.

That initial pilot did not change the runtime API or benchmark exclusions. The
expanded training and integration below supersede its checkpoint. Wider spectral
coverage, improved long-horizon accuracy, and separate 2D/obstacle operators remain
future work.

### Expanded training and long-horizon stability

Validation endpoints now cycle through flow families before taking another parent
from the same family. Parent ordering is deterministic, independent of training
sampling, and respects the original parent split. Reports include each selected
parent's normalized error, not only the case mean. Larger validation sets are
necessary: the first pilot's first two validation rows were unrepresentative.

A direct check of the original unprojected pilot at N=20, nu=0.001, dt=0.05 on
the benchmark random IC was finite at 40 steps, grew to a maximum velocity of
81.8 at 320 steps, and became nonfinite at 10,240 steps. That checkpoint is not a
production model. Model version 2 adds opt-in `--conserve-energy` training: after
any optional Helmholtz projection, it restores the input mean velocity and limits
fluctuating kinetic energy to its previous value. This encodes the momentum and
energy properties of unforced periodic flow. The same operation and gradients
are used in training and inference. It can differ from weakly compressible XLB
transients, and bounded energy does not guarantee accurate fields or gradients.

The expanded manifest uses 384 parents per N=8/16/32 training case and 128 per
N=20/48/64 holdout case, totaling 5,376 trajectories. Training horizons extend
through 100 steps; resolution holdouts extend through 200/300/320 steps. Early
and late 20-step dense windows support four updates at span five, with sparse
intermediate snapshots. This retains about 55.4 GiB of float32 fields; it does
not store dense 10,240-step trajectories. The separate frozen-checkpoint evaluator
covers all registered horizons, including 10,240, using exact benchmark ICs only
after training.

`export_operator.py` removes optimizer state and preserves model source hash,
architecture, selected step, validation score, and training identity. The candidate
`operator_api.py` accepts positive viscosity/dt/domain length, cubic N>=8 grids,
and nonnegative horizons. It returns exact identity at zero steps and handles a
partial final macro-step. Its VJP supports initial velocity, viscosity, and dt;
periodic drag is zero. It rejects 2D fields, obstacles, and inflow. Exported weights
must match the exact model source hash. `evaluate_operator.py` records finite
status, teacher-relative error, energy, cold/warm forward timings, and gradient
checks against all registered 3D cases; this direct comparison does not replace
the complete Mosaic benchmark harness.


### Final 3D integration (2026-10-07)

Prepare the expanded plan with `operator_dataset.py --output coverage.json
--training manifest.json`. The launcher defaults to the validated width-24,
mode-4, four-layer architecture, 10,000 updates, span-five evaluation, and energy
conservation; it runs projected and unprojected variants. `OPERATOR_UPDATES` and
`OPERATOR_VALIDATION_SAMPLES` override the corresponding budgets. To reuse an
immutable dataset already cached on a node, set `OPERATOR_LOCAL_DATA` and schedule
on that node. The manifest must match every cached case. This skips regeneration
and duplicate publication; it does not copy the cache back through shared storage.

Generation job `2911822` produced 5,376 finite trajectories in 336 shards, with
101.50s summed native teacher execution (excluding compilation, checks and I/O).
Its unconstrained training attempts were stopped during the old loader's large
unpack operation. The complete dataset was archived and cached on rtx02. Job
`2911959` reused that node-local dataset with direct NPZ mapping and completed both
constrained 10,000-update runs; the projected update-8,000 checkpoint won on
validation. Its training plus validation timer was 174.3s, excluding loader setup,
final tests and publication. The selected model and direct evaluation results are
summarized in [MODEL.md](MODEL.md) and [operator_validation.json](operator_validation.json).

Initial evaluator sweeps inherited the XLB module's global x64 setting. The final
sweep (`2912042`) explicitly uses float32 IC construction and the same float32
operator arithmetic as training. Both API forward and VJP isolate themselves from
other solvers' global JAX precision settings. Benchmark comparisons include two
teacher-VJP probes and the full registered energy-FD epsilon/direction grid.

The runtime now packages `operator_api.py`, `operator_model.py` and
`operator_weights.npz`. All existing 3D experiments admit it, with XLB-equivalent
LBM scaling in resolution sweeps. The fixed model is retained under `legacy_api.py`.
The declared runtime image still requires a fresh build; validation here used the
cached GPU image and the in-process Tesseract/JAX adapter, not a newly built image.
2D/obstacle support and accurate long-horizon dynamics remain unfinished.
