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
