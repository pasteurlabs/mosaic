# Conditioned XLB 3D surrogate

This Tesseract advances periodic cubic 3D velocity fields with one shared Fourier
neural operator across grid resolutions, viscosities, timesteps, and horizons.
It now participates in every registered `ns-3d-grid` experiment. Benchmark
admission means that the input and derivative contracts are supported; it does
not imply uniformly accurate XLB predictions. Long-horizon accuracy remains poor.
The 2D and obstacle suites are still excluded and need separate training/support.

## Runtime contract

Inputs are finite `(N, N, N, 3)` velocity fields with `N >= 8`, positive viscosity,
timestep and domain length, nonnegative step count, and fully periodic boundaries.
There is no forcing, obstacle, inflow, or learned drag head. Drag is zero.
The output has the input shape and float32 dtype. Zero steps return the input.
The API provides VJPs for initial velocity, viscosity, and timestep.

The packaged model has width 24, four residual spectral blocks, and a shared bank
of signed Fourier modes through mode four. Unsupported modes are cropped on small
grids; even-grid Nyquist modes do not alias into the retained bank. Physical
wave numbers drive the diffusion skip. Conditioning includes viscosity, timestep,
elapsed interval, grid spacing, and domain extent. A macro-step normally spans
five requested steps, with a shorter final update for any remainder.

Each update applies a Helmholtz projection, restores the input mean velocity,
and limits fluctuating kinetic energy to its previous value (up to floating-point
and regularization error). This encodes properties of unforced incompressible
periodic flow. It can differ from weakly compressible XLB transients. Energy
bounds ensure neither field accuracy nor accurate teacher gradients; in particular,
the learned dynamics can retain too much energy at long horizons.

XLB velocity omits the lattice populations, so it is not a closed representation
of the teacher's numerical state. Training labels come from continuous native
float64 XLB population trajectories, decoded and saved as float32. The teacher is
not restarted at each snapshot. Both training and inference use `operator_model.py`.

## Training and selection

The expanded dataset has 5,376 trajectories across 16 physics/grid combinations.
The 13 N=8/16/32 training cases each use the same 384 parent identities, split into
304 training, 42 validation and 38 test parents. The three N=20/48/64 resolution
holdouts each use 128 parents (102/17/9 by split). Parent splits are consistent
across physics and grids; held-out resolutions never participate in updates or
checkpoint selection. Families are randomized TGV, ABC, solenoidal waves, mean
flows, near-zero fields and small longitudinal perturbations. Spectral breadth
is still limited; this is not a universal fluid model.

Two variants trained for 10,000 updates each on RTX 5090s (job `2911959`). The
projected variant at update 8,000 won by mean validation normalized RMS error,
using up to 24 parents per case balanced across families. Its selection was fixed
before inspecting benchmark comparisons. Training plus validation took 174.3s;
this excludes setup, final tests, and archival. The unprojected comparison took
173.2s. These timings describe this dataset/model and are not end-to-end job times.

The selected model's mean validation error is 0.2405 and its training-resolution
test error is 0.2372. On held-out test parents, errors are 1.1538 at N=20 (320 steps),
0.3221 at N=48 (300 steps), and 0.2127 at N=64 (200 steps). These are RMS-normalized
errors with denominator floor 0.01, averaged over parents and then cases;
they are not strict relative L2 percentages. Holdout tests contain nine parents
per resolution. All predictions in these tests are finite.

Training spans viscosities 0.001–0.1, training timesteps 0.005–0.05, and domain
length `2π`. Smaller timesteps 0.0025 and 0.003333 are covered by resolution
holdouts. Other positive values are accepted but represent extrapolation.
See [TRAINING.md](TRAINING.md) for generation, local storage, training and export.

## Benchmark integration and reproducibility

Resolution sweeps apply the same `lbm_N_base` timestep/horizon scaling to the
surrogate and XLB. Existing experiments and metrics are retained. The real
Tesseract/JAX adapter passes gradient checks, and a three-iteration projected
recovery smoke test reduces the objective and IC error. Recovery uses the
surrogate's own observation; this does not establish XLB-target inversion.

The full direct API evaluation covers all 46 registered 3D physics/IC cases and
VJPs on 15 gradient/recovery cases, including 10,240 steps. It complements, rather
than replaces, the full cost, Jacobian-SVD and optimization benchmark harnesses.
The evaluator explicitly generates benchmark ICs in float32 before invoking the
float64 XLB teacher, since JAX precision changes random-number draws.

The final float32-runtime evaluation (job `2912042`) produced 46/46 finite fields
and 15/15 finite VJPs. Median teacher-relative L2 error across the 46 cases was
0.0543, but the recovery case was 0.6689. At 10,240 steps it reached 198.4 relative
L2 / 0.2832 absolute RMS: the surrogate retained about 96% of initial energy while
XLB decayed substantially. Stability is not long-horizon fidelity.

VJP cosine agreement with XLB was 0.9763 on the short FD case and 0.5726 on the
100-step recovery case. The native GPU energy-FD sweep, using the registered
benchmark's ten directions and epsilon grid, achieved best median relative error
`7.05e-4` and cosine `0.99999955`. The actual CPU FD benchmark harness achieved
`3.07e-4` and cosine above `0.9999999`, passing its thresholds. Smaller perturbations
show float32 noise; the complete sweep is retained in
[operator_validation.json](operator_validation.json), alongside every direct case.

The 7.2 MiB inference artifact contains parameters and provenance, without Adam
state. SHA-256:
`279a30aab464c6975737c0bef12a8dc0fb51a7b95b409a436a0463c973ac98cb`.
The loader validates parameter shapes, finiteness, model version and source hash.
Training used the cached XLB image with JAX 0.10; local API/harness checks also run
with JAX 0.11.2. These runs did not rebuild the declared runtime image.

The retired fixed N=16 model, checkpoint, and experimental training recipes are
preserved in [Git history](https://github.com/pasteurlabs/mosaic/tree/3a8c4cc/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate).
The active tree contains only the general operator pipeline. `operator_model.py`
is frozen byte-for-byte to preserve its checkpoint source hash, including its
historical module header.
