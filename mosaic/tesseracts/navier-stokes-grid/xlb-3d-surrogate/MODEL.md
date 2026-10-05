# XLB 3D initial-condition recovery surrogate

This Tesseract is a task-specific autoregressive full-field surrogate for the
`N=16`, `ν=0.01`, `dt=0.02`, 100-step periodic 3D initial-condition recovery
benchmark. It is not a general Navier–Stokes solver and is excluded from other
resolutions, physical parameters, horizons, geometries, and benchmark cells.
The canonical drag output is zero because this triply-periodic task has no
obstacle; there is no learned drag head.

## Model contract

The differentiable state is the complete `16 × 16 × 16 × 3` velocity field.
One periodic 3D Fourier neural operator advances that field by a macro-step of
five XLB solver steps (`ΔT=0.1`). The same weights are reused autoregressively
20 times to produce the full-horizon result. This is not an IC-to-final-state
regressor.

The neural operator has width 32, six retained Fourier modes per axis, and six
residual spectral blocks. An exact one-macro-step viscous-diffusion operator
supplies a physics skip while the neural operator learns the finite-amplitude
nonlinear correction. Every macro-step applies the implemented spectral Helmholtz projection.

XLB's velocity alone omits the lattice populations and is therefore not a
closed representation of the teacher's numerical state. The training targets
are nevertheless decoded from one continuous native XLB population rollout;
the teacher is never restarted from equilibrium at macro-step boundaries.

## Benchmark integration

This solver is discovered as `xlb-3d-surrogate` (display name
`XLB 3D surrogate`) by the regular `ns-3d-grid` registry. Default API inputs
are a zero `16×16×16×3` field with the supported viscosity, time step, and
horizon. Other physics are rejected before inference. All 2D and unmatched
3D benchmark cells are excluded.

The surrogate participates in the existing
`optimization/recovery_constant_ic_bfgs_proj` experiment, using the same random
divergence-free ICs (seeds 0/1/2), physics, and 100-update projected L-BFGS
optimizer as the other eligible solvers. It uses the standard result envelopes,
field snapshots, status reporting, and recovery plots. No experiments or shared
benchmark kernels are added or changed for this solver.

Run it alongside XLB through the normal CLI (Docker with GPU support required):

```bash
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e optimization/recovery_constant_ic_bfgs_proj
```

Recovery uses each solver's own observation; it does not establish XLB-target
inversion. Existing forward, gradient, and cost experiments have unsupported
physics and are excluded. The comparisons below are archived offline validation,
not additional registered cases in this PR. Supporting more existing benchmark
settings requires a model trained for those settings.

## Training and checkpoint

All data-generation and training recipes live in this directory; see
[TRAINING.md](TRAINING.md). The runtime image includes only the inference API,
shared model, and weights. The checkpoint was trained on 16,384 continuous native
XLB trajectories, split 12,288/2,048/2,048 for training/validation/test, with
benchmark seeds 0/1/2 excluded. Each trajectory contains the IC and 20 snapshots.
Training used unroll curriculum `1 → 2 → 4 → 8 → 12 → 20`, then full-horizon
fine-tuning.

Checkpoint SHA-256:
`1ea04a7333981d1bfb836461d6fd6d89ae12f31c64ea40701c2607f03fb4107f`.

## Offline validation

An offline comparison (Slurm job `2869896`, RTX 5090) used
seeds 0/1/2, 100 recovery updates, and 20 warm timing trials per seed. The
checkpoint was unchanged. Means across the three seeds, or all 60 timing
trials, are:

Only projected recovery corresponds to a currently registered experiment. The
other measurements used temporary registrations preserved in the archived
[source](https://github.com/pasteurlabs/mosaic/tree/f5b89e1) and execution scripts.

| Metric                                               |              XLB |        Surrogate |
| ---------------------------------------------------- | ---------------: | ---------------: |
| Forward field error against XLB                      |        reference |            4.39% |
| Unconstrained L-BFGS IC recovery error               |            5.38% |           16.74% |
| Projected L-BFGS IC recovery error                   |            5.21% |           16.67% |
| Forward API time, mean / median                      | 10.26 / 10.27 ms | 11.57 / 10.32 ms |
| Forward + VJP API time, mean / median                | 48.31 / 48.37 ms | 38.92 / 38.32 ms |
| Best-epsilon median directional FD error, seed range | 0.00033–0.00142% |     0.532–0.904% |

Both solvers are internally FD-consistent, but the full gradients of their
respective `sum(u_T²)` objectives are not interchangeable: surrogate-to-XLB
relative L2 differences are 222%, 207%, and 150%, with cosines
0.405, 0.438, and 0.568. These are objective-gradient comparisons, not
full-Jacobian or common-cotangent VJP errors. They must not be conflated with
restricted JVP statistics in the historical study linked below.

Timing used HTTP/base64 services and a CPU JAX client, with XLB's default
float64 path and the float32 surrogate. One allocation and fixed solver
order establish a measured comparison, not a hardware-independent speedup.
The run used existing runtime images with the current source/API/shared
model/weights mounted in; it did not rebuild those images. The generated
Docker build context was checked separately.

[Raw benchmark envelopes, field snapshots, plots, summary script, and provenance](https://github.com/pasteurlabs/mosaic/tree/surrogate-standalone-results)
are available separately from the solver source.

![Archived offline comparison: forward accuracy, API timings, recovery and finite differences](https://raw.githubusercontent.com/pasteurlabs/mosaic/37c9d6b/comparison.png)

## Inverse-model limitations

The surrogate's internally consistent derivatives do not make it a drop-in
inverse model for XLB observations. Its amplitude gate strongly suppresses the
learned correction's derivative near the zero recovery start; the square-root
floor makes that suppression finite. Similar scalar condition numbers also do
not establish agreement with XLB's Jacobian.

Eight terminal-only and six trajectory-replay fine-tuning pilots tested VJP
supervision, central secants, recovery-path labels, and an optional linear
correction. None improved mean XLB-target recovery on the three common
validation cases. The closest replay candidate reached 35.12% IC error versus
35.03% for the original checkpoint. These native SciPy validation runs use
different cases and optimizer settings from the registered self-target benchmark.
The original weights remain packaged; see [training results and plots](TRAINING.md#results).

The [historical study in PR #118](https://github.com/pasteurlabs/mosaic/pull/118)
contains the 4k/16k data comparison, restricted Jacobian spectra, and additional
inverse-path diagnostics. Larger training data improved forward accuracy there
without improving recovery. Neither these results nor the small fine-tuning
pilots establish that longer training cannot help.
