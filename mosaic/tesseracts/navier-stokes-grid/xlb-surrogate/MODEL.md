# XLB cylinder-flow surrogate

This Tesseract is a task-specific surrogate for Mosaic's `N=32`, `Re=20`
cylinder-flow drag optimization. It is not a general Navier–Stokes solver and
is excluded from benchmark cells with other resolutions, geometries, physical
parameters, time horizons, or initial conditions.

## Model contract

The differentiable input is the 32-value inflow profile. A residual MLP maps
that profile to 128 POD coefficients. The POD decoder reconstructs four full
`32 × 32` fields:

1. final x velocity,
2. final y velocity,
3. tail-averaged x velocity, and
4. tail-averaged pressure.

The final velocity is returned as the canonical Mosaic `result`. The
tail-averaged velocity and pressure are internal fields used by the shared
`drag_jax` surface integral. There is no independent scalar drag head.

## Training provenance

The packaged `weights.npz` was trained outside this repository from 4,096
stable XLB KBC D2Q9 trajectories. Profiles span smooth Fourier, multiscale,
localized, and near-flat controls within the optimizer's `[0, 1.5]` bounds.
Another 653 candidate profiles were rejected because the XLB teacher became
non-finite or physically exploded.

The split contains 3,072 training, 512 validation, and 512 test profiles. The
training dataset SHA-256 is
`2832fecfb85d1ced442b38b608c512c71aca224c9a3b982d39e736d2bdf16ac8`.
The packaged weights SHA-256 is
`6089ba29d9644e1f3b87404302b0b19aaeb7ea6ee0b359d7aa5fe395ef04626c`.

XLB computes drag through lattice momentum exchange, whereas Mosaic's
projection solvers use a velocity/pressure surface integral. For each stable
teacher trajectory, the RANS pressure amplitude was calibrated so the shared
surface integral reproduced XLB's momentum-exchange drag. The spatial pressure
structure remains the one recovered from XLB density.

## Offline validation

On the 512-profile held-out split:

- mean final-velocity relative L2 error: 0.99%;
- 95th-percentile final-velocity relative L2 error: 4.47%;
- drag MAE: `2.39e-4`;
- drag R²: `0.99925`.

Against float64 XLB on eight held-out profiles:

- mean final-velocity relative L2 error: 0.30%;
- drag MAE: `5.81e-5`;
- mean drag-gradient cosine: `0.9950` (minimum `0.9885`);
- mean drag-gradient relative L2 error: 9.06%.

A 250-update surrogate optimization reached drag `-0.04689`. Re-evaluating
the final profile with float64 XLB gave `-0.04758`.
