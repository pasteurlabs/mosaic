# PR 118 offline 3D autoregressive-surrogate results

These files are PR-description assets and reproducibility records for
`feat/ns-3d-grid-recovery-surrogate`. They are intentionally kept off the
source branch. The source branch contains the non-cluster trajectory generator
and training program beside the Tesseract, plus the final inference checkpoint.
The raw dataset, intermediate checkpoints, and cluster orchestration are not
committed.

All experiments ran offline on Kander through the shared Mosaic Slurm runner
and Pyxis/Enroot solver images. The hosted benchmark was not triggered.

## Model and training data

- Fixed task: `N=16`, `ν=0.01`, `dt=0.02`, 100 XLB steps, periodic
  `[0,2π]³`.
- Full differentiable velocity state: `16×16×16×3`.
- One width-32, six-mode, six-block 3D FNO advances five XLB steps and is
  reused autoregressively for 20 macro-steps.
- An exact viscous skip and Helmholtz projection are applied at every
  macro-step.
- Runtime package data consists only of the shared inference model and
  checkpoint. Training and generation files remain ordinary source files.
- Drag is exactly zero because the periodic 3D task has no obstacle; there is
  no learned drag head.

The checkpoint uses 16,384 continuous native XLB KBC D3Q27 trajectories,
split 12,288/2,048/2,048. Every trajectory stores the IC plus 20 snapshots at
five-solver-step intervals without reconstructing XLB populations between
snapshots. The generated final frame matched the canonical 100-step float32
teacher call with maximum absolute difference zero.

Training used rollout curriculum `1 → 2 → 4 → 8 → 12 → 20`, followed by
12,000 full-horizon fine-tuning updates. Dataset SHA-256:
`4836fba4e6a8524af7a552c5977721118e726afa21db9a9f4d0b612a879a0005`.
Checkpoint SHA-256:
`1ea04a7333981d1bfb836461d6fd6d89ae12f31c64ea40701c2607f03fb4107f`.

## Forward and derivative accuracy

On the three excluded recovery seeds, final-field relative L2 errors are
4.418%, 4.191%, and 4.317% (4.309% mean), with mean cosine 0.999076. Across
1,982 nonzero held-out test trajectories, mean final error is 7.219% and
median final error is 5.348%.

Increasing the dataset from 4,096 to 16,384 trajectories improves excluded
forward error from 7.473% to 4.309% and full-spectrum JVP error from 38.28% to
26.00%. Held-out and validation errors improve with training, so this is not
conventional train/test overfitting.

## Recovery

Both paper optimizer variants use the exact zero cold start, 100 iterations,
the same zoom line search, and seeds 0/1/2.

| optimizer / solver | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| L-BFGS / XLB | 4.96% | 5.17% | 6.12% | **5.42%** |
| L-BFGS / surrogate | 17.31% | 15.70% | 17.10% | **16.70%** |
| L-BFGS + projection / XLB | 4.70% | 5.02% | 5.94% | **5.22%** |
| L-BFGS + projection / surrogate | 17.32% | 15.59% | 17.19% | **16.70%** |

For surrogate seed 0, unconstrained/projected final `max|∇·u₀|` is
1.49e-2/1.43e-2; the maximum over each optimization is 1.52e-2/1.43e-2.
The projected optimizer follows the paper definition: it projects the
gradient before the L-BFGS update, not the quasi-Newton iterate itself.

The larger checkpoint is a better forward model but a worse global inverse
than the 4,096-trajectory checkpoint, whose self-recovery mean was about 8.1%.
At exactly zero, mean self-target descent alignment with the direction to the
true IC is almost unchanged (cosine 0.845 versus 0.847). At IC amplitude 0.05,
the 16k model falls to 0.419 versus 0.825; at amplitude 0.10 it falls to 0.150
versus 0.807. Forward-only rollout selection does not constrain this
off-manifold inverse-gradient geometry. Similar local conditioning at the
true IC therefore does not predict cold-start recovery.

## Jacobian conditioning

The recovery-subspace audit retains the full output and restricts the input to
the complete 512D real divergence-free Fourier subspace through `|k|≤4`:

- `κ(J)`: 6.864 XLB, 7.142 surrogate.
- `κ(JᵀJ)`: 47.17 XLB, 51.05 surrogate.
- Frobenius cosine: 0.996096.
- Frobenius relative error: 8.840%.
- Diffusion-subtracted residual cosine: 0.994122.

The similar scalar condition numbers are real but do not imply identical
Jacobians. They summarize only spectral extremes in a smooth restricted
subspace, while Frobenius, directional, amplitude-path, and inverse-gradient
diagnostics measure different structure.

The paper-style audit constructs dense 1,536-dimensional raw-state Jacobians
using one VJP per output degree of freedom and a dense SVD. Because this
surrogate is fixed at `N=16`, an explicit orthonormal `8³` block-grid
lift/restriction preserves the fixed recovery physics:

- Raw `κ(J)`: 2.790e7 XLB, 7.476e10 surrogate.
- Float32-resolved `κ(J)`: 5.319e3 XLB, 5.445e3 surrogate.
- Frobenius cosine: 0.944259.
- Frobenius relative error: 32.935%.

The raw tails lie below the float32 rank tolerance. A native `N=8` TGV XLB
control and all three dense matrices are included.

## Timing and XLB-target limitation

Warm packaged RTX 5090 medians:

| kernel | XLB | surrogate |
|---|---:|---:|
| forward | 4.812 ms | 7.243 ms |
| VJP | 14.368 ms | 14.779 ms |

The autoregressive surrogate is 1.51× slower for forward and 1.03× slower for
VJP. Complete three-seed recovery wall times remain similar because RPC,
callbacks, optimizer, projection, and line-search overhead dominate:
34.73 s surrogate versus 34.21 s XLB for L-BFGS, and 33.38 s versus 35.78 s
with projection.

Surrogate inversion of XLB-generated targets remains a negative result:
37.43% final mean IC error and 17.99% mean at the best saved snapshots. The
surrogate residual reaches 1.78%, but XLB re-evaluation leaves 17.77% mean
final-field residual.

## Figures

- `data_scaling_recovery_path.png`: forward error, radial JVP error, and
  self-recovery descent alignment for the 4k and 16k checkpoints.
- `recovery_optimizer_ablation.png`: objective, true IC error, and optimized-IC
  divergence for both L-BFGS variants and all five selected solvers.
- `recovery_convergence_comparison.png`: projected recovery loss and true IC
  error across PhiFlow, XLB, Warp-NS, Exponax, and the surrogate.
- `full_field_recovery.png`: true/recovered ICs and final fields.
- `forward_full_field_slices.png`: three orthogonal final-field slices.
- `recovery_evolution.gif`: full 3D projected-recovery evolution.
- `jacobian_conditioning.png`: restricted and paper-style Jacobian spectra.
- `cross_model_recovery.png`: XLB-target objective and IC-error divergence.
- `autoregressive_curriculum.png`: curriculum and rollout training metrics.
- `optimization_summary.png`: recovery accuracy and kernel/harness timing.

See `manifest.json` for offline job IDs and SHA-256 checksums.
