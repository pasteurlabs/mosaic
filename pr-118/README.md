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
the 1,982 held-out trajectories with IC amplitude `≥0.05`, mean final error
is 7.219% and median final error is 5.348%. The remaining 66 lower-amplitude
cases are reported separately with absolute RMS error because relative error
is ill-conditioned as the target norm approaches zero.

Slurm follow-up `1697360` reruns this audit against the current recurrent-state
XLB image and reproduces all forward and JVP headline values exactly.

Increasing the dataset from 4,096 to 16,384 trajectories improves excluded
forward error from 7.473% to 4.309% and full-spectrum JVP error from 38.28% to
26.00%. Held-out and validation errors improve with training, so this is not
conventional train/test overfitting.

## Finite-difference verification

The packaged surrogate and the current solver-loop XLB image were checked on
the exact recovery physics with central finite differences. The protocol uses
seeds 0/1/2, ten shared unit-norm random directions per seed, and the same
12-point relative-ε sweep. Perturbations are scaled by the true IC RMS.

| objective / best-ε aggregate | XLB | surrogate |
|---|---:|---:|
| paper energy `sum(u_T²)` median error | 6.77e-6 | 6.63e-3 |
| paper energy mean cosine | ≈1.000000 | 0.999984 |
| recovery MSE at zero median error | 1.32e-1 | 2.56e-4 |
| recovery MSE at zero mean cosine | 0.991669 | ≈1.000000 |

The surrogate is less precise on the energy-objective magnitude check but has
near-perfect direction agreement. More importantly, its VJP is exceptionally
consistent with finite differences for the actual zero-start recovery
objective. The poor recovered IC is therefore not an autodiff bug: the VJP is
an accurate derivative of a learned map whose global inverse geometry is
wrong. XLB has a larger zero-start recovery-MSE magnitude discrepancy, but its
direction remains closely aligned and its optimizer reaches the better IC.

## Recovery

Both paper optimizer variants use the exact zero cold start, 100 iterations,
the same zoom line search, and seeds 0/1/2.

| optimizer / solver checkpoint | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| L-BFGS / XLB | 4.96% | 5.17% | 6.12% | **5.42%** |
| L-BFGS / surrogate 4k | 6.94% | 7.81% | 9.56% | **8.10%** |
| L-BFGS / surrogate 16k | 17.31% | 15.70% | 17.10% | **16.70%** |
| L-BFGS + projection / XLB | 4.70% | 5.02% | 5.94% | **5.22%** |
| L-BFGS + projection / surrogate 4k | 6.97% | 7.85% | 9.54% | **8.12%** |
| L-BFGS + projection / surrogate 16k | 17.32% | 15.59% | 17.19% | **16.70%** |

For surrogate seed 0, unconstrained/projected final `max|∇·u₀|` is
1.49e-2/1.43e-2; the maximum over each optimization is 1.52e-2/1.43e-2.
The projected optimizer follows the paper definition: it projects the
gradient before the L-BFGS update, not the quasi-Newton iterate itself.

The larger checkpoint is a better forward model but a worse global inverse
than the 4,096-trajectory checkpoint. The 16k seed-0 final objectives are
slightly lower (`1.15e-8`/`1.18e-8` versus `1.28e-8`/`1.33e-8` for
unconstrained/projected), and its final divergence is also lower, despite its
roughly twofold larger IC error.

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

The matched checkpoint comparison is:

| audit | metric | XLB | surrogate 4k | surrogate 16k |
|---|---|---:|---:|---:|
| restricted 512D | `κ(J)` | 6.864 | 7.740 | 7.142 |
| restricted 512D | `κ(JᵀJ)` | 47.17 | 59.98 | 51.05 |
| restricted 512D | Frobenius relative error | — | 13.50% | 8.84% |
| restricted 512D | Frobenius cosine | — | 0.9914 | 0.9961 |
| adapted block-grid 1536D | raw `κ(J)` | 2.790e7 | 3.747e10 | 7.476e10 |
| adapted block-grid 1536D | resolved `κ(J)` | 5.319e3 | 4.890e3 | 5.445e3 |
| adapted block-grid 1536D | Frobenius relative error | — | 32.98% | 32.94% |
| adapted block-grid 1536D | Frobenius cosine | — | 0.9442 | 0.9443 |

Scaling the data improves the restricted Jacobian, but leaves the adapted
block-grid agreement nearly unchanged and worsens the unresolved raw spectral
tail. Neither local audit predicts the observed cold-start inverse basin.

The adapted audit constructs dense 1,536-dimensional Jacobians using one VJP
per coarse output degree of freedom and a dense SVD. Because this surrogate is
fixed at `N=16`, an explicit orthonormal `8³` block-grid lift/restriction
preserves the fixed recovery physics. It covers every coordinate of that
coarse map, but it is not the complete 12,288-dimensional production
Jacobian and is distinct from the paper's native `N=8` TGV physics:

- Raw `κ(J)`: 2.790e7 XLB, 7.476e10 surrogate.
- Float32-resolved `κ(J)`: 5.319e3 XLB, 5.445e3 surrogate.
- Frobenius cosine: 0.944259.
- Frobenius relative error: 32.935%.

The raw tails lie below the float32 rank tolerance. A native `N=8` TGV XLB
control and all three dense matrices are included.

The rebased source passes the full suite (`522 passed, 3 skipped`) in Slurm job
`1697321`; the rebuilt runtime image passes its generated API check and
Pyxis/Enroot round-trip in `1697322`.

Follow-up JSON:

- `followup_evaluation_16k.json`: current-XLB forward/JVP reproduction with
  explicit amplitude-threshold and low-amplitude absolute-error reporting.
- `followup_timing_xlb.json` and `followup_timing_surrogate.json`: 40-trial
  order-balanced timing aggregates from completed jobs `1697367` and
  `1697363`.

## Timing and XLB-target limitation

Two matched RTX 5090 blocks counterbalance solver order and contribute 40 warm
trials per solver:

| kernel | XLB | surrogate |
|---|---:|---:|
| forward | 4.794 ms | 7.350 ms |
| VJP | 15.198 ms | 14.771 ms |

The autoregressive surrogate is 1.53× slower for forward, while VJP cost is at
parity: 0.97× in the counterbalanced result and 1.03× in the earlier
independent measurement. Solver-scoped three-seed harness times remain similar:
32.81 s surrogate versus 32.09 s XLB for L-BFGS, and 31.51 s versus 32.50 s
with projection. Including result-script setup and serialization gives 34.73 s
versus 34.21 s and 33.38 s versus 35.78 s, respectively. RPC, callbacks,
optimizer, projection, and line-search overhead dominate these wall times.

Surrogate inversion of XLB-generated targets remains a negative result:
37.43% final mean IC error and 17.99% mean at the best saved snapshots. The
surrogate residual reaches 1.78%, but XLB re-evaluation leaves 17.77% mean
final-field residual.

## Task-aware Sobolev follow-up

The derivative follow-up supervises projected directional derivatives of the
actual recovery MSE along XLB-generated optimization paths. This is
first-derivative Sobolev supervision; differentiating its loss with respect to
network parameters uses mixed second-order autodiff. It does not train against
an explicit teacher Hessian.

The leakage-safe dataset contains 96 training and 24 validation paths, eight
snapshots per path, and four deterministic divergence-free directions per
state (two low-frequency and two full-spectrum). Benchmark seeds 0–2 and Adam
calibration seeds 100–102 are excluded. All arms start from the packaged gated
16k checkpoint and use 1,000 fine-tuning updates. The derivative weight is
selected on validation field plus task-gradient relative error; only the
selected `λ=1e-3` checkpoint sees the benchmark test seeds.

| metric | packaged start | field-only continuation | task-Sobolev `λ=1e-3` |
|---|---:|---:|---:|
| validation task-gradient relative L2 | 5.305 | 4.568 | **4.088** |
| validation task-gradient cosine | 0.553 | 0.571 | **0.599** |
| held-out final-field error | 7.219% | **7.219%** | 7.506% |
| excluded-seed forward error | **4.309%** | 4.343% | 4.528% |
| low-frequency JVP error | 18.24% | 17.55% | **17.17%** |
| full-spectrum JVP error | 26.00% | 25.67% | **25.26%** |

The proxy improvement does not transfer to inverse recovery:

| recovery metric | packaged checkpoint | task-Sobolev |
|---|---:|---:|
| projected L-BFGS self-recovery | **16.70%** | 17.05% |
| projected Adam, 300 updates, LR `1e-2` | **33.99%** | 34.10% |
| projected Adam, 300 updates, LR `3e-2` | 63.43% | **63.06%** |
| XLB-target cross-model final IC error | **37.43%** | 42.19% |
| XLB re-evaluation residual | **17.77%** | 19.77% |

Residual-only Adam calibration selects `1e-2` for the packaged checkpoint and
the upper-grid `3e-2` for task-Sobolev. The crossed fixed-rate controls show
that the apparent 33.99% versus 63.06% regression is a learning-rate/branch
effect: within each matched rate, the two checkpoints are nearly identical.
The larger rate obtains a slightly smaller surrogate residual while recovering
a much worse IC. Improving an average directional sensitivity metric on
held-out Adam paths is therefore insufficient for this inverse problem. This
does not establish that Sobolev training is generally ineffective: the current
study uses four directional sketches, Adam-path data only, one fine-tuning
seed, and retains the architecture's zero-start amplitude gate.

Jobs: dataset `1697734`; field control `1697737`; Sobolev weight sweep
`1697735`, `1697741`, `1697740`; evaluations `1697745`, `1697749`; selected
image build and Pyxis round-trip `1697751`; L-BFGS `1697752`; Adam `1697753`;
cross-model recovery `1697754`; crossed fixed-rate Adam controls `1697999`,
`1698000`; final render `1698013`.

## Figures

- `task_sobolev_followup.png`: validation task-gradient agreement, ordinary
  forward/JVP metrics, selected-checkpoint self-recovery, and XLB-target
  cross-model recovery.
- `finite_difference_checks.png`: matched XLB/surrogate FD U-curves and
  directional agreement for both the paper energy objective and zero-start
  recovery MSE.
- `inverse_diagnostic_explanation.png`: recovery-gradient alignment and radial
  loss along the true-IC ray, followed by the optimizer objective and actual IC
  error. It shows why a smooth, decreasing forward objective is insufficient.
- `checkpoint_conditioning_comparison.png`: matched 4k/16k restricted and
  adapted block-grid singular spectra, condition numbers, and Jacobian
  agreement.
- `checkpoint_optimization_comparison.png`: complete 4k/16k L-BFGS and
  projected-L-BFGS objective, IC-error, and divergence histories.
- `data_scaling_recovery_path.png`: forward error, radial JVP error, and
  self-recovery descent alignment for the 4k and 16k checkpoints.
- `recovery_optimizer_ablation.png`: objective, true IC error, and optimized-IC
  divergence for both L-BFGS variants and all five selected solvers.
- `recovery_convergence_comparison.png`: projected recovery loss and true IC
  error across PhiFlow, XLB, Warp-NS, Exponax, and the surrogate.
- `full_field_recovery.png`: true/recovered ICs and final fields.
- `forward_full_field_slices.png`: three orthogonal final-field slices.
- `recovery_evolution.gif`: full 3D projected-recovery evolution.
- `jacobian_conditioning.png`: restricted and adapted block-grid Jacobian
  spectra.
- `cross_model_recovery.png`: XLB-target objective and IC-error divergence.
- `autoregressive_curriculum.png`: curriculum and rollout training metrics.
- `optimization_summary.png`: recovery accuracy and kernel/harness timing.

See `manifest.json` for offline job IDs and SHA-256 checksums.
