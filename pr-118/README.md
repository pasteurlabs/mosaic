# PR 118 offline 3D autoregressive-surrogate results

These files are PR-description assets and reproducibility records for
`feat/ns-3d-grid-recovery-surrogate`. They are intentionally kept off the
source branch. Dataset generation and training code are not included.

All experiments ran offline on Kander through the shared Mosaic Slurm runner
and Pyxis/Enroot solver images. The hosted benchmark was skipped.

## Model and forward result

- Full velocity state: `16×16×16×3`.
- One shared FNO advances five XLB steps and is reused for 20 macro-steps.
- Rollout curriculum: `1 → 2 → 4 → 8 → 12 → 20`, followed by full-horizon
  fine-tuning.
- Exact excluded recovery seeds: 7.473% mean final-field relative L2 and
  0.997619 mean field cosine.
- Nonzero held-out test split: 9.905% mean final-field relative L2.

## Recovery and optimizer ablation

The exact cold-start task was run with both unconstrained L-BFGS and L-BFGS
with divergence-free gradient projection. Both use 100 iterations, the same
zoom line search, seeds 0/1/2, and the same solver-specific self-target.

Seed-0 final IC relative L2:

| Solver | L-BFGS | L-BFGS + projection |
|---|---:|---:|
| PhiFlow | 1.729% | 0.283% |
| XLB | 4.961% | 4.699% |
| Warp-NS | 3.834% | 0.262% |
| Exponax | 1.390% | 0.786% |
| Surrogate | 6.939% | 6.967% |

Seed-0 final/max spectral `max|div(u0)|`:

| Solver | L-BFGS final/max | L-BFGS + projection final/max |
|---|---:|---:|
| PhiFlow | 7.29e-2 / 1.46e-1 | 9.01e-3 / 4.31e-2 |
| XLB | 8.89e-2 / 8.92e-2 | 3.17e-2 / 3.17e-2 |
| Warp-NS | 1.76e-1 / 5.42e-1 | 9.72e-3 / 1.85e-2 |
| Exponax | 4.90e-2 / 1.49e0 | 2.72e-5 / 2.72e-5 |
| Surrogate | 1.77e-2 / 2.26e-2 | 1.77e-2 / 1.97e-2 |

The projected variant follows the paper’s gradient-projection definition; it
does not project the L-BFGS iterate after the quasi-Newton update. The raw
per-iteration loss, IC error, and optimized-IC divergence histories are in the
paired recovery JSON files.

## Jacobian conditioning

Two complementary audits characterize the fixed recovery map. The
recovery-subspace audit restricts the input to a complete 512D smooth
divergence-free Fourier subspace through `|k|≤4`, excludes longitudinal and
mean modes, and retains the full output:

- `kappa(J)`: 6.864 XLB, 7.740 surrogate.
- Frobenius cosine: 0.991392.
- Frobenius relative error: 13.498%.

The paper-protocol audit uses the full raw-state construction: one VJP per
output degree of freedom followed by a dense per-solver SVD. Because the
packaged surrogate is fixed at `N=16`, an explicit orthonormal `8³` block-grid
lift/restriction gives a complete 1,536-dimensional raw state while retaining
the fixed recovery physics.

- Raw `kappa(J)`: 2.790e7 XLB, 3.747e10 surrogate.
- Float32-resolved `kappa(J)`: 5.319e3 XLB, 4.890e3 surrogate.
- Frobenius cosine: 0.944208.
- Frobenius relative error: 32.984%.

The raw condition numbers differ by approximately three orders of magnitude
and are dominated by singular values below the float32 rank tolerance. Above
that tolerance, both resolved condition numbers are approximately `5e3`. The
normalized spectra and both raw/resolved statistics are provided. A native
`N=8` TGV XLB control using the paper settings is included in the same JSON.
The 24 MiB NPZ contains all three dense Jacobian matrices.

## Timing and cross-model limitation

- Packaged RTX 5090 medians:
  - XLB: 4.812 ms forward, 14.368 ms VJP.
  - surrogate: 7.116 ms forward, 14.862 ms VJP.
- Exact projected recovery harness: 30.96 s XLB, 33.17 s surrogate.
- Surrogate inversion of XLB-generated targets remains a negative result:
  51.47% final mean IC error, 21.84% mean at the best saved snapshots.

## Figures

- `recovery_optimizer_ablation.png`: loss, true IC error, and optimized-IC
  divergence for both L-BFGS variants and all five selected solvers.
- `recovery_convergence_comparison.png`: paper-style projected-recovery loss
  and true IC-error comparison.
- `jacobian_conditioning.png`: paper-protocol full-state spectrum contrasted
  with the earlier restricted recovery-subspace spectrum.
- `optimization_summary.png`: recovery accuracy and kernel-versus-harness
  timing.
- `full_field_recovery.png`: true/recovered ICs and final fields.
- `forward_full_field_slices.png`: three orthogonal held-out final-field
  slices.
- `recovery_evolution.gif`: full 3D projected-recovery evolution.
- `cross_model_recovery.png`: XLB-target inverse-model failure.
- `autoregressive_curriculum.png`: curriculum and rollout training metrics.

See `manifest.json` for offline job IDs and SHA-256 checksums.
