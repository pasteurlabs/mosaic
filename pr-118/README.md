# PR 118 offline 3D recovery-surrogate results

These files are PR-description assets and reproducibility records for
`feat/ns-3d-grid-recovery-surrogate`. They are intentionally kept off the
source branch. Dataset generation and training code are not included.

All experiments ran offline on Kander through the shared Mosaic Slurm runner
and Pyxis/Enroot solver images. The hosted benchmark was skipped.

## Main results

- Direct held-out full-field error: 7.151% relative L2; field cosine 0.997441.
- Exact self-recovery IC error, seeds 0/1/2:
  - XLB: 4.70%, 5.02%, 5.94% (5.22% mean).
  - surrogate: 6.20%, 6.92%, 8.15% (7.09% mean).
- Exact harness wall time: 30.96 s XLB, 24.48 s surrogate (1.26x).
- Warm in-process kernel medians:
  - XLB: 4.812 ms forward, 14.368 ms VJP.
  - surrogate: 0.920 ms forward, 1.082 ms VJP.
- Recovery-subspace Jacobian conditioning:
  - `kappa(J)`: 6.86 XLB, 6.56 surrogate.
  - `kappa(J^T J)`: 47.17 XLB, 43.09 surrogate.
  - Jacobian Frobenius cosine: 0.9901.
- Cross-model recovery (surrogate inversion of XLB targets):
  - 41.5% final mean IC error after 100 iterations.
  - 19.0% mean at the best saved snapshots.
  - This is a negative result and establishes that the checkpoint is not a
    drop-in inverse for XLB-generated observations.

## Figures

- `optimization_summary.png`: recovery accuracy and kernel-versus-harness timing.
- `full_field_recovery.png`: true/recovered ICs and final fields.
- `forward_full_field_slices.png`: three orthogonal slices of the held-out final field.
- `recovery_evolution.gif`: full 3D recovery evolution over saved optimizer snapshots.
- `jacobian_conditioning.png`: singular spectra and per-seed condition numbers.
- `cross_model_recovery.png`: the cross-model inversion failure mode.

Raw harness envelopes, field snapshots, conditioning spectra, timing samples,
held-out evaluation, and cross-recovery output are included alongside the
figures. See `manifest.json` for job IDs and checksums.
