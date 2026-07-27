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
nonlinear correction. Every macro-step is Helmholtz-projected so both the
rollout and its recovery gradients remain on the divergence-free periodic
manifold.

XLB's velocity alone omits the lattice populations and is therefore not a
closed representation of the teacher's numerical state. The training targets
are nevertheless decoded from one continuous native XLB population rollout;
the teacher is never restarted from equilibrium at macro-step boundaries.

## Training provenance

`generate_trajectories.py`, `training_data.py`, and `train.py` live beside the
Tesseract source for reproducibility but are not copied into its runtime
image. The generator must run against the XLB teacher API; cluster submission
and container orchestration remain external. The runtime image contains only
the inference API, shared model definition, and weights. The packaged weights
were trained from 16,384 native XLB KBC D3Q27 trajectories, split
12,288/2,048/2,048 for
training/validation/test. Each trajectory contains the IC plus 20 full-field
snapshots, one every five XLB steps. The final generated snapshot matched the
canonical 100-step float32 teacher call with maximum absolute difference zero.
Benchmark IC seeds 0, 1, and 2 were excluded.

Training used autoregressive unroll curriculum
`1 → 2 → 4 → 8 → 12 → 20`, followed by a full-20-step fine-tune. The
distribution contains random divergence-free fields around the benchmark's
`|k|=2` energy shell, broader spectra for optimizer-path coverage, and
amplitudes from zero to 1.25.

The native trajectory dataset SHA-256 is
`4836fba4e6a8524af7a552c5977721118e726afa21db9a9f4d0b612a879a0005`.
The packaged checkpoint SHA-256 is
`1ea04a7333981d1bfb836461d6fd6d89ae12f31c64ea40701c2607f03fb4107f`.

The reproducible program boundary is:

```bash
python generate_trajectories.py \
  --teacher-api /tesseract/tesseract_api.py \
  --output /surrogate-output/recovery_3d_xlb_trajectories.npy
python train.py \
  --dataset /surrogate-output/recovery_3d_xlb_trajectories.npy \
  --weights /surrogate-output/recovery_3d_autoregressive_weights.npz
```

The first command is run in the XLB solver image so
`/tesseract/tesseract_api.py` is the teacher implementation. The second
command uses the model definition that the inference API imports.

## Offline validation

All experiments were run offline through the shared Slurm and Pyxis/Enroot
cluster path. On the three excluded recovery seeds, mean final-field relative
L2 error is 4.309% and mean field cosine is 0.999076. Across the 1,982 nonzero
held-out test trajectories, mean final-step relative L2 error is 7.219%.

Directional derivatives were evaluated end to end through all 20 shared
operator applications. On low-frequency projected directions (`|k|≤4`), mean
JVP cosine is 0.9811 and relative L2 error is 18.24%. On projected white
full-spectrum directions, the stricter values are 0.9654 and 26.00%.
Subtracting the exact full-horizon viscous-diffusion derivative leaves
nonlinear-residual JVP cosines of 0.9717 and 0.9881, respectively.

The full-output Jacobian was also evaluated on the complete 512-dimensional
real divergence-free Fourier subspace through `|k|≤4`. Across the three
recovery ICs, mean condition number is 7.142 for the surrogate versus 6.864
for XLB; the corresponding Gauss–Newton condition numbers are 51.05 and
47.17. Total restricted-Jacobian Frobenius cosine is 0.9961. After subtracting
the shared viscous baseline it is 0.9941, while the XLB nonlinear residual has
81.2% of the total Jacobian Frobenius norm. The restricted conditioning result
must not be generalized to the full 12,288-dimensional input space, and the
similar scalar condition numbers do not establish Jacobian equality.

The paper's conditioning protocol instead retains the complete raw velocity
state, constructs one Jacobian row per output degree of freedom by sequential
VJP, and applies a dense SVD. Because this solver is fixed at `N=16`, the
paper-protocol audit uses an explicit orthonormal `8³` block-grid
lift/restriction around the recovery IC, producing the same 1,536-dimensional
raw state as the paper while retaining the fixed recovery physics. Under this
protocol, the Jacobian Frobenius cosine falls to 0.9442 and relative error
rises to 32.94%. Raw condition numbers are `2.79e7` for XLB and `7.48e10` for
the surrogate. These tails fall below the float32 rank tolerance; condition
numbers over the resolved singular values are `5.32e3` and `5.45e3`,
respectively. The draft PR provides both normalized spectra, rank diagnostics,
and the dense matrices. This adapted fixed-task audit is distinct from the
paper's native `N=8` TGV physics; an XLB control at those native settings is
reported separately.

The larger dataset improves excluded-seed forward error from 7.473% to 4.309%
and full-spectrum JVP error from 38.28% to 26.00%; validation and held-out
errors improve together, so the observed inverse-model gap is not evidence of
conventional train/test overfitting. A remaining architectural limitation is
the amplitude gate on the learned correction: at the zero-velocity cold start,
the correction is second-order and the full model's first derivative is fixed
by the viscous skip. Consequently, more trajectory data cannot correct the
measured 49.34% radial JVP error at zero, where recovery begins.

Both optimizer variants from the paper were run on the exact self-recovery
benchmark. With unconstrained L-BFGS, the surrogate recovers the three ICs to
17.31%, 15.70%, and 17.10% relative L2 error (16.70% mean), compared with
4.96%, 5.17%, and 6.12% for XLB (5.42% mean). With divergence-free gradient
projection, the surrogate reaches 17.32%, 15.59%, and 17.19% (16.70% mean),
compared with 4.70%, 5.02%, and 5.94% for XLB (5.22% mean). Per-iteration
objective, true IC error, and optimized-IC divergence histories are recorded
for PhiFlow, XLB, Warp-NS, Exponax, and the surrogate.

The 16,384-trajectory checkpoint is therefore a better forward model but a
worse global inverse than the 4,096-trajectory checkpoint. The initial
self-target descent direction at exactly zero has nearly unchanged mean
cosine with the direction to the true IC (0.845 versus 0.847). Immediately
away from zero, however, the larger model's mean alignment falls to 0.419 at
IC amplitude 0.05 and 0.150 at amplitude 0.10, versus 0.825 and 0.807 for the
smaller checkpoint. Forward-only rollout selection does not constrain this
off-manifold inverse-gradient geometry; similar conditioning at the true IC
therefore does not predict recovery from the zero cold start.

Warm packaged in-process RTX 5090 medians are 7.24 ms for the 20-macro-step
forward rollout and 14.78 ms for its end-to-end VJP. XLB takes 4.81 ms and
14.37 ms on the same task. The autoregressive surrogate is therefore not a
kernel-level speedup: repeatedly applying the shared operator makes it 1.51×
slower for forward and 1.03× slower for VJP. Complete three-seed recovery
harness times remain similar (34.73 s surrogate versus 34.21 s XLB for
L-BFGS, and 33.38 s versus 35.78 s with projection) because RPC, optimizer,
projection, callbacks, and line search dominate these single-run wall times.

## Limitation: XLB-target inversion

Self-recovery follows the benchmark contract: each solver produces and inverts
its own final field. It does not establish that the surrogate can safely invert
an XLB-generated observation. In a separate cross-model test, projected L-BFGS
through this checkpoint against XLB final fields finishes at 32.1–41.0% IC
error (37.4% mean), despite reducing the surrogate residual to 1.6–2.0%.
The best saved IC errors are 17.5–18.9%, and XLB re-evaluation of the final
recovered ICs leaves 14.9–19.8% final-field residual. The checkpoint must not
be presented as a drop-in inverse model for XLB observations.

The draft PR contains both L-BFGS recovery variants, per-iteration loss, IC
error and divergence histories, full-field plots, animation, timing
decomposition, restricted and paper-protocol Jacobian spectra, and external
offline artifact provenance.
