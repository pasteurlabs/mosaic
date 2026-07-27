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

Training and dataset-generation code remain outside this repository. The
packaged weights were trained from 4,096 native XLB KBC D3Q27 trajectories,
split 3,072/512/512 for training/validation/test. Each trajectory contains the
IC plus 20 full-field snapshots, one every five XLB steps. The final generated
snapshot matched the canonical 100-step float32 teacher call with maximum
absolute difference zero. Benchmark IC seeds 0, 1, and 2 were excluded.

Training used autoregressive unroll curriculum
`1 → 2 → 4 → 8 → 12 → 20`, followed by a full-20-step fine-tune. The
distribution contains random divergence-free fields around the benchmark's
`|k|=2` energy shell, broader spectra for optimizer-path coverage, and
amplitudes from zero to 1.25.

The native trajectory dataset SHA-256 is
`f94390c512c44d893581017007720077d85f2c153d28d8841630e0adc1e60dc8`.
The packaged checkpoint SHA-256 is
`03178cee7457849e28ecd4f6f97bfa70b111cebedc0847f882d8362abf493835`.

## Offline validation

All experiments were run offline through the shared Slurm and Pyxis/Enroot
cluster path. On the three excluded recovery seeds, mean final-field relative
L2 error is 7.473% and mean field cosine is 0.997619. Across the 491 nonzero
held-out test trajectories, mean final-step relative L2 error is 9.905%.

Directional derivatives were evaluated end to end through all 20 shared
operator applications. On low-frequency projected directions (`|k|≤4`), mean
JVP cosine is 0.9846 and relative L2 error is 17.64%. On projected white
full-spectrum directions, the stricter values are 0.9253 and 38.28%.
Subtracting the exact full-horizon viscous-diffusion derivative leaves
nonlinear-residual JVP cosines of 0.9752 and 0.9757, respectively.

The full-output Jacobian was also evaluated on the complete 512-dimensional
real divergence-free Fourier subspace through `|k|≤4`. Across the three
recovery ICs, mean condition number is 7.740 for the surrogate versus 6.864
for XLB; the corresponding Gauss–Newton condition numbers are 59.98 and
47.17. Total restricted-Jacobian Frobenius cosine is 0.9914. After subtracting
the shared viscous baseline it is 0.9861, while the XLB nonlinear residual has
81.2% of the total Jacobian Frobenius norm. The restricted conditioning result
must not be generalized to the full 12,288-dimensional input space.

On the exact self-recovery benchmark, projected L-BFGS recovers the three ICs
to 6.97%, 7.85%, and 9.54% relative L2 error (8.12% mean), compared with
4.70%, 5.02%, and 5.94% for XLB (5.22% mean).

Warm packaged in-process RTX 5090 medians are 7.12 ms for the 20-macro-step
forward rollout and 14.86 ms for its end-to-end VJP. XLB takes 4.81 ms and 14.37 ms on
the same task. Unlike the replaced direct IC-to-final checkpoint, the
autoregressive surrogate is therefore not a kernel-level speedup: repeatedly
applying the shared operator makes it 1.48× slower for forward and 1.03×
slower for VJP. The matched RTX 5090 recovery harness takes 33.17 s versus
30.96 s for XLB (1.07× slower); shared RPC, optimizer, projection, and
line-search overhead narrows the kernel difference.

## Limitation: XLB-target inversion

Self-recovery follows the benchmark contract: each solver produces and inverts
its own final field. It does not establish that the surrogate can safely invert
an XLB-generated observation. In a separate cross-model test, projected L-BFGS
through this checkpoint against XLB final fields finishes at 49.2–53.4% IC
error (51.5% mean), despite reducing the surrogate residual to 1.7–2.4%.
The best saved IC errors are 20.4–24.6%, and XLB re-evaluation of the final
recovered ICs leaves 11.2–16.1% final-field residual. The checkpoint must not
be presented as a drop-in inverse model for XLB observations.

The draft PR contains the exact self-recovery results, paper-style loss and IC
error histories, full-field plots, animation, timing decomposition, Jacobian
spectra, and external offline artifact provenance.
