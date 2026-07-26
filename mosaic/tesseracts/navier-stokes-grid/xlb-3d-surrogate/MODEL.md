# XLB 3D initial-condition recovery surrogate

This Tesseract is a task-specific, direct full-field surrogate for the
`N=16`, `ν=0.01`, `dt=0.02`, 100-step periodic 3D initial-condition recovery
benchmark. It is not a general Navier–Stokes solver and is excluded from other
resolutions, physical parameters, horizons, geometries, and benchmark cells.
The canonical drag output is zero because this triply-periodic task has no
obstacle; there is no learned drag head.

## Model contract

The differentiable input is the complete `16 × 16 × 16 × 3` initial velocity
field. A periodic 3D Fourier neural operator maps that initial condition
directly to the full final velocity field. It is not an autoregressive or
one-step model.

The network has width 32, six retained Fourier modes per axis, and six residual
spectral blocks. An exact linear viscous-diffusion operator supplies the
zero-state Jacobian; the neural operator learns the finite-amplitude nonlinear
correction. Inputs and outputs are Helmholtz-projected so the model and its
recovery gradients remain on the divergence-free periodic manifold.

## Training provenance

Training and dataset-generation code remain outside this repository. The
packaged weights were trained from 8,192 XLB KBC D3Q27 trajectories, split
6,144/1,024/1,024 for training/validation/test. Benchmark IC seeds 0, 1, and 2
were excluded. The training distribution contains random divergence-free
fields around the benchmark's `|k|=2` energy shell, broader off-manifold
spectra, and amplitudes from zero to 1.25.

Directional derivatives from 4,096 float64 XLB JVPs were used for gradient
distillation. This is important for the inverse task: field accuracy alone does
not establish that an optimizer sees the teacher's local geometry.

The packaged checkpoint SHA-256 is
`8669ebaf92d920668c945b04722021fef12d5fc6fa4aa182abc63c81780ed0c9`.

## Offline validation

All validation was run offline through the same Slurm and Pyxis/Enroot path as
the solver benchmarks. On benchmark seeds 0, 1, and 2, which were excluded from
training, mean final-field relative L2 error is 7.151% and mean field cosine is
0.997441. Mean random projected-JVP cosine is 0.849667; these white-spectrum
directions are harsher than the low-frequency recovery manifold.

On the exact self-recovery benchmark, projected L-BFGS recovers the three ICs
to 6.20%, 6.92%, and 8.15% relative L2 error (7.09% mean), compared with 4.70%,
5.02%, and 5.94% for XLB (5.22% mean).

The full-output Jacobian was also evaluated on the complete 512-dimensional
real divergence-free Fourier subspace through `|k| <= 4`. Across the three
recovery ICs, the surrogate's mean condition number is 6.56 versus 6.86 for
XLB; the corresponding Gauss–Newton condition numbers are 43.09 and 47.17.
The restricted Jacobians have mean Frobenius cosine 0.9901.

Warm in-process packaged medians are 0.920 ms for forward and 1.082 ms for VJP,
versus 4.812 ms and 14.368 ms for XLB. The exact remote harness takes 24.48 s
versus 30.96 s because RPC, callbacks, L-BFGS bookkeeping, projection, and line
search dominate this small cell.

## Limitation: cross-model inversion

Self-recovery follows the benchmark contract: each solver produces and inverts
its own final field. It does not imply that the surrogate can safely invert an
XLB-generated observation. In a separate cross-model test, optimizing through
the surrogate against XLB final fields drives surrogate residuals to 1.3–1.6%
but finishes at 40.3–42.5% IC error (41.5% mean); the best saved snapshots are
17.8–20.9% IC error. XLB re-evaluation of those recovered ICs leaves 6.5–9.2%
final-field residual. This checkpoint must therefore not be presented as a
drop-in inverse model for XLB observations.

The draft PR contains the full per-seed results, plots, animation, and external
offline artifact provenance.
