# XLB 3D initial-condition recovery surrogate

This Tesseract is a task-specific, direct full-field surrogate for the
`N=16`, `ν=0.01`, `dt=0.02`, 100-step periodic 3D initial-condition recovery
benchmark. It is not a general Navier–Stokes solver and is excluded from other
resolutions, physical parameters, horizons, geometries, and benchmark cells.

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

Dataset, weight, accuracy, gradient, conditioning, and recovery metrics are
reported in the draft PR and its external offline artifacts.
