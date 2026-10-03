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

The matched cells use the same random divergence-free ICs (seeds 0, 1, 2)
and physical parameters for every participating solver:

| Experiment                                    | Quantity measured                                                                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `forward/recovery_teacher`                    | Final-field relative L2 error against XLB, not an exact PDE solution                                                               |
| `gradient/recovery_fd_check`                  | Each solver's `sum(u_T²)` gradient against its own central finite differences, ten shared directions and eight relative step sizes |
| `cost/recovery_forward`                       | Twenty warm forward API calls per seed                                                                                             |
| `cost/recovery_vjp`                           | Twenty warm `sum(u_T²)` gradient calls per seed, including forward and VJP                                                         |
| `optimization/recovery_constant_ic_bfgs`      | Self-target IC recovery from zero, 100 unconstrained L-BFGS updates                                                                |
| `optimization/recovery_constant_ic_bfgs_proj` | The same recovery with divergence-free gradient projection                                                                         |

The forward comparison requires an XLB result; it never falls back to a
surrogate-only consensus or a Taylor–Green analytic reference. Derivative
consistency verifies the learned map's implementation, not agreement with
the teacher's Jacobian. Recovery uses each solver's own observation, as in
the existing benchmark, so it does not establish XLB-target inversion.

Run both solvers through the normal CLI (Docker with GPU support required):

```bash
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e forward/recovery_teacher
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e gradient/recovery_fd_check --no-build
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e cost/recovery_forward --no-build
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e cost/recovery_vjp --no-build
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e optimization/recovery_constant_ic_bfgs --no-build
mosaic run -p ns-3d-grid -s xlb,xlb-3d-surrogate -e optimization/recovery_constant_ic_bfgs_proj --no-build
```

Each matched forward/gradient/cost experiment has `seed_0`, `seed_1`, and
`seed_2` sub-results, with the standard JSON envelope, field snapshots, and
plots. Timings exclude the warmup call but include the chosen API transport;
they are not isolated neural-network kernel timings. XLB retains its default
float64 path and the surrogate uses float32. Record the runtime, precision,
GPU, transport, and raw trials when comparing results.

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

## Matched validation on current main

A fresh run of the registered cells (Slurm job `2869896`, RTX 5090) used
seeds 0/1/2, 100 recovery updates, and 20 warm timing trials per seed. The
checkpoint was unchanged. Means across the three seeds, or all 60 timing
trials, are:

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
the restricted JVP statistics from the historical study below.

Timing used HTTP/base64 services and a CPU JAX client, with XLB's default
float64 path and the float32 surrogate. One allocation and fixed solver
order establish a measured comparison, not a hardware-independent speedup.
The run used existing runtime images with the current source/API/shared
model/weights mounted in; it did not rebuild those images. The generated
Docker build context was checked separately.

[Raw benchmark envelopes, field snapshots, plots, summary script, and provenance](https://github.com/pasteurlabs/mosaic/tree/surrogate-standalone-results)
are available separately from the solver source.

## Historical offline validation (PR #118)

All experiments were run offline through the shared Slurm and Pyxis/Enroot
cluster path. On the three excluded recovery seeds, mean final-field relative
L2 error is 4.309% and mean field cosine is 0.999076. Across the 1,982
held-out trajectories with IC amplitude at least 0.05, mean final-step
relative L2 error is 7.219%. The 66 lower-amplitude cases are reported with
absolute RMS error because relative error is ill-conditioned near zero.

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

Against the 4,096-trajectory checkpoint under this same restricted audit,
increasing the dataset changes the surrogate condition number from 7.740 to
7.142 and the Gauss–Newton condition number from 59.98 to 51.05. The
restricted-Jacobian Frobenius relative error improves from 13.50% to 8.84%
and cosine from 0.9914 to 0.9961.

An adapted dense audit uses an explicit orthonormal `8³` block-grid
lift/restriction around the fixed `N=16` recovery map, producing the same
1,536-dimensional matrix size as the paper's raw-state protocol. It covers
every coordinate of that coarse block-grid map, but it is not the complete
12,288-dimensional production Jacobian and is distinct from the paper's
native `N=8` TGV physics. Under this audit, the Jacobian Frobenius cosine falls
to 0.9442 and relative error rises to 32.94%. Raw condition numbers are
`2.79e7` for XLB and `7.48e10` for the surrogate. These tails fall below the
float32 rank tolerance; condition numbers over the resolved singular values
are `5.32e3` and `5.45e3`, respectively. The draft PR provides both normalized
spectra, rank diagnostics, the dense matrices, and a native `N=8` TGV XLB
control.

The adapted block-grid comparison is essentially unchanged by scaling the
training set: the 4k/16k Frobenius errors are 32.98%/32.94%, while the
float32-resolved condition numbers are `4.89e3`/`5.45e3`. Their raw condition
numbers are `3.75e10`/`7.48e10`, so the unresolved spectral tail becomes worse
even as the restricted Jacobian improves.

The larger dataset improves excluded-seed forward error from 7.473% to 4.309%
and full-spectrum JVP error from 38.28% to 26.00%; validation and held-out
errors improve together, so the observed inverse-model gap is not evidence of
conventional train/test overfitting. A remaining architectural limitation is
the amplitude gate on the learned correction: at the zero-velocity cold start,
the amplitude factor strongly suppresses the learned correction's first
derivative, leaving the viscous skip dominant. The implemented square-root
floor makes this suppression finite, rather than an exactly fixed derivative.
More trajectory data alone did not correct the measured 49.34% radial JVP
error at zero, where recovery begins.

End-to-end VJPs were also checked against central finite differences on the
exact recovery physics, using three IC seeds, ten shared unit-norm random
directions per seed, and a 12-point relative-ε sweep. For the paper's
energy-like objective `sum(u_T²)` at the true IC, the best aggregate median
directional error is `6.63e-3` for the float32 surrogate versus `6.77e-6` for
XLB; the corresponding mean direction cosines are 0.999984 and effectively
1.0. For the actual self-recovery MSE at the zero cold start, the surrogate is
more FD-consistent: its best median error is `2.56e-4` with mean cosine
1.000000, versus `1.32e-1` and 0.991669 for XLB. The surrogate recovery
failure is therefore not caused by an incorrect VJP implementation: finite
differences verify the derivative of the learned forward map, while the
inverse-path diagnostic shows that this internally accurate derivative points
toward the wrong learned inverse basin.

Both optimizer variants from the paper were run on the exact self-recovery
benchmark. With unconstrained L-BFGS, the surrogate recovers the three ICs to
17.31%, 15.70%, and 17.10% relative L2 error (16.70% mean), compared with
4.96%, 5.17%, and 6.12% for XLB (5.42% mean). With divergence-free gradient
projection, the surrogate reaches 17.32%, 15.59%, and 17.19% (16.70% mean),
compared with 4.70%, 5.02%, and 5.94% for XLB (5.22% mean). Per-iteration
objective, true IC error, and optimized-IC divergence histories are recorded
for PhiFlow, XLB, Warp-NS, Exponax, and the surrogate.

The 16,384-trajectory checkpoint is therefore a better forward model but a
worse global inverse than the 4,096-trajectory checkpoint. Under complete
matched runs, the smaller checkpoint reaches 8.10% mean IC error with L-BFGS
and 8.12% with projected L-BFGS, versus 16.70% for both variants with the
larger checkpoint. The seed-0 final objectives are nevertheless comparable:
`1.28e-8`/`1.33e-8` for the 4k checkpoint and `1.15e-8`/`1.18e-8` for the 16k
checkpoint (unconstrained/projected). The 16k seed-0 final divergence is also
slightly lower, at `1.49e-2`/`1.43e-2` versus `1.77e-2`/`1.77e-2`.

The initial self-target descent direction at exactly zero has nearly
unchanged mean cosine with the direction to the true IC (0.845 versus 0.847).
Immediately away from zero, however, the larger model's mean alignment falls
to 0.419 at IC amplitude 0.05 and 0.150 at amplitude 0.10, versus 0.825 and
0.807 for the smaller checkpoint. Forward-only rollout selection does not
constrain this off-manifold inverse-gradient geometry; similar conditioning
at the true IC therefore does not predict recovery from the zero cold start.

Two matched RTX 5090 timing blocks counterbalance solver order and contribute
40 warm trials per solver. Their combined medians are 7.35 ms for the
20-macro-step surrogate forward and 14.77 ms for its end-to-end VJP; XLB takes
4.79 ms and 15.20 ms on the same task. The surrogate is therefore 1.53× slower
for forward, while VJP cost is effectively at parity (0.97× here; an earlier
independent run measured 1.03×). Solver-scoped three-seed harness times remain
similar: 32.81 s surrogate versus 32.09 s XLB for L-BFGS, and 31.51 s versus
32.50 s with projection. Including result-script setup and serialization gives
34.73 s versus 34.21 s and 33.38 s versus 35.78 s, respectively. RPC,
optimizer, projection, callbacks, and line search dominate these single-run
wall times.

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
decomposition, 4k/16k restricted and paper-protocol Jacobian spectra, the
inverse-gradient diagnostic, finite-difference U-curves, and external offline
artifact provenance.

## Experimental input-derivative supervision

`sobolev.py` provides an offline fine-tuning pilot, separate from the shipped
checkpoint. It matches terminal velocity and full 100-step VJPs using identical
output cotangents for XLB and the surrogate. This is input-Jacobian supervision,
whereas the existing trainer's spectral loss weights spatial frequencies.
See [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859).

The teacher supplies first derivatives only, cached as fixed labels. Optimizing
the student's VJP loss requires mixed input/parameter second derivatives through
the native JAX surrogate. It does not require second derivatives through the
Tesseract API. Small-model tests check these mixed derivatives against finite
differences; full-size GPU training and improvement on recovery are not yet
validated.

Generate labels in the XLB environment, with this directory on `PYTHONPATH`:

```bash
python sobolev.py generate --teacher-api /tesseract/tesseract_api.py \
  --dataset /surrogate-output/recovery_3d_xlb_trajectories.npy \
  --output /surrogate-output/sobolev-labels.npz
```

Then run matched pilots in the surrogate environment, using distinct output
files for weights `0`, `0.01`, `0.1`, and `1`:

```bash
python sobolev.py train --labels /surrogate-output/sobolev-labels.npz \
  --init-weights weights.npz --weight 0.1 --updates 500 \
  --output /surrogate-output/sobolev-0.1.npz
```

All pilots use identical starting weights, normalization, batches, and update
counts. Weight zero is the additional-training control with the same terminal
field loss and data; it is not the original curriculum loss. It also evaluates
VJPs, so pilot training timings do not measure the minimal field-only cost.
Labels sample 128 training and 32 validation trajectories by default, with
amplitude factors drawn from zero, 0.1, 0.5, and 1. Test trajectories are excluded.
Teacher calculations use float64; saved labels use float32. Every sample has
one fixed random unit output cotangent. This is a small derivative-supervision
pilot, not a complete Jacobian dataset or coverage of arbitrary recovery states.

Checkpoints are selected by validation field relative MSE plus the weighted VJP
relative MSE; both components are logged. Scores with different weights are not
directly comparable. Compare selected checkpoints on fresh cotangents, forward
error, gradient cosine/error, both existing recovery benchmarks, and runtime.
Keep final test results separate from hyperparameter selection. Follow-up arms
should compare more cotangents, recovery-residual cotangents, broader input
coverage, and an architectural change permitting a stronger learned derivative
near zero. The existing amplitude gate strongly suppresses that derivative;
Sobolev training alone is not guaranteed to remove this restriction.

### Alternatives and experiment order

The objective is teacher-faithful sensitivities and better held-out recovery.
A smaller condition number alone is not success: viscosity physically damps
high-frequency modes, and the implemented projection removes some input
directions. Forcing all singular values toward one would change the solver.
Report resolved rank, singular-value spectra with explicit precision thresholds,
and derivative alignment along recovery paths, rather than raw condition numbers
below the numerical noise floor.

| Priority | Experiment                            | Hypothesis and limitation                                                                                                                                                                                                              |
| -------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Central-secant supervision            | Match `F(x + h d) - F(x - h d)` to XLB; avoids mixed second derivatives in training. Sweep perturbation sizes because finite differences introduce truncation and cancellation errors.                                                 |
| 1        | Near-zero linear correction           | Add a zero-preserving learnable linear branch beside the gated nonlinear correction. Test its zero-state Jacobian against XLB before training. Simply removing the gate is a separate control, not an established fix.                 |
| 2        | Recovery-path data                    | Query XLB at actual optimization iterates from training targets, including perturbations away from amplitude-scaled ICs. Split by original target before generating paths; never train on final benchmark trajectories.                |
| 2        | Task-directed derivative labels       | Supplement random cotangents with normalized XLB recovery residuals. Match the same cotangent for teacher and student, then independently compare their actual objective gradients. Residual-only training may overfit one objective.  |
| 2        | Reduced-space recovery and damping    | Optimize divergence-free Fourier coefficients, progressively admit higher frequencies, and compare damped Gauss–Newton with L-BFGS. This changes the inverse algorithm/prior, not the learned solver; apply the same procedure to XLB. |
| 3        | Teacher-informed spectral supervision | Match Jacobian actions in selected resolved directions, including weak directions. Avoid penalties demanding arbitrary invertibility or suppressing every large derivative. Probe unexplored directions separately.                    |
| 3        | Shorter rollout or improved state     | Compare fewer learned macro-steps or an augmented recurrent state. XLB evolves hidden populations, whereas the surrogate advances velocity alone; test whether this limits derivative fidelity rather than assuming it does.           |

Reduced derivative representations are motivated by
[Derivative-Informed Neural Operators](https://arxiv.org/abs/2206.10745), which
compress derivative information to make training practical. Their published
results do not establish an improvement for this XLB checkpoint. The experiments
above are hypotheses, not measured improvements.

The central-secant arm is implemented in `sobolev.py`. Add `--secant-step 0.01`
to label generation, then `--method secant` to training. It uses projected random
input directions normalized to unit RMS. The perturbation RMS is the requested
fraction of `max(input RMS, 0.01)`, so zero anchors still receive nonzero probes.
Labels store the normalized central difference, computed with float64 XLB and
saved as float32. VJP labels are also generated for the same ICs. Try relative
steps `0.003`, `0.01`, and `0.03` on validation data; generate each label set to a
separate file. Unit tests check central-difference convergence and optimizer
updates, not empirical performance on XLB.

For each candidate, record validation forward error; fresh-cotangent VJP and
fresh-direction JVP errors/cosines; gradients of the actual recovery objective at
zero and along optimization paths; resolved Jacobian spectra; both self-target
and XLB-target recovery; runtime and peak training memory. Compare methods under
both fixed update counts and a fixed compute budget. Select using validation
cases, then run the existing held-out solver benchmarks once. A successful
candidate must reduce recovery error without an unacceptable forward-accuracy
or runtime regression. Keep architecture and inverse-optimizer changes as
separate ablations before combining them.

### October 3 pilot submission

The first GPU pilot uses eight 500-update arms from the shipped checkpoint:
field-only; VJP weights 0.01, 0.1, and 1; central-secant weight 0.1;
VJP weight 0.1 with a zero-initialized linear correction; and field-only/VJP
weight 0.1 on recovery-path data. The linear branch learns real 3×3 channel
mixing per squared-wavenumber shell, preserving the zero state and permitting
a learned first derivative there. It is enabled by `--linear-correction` and
stored as the optional `w_linear` checkpoint parameter. Existing checkpoints
retain their original behavior. No retrained checkpoint has replaced the
packaged weights.

The offline validation pilot uses three original validation trajectories,
fresh output cotangents, gradient probes along amplitude-scaled paths, and
100-iteration SciPy L-BFGS-B recovery with self and XLB targets. A separate
Fourier-restricted recovery arm is applied to XLB and every surrogate. These
SciPy runs differ from the registered benchmark optimizer and are labeled
accordingly. Candidate selection uses full-space XLB-target validation recovery
error; the selected checkpoint subsequently runs the standard registered
experiments. All pilot jobs use the preemptible `nice` queue because the regular
RTX pool is at its per-user GPU limit. Results remain pending until those jobs
complete; submission is not evidence of improvement.
