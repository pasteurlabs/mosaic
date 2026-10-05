# Frozen corrector transfer

The independent INS confirmation met the registered accuracy criterion. This
follow-up tests whether its selected neural-corrector recipe transfers to
JAX-CFD, PhiFlow, PICT, Warp and XLB. It does not repeat hyperparameter selection.

## Fixed scientific comparison

- Coarse grid 64², same-solver reference grid 192²; viscosity 0.001, coarse
  timestep 0.01, four solver steps per correction interval.
- Kolmogorov forcing amplitude 1 and wavenumber 6; burn-in time 75;
  initial multimode amplitude 0.05, k0=4, sigma=1, periodic domain 2π.
- Training ICs 0–7, 24 training frames; confirmation ICs 20000–20031,
  48 evaluation frames; model seeds 8–15. IC seed identities match INS,
  but reference trajectories come from each evaluated solver.
- Same periodic residual CNN, architecture and optimizer definitions as INS.
- Full and stopped gradients: 3000 updates, unroll 16, lr=1e-4, from scratch.
- Fixed-pair supervision: 1000 updates, eight-pair windows, lr=1e-5.
- The stopped model is both the frozen selected baseline and the matched
  derivative ablation. There is no need to train it twice.

Compare methods within each solver. Report every solver, including admission
failures and runtime limits. Do not pool solver-specific reference errors into
a single cross-solver winner. Use the registered paired model/IC bootstrap and
the ≥5% mean reduction plus interval-upper-bound <1 criterion against supervision.
Report the matched stopped-gradient comparison separately. Per-solver intervals
are not a simultaneous familywise guarantee.

## Numerical gates

Before training, require finite reference trajectories, temporal convergence,
native-state closure and reference-accuracy checks. Preserve the existing 0.5%
reference-convergence threshold and 5% primary training-gradient threshold.
Every training and confirmation IC must pass admission, not only the preflight
ICs. A passing preflight is not a completed transfer experiment.

The initial preflight uses ICs 0 and 20000 and, if admitted, a ten-update
full-gradient run at the frozen horizon and learning rate. Based on earlier
reference failures, temporal refinement factors are prespecified as 48/96 for
JAX-CFD, 24/48 for XLB and 3/6 for PhiFlow, PICT and Warp. These change reference
integration accuracy, not grid resolution, physical time or model settings.

PhiFlow and Warp initially produced nonfinite values during the first 192²
burn-in, before reference trajectories or training. Preserve those failures.
Instrumented stability diagnostics may investigate implementation defects and
reference timesteps; they cannot silently replace the scientific configuration.
Any numerical repair or additional refinement must have a recorded rationale,
validation and frozen source/image/configuration identities before confirmation.
Changing the coarse numerical schedule must be explicitly reported as a transfer
variant, not described as the identical INS settings.

## Runtime and reproducibility

Prepare references independently by IC and assemble in a fixed order, verifying
source/image/physics identities and array checksums. Compute common training
normalization from all eight training ICs with the same original reduction.
Held-out fields never enter training or hyperparameter selection.

If a training arm exceeds an allocation, continuation must preserve the model,
Adam state, sampling RNG, schedule position, initial gradient diagnostics and
cumulative costs. Validate interrupted-versus-uninterrupted parity before use.
Restarting Adam or reducing the 3000-update budget is not equivalent continuation.
Separate fitting from batched evaluation so allocation limits do not alter the
test set. Keep all attempts and charge actual execution costs.

All simulations, numerical tests, bootstrap analysis and plotting run on Slurm.
The local machine and login controller perform only orchestration and metadata
handling. Published INS results and their frozen numerical source stay unchanged.

## Follow-up numerical diagnostics (2026-10-05)

The completed JAX-CFD transfer uses the original frozen recipe and image. The
remaining admission failures are retained; they are not negative training
results because fitting has not yet been admitted.

- PICT: test the recorded reference-only timestep ladder 6/12, then 12/24 if
  needed. The original 3/6 comparison had maximum discrepancy 1.99%.
- XLB: compare saved post-burn states at factors 12/24/48/96, then diagnose
  96/192/384 if needed. A separate direct-kernel comparison retains populations
  in float64 to isolate recurrent checkpoint quantization. These saved-state
  diagnostics do not admit a full dataset or replace the required burn-in.
- PhiFlow: investigate the failed finite-difference checks using complete
  perturbation sweeps, matched directions and precision comparisons through
  the full training duration. Preserve both original failed checks. The actual
  corrector model-parameter gradient check remains mandatory and unchanged.
- Warp: preserve Euler and SSPRK3-only failures. Test separately identified
  variants with a Poisson symbol matching the centered divergence/gradient,
  then skew-symmetric centered advection if required. Validate discrete
  projection, null modes, energy identities and derivatives before burn-in.

Any solver variant needs an immutable adapter/image identity and all reference
and training gates before its results can enter the transfer comparison. No
filtering, clipping, reduced update budget or relaxed threshold is authorized by
these diagnostic ladders.
