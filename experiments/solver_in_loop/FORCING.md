# Forced periodic flow: reference and gradient preflight

This extends the common benchmark stepping map with the periodic acceleration
`f(x,y) = (a sin(2π k y/L), 0)`. It is inspired by the Kolmogorov case in
[List et al.](https://arxiv.org/html/2402.12971v2#S3.SS3), but uses our existing
solvers and the common 64²→192² resolution pair. It is not an exact replication.

For each native timestep, the harness applies a half force kick, advances the
solver once, then applies the second half kick. A solver's native checkpoint is
preserved between calls; the next call assimilates the canonical velocity change
just as it does a neural correction. Both the forward and backward paths use
this composition. Setting force amplitude to zero retains the original batched
solver call exactly.

The force is zero-mean and divergence-free. Its integer wavenumber must lie below
Nyquist. The canonical grid uses y=j L/N, consistent with existing ICs. Halving
the reference dt halves both the solver timestep and force-splitting interval.
Temporal audit and closure checks therefore cover the full forced stepping map.

A burn-in phase evolves each fine-grid IC before saving training or evaluation
frames. The initial condition is a small multimode perturbation plus 0.1 f.
References and their temporal audits start from the same saved post-burn-in
canonical field with fresh native checkpoints. The time audit checks evolution
from that initial field, not convergence of burn-in itself. Burn-in diagnostics
record initial/final velocity RMS; they do not certify statistical stationarity.

First campaign: `pr116-forced-preflight-20261001`.

- JAX-CFD and INS-JL, same resolution 64² coarse and 192² reference.
- Force amplitude 1, wavenumber 6, viscosity 0.001, random perturbation amplitude
  0.05. Coarse dt=0.01, reference dt=0.01/3, audit dt=0.01/6.
- Four coarse steps per correction interval, 48 evaluation intervals.
- Short physical burn-in of 5, one training IC (seed 0), one held-out IC (100).
- Reference-only mode: generate references, check closure, finite states and
  time accuracy; if admitted, check a two-interval end-to-end directional VJP.
  No neural training is performed.

The 5-unit burn-in is an engineering preflight, not a developed-turbulence claim.
The published forced-flow setup used a much longer burn-in. Successful preflight
must be followed by longer burn-in, inspection of flow development and production
reference checks before a training comparison is meaningful. Per-native-step
remote calls also add cost; measure that cost before scaling the dataset.

```sh
# With a frozen campaign and validation dependency:
PYTHONPATH=/home/andrinr/slurm-runner .venv/bin/python \
  experiments/solver_in_loop/submit.py train --campaign CAMPAIGN \
  --solvers jax-cfd,ins-jl --regimes 64:4:8 --seeds 0 --updates 1 \
  --forcing-amplitude 1 --forcing-wavenumber 6 --amplitude 0.05 \
  --dt 0.01 --burn-in-time 5 --reference-only --after VALIDATION_JOB
```

Reference-only outcomes are stored under `reference_checks` in the report,
separately from trained comparisons and runtime failures. All numerical testing,
reference generation, VJP checks and rendering run through Slurm on the cluster.

## Preflight results and next check (2026-10-01)

Both short-burn-in probes completed without training. INS-JL passed the temporal
reference audit (maximum relative difference 0.00004247) and recurrent directional
VJP check (relative difference 0.0001185). JAX-CFD failed the temporal audit
(0.09574 against the unchanged 0.005 limit), so its gradient check was not run.
These are engineering checks, not evidence that learned correction helps.

The next reference-only campaign, `pr116-forced-longburn-20261001`, keeps both
solvers at 64²→192², coarse dt=0.01 and four steps per correction. It increases
burn-in to 75 physical units and refines reference/audit timesteps to 0.01/12
and 0.01/24 respectively. Both solvers use identical settings and seeds 0/100.
The rollout duration stays 1.92; the accuracy threshold stays 0.005. This tests
longer-developed flow and finer reference integration together; it does not
isolate which change explains a different audit outcome. Passing still requires
flow inspection before choosing a training protocol, and does not certify
stationarity or spatial convergence.
