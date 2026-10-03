# Frozen follow-up: confirmation and stronger baselines

This round follows the three-seed forced-INS result (full 3.751%, stopped
4.476%, supervised 39.061% mean rollout error). All numerical work and plotting
runs through Slurm. No benefit is assumed in advance. Resolution remains
64² coarse and 192² same-solver reference, with the unchanged reference and
gradient thresholds.

## INS replication

Campaign `pr116-ins-confirm-20261002` freezes the pilot's physical setup,
training ICs 0–7, 1,000 updates, horizon 8 and lr=1e-4. Eight model seeds 0–7
use new test ICs 2000–2003. Keeping four test ICs preserves the dataset size and
fits the four-hour allocation limit. The historical submission filename prefix
is `explore`; this is a prospectively fixed confirmation, not a new selection.

Initial jobs 2860033–2860041 included five container-start failures on `rtx03`
(before numerical work). Those logs remain; seeds 3–7 were resubmitted unchanged
as 2860047–2860051 with that node excluded. Report job: 2860053.

## Supervised baseline selection and final comparison

Campaign `pr116-ins-baseline-validation-20261002` uses training ICs 0–7 and
**validation ICs 3000–3003**, model seeds 0–2, horizon 8, learning rates
{1e-5, 3e-5, 1e-4} and update counts {1,000, 3,000}. All 18 jobs and six
candidates are retained. The full/stopped arms take only one update in this
selection campaign to avoid redundant training; these are **not three-arm
performance comparisons**. Only supervised validation rollout error selects
settings. Ties prefer fewer updates, then a lower learning rate. Nonfinite or
incomplete candidates stop automatic selection rather than disappearing.

The code audit confirms fixed pairs advance each reference frame to the next
frame's target, preserving and reconciling native state as in rollout. New
metrics measure teacher-forced one-step error on training and validation/test
ICs, alongside free-running error. Evaluation pairs never enter optimization.
This distinguishes an inaccurate one-step map from accumulated rollout drift;
it does not presume the latter explains the pilot failure.

Before reading new test outcomes, freeze the selected rate/update count in
`baseline-selection.json`. Two new campaigns then use **test ICs 4000–4003**
and eight model seeds 0–7:

- `pr116-ins-tuned-confirmation-20261002`: full/stopped retain the pilot's
  1,000-update schedule; supervision uses the selected validation settings.
- `pr116-ins-compute-confirmation-20261002`: supervision uses the selected rate
  with a training-time budget equal to measured full-gradient training time,
  subtracting fixed-pair generation cost. Stop before the next update after
  exhausting that budget; report actual time and update counts. One update can
  overshoot the budget. Shared reference generation and post-training evaluation
  are excluded. The measured full-gradient time includes its initial finite-
  difference check, giving supervision a conservative extra allowance. The
  entire first update bounds this overhead by 2.61% in the three pilot runs;
  the diagnostic alone costs less. The optimizer has a 100,000-update safety cap.

The settings, sample sizes and selection metric are fixed before those tests.
Report both comparisons even if the advantage disappears. Separate supervised
budgets are explicit in report metadata. Validation jobs: 2860063–2860080;
report 2860081. Cluster validation of baseline changes: 696 passed, 3 skipped.

## Warp derivative refinement

Campaign `pr116-warp-seed7-check-20261002`, job 2860031, repeats the fresh-IC
protocol's initial seed-7 gradient check with one training update. Epsilons
0.01, 0.003, 0.001, 0.0003 and 0.0001 are evaluated at the **same model, window,
gradient and random direction**. Autodiff was -0.0174129 throughout; relative
discrepancies were 0.0927893, 0.00367609, 0.00931098, 0.00487107 and 0.0302218.
The intermediate steps support the derivative; the curve is not monotonically
convergent at smaller steps. Preserve the failed primary 0.01 check and the
whole curve rather than replacing the primary metric with its best value.
This checks the initial point, not every trained state. Report job: 2860054.

## JAX-CFD transfer, conditional on reference admission

Job 2860032 in `pr116-jax-reference-refine-20261002` refines temporal factors
from 12/24 to 24/48, preserving forcing, burn-in, coarse dt, grids and admission
threshold. Report: 2860055. If it fails, the frozen next probe uses 48/96; a
second failure stops automatic transfer instead of relaxing the threshold.

Campaign `pr116-jax-forced-transfer-20261002` contains a tested optional cache
of canonical post-burn-in fields, needed to fit the finer reference into Slurm's
four-hour limit. Keys bind source archive, immutable solver-image path, physics,
initial-field hash, integration grouping and domain. Stored shape, dtype and
content checksum are verified; writes are atomic. Cached states start with a
fresh native checkpoint exactly as before. Reference rollouts, temporal audits
and closure checks still run fresh. Cache hits and field hashes are recorded.

After code validation 2860062 and a passing temporal probe, prepare all ICs
0–7 and 100–103 in separate reference-only jobs, then repeat the three-arm
1,000-update INS pilot protocol with JAX-CFD for three model seeds. Cache
preparation also audits each IC; any admission failure stops automatic training.
Temporal factors may differ between solvers to meet the same accuracy limit;
resolutions do not. Compare methods within a solver, not absolute errors across
solver-specific references.

## Continuing without an open chat turn

The one-shot controller and frozen launcher run on `kander-login` in
`/data/personal/andrinr/runner/results/mosaic/pr116-controller-20261002/`
(PID 3163552); local drafts are in `mosaic-results/pr116-next-round/`. `status.json`, `submissions.jsonl`,
`baseline-selection.json` and `controller.log` record progress and decisions.
It performs only lightweight JSON and Slurm submission on the login node. Training,
reference generation, tests and rendering remain cluster jobs. Failed gates
stop their dependent branch and remain reported. Cluster validation including the cache passed 697 tests, with three skips.
The publication step checks
for intervening PR edits before replacing its description.

## Speed work

INS final comparisons also use the verified burn-in cache, prefilled in
`pr116-ins-burn-cache-20261002` before reading final test outcomes. This changes
reference-generation cost, not the dynamics or training protocol. Every IC is
still audited, and state hashes are recorded. Common training IC 0 is prepared
once before dependent jobs to avoid simultaneous duplicate burn-in. The cached
campaign source is the same validated source as the JAX transfer campaign;
solver images are unchanged for these frozen comparisons.

The INS recurrent VJP formerly ran two extra forward solves for a viscosity
finite difference even when viscosity was absent from `vjp_inputs`. The adapter
now passes a request flag and skips only those unused solves. The requested
viscosity path remains unchanged. Cluster check 2860184 compares old/new forward
outputs and all requested derivatives in 2D at 64/192 and 3D at 8, with initial
and recurrent state. Requested velocity/state/timestep gradients match exactly
between the candidate's complete and reduced-gradient paths; old/new checks
also pass. Five interleaved warmed-call samples give 1.34× median speedup for
64² recurrent four-step VJPs, 1.38× for 64² initial one-step VJPs, and 1.28–1.30×
in the 3D checks. The 192² timings have substantial variability. These are
in-process VJP timings, not measured end-to-end training speedups. Raw timings
are in `pr116-ins-vjp-speed-20261002/verification.json`. The initial test job
2860159 failed in its Python assertion helper before any gradient comparison;
the corrected test passed. Existing experiment images remain frozen; the
optimization is built and verified separately before future use.

## Completed checks and selected baseline (2026-10-02)

The eight-seed confirmation on ICs 2000–2003 completed with mean rollout errors
3.582% full, 4.447% stopped, 50.211% original supervision, and 5.407% uncorrected.
All training gradient checks passed. This confirms the original recipe's result,
not superiority to tuned supervision. Validation selected supervision lr=1e-5,
1,000 updates (2.663% validation error). Final tuned and compute comparisons on
ICs 4000–4003 are separate runs; validation and confirmation errors must not be
compared as though they used the same test data.

Reference-only diagnostics now honor the configured perturbation and retain raw
AD/FD values for every requested step. Cluster validation 2860645 passed 698
tests, with three skipped. The earlier INS cache IC6 primary check remains failed
(72.33% at epsilon .01). Exact saved-state probes reproduce it. Four additional
fixed directions pass at .01 (0.13–0.47%); the original directional derivative
is only 2.62e-8. A near-null direction in control IC5 exhibits the same issue.
This supports float32 cancellation as the explanation, without replacing the
failed primary check. Raw curves: `pr116-saved-reference-fd-20261002`, retry
jobs 2860656 and 2860830; the initial diagnostic-wrapper failure is retained.

The optimized INS image gives 1.064× end-to-end speedup in a 30-sample repeat,
with exactly matching loss, gradients and Adam updates. Optional RPC-boundary
JIT gives another 1.080× in an isolated measurement with strict comparisons
passing. Full-graph JIT gives 2.16–2.35× but fails strict numerical equivalence;
it is exploratory and is not used in frozen scientific comparisons.

PhiFlow's periodic pressure projection now uses the exact discrete staggered
FFT operator instead of its unstable singular float32 CG solve. Isolated GPU
checks cover native divergence/gradient operators, rectangular 2D/3D grids,
self-adjointness, whole-step forward agreement and finite differences (2860841).
Repeated reference jobs 2860836/37 produce identical dataset hashes, zero prefix
closure error, 0.0614% temporal discrepancy and 0.0168% gradient discrepancy.
Reference generation takes 14–15 seconds versus 275–341 seconds with the original
image. These are reference-generation timings, not training speedups. A rebuilt
image must pass admission before training; existing campaigns retain old images.

The final tuned comparison has now completed on all eight seeds and fresh ICs
4000–4003: full 3.4984%, stopped 4.2997%, tuned supervision 2.8319%, native 5.9978%.
All run admission and training-gradient checks pass. The paired supervision/full
error ratio is 0.8095 (95% CI 0.7519–0.8770): tuned supervision is better. Full
beats stopped by 18.6%, but does not establish superiority over strong supervision.
Full training took 13,818 seconds summed across seeds, versus 4,410 seconds for
supervision including pair generation. The compute-matched comparison cannot
reverse this conclusion: forcing supervision to spend more time can overtrain,
and a within-budget baseline may select the cheaper validation-optimal schedule.

## PICT runtime investigation

The completed 300-update run spent 371.9 seconds generating references, 895.0
training full gradients, 439.8 on stopped gradients and 178.9 on supervision.
Scaling the training portion to 3,000 updates predicts about 4.3 hours, exceeding
the four-hour allocation limit. The timeout is consistent with cumulative work;
it does not by itself indicate a stalled solver. Splitting or resuming completed
arms is the practical next step.

GPU profile 2861273 (`pr116-pict-profile-20261002`) measured 64², four-step
periodic calls: median forward 50.3 ms, VJP 182.0 ms, including a 118.7 ms
adjoint. Domain construction takes only 0.085 ms and setup outside the simulation
less than 1 ms. Sparse linear solves dominate, so caching mutable domains would
add correctness risk for negligible benefit. Four pressure correctors and the
original linear-solver tolerances remain unchanged.

The following candidates were measured and retained without changing the
production PICT adapter or frozen images:

- Upstream BiCG preconditioning (2861302, `pr116-pict-precondition-20261002`)
  was slower and failed strict forward parity; reject it.
- Removing an unused scalar GPU synchronization (2861310,
  `pr116-pict-sync-cleanup-20261002`) passed forward and velocity/viscosity VJP
  comparisons across random periodic fields, 32²/64² grids, different timesteps,
  viscosities and step counts. Timings were unchanged within variability;
  the adapter edit was reverted.
- Optional RPC-boundary JIT (2861280, `pr116-pict-rpc-boundary-20261002`) gave
  1.081× median speedup over 30 paired horizon-eight training-step samples.
  Loss matched and gradients passed strict comparison, but two Adam-updated
  parameters failed the elementwise check (maximum difference 8.64e-6).
  Preserve that failure; PICT scientific runs keep the option off. A subsequent
  eager/eager control job 2861324 was cancelled before completing to prioritize
  the allocation-limit fix.

The profiling scripts are `check_pict_cost.py` and `check_ins_rpc_cost.py`
(the latter also accepts a solver and physical parameters). All numerical work
ran on Slurm. These probes do not provide new training-quality results.

## Approach discontinued

On 2026-10-02 the user requested abandoning correction learning in favor of
neural fluid control. The old automatic controller was stopped; remaining INS
compute-matched, PhiFlow correction and JAX reference jobs were cancelled.
Completed results and partial logs remain preserved. The PICT continuation
prototype is parked in git stash and frozen cluster archives, not integrated
into this benchmark. The replacement hypothesis is described in
`experiments/flow_control/PROTOCOL.md`.
