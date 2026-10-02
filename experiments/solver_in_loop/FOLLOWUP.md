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
  overshoot the budget. Reference generation and diagnostic evaluation are
  excluded for every arm. The optimizer has a 100,000-update safety cap.

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

The one-shot controller and frozen launcher are in
`mosaic-results/pr116-next-round/`. `status.json`, `submissions.jsonl`,
`baseline-selection.json` and `controller.log` record progress and decisions.
It performs only lightweight JSON/SSH orchestration on this machine. Training,
reference generation, tests and rendering remain cluster jobs. Failed gates
stop their dependent branch and remain reported. The publication step checks
for intervening PR edits before replacing its description.
