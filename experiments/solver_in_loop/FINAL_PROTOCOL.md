# Final bounded INS corrector comparison

## Purpose and stopping rule

This is the final correction-training comparison for PR #116. Earlier INS
confirmation favored tuned supervision (2.8319% mean rollout error versus 3.4984%
full gradients); retain that negative result. The question is whether fair tuning
of recurrent training, including a supervised warm start, produces a convincing
advantage on untouched initial conditions.

Freeze the manifest, source, image, datasets and this protocol before inspecting
validation outcomes. Run only the search and confirmation below. If the final
comparison does not meet the stated evidence criterion, close this research line
as **no convincing advantage demonstrated**, without further architecture,
physics, horizon or learning-rate searches. A confidence interval crossing the
null does not establish equivalence or prove that no beneficial setting exists.

## Shared problem and admission

Use the existing forced-INS correction setup unchanged: the same solver at 64²
coarse and 192² reference resolution, identical forcing, viscosity, timestep,
burn-in, trajectory length, corrector architecture, correction convention and
training ICs 0–7. The executable manifest must contain every resolved setting
and the immutable solver-image/source hashes; no implicit new defaults.
Training horizon varies below; evaluation trajectory length does not vary with it.

Generate, audit and freeze one shared dataset, including native-state/cache
semantics and checksums. Every arm receives the same reference fields. Preserve
existing temporal, closure and primary finite-difference admission thresholds;
retain failed primary checks and supplemental perturbation curves. No solver
change, threshold relaxation, IC replacement or silent failure filtering is
allowed after outcomes are inspected. A common reference failure blocks the
comparison until a separately documented numerical repair and new freeze.

Validation ICs are 10000–10015. Confirmation ICs are 20000–20031; none may be
read for selection. Use tuning model seeds 0–2 and fresh confirmation model seeds
8–15. Evaluate the 32 confirmation ICs in separate four-IC Slurm shards as needed.
Training ICs are reused intentionally; the claim
is conditional on this training dataset, not variation across training datasets.

## Finite search

All schedules use the existing optimizer and loss definitions. No architecture
search, learning-rate decay search or new teacher data is included.

| Arm | Initial candidates, each on three tuning seeds |
| --- | --- |
| Full solver gradients | learning rate {1e-5, 3e-5, 1e-4} × horizon {4, 8, 16}, 1000 updates |
| Stopped solver gradients | the same nine candidates |
| Fixed-pair supervision | learning rate {1e-5, 3e-5, 1e-4} × updates {1000, 3000} |

For full and stopped training separately, rank the nine initial candidates by
mean free-running validation relative L2 error over all 16 validation ICs and all
three model seeds. Extend the best two settings per arm to 3000 updates, trained
from the same seed-specific initialization. Their first 1000 updates must follow
the identical schedule; retaining a checkpoint or rerunning from scratch is an
implementation choice, with actual compute recorded. Keep the original 1000-step
candidates eligible. Do not extend additional candidates after seeing results.
Ties prefer fewer updates, then smaller learning rate, then smaller horizon,
then candidate ID.

Also test two prospective warm-start candidates per recurrent arm: supervised
pretraining for 1000 updates at lr=1e-5, followed by 1000 full or stopped updates
at lr={1e-5,3e-5}, horizon 8. Reset optimizer state at the transition for both
arms. For each seed, full and stopped warm starts use exactly the same pretrained
parameter bytes. Retain the pretrained/no-finetuning model as the supervised
1000-update lr=1e-5 baseline; it is already part of the grid. Charge pretraining
and fixed-pair construction to each warm-start method even if the checkpoint is
physically shared. Save the transition checkpoint and its hash.

This is 96 arm runs: 54 initial recurrent, 18 supervised, 12 recurrent extensions
and 12 warm-start runs. Shared cached preparation does not create extra candidates.
Warm candidates do not enter the top-two extension decision. If a required tuning
run fails numerically, keep its result and mark that candidate ineligible; do not
average its surviving seeds or substitute an unregistered setting. Infrastructure
retries may repeat the identical manifest, with all attempts retained.

## Selection, compute and confirmation

Choose one eligible candidate per arm using the same validation metric over all
ICs and seeds. A candidate must finish and pass required admission on every
registered tuning seed. Never choose by final-test error. Resolve exact ties by
fewer total updates (including pretraining), then lower learning rate, then lower
horizon, then candidate ID. Freeze selections and their content hash
before confirmation. If an arm has no eligible candidate, report that failure
and do not claim a complete three-arm comparison.

Train the three frozen winners with each of the eight confirmation model seeds.
Also evaluate a matched stopped-gradient arm using exactly the selected full
recipe, including initialization, pretraining, horizon, learning rate and updates.
Reuse the selected stopped arm if its recipe is identical; otherwise train eight
additional models. This adds at most eight confirmation fits and no new search.
Share the initial model parameters and, wherever applicable, minibatch/window RNG
streams for paired methods; record realized sample schedules. Different training
horizons expose different numbers of solver steps and targets, so equal updates
are not equal compute or equal target exposure. Warm and cold candidates also
have different histories; do not call their final checkpoints identical starts.

The primary comparison is validation-selected best performance under the finite
registered search. It is **not** a matched-wall-time efficiency claim. Report
training wall time, update count, analytical solver-step/target-exposure estimates,
fixed-pair construction and supervised pretraining separately. Shared reference
preparation, diagnostics and evaluation are separate costs, with an end-to-end
view adding common preparation equally. Include total tuning-search cost.
Do not force supervision to run longer than its validation-optimal schedule merely
to spend a full-gradient budget; longer training can overfit. Any later compute
comparison must select a baseline within the budget using validation only and is
outside this finite protocol unless registered before this launch.

## Evidence and interpretation

For every model seed and test IC, save the free-running relative L2 error using
the existing fixed evaluation reduction. Aggregate equally over ICs and seeds.
The primary quantity is the ratio of mean full-gradient error to mean supervised
error, with a paired crossed bootstrap: independently resample model-seed indices
and IC indices, apply the same sampled indices to both methods, and recompute the
ratio. Use 10000 bootstrap replicates and a fixed bootstrap RNG seed 11620261003.
Report the 95% interval, every seed/IC result and paired win counts. Frames within
one trajectory are not independent experimental samples.

The prospectively defined positive criterion is at least 5% lower mean error
(full/supervised ratio ≤.95) and a 95% ratio interval with upper endpoint <1.
Otherwise conclude no convincing advantage under this protocol and stop. This
combines a point-estimate practical margin with evidence of a nonzero benefit;
it does not claim the entire interval establishes a ≥5% benefit.

Also report full versus the globally selected stopped arm using the same paired
analysis; this combines derivative choice with selected hyperparameters. The
matched stopped arm isolates derivative choice at the selected full recipe.
A mechanistic claim that solver derivatives help requires its paired ratio
interval upper endpoint <1. Retain matched-setting tuning diagnostics as well.
Report all comparisons without hiding an unfavorable ablation. Separate 95%
intervals are not a simultaneous familywise guarantee; superiority to supervision
alone is not sufficient evidence for the derivative-specific mechanism.

Include native-solver error, validation/test rollout errors, measured training
costs and a fixed full-field example (test IC 20000, model seed 8). Use shared
color scales and show failures alongside results. Solver-step and target-exposure
estimates are analytical quantities, not instrumented forward/reverse query
counts. No claim of measured adjoint replay cost is made.
All numerical work, bootstrap analysis, tests and rendering run on Slurm; this
machine performs only orchestration and lightweight metadata inspection.
