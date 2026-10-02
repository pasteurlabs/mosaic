# Neural fluid control through a differentiable solver

This replaces the correction-learning objective of PR #116. The final INS
correction comparison favored validation-tuned supervision (2.8319% mean rollout
error versus 3.4984% with full solver gradients). Those results are retained;
control is a separate hypothesis, not a reinterpretation of that comparison.
Remaining correction-training campaigns were cancelled on 2026-10-02 at the
user's request. Their logs and partial results remain in the campaign archives.

## Question

Can solver gradients train an amortized controller to reach requested flow states
with less objective error or less total compute than useful alternatives?
A policy receives initial and target velocity fields and predicts bounded force
coefficients. The objective measures the resulting terminal velocity plus control
effort. No assumption of a unique optimal action sequence is made.

This is an adaptation of Holl, Koltun and Thuerey, *Learning to Control PDEs with
Differentiable Physics* (https://arxiv.org/abs/2001.07457), not a reproduction of
its hierarchical smoke-control architecture.

## Initial physical setup

- Repaired PhiFlow image `phiflow-2861222.sqsh` (SHA256
  `b5317e9438b4487646f62e090a36e5ef546d9fcc30d9881b7637fb4e71f65462`).
- Periodic square of side 2π, viscosity .001. Optimize at 64², generate targets
  and evaluate at 192² using the same solver.
- Coarse dt=.02, four native steps per control interval, eight intervals (T=.64).
  Fine dt=.02/3; temporal audit dt=.02/6 at the same final physical time.
- Eight zero-mean Fourier actuator fields from sine/cosine streamfunctions with
  wavevectors (1,0), (0,1), (1,1), (1,-1). The same physical fields and force
  durations apply at each grid. Coefficients are bounded by .25 using tanh.
- Symmetric force half-kicks surround every native solver step. Native state is
  retained and reconciled with each canonical velocity change.
- Loss: terminal velocity mean-square error divided by a fixed velocity scale
  squared, plus .01 times consistently normalized integrated physical force
  energy. Report both components separately, not only the weighted objective.

## Gates before training

Use eight fixed development tasks, separate from later train/validation/test
splits. Save their generating controls, initial fields and full goal trajectories.
Check actuator constraints, finite rollouts, directional derivatives at one fixed
coefficient vector with multiple directions/perturbation sizes, direct-shooting
objective reduction, and 192² temporal convergence. Keep every failure and
perturbation curve. Do not optimize a threshold after seeing its outcome.

Initial numerical tolerances: relative time-refinement terminal discrepancy
below .5%; directional derivative relative discrepancy below 5%, with absolute
errors and derivative magnitudes retained to expose near-null directions.
Check three fixed directions at epsilon .001 as primary, and .01/.0001 as
additional diagnostics. Refinement never replaces a failed primary scalar.

## Comparisons after gates

1. Identical policy trained through solver gradients.
2. Action-space antithetic SPSA followed by the exact policy VJP, using the same
   objective and action parameterization. Tune perturbation scales and learning
   rates on validation tasks and count all perturbed simulations.
3. Supervised imitation using the generating controls and improved controls from
   per-instance optimization. Generating demonstrations remain available to this
   baseline; we do not hide labels to create an artificial advantage. Include
   label-generation cost and report it separately.
4. Zero control and direct per-instance gradient/gradient-free control show task
   difficulty and the cost/benefit of amortization.

Training/evaluation task sizes, model seeds and update/time budgets will be
frozen before that stage. The intended larger split is 64 training, 16 validation
and 32 test tasks with disjoint initial-condition and goal seeds. Gate tasks
cannot be relabeled as held-out confirmation. Select settings on validation only.
Use the same architecture and paired initialization for learned methods. Report
both objective-versus-compute curves and final held-out performance, full-field
plots, force effort, failures, inference cost and label/training cost.

All numerical work, tests and rendering run through Slurm. Frozen source,
configuration and image identities accompany every campaign. Existing correction
experiments and solver fixes remain archived separately.

## Development gate outcomes and next probe

All eight short-horizon (T=.64) tasks passed the numerical gates in
`pr116-control-gate-v2-20261002` (jobs 2861675–2861678). The cheap linear controller
had a lower fine-grid objective than 25 Adam updates initialized at zero on all
eight tasks. No neural training was launched on that basis. This does not rule
out better optimization: zero initialization and 25 updates can be insufficient.
The first source failed one actuator-precision unit test; immutable Fourier
constants now use float64 construction and one float32 cast. All 24 tests passed
on Slurm 2861674 without relaxing tolerances.

The next development probe changes only native steps per control slot from 4 to 20,
so T becomes 3.2. Image, numerical source, all eight development tasks, actuator
bound, dt, spatial grids and admission thresholds stay fixed. Campaign:
`pr116-control-long-gate-20261002`, jobs 2861713–2861716. Comparisons are within each
horizon: normalized effort divides by duration, so scores across horizons do not
represent equal physical integrated effort. A separate linear-initialized
shooting diagnostic will distinguish optimizer convergence from task difficulty.
Neither development probe is held-out confirmation.

The prepared neural pilot uses 16 training tasks 1000–1015 and 8 validation tasks
2000–2007, model seeds 0–2. Full-gradient and action-SPSA policies use 300 updates;
imitation uses one uninterrupted 1000-update run with 250/500/1000 checkpoints.
This is an explicitly untuned development pilot with unequal update budgets and
measured costs, not a superiority test. The T=3.2 pilot is selected for launch after source validation: its direct-control
gate beats the cheap linear controller on every development task. Final test tasks remain untouched.

The longer-horizon gate passed all eight tasks. Mean fine-grid objective was
0.008297 for 25 gradient updates from zero versus 0.025083 for the linear
controller (66.9% lower); all eight paired comparisons favored optimization.
Maximum goal/controlled time-refinement discrepancy was 0.311%. This supports
launching the neural development pilot, but establishes neither amortized policy
quality nor an advantage over imitation or gradient-free learning. Both horizon
results remain reported. Separate short/long linear-warm-start diagnostics are
engineering checks and do not select neural checkpoints.

## Pilot findings and fixed follow-up

The first 12 neural jobs completed and passed temporal admission. All learned
controllers were worse than the cheap linear controller on validation tasks.
Mean final objective: full gradients .163527, action SPSA .239359, imitation
.171955, versus linear .025406. Imitation at 500 updates scored .165283, so the
final checkpoint comparison is not a tuned-supervision claim. The 73,920-parameter
field MLP fit only 16 training tasks: imitation training action MSE was
.00056–.00195 while validation action MSE was .079–.088. Longer imitation training
worsened validation in every seed. The original expert optimizer replaced none
of the demonstration labels, despite about 31 minutes of extra work per seed.

Separate linear-initialized direct optimization improved all eight development
tasks at both horizons. At T=3.2, 25 updates reduced mean objective from .025083
to .001868 (92.55%); at T=.64 the reduction was only 3.67%. These are per-instance
optimization results, not learned-controller performance.

Dataset audit job 2862236 found identical initial fields, controls and seeds,
but two bytewise goal groups. Goal differences were about 1e-6 relative L2;
common-target rescoring shifted mean terminal MSE by at most 5e-8. This is far too
small to explain the observed validation gap, but does not establish bitwise
identical training trajectories. Original files remain intact.

The follow-up changes the model and dataset together to address this failure;
it does not isolate their individual effects. Freeze 64 training tasks 1000–1063
and 16 validation tasks 2000–2015. Generate and audit each task once, save its
coarse unforced terminal field, and assemble one hashed dataset. Every method
must read exactly those bytes, with matching physics/image and split identities.
Any task admission failure blocks assembly; tasks cannot be silently excluded.
Final test tasks remain untouched.

All four methods use the same 4,192-parameter residual policy: 64 physical features,
a 32-unit hidden layer and a zero-initialized output layer. It begins at the
linear controller, up to the documented inward tanh-bound transformation.
Features use initial low-frequency modes, actuator projections of the observed
goal discrepancy, and linear control coefficients. Generating controls and fine
targets cannot enter these features. Deployment requires one coarse free rollout;
report that cost separately from neural inference.

Model seeds remain 0–2. Full gradients and SPSA use 300 updates at lr=1e-3,
with checkpoints 0/100/300. SPSA uses one antithetic direction and perturbation
.05. Imitation uses 1000 updates at lr=1e-3, checkpoints 0/250/500/1000. Every
checkpoint is evaluated only after training. Report final training objectives
at 64² and validation objectives at 192² to expose generalization gaps.
The zero-update checkpoint is the untrained baseline, not an optimization success.
This remains an untuned, unequal-budget development pilot.

For improved imitation, refine each public training demonstration with 25
coarse-grid Adam updates at lr=.01 and retain whichever has lower coarse loss.
Never generate optimized validation labels. Save the actual choice, trace and
label cost. Common dataset preparation is reported once as shared work and
included in each standalone method comparison; only improved imitation includes
expert refinement cost. Per-job costs exclude precomputed shared work, preventing
double-counting. Full-grid validation and fit diagnostics are separate from
training cost. No revised superiority claim is justified until these comparisons
have completed and baselines have been tuned on validation data.

## Residual pilot closure

All 12 residual-policy jobs completed and passed admission on the shared dataset
`deead2fb02bedbe3c3b4a2199114e0d08a0ebe00ac66ced553316cead8ba4a83`.
Mean final objectives were full gradients .02729366, SPSA .02639340,
demonstration imitation .02787517 and refined-label imitation .02810713,
versus linear initialization .02605520. For every method, the best mean over
its registered validation checkpoints was update zero. Expert refinement did
improve all 64 training labels, so the residual pilot is no longer affected by
the earlier unchanged-label issue. These outcomes close this policy architecture
without a neural-training advantage; no additional runs of it are planned.

The separate direct-control experiment in [DIRECT_PROTOCOL.md](DIRECT_PROTOCOL.md)
now tests solver-gradient optimization against tuned SPSA and Powell at matched
wall-time budgets. Its untouched confirmation runs are gated on development-only
selection. Direct optimization results cannot be relabeled as neural training.
