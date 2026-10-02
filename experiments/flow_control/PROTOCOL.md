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
