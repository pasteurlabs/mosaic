# Replicate the Warp correction result

The repaired Warp solver's first 3,000-update run produced mean held-out relative
L2 errors of 0.01266 (full gradients), 0.01369 (stopped gradients), 0.01778
(supervised), and 0.02668 (native solver). Reference admission and the initial
end-to-end gradient check passed. This is one exploratory model seed, not a
confirmed advantage. Its 300-update run still favored supervision.

The next runs preserve the campaign's frozen numerical source, image and
hyperparameters. They do not select checkpoints using held-out errors.

- Campaign: `pr116-warp-repaired-20261001`; source SHA256
  `96bc14fb5e0f2da11c7d71a16ca3d930512d5347402d8899dc68ca1f90e3a1b6`.
- Warp image: `warp-ns-2856606.sqsh`, with the validated Bluestein FFT repair.
- Same-solver reference: 64² coarse, 192² fine; original temporal audit unchanged.
- All learned arms: 3,000 updates, horizon 8, Adam lr=1e-4, same model
  initialization and sampled windows within each seed.
- Replication: add model seeds 1–4 to existing seed 0. Keep training ICs 0–7
  and evaluation ICs 100–103. Report all five seeds together.
- Separate confirmation: model seeds 0–2, training ICs 0–15, fresh evaluation
  ICs 1000–1007. The expanded training set overlaps the exploration set; only
  the evaluation ICs are fresh. Report this protocol separately.
- Require valid reference and gradient checks and paired seed/IC bootstrap
  intervals favoring full gradients over supervision, stopped gradients and
  the native solver. Retain failed runs and negative results.

The comparisons match optimizer updates and target counts, not wall time. In
the initial 3,000-update run, full-gradient training took 39.6 minutes versus
26.2 minutes for supervision including pair generation (about 1.51×). A win
would establish an advantage under this fixed protocol, not equal-compute
superiority or superiority over every tuned supervised baseline.

Slurm jobs: exploration 2858339–2858342; confirmation 2858344–2858346;
CPU report 2858348 runs after all seven jobs terminate. Numerical work and
rendering run on the cluster. Existing unforced warm-start/curriculum losses
and forced-reference failures remain part of the PR evidence.

## Learning-rate robustness check

Before the replication results are available, launch a separate three-seed
comparison on the exploration ICs: 1,500 updates at lr=1e-4, then 1,500 at
lr=1e-5, with horizon 8 throughout. Apply the identical schedule to all three
methods and retain Adam state across the change. This tests sensitivity to the
constant-rate optimizer without changing the update budget. It is not
hyperparameter selection on a validation set; report the schedule separately
whether or not it favors full gradients. Jobs 2858621–2858623 use the same frozen
source and image as the original run.

## Longer credit-assignment horizon

The five-seed original-protocol report gives baseline/full error ratios of
1.166 [1.011, 1.367] versus supervision and 1.043 [0.983, 1.115] versus stopped
gradients (paired bootstrap 95% intervals). Thus the additional derivative
benefit is unresolved, despite the favorable mean. This motivates a separate
exploratory horizon-16 comparison: three model seeds, 3,000 updates, constant
lr=1e-4, original training and evaluation ICs. All methods receive horizon 16
and therefore 48,000 target examples per seed. Update budgets match horizon 8;
target counts and compute do not. Jobs 2858727–2858729 use the existing frozen
source and image. Report this as a new configuration, not part of the original
replication or fresh-IC confirmation.

Fresh-IC confirmation also completed: mean errors are 0.01166 (full), 0.01187
(stopped), 0.01427 (supervised), and 0.02776 (native). Baseline/full ratios are
1.224 [1.154, 1.298] versus supervision and 1.018 [0.995, 1.042] versus stopped.
All reference and initial gradient checks pass. The supervised advantage
replicates on fresh held-out ICs; the added solver-derivative advantage remains
unresolved. The longer-horizon and learning-rate checks remain exploratory.

## Directional derivative refinement

A separate diagnostic campaign (`pr116-warp-gradientcheck-20261001`) repeats
only the first training update. For each matched initial-model check, the
solver, image, source, ICs and complete run configuration are identical after
removing only `max_updates` and `fd_epsilon`. These short runs are not evidence
of training quality. `gradient_checks.json` records the matching configuration
hashes and dataset hashes. The original validation metrics remain unchanged.

Seed 2 has the largest original-step discrepancy. Refining the finite-difference
perturbation gives these relative errors (fractions, not percentages):

| Epsilon |  Horizon 8 | Horizon 16 |
| ------- | ---------: | ---------: |
| 0.03    |   0.505708 |    not run |
| 0.01    |   0.038669 |   0.063100 |
| 0.003   |   0.003271 |   0.005345 |
| 0.001   |   0.000528 |   0.000707 |
| 0.0003  | 0.00000205 |  0.0000527 |

This convergence supports the initial-model derivative and indicates truncation
error at larger perturbations. It does not validate derivatives everywhere along
a trained trajectory. Horizon-16 seeds 0 and 1 also improve from errors 0.00136
and 0.00445 at epsilon 0.01 to 0.0000524 and 0.0000449 at epsilon 0.003. The
original horizon-16 seed-2 check exceeds the unchanged 0.05 threshold; retain
that failure and report this independent refinement alongside it. The CLI's
`--fd-epsilon` controls this diagnostic, with the original default 0.01 retained.

## Fixed expansion of confirmation seeds

After the first three confirmation seeds, fix the final confirmation sample at
**eight model seeds (0–7)**. Add seeds 3–7 with exactly the same source, image,
16 training ICs, eight held-out ICs, horizon 8 and 3,000-update budget. Jobs
2858782–2858786 perform this expansion. Report all eight seeds regardless of
outcome; do not stop adding seeds when an interval first excludes a tie. This
addresses the unresolved small derivative effect, and the earlier three-seed
summary remains an interim result. This expansion is chosen after inspecting
that summary and should be described as such.
