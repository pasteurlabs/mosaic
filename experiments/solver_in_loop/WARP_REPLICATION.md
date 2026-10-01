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
