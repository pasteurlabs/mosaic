# PR 116 local Slurm artifacts

These generated artifacts support draft PR 116 and intentionally live outside
the pull-request source diff. They correspond to source commit `985584a`.

- `multimode/`: nonlinear decaying periodic flow, using a benchmark-owned
  64² pseudo-spectral reference restricted to the shared 32² grid. Sixteen ICs
  train the corrector; eight disjoint ICs evaluate it to `t=2.88`.
- `tgv/`: translated Taylor–Green vortices with exact analytic viscous decay.
  This is a reference/time sanity case, not a second broad IC-generalization
  claim.

Each directory contains:

- `comparison_summary.json`: paired seed × IC bootstrap comparisons
- `result.json`, `params.json`, and `corrector_fields.npz`: merged metrics,
  configuration, and complete plot inputs
- `solver_in_loop.png`: training, rollout, and time-to-quality overview
- `solver_in_loop_fields.png`: reference, solver-only, and corrected fields
- `solver_in_loop_fairness.png`: raw quality, correctability, VJP lift, and cost
- `solver_in_loop_diagnostics.png`: first interval, restart, held-out IC, and
  temporal-extrapolation diagnostics
- `solver_in_loop_physics.png`: energy, enstrophy, and common divergence measures
- `solver_in_loop_trajectory.gif`: reference, solver-only, and corrected
  trajectories over the full horizon

The nonlinear result finds a statistically positive VJP contribution only for
PICT and Warp-NS; their lifts are indistinguishable, while Warp-NS reaches that
tier at roughly half PICT's steady update time. The exact TGV case shows why
the result is reference-sensitive: JAX-CFD, PhiFlow, and INS.jl have large
canonical-state restart errors that local stop-gradient correction learns
well, while full-VJP training is significantly worse.

The six nonlinear production jobs were `1532904`, `1533251`, `1533252`,
`1533265`, `1533254`, and `1533255`. The analytic-reference jobs were
`1537368`, `1537369`, `1537370`, `1537371`, `1537373`, and `1537374`. Final
merge/render jobs were `1537505` and `1537535`.
