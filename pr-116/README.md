# PR 116 local Slurm artifacts

These generated artifacts support draft PR 116 and intentionally live outside
the pull-request source diff. Training used source commit `4cfa5d1`; the
normalization-only rendering fix and its regression tests are source commit
`96546b0`.

- `multimode/`: nonlinear decaying periodic flow, using a benchmark-owned
  64² pseudo-spectral reference restricted to the shared 32² grid. Sixteen ICs
  train the corrector; eight disjoint ICs evaluate it to `t=2.88`.
- `tgv/`: translated Taylor–Green vortices with exact analytic viscous decay.
  This is a reference/time sanity case, not a second broad IC-generalization
  claim.
- `recurrence/`: direct fixed-time semigroup admission for the canonical
  recurrent state, comparing two successive calls with one uninterrupted call.
- `self-reference/`: recurrence-admitted PICT and Warp-NS correctors trained
  against each solver's own 64², half-step reference restricted to 32².

The experiment result directories contain:

- `comparison_summary.json`: paired seed × IC bootstrap comparisons
- `result.json`, `params.json`, and `corrector_fields.npz`: merged metrics,
  configuration, and complete plot inputs
- `solver_in_loop.png`: training, rollout, and time-to-quality overview
- `solver_in_loop_fields.png`: reference, solver-only, and corrected fields
- `solver_in_loop_fairness.png`: target-normalized quality, correctability,
  VJP lift, and cost
- `solver_in_loop_physics.png`: energy, enstrophy, and common divergence measures
- `solver_in_loop_trajectory.gif`: reference, solver-only, and corrected
  trajectories over the full horizon

The shared-reference directories additionally contain
`solver_in_loop_diagnostics.png` for first-interval, restart, held-out-IC, and
temporal-extrapolation diagnostics. The self-reference directory deliberately
omits that absolute-error panel because its targets differ by solver.

Among recurrence-admitted cells, the nonlinear shared-reference result finds
a statistically positive VJP contribution for both PICT and Warp-NS. Their
lifts are indistinguishable, while Warp-NS reaches that tier at roughly half
PICT's steady update time. The recurrence audit explains the other four cells:
JAX-CFD, PhiFlow, and INS.jl repeatedly convert between collocated canonical
velocities and native staggered state, while XLB drops its population state.
Their current full-VJP/stop-gradient numbers therefore measure a lossy adapter
boundary and are not used to rank the native solvers.

The solver-specific refined-reference result asks the narrower upscaling
question only for recurrence-admitted cells. Its absolute errors are not
comparable across solvers because each target is different; the comparable
quantity is each solver's paired within-cell full-VJP lift over stop-gradient.
PICT gains 4.21% (paired bootstrap 95% CI 2.53–6.15%) and Warp-NS gains
4.19% (2.80–5.54%). Their pairwise lift ratio differs by only 0.016%, with a
95% interval from −1.19% to +1.50%, so the two VJP benefits are not
distinguishable. Warp-NS's median full-VJP update is 0.812 s versus PICT's
1.612 s.

The six nonlinear production jobs were `1532904`, `1533251`, `1533252`,
`1533265`, `1533254`, and `1533255`. The analytic-reference jobs were
`1537368`, `1537369`, `1537370`, `1537371`, `1537373`, and `1537374`. Final
merge/render jobs were `1537505` and `1537535`. Direct recurrence audits were
`1540793`, `1540730`, `1540810`, `1540835`, `1540791`, and `1540745`; their
merge/render job was `1540887`. Self-reference production jobs were PICT
`1540834` and Warp-NS `1540792`; merge/render was `1543344`.
