# PR 116 local Slurm artifacts

These generated artifacts support draft PR 116 and intentionally live outside
the pull-request source diff. The matched-forward control and repaired PICT
projection are source commit `eb7492e`; the final admission-aware rendering is
source commit `65b30e1`.

- `multimode/`: nonlinear decaying periodic flow, using a benchmark-owned
  64² pseudo-spectral reference restricted to the shared 32² grid. Sixteen ICs
  train the corrector; eight disjoint ICs evaluate it to `t=2.88`.
- `tgv/`: translated Taylor–Green vortices with exact analytic viscous decay,
  using the `forward/agreement` cell `N=64`, `nu=0.005`, `dt=0.05`, and
  `steps=20`. This is a matched-forward, near-error-floor control rather than a
  second broad IC-generalization claim.
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

The nonlinear shared-reference comparison first gates on recurrent-state
closure. PICT and Warp-NS then begin at matched first-interval error (0.05970
and 0.05912), so raw forward quality does not confound their VJP comparison.
Their correctors reduce geometric rollout error by 2.102× and 1.588×. The
solver VJP contributes 2.92% (paired bootstrap 95% CI 1.28–4.76%) for PICT and
2.65% (1.18–3.93%) for Warp-NS. Their lift ratio differs by only +0.26%, with a
95% interval from −1.18% to +1.58%, so gradient utility is not distinguished.
Warp-NS is instead distinguished on cost: its 0.801 s median full-VJP update is
about 3.1× faster than PICT's 2.485 s.

The recurrence audit explains the other four cells. JAX-CFD, PhiFlow, and
INS.jl repeatedly convert between collocated canonical velocities and native
staggered state, while XLB drops its population state. Their current
full-VJP/stop-gradient numbers therefore measure a lossy adapter boundary and
are not used to rank the native solvers. The fairness plot retains their
absolute interface diagnostics but restricts its VJP panels to admitted cells.

The solver-specific refined-reference result asks the narrower upscaling
question only for recurrence-admitted cells. Its absolute errors are not
comparable across solvers because each target is different; the comparable
quantity is each solver's paired within-cell full-VJP lift over stop-gradient.
PICT gains 3.95% (paired bootstrap 95% CI 1.94–6.20%) and Warp-NS gains
4.19% (2.80–5.54%). Their pairwise lift ratio differs by −0.23%, with a 95%
interval from −1.71% to +1.57%, so the two VJP benefits are not
distinguishable. Warp-NS's median full-VJP update is 0.812 s versus PICT's
2.476 s.

The analytic TGV control starts even more tightly matched: PICT has
5.89e-4 first-interval error and Warp-NS has 6.10e-4. Both learned correctors
worsen the near-floor trajectory (geometric reductions 0.211× and 0.155×).
PICT's +18.8% point VJP lift has a wide −10.8% to +98.1% confidence interval;
Warp-NS's −1.0% point lift has a −44.7% to +57.0% interval. This is a useful
null control: matched forward accuracy is necessary for comparison, but a
training task also needs learnable residual error.

The nonlinear PICT refresh was job `1638650`; the other five unchanged
diagnostic cells came from jobs `1532904`, `1533251`, `1533252`, `1533254`,
and `1533255`; final merge/render was `1640008`. Matched TGV production was
PICT `1638651` and Warp-NS `1636837`, with merge/render `1639119`.
Self-reference production was PICT `1638649` and Warp-NS `1540792`, with
merge/render `1639980`. Direct recurrence audits were `1540793`, `1540730`,
`1540810`, `1540835`, `1540791`, and `1540745`; their merge/render job was
`1540887`. The repaired PICT image was built in Slurm job `1636707` and
verified by forward and solver-loop canaries `1638643` and `1638642`.
