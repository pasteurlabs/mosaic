# PR 116 local Slurm artifacts

These generated artifacts support draft PR 116 and intentionally live outside
the pull-request source diff.

- `multimode/`: nonlinear decaying periodic flow, using a benchmark-owned
  64² pseudo-spectral reference restricted to the shared 32² grid.
- `tgv/`: translated Taylor–Green vortices with exact analytic viscous decay.

Each directory contains the merged result and parameters, the full plot input
archive, static comparison figures, and an animation with columns for the
reference, solver-only rollout, and full solver-in-the-loop corrector.

The six nonlinear solver jobs were `1502100`, `1502277`, `1502279`, `1502280`,
`1502282`, and `1502283`. The analytic-reference jobs were `1502729`, `1502732`,
`1502735`, `1502738`, `1502739`, and `1502740`. Final merge/render jobs were
`1503335` and `1503336`.
