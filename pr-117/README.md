# PR 117 offline surrogate results

These artifacts support
[`feat(solver): add full-field XLB cylinder surrogate`](https://github.com/pasteurlabs/mosaic/pull/117).
They were generated offline on Kander RTX 5090 nodes. The feature PR contains
only inference code, model documentation, and trained weights; training code
and the training dataset remain outside the repository.

## Offline jobs

- XLB optimization: Slurm `1533241`
- XLB-surrogate optimization: Slurm `1533244`
- PICT optimization: Slurm `1533248`
- PhiFlow optimization: Slurm `1533243`
- float64 XLB final-profile re-evaluation: Slurm `1533250`
- packaged-surrogate field-history replay: Slurm `1540226`
- plot/GIF rendering: Slurm `1540227`

Each optimization used the unmodified `optimization/drag_opt` harness for 250
Adam updates. The remote Tesseract services and client ran on RTX 5090 nodes
through the same Slurm runner and Enroot/Pyxis workflow as PR 116. The
`benchmark:none` disposition was applied only after the offline runs; CI solver
builds and benchmark execution remained skipped.

## Media

- `optimization_summary.png`: drag reduction, final controls, wall time, and
  the compiled-kernel versus remote-harness speed distinction.
- `full_field_comparison.png`: final full `u_x`, `u_y`, and vorticity fields
  for all four solvers.
- `profile_evolution.gif`: saved inflow-control checkpoints for all solvers.
- `surrogate_full_field_evolution.gif`: 14 actual full-field surrogate
  evaluations at optimizer updates 0, 20, ..., 240, and 250.

The full-field animation shows optimization progress, not physical simulation
time: this surrogate directly maps the fixed task's inflow profile to its
200-step final state and tail-averaged traction fields.

## Supporting data

`runs/<solver>/` contains each harness `result.json`, `profiles.npz`, and
`flow_fields.npz`. The surrogate directory additionally contains the replayed
field history. `surrogate_evaluation.json`, `training_metrics.json`, and
`teacher_reevaluation.json` contain the held-out, gradient, kernel timing, and
independent float64-XLB checks quoted in the PR.

Weights SHA-256:
`6089ba29d9644e1f3b87404302b0b19aaeb7ea6ee0b359d7aa5fe395ef04626c`.

Dataset SHA-256:
`2832fecfb85d1ced442b38b608c512c71aca224c9a3b982d39e736d2bdf16ac8`.
