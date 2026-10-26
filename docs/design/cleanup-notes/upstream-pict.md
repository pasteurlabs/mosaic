# Draft: tum-pbs/PICT

**Status:** draft, not filed. File only after Dion approves.

**Title:** CG pressure solve returns a NaN residual in the first step with `non_orthogonal=True` on a body-fitted O-grid

## Body

On a 2-D channel with an O-grid around a cylinder (Schäfer–Turek DFG 2D-1, 10 cells per diameter, 9 blocks), running `PISOtorch_simulation.Simulation` with the default `non_orthogonal=True` fails in the first step:

```
Linear solve (T:False, BiCG:False, tol:1.000e-06) reported non-finite residual for RHS [0] of 1.
Solver infos:
  RHS 0: LinearSolverResultInfo(finalResidual=nan, usedIterations=2157, converged=0, isFiniteResidual=0)
```

With `non_orthogonal=False` and otherwise identical settings, the same case runs (10 steps, finite forces). Our reading is that the skew correction makes the pressure matrix non-symmetric while it stays singular, since pressure has Neumann conditions on every boundary, so CG is not applicable. BiCGStab also broke down for us in an earlier attempt, though we didn't record that output.

Setup: `substeps=1`, `corrector_steps=2`, `differentiable=False`, `pressure_tol` default, PICT's stock `linear_solve`, CUDA, float32. The skewed cells are at the corners of the square around the cylinder, about one diameter from the wall.

Questions:

1. Is `non_orthogonal=True` expected to work with an all-Neumann pressure system, or is a reference pressure or a different solver needed?
2. Is reverse-mode differentiation supported with `non_orthogonal=True`? The orthogonal branches are labelled "orthogonal version with gradient/backprop support".

Mesh and driver: Mosaic's PICT tesseract (`mosaic/tesseracts/navier-stokes-grid/pict/tesseract_api.py`, `_run_channel`), at PICT commit 72a6e86. We have not tried PICT `master` (e019ebe at the time of writing).

## Notes for Dion (not part of the issue)

- Reproduced 2026-10-26 on the A100 with the tesseract's own channel, `_simulate` patched to set `non_orthogonal` and disable Mosaic's warm-start override, so only PICT's stock solver runs (`diag/pict/repro.py`, inputs from `cylinder.channel_inputs(..., 10, t_end=0.1)`).
- A standalone script without Mosaic's mesh builder would make a better report. That takes extracting the block construction from `_run_channel`; not done.
- Check `master` first: it is six days newer than the pin.
