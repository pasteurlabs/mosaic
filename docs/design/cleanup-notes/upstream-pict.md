# Draft: tum-pbs/PICT

**Status:** ready to file (approved 2026-10-27). Re-checked 2026-10-30: master is still e019ebe.

**Title:** Pressure matrix becomes indefinite with `non_orthogonal=True` on a multi-block O-grid; CG and float32 BiCGStab fail in the first step

**Attach:** `pressure_system_true.npz`, `pressure_system_false.npz` (saved by `diag/pict/capture.py`, not in the repo) and `diag/pict/standalone.py`.

## Body

On a 2-D channel with an O-grid around a cylinder (Schäfer–Turek DFG 2D-1, 10 cells per diameter), `PISOtorch_simulation.Simulation` with `non_orthogonal=True` fails in the first pressure solve:

```
Linear solve (T:False, BiCG:False, tol:1.000e-06) reported non-finite residual for RHS [0] of 1. Solver infos:
  RHS 0: LinearSolverResultInfo(finalResidual=nan, usedIterations=2157, converged=0, isFiniteResidual=0)
```

With `non_orthogonal=False` the same case runs. The mesh has 12 blocks: four O-grid blocks between the cylinder and a square box of side 2D, and eight Cartesian blocks around the box. At each box corner, five blocks share a vertex: the Cartesian corner block, two Cartesian side blocks and two O-grid blocks.

I saved the first pressure system (CSR values, indices, row pointers, RHS; float32, 9420 rows) in both modes. `standalone.py` loads one, prints its properties and solves it with `PISOtorch_diff._linear_solve_wrapper` and with SciPy:

|                            | `non_orthogonal=False` | `non_orthogonal=True`                |
| -------------------------- | ---------------------- | ------------------------------------ |
| max \|A − Aᵀ\| / max \|A\| | 2e-8                   | 4.2e-2                               |
| max \|row sum\|            | 3e-7                   | 0.12                                 |
| eigenvalues nearest 0      | all ≤ 0, one ≈ −2e-9   | five < 0, one +2.1e-4                |
| GPU CG, float32 / float64  | converges              | no convergence after 5000 iterations |
| GPU BiCGStab, float32      | converges              | NaN at iteration 2210                |
| GPU BiCGStab, float64      | converges              | converges                            |
| SciPy `spsolve`            | residual 4e-14         | residual 1e-13                       |

The largest row-sum defects (0.116 and −0.039) sit in the cells around the four vertices that five blocks share. Smaller ones (up to 0.008) line the curved wall.

For comparison, the first pressure systems of the new `cylinder_channel_sample.py` (master e019ebe, `res=16`, `Re=20`) are also non-symmetric (about 10%) and have non-zero row sums, but all of their eigenvalues near 0 are negative. That sample's CG also stops at 1000 iterations without converging (residual 9.5e-5) and continues through `pressure_return_best_result=True`.

In the simulation, `scipy_solve_pressure=True` runs our case, while every GPU variant we tried still fails in step 1. The variants were `pressure_use_BiCG`, `solver_double_fallback`, `pressure_return_best_result` and an initial `make_divergence_free()`. `pressure_use_BiCG` with `solver_double_fallback` reports `usedIterations=0` and a NaN residual, although float64 BiCGStab converges on the saved matrix in `standalone.py`. We haven't found out why.

Questions:

1. Are vertices shared by five blocks supported with `non_orthogonal=True`?
2. Is it expected that the non-orthogonal pressure matrix can become indefinite, and is there a recommended solver configuration for that case?

Setup: PICT 72a6e86. The solver core is unchanged on master e019ebe, and the sample above ran with the master sources. CUDA, float32, `substeps=1`, `corrector_steps=2`, `differentiable=False`, `pressure_tol` default.

## Notes for Dion (not part of the issue)

- Master (e019ebe) changes only samples and library utilities; the solver core matches our pin, so the failure persists there.
- The matrices are on the sandbox at `~/localfiles/runs/cleanup/upstream/pict/`. `diag/pict/capture.py` regenerates them in the PICT image.
- Not resolved: why `pressure_use_BiCG` with `solver_double_fallback` gives a NaN residual at iteration 0 in the simulation. The rank-deficiency flag the simulation passes is a guess, not checked.
