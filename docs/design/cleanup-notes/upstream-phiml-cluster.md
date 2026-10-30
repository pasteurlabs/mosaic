# Draft: tum-pbs/PhiML, `'cluster'` preconditioner on a periodic pressure solve

**Status:** ready to file (decided 2026-10-30). Re-checked 2026-10-30 on phiml 1.16.1 (latest release): both failures reproduce as written. Found while measuring PhiFlow's 2-D cost (`phiflow-cost.md`); Mosaic does not use the preconditioner.

**Title:** `Solve(preconditioner='cluster')` fails in `make_incompressible` on a periodic grid: "shape of weights must match shape of x"

## Body

On a fully periodic grid, `fluid.make_incompressible` with the `'cluster'` preconditioner raises in the preconditioner's `bincount`. The same solve without a preconditioner works. With `rank_deficiency=0` the error goes away, but on this field the solve then raises `NotConverged`, which we would expect for a singular system whose right-hand side is not compatible.

```python
from phi.jax.flow import Box, Noise, Solve, StaggeredGrid, extrapolation, fluid

vel = StaggeredGrid(Noise(), extrapolation.PERIODIC, x=16, y=16, bounds=Box(x=1, y=1))
fluid.make_incompressible(vel, solve=Solve("CG", 1e-5, 1e-5))  # works
fluid.make_incompressible(vel, solve=Solve("CG", 1e-5, 1e-5, preconditioner="cluster", rank_deficiency=0))  # NotConverged
fluid.make_incompressible(vel, solve=Solve("CG", 1e-5, 1e-5, preconditioner="cluster"))  # fails
```

```
  File ".../phiml/math/_optimize.py", line 694, in solve_linear
  File ".../phiml/math/_optimize.py", line 791, in _linear_solve_forward
  File ".../phiml/backend/_backend.py", line 1522, in conjugate_gradient
  File ".../phiml/backend/_linalg.py", line 63, in cg
  File ".../phiml/backend/_linalg.py", line 752, in apply
  File ".../phiml/backend/_backend.py", line 1087, in batched_bincount
  File ".../phiml/backend/jax/_jax_backend.py", line 595, in bincount
ValueError: shape of weights must match shape of x.
```

Versions: phiml 1.16.1, phiflow 3.4.0, jax 0.10.0, CPU (`JAX_PLATFORMS=cpu`); the same error on an A100.

Our reading, not verified in the source beyond the traceback: for the rank-deficient periodic Poisson matrix, the solve appends one row and column to fix the null space, so the vector has N² + 1 entries while the clusters cover the N² grid cells. The `bincount` in the preconditioner then gets weights of one length and indices of the other.

## Notes for Dion (not part of the issue)

- Reproducer: `~/localfiles/runs/cleanup3/phiml-cluster/repro.py` on the sandbox, run in any PhiFlow tesseract image with `JAX_PLATFORMS=cpu`.
- On the benchmark's own 2-D fields, `rank_deficiency=0` with `'cluster'` ran but needed more iterations than plain CG (`phiflow-cost.md`); on the noise field above it does not converge. Either way, a fix would not change Mosaic's PhiFlow entry.
