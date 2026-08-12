# ns-3d-grid converged references

## `forward_agreement.npz`

The accuracy reference for the `forward/agreement` experiment (F3 forward).

Each `reference_{i}` is the converged final velocity field for the `i`-th
viscosity in the sweep (`nu ∈ [0.001, 0.01, 0.05]`), on the benchmark grid
`N=16`. It is produced by integrating the **full nonlinear** 3D Navier–Stokes
equations with the spectral ETDRK solver (Exponax, incompressibility to machine
precision) at `N=64`, then spectrally truncating the final field back to `N=16`.

### Why not the analytic decay?

The linearized decay `u(t) = u(0)·exp(-2νt)` is exact for the _2D_ TGV but not
the _3D_ one: the 3D initial condition is not advection-free, so vortex
stretching immediately generates z-structure. At the benchmark parameters the
analytic field is essentially the frozen initial condition, so every solver
that correctly integrates the nonlinear dynamics is penalized by a large
common-mode bias (~0.11 relative) that swamps the real per-solver differences.
See issue #123.

Cross-validation at generation time (`generate.py`):

| nu    | ‖converged − analytic‖ | ‖bench − converged‖ (truncation err) |
| ----- | ---------------------- | ------------------------------------ |
| 0.001 | 1.24e-01               | 7.6e-06                              |
| 0.01  | 1.22e-01               | 6.4e-06                              |
| 0.05  | 1.14e-01               | 3.0e-06                              |

The left column is the common-mode bias the analytic reference was measuring;
the right column shows the converged reference is resolution-independent to
~1e-5, far below any meaningful solver error.

### Regenerating

```
python -m mosaic.benchmarks.problems.navier_stokes_3d_grid.references.generate
```

Requires `exponax` and `equinox` importable (the same packages the Exponax
tesseract pins).
