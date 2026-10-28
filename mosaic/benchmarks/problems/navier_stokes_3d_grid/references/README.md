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

Cross-validation (measured when the reference was generated):

| nu    | ‖converged − analytic‖ | ‖bench − converged‖ (truncation err) |
| ----- | ---------------------- | ------------------------------------ |
| 0.001 | 1.24e-01               | 7.6e-06                              |
| 0.01  | 1.22e-01               | 6.4e-06                              |
| 0.05  | 1.14e-01               | 3.0e-06                              |

The left column is the common-mode bias the analytic reference was measuring;
the right column shows the truncation to `N=16` loses nothing meaningful (the
field is band-limited well below the `N=16` Nyquist cutoff), so the downsampled
high-`N` run is a faithful ground truth on the benchmark grid.

### Regenerating

```
mosaic reference -p ns-3d-grid -e forward/agreement
```

The converged strategy (reference solver + resolution) is registered in
`CONVERGED_REFERENCES` in `mosaic/benchmarks/core/reference.py`. Regeneration
builds the reference solver's tesseract image (Docker required).

## `forward_baseline.npz`

The accuracy reference for the `forward/baseline` resolution sweep
(`N ∈ [8, 16, 32]`). Each `reference_{i}` is the velocity field at `t = 1.28`
on the `i`-th grid, spectrally truncated from one Exponax run at `N=64`
(`dt = 0.02`, 64 steps, the benchmark's CFL number).

Generating the reference from an `N=128` run instead changes it by `1.2e-5`
(relative L2) on every grid, far below the errors the experiment measures.

Regenerate with `mosaic reference -p ns-3d-grid -e forward/baseline`.

Both references use Exponax's default 2/3-rule dealiasing. Turning dealiasing
off changes them by 6e-7 (relative L2) at every grid and viscosity.
