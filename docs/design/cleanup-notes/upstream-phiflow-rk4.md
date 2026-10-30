# Draft: tum-pbs/PhiFlow

**Status:** filed 2026-10-30 (see the handoff for the issue number).

**Title:** `fluid.incompressible_rk4` is third-order: stage 4 uses `rhs2` instead of `rhs3`

## Body

In `phi/physics/fluid.py` (3.4.0, unchanged on `develop`), the fourth stage of `incompressible_rk4` is built from the second slope:

```python
rhs3 = pde(v3, **pde_aux_kwargs) - p3.gradient(at=v1.sampled_at, order=pressure_order)
v4_old = velocity + dt * rhs2
```

Classical RK4 uses `velocity + dt * rhs3` there. With `rhs2`, the Butcher tableau has a42 = 1 instead of a43 = 1, which satisfies every order condition up to three but not b·A·A·c = 1/24, so the scheme is third-order. `rhs3` is still computed and used in the final combination.

Measured on a case where the projection does nothing: a shear wave u = (sin y, 0) on a periodic 8 × 8 grid, ν = 1, t = 1, which is divergence-free and advection-free, so the semi-discrete solution decays exactly as exp(−ν λ t) with λ = (2 − 2 cos Δy)/Δy². In float64 (`math.set_global_precision(64)`), with `advect.finite_difference(order=2) + diffuse.finite_difference(order=2)` as the PDE and `pressure_order=2`:

| steps | `incompressible_rk4` max error | classical RK4, same PDE |
| ----- | ------------------------------ | ----------------------- |
| 8     | 2.6e-5                         | 6.2e-7                  |
| 16    | 3.1e-6                         | 3.7e-8                  |
| 32    | 3.8e-7                         | 2.2e-9                  |
| 64    | 4.7e-8                         | 1.4e-10                 |
| 128   | 5.8e-9                         | 8.6e-12                 |
| order | 3.0                            | 4.0                     |

Script attached (`rk4order.py`, 40 lines). The higher-order notebooks in the docs (`Taylor_Green_Comparison`, `HigherOrder_Demo`) call this function, so their time integration is third-order too, although on those setups the spatial error may well dominate.

Versions: phiflow 3.4.0, phiml 1.16.1, jax 0.10.0, CPU.

## Notes for Dion (not part of the issue)

- Script: `diag/phiflow/rk4order.py`, run in the phiflow image with `JAX_PLATFORMS=cpu`.
- Effect on Mosaic is negligible: on the ns-grid TGV setups the spatial error dominates (RK4 and forward Euler give the same error to three digits at N = 16 to 64, `diag/phiflow/probe.py`). No workaround is needed under rule 2; the issue is for upstream.
- That the notebooks call `fluid.incompressible_rk4` is from reading them (see `policy1-phiflow.md`); I have not checked their convergence plots.
