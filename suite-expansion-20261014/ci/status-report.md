**Coverage:** partial run — some cells are shown from the baseline rather than measured this run.
## Status diff vs base

**Legend** · ✅ ok · 🟠 anom · ❌ fail · · missing · 🚫 excluded (permanent — out of score denominator) · ⚪ excluded (work-to-do) · **\*** stale — result predates current benchmark run

**0 regression(s)** · **0 improvement(s)** · 0 metric change(s) · 0 other transition(s) · 0 resource-frontier shift(s) · 11 new row(s) · 0 removed row(s) · score 0.94 → 0.95

### Added experiments

- `buoyancy-grid` · `forward/diffusion`
- `buoyancy-grid` · `forward/physical_laws`
- `buoyancy-grid` · `forward/refinement`
- `buoyancy-grid` · `cost/spatial_cost`
- `buoyancy-grid` · `cost/temporal_cost`
- `buoyancy-grid` · `cost/vjp_cost/by_N`
- `buoyancy-grid` · `cost/vjp_cost/by_steps`
- `buoyancy-grid` · `gradient/fd_check`
- `buoyancy-grid` · `gradient/horizon_sweep`
- `buoyancy-grid` · `gradient/jacobian_svd`
- `buoyancy-grid` · `optimization/solver_in_loop`

<details><summary>Full Mosaic status</summary>

## Mosaic status

**Legend** · ✅ ok · 🟠 anom · ❌ fail · · missing · 🚫 excluded (permanent — out of score denominator) · ⚪ excluded (work-to-do) · **\*** stale — result predates current benchmark run

Each solver is run against every experiment in the suite. **ok** = produced valid results; **fail** = crashed or returned invalid data; **anom** = ran successfully but tripped an automated quality check (e.g. poor gradient accuracy, outlier wall-clock time, or diverged optimisation). Thresholds are defined per-problem in the problem config.

| problem | ok | anom | fail | missing | excl (work) | excl (perm) | stale | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `buoyancy-grid` | 54 | 1 | 0 | 0 | 0 | 0 | 0 | 🟢 **0.99** |
| `ns-3d-grid` | 90 | 6 | 7 | 0 | 0 | 9 | 0 | 🟢 **0.90** |
| `ns-grid` | 106 | 15 | 3 | 0 | 0 | 16 | 0 | 🟢 **0.92** |
| `structural-mesh` | 55 | 0 | 0 | 0 | 0 | 5 | 0 | 🟢 **1.00** |
| `thermal-mesh` | 63 | 0 | 0 | 0 | 0 | 7 | 0 | 🟢 **1.00** |
| **overall** | **368** | **22** | **10** | **0** | **0** | **37** | **0** | 🟢 **0.95** |

### Failures & anomalies

- 🟠 `buoyancy-grid` · `forward/physical_laws` · **PhiFlow** — error 0.0183 at sweep=128 > max_error=0.01
- 🟠 `ns-3d-grid` · `forward/baseline` · **INS.jl** — error 0.147 at sweep=8 is 43.1× peer median (0.00341); threshold k=3.0
- 🟠 `ns-3d-grid` · `forward/baseline` · **PhiFlow** — error 0.147 at sweep=8 is 43.1× peer median (0.00341); threshold k=3.0
- 🟠 `ns-3d-grid` · `forward/baseline` · **XLB** — error 0.273 at sweep=8 is 80.2× peer median (0.00341); threshold k=3.0
- ❌ `ns-3d-grid` · `cost/spatial_cost` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- ❌ `ns-3d-grid` · `cost/temporal_cost` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- ❌ `ns-3d-grid` · `cost/vjp_cost/by_N` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- 🟠 `ns-3d-grid` · `cost/vjp_cost/by_N` · **XLB** — median time 171.8s is 64× peer median (2.67s); threshold k=20.0
- ❌ `ns-3d-grid` · `cost/vjp_cost/by_steps` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- 🟠 `ns-3d-grid` · `cost/vjp_cost/by_steps` · **XLB** — median time 564.5s is 208× peer median (2.72s); threshold k=20.0
- 🟠 `ns-3d-grid` · `gradient/fd_check` · **INS.jl** — best-ε median FD rel_err 2.36e-03 > max_rel_err=1e-03
- ❌ `ns-3d-grid` · `gradient/horizon_sweep_limits` · **PhiFlow** — nan
- ❌ `ns-3d-grid` · `gradient/horizon_sweep_limits` · **PICT** — error
- ❌ `ns-3d-grid` · `gradient/horizon_sweep_limits` · **Warp-NS** — nan
- ❌ `ns-grid` · `forward/agreement/multimode` · **XLB** — all entries failed
- 🟠 `ns-grid` · `forward/agreement/tgv` · **jax-cfd** — error 0.0145 at sweep=0.001 is 6.0× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/agreement/tgv` · **PhiFlow** — phiflow's double CenteredGrid↔StaggeredGrid resampling gives 4.18% amplitude damping (ratio=0.9582); cosine=0.9999924 (pattern correct); arithmetic-average output conversion fix worsened error 9×; upstream library change required
- 🟠 `ns-grid` · `forward/agreement/tgv` · **XLB** — error 0.274 at sweep=0.001 is 113.7× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/baseline` · **INS.jl** — staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) round-trip error at all N; 35-40x above collocated peers
- 🟠 `ns-grid` · `forward/baseline` · **jax-cfd** — staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) round-trip error at all N; 35-40x above collocated peers
- 🟠 `ns-grid` · `forward/baseline` · **XLB** — error 0.00704 at sweep=128 is 11.7× peer median (0.000602); threshold k=3.0
- 🟠 `ns-grid` · `forward/cylinder` · **OpenFOAM** — error 0.694 at sweep=0.05 > max_error=0.5
- 🟠 `ns-grid` · `forward/cylinder` · **PICT** — error 0.672 at sweep=0.05 > max_error=0.5
- 🟠 `ns-grid` · `forward/tgv_nu_sweep` · **jax-cfd** — error 0.0145 at sweep=0.0001 is 6.0× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/tgv_nu_sweep` · **XLB** — error 0.275 at sweep=0.0001 is 114.2× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `cost/spatial_cost` · **OpenFOAM** — median time 10.7s is 36× peer median (0.30s); threshold k=20.0
- ❌ `ns-grid` · `cost/spatial_cost` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- 🟠 `ns-grid` · `cost/temporal_cost` · **PhiFlow** — median time 11.7s is 39× peer median (0.30s); threshold k=20.0
- ❌ `ns-grid` · `cost/vjp_cost/by_N` · **PhiFlow** — error: RuntimeError: Error 500 from Tesseract: Traceback (most recent call last):
- 🟠 `ns-grid` · `optimization/drag_opt` · **PhiFlow** — final/initial = 0.85 (> 0.5)
- 🟠 `ns-grid` · `optimization/drag_opt` · **PICT** — final/initial = 0.74 (> 0.5)
- 🟠 `ns-grid` · `optimization/drag_opt` · **XLB** — final/initial = 0.69 (> 0.5)

<details><summary>buoyancy-grid — 11 experiment(s)</summary>

| experiment | `INS.jl` | `JAX-CFD` | `Oceananigans` | `PhiFlow` | `WaterLily` |
|---|:---:|:---:|:---:|:---:|:---:|
| `forward/diffusion` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws` | ✅ | ✅ | ✅ | 🟠 | ✅ |
| `forward/refinement` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gradient/horizon_sweep` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `optimization/solver_in_loop` | ✅ | ✅ | ✅ | ✅ | ✅ |

</details>

<details><summary>ns-3d-grid — 16 experiment(s)</summary>

| experiment | `Exponax` | `INS.jl` | `OpenFOAM` | `PhiFlow` | `PICT` | `Warp-NS` | `XLB` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/baseline` | ✅ | 🟠 | ✅ | 🟠 | ✅ | ✅ | 🟠 |
| `forward/physical_laws/vs_N` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_nu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_steps` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | ✅ | ✅ | 🚫 | ❌ | ✅ | ✅ | 🟠 |
| `cost/vjp_cost/by_steps` | ✅ | ✅ | 🚫 | ❌ | ✅ | ✅ | 🟠 |
| `gradient/fd_check` | ✅ | 🟠 | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/horizon_sweep_limits` | ✅ | ✅ | 🚫 | ❌ | ❌ | ❌ | ✅ |
| `gradient/jacobian_svd` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_nu01` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps20` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps40` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/recovery_constant_ic_bfgs_proj` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |

</details>

<details><summary>ns-grid — 20 experiment(s)</summary>

| experiment | `INS.jl` | `jax-cfd` | `OpenFOAM` | `PhiFlow` | `PICT` | `Warp-NS` | `XLB` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement/multimode` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `forward/agreement/tgv` | ✅ | 🟠 | ✅ | 🟠 | ✅ | ✅ | 🟠 |
| `forward/baseline` | 🟠 | 🟠 | ✅ | ✅ | ✅ | ✅ | 🟠 |
| `forward/cylinder` | 🚫 | 🚫 | 🟠 | ✅ | 🟠 | 🚫 | ✅ |
| `forward/physical_laws/vs_N` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_nu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_steps` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/tgv_nu_sweep` | ✅ | 🟠 | ✅ | ✅ | ✅ | ✅ | 🟠 |
| `cost/spatial_cost` | ✅ | ✅ | 🟠 | ❌ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | ✅ | ✅ | 🚫 | ❌ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/horizon_sweep` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_nu01` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps20` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps40` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/param_sweep` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/drag_opt` | 🚫 | 🚫 | 🚫 | 🟠 | 🟠 | 🚫 | 🟠 |

</details>

<details><summary>structural-mesh — 10 experiment(s)</summary>

| experiment | `deal.II` | `FEniCS` | `Firedrake` | `JAX-FEM` | `TopOpt.jl` | `torch-fem` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/baseline` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | 🚫 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | 🚫 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | 🚫 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `gradient/param_sweep` | 🚫 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `optimization/topopt` | 🚫 | ✅ | ✅ | ✅ | ✅ | ✅ |

</details>

<details><summary>thermal-mesh — 14 experiment(s)</summary>

| experiment | `deal.II` | `FEniCS` | `Firedrake` | `JAX-FEM` | `torch-fem` |
|---|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/baseline` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/source_baseline` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/source_linearity` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/param_sweep` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/source_fd_check` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/source_width_sweep` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/conductivity_recovery_bfgs` | 🚫 | ✅ | ✅ | ✅ | ✅ |

</details>

</details>
