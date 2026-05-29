## Status diff vs base

**Legend** · ✅ ok · 🟠 anom · ❌ fail · · missing · 🚫 excluded (permanent — out of score denominator) · ⚪ excluded (work-to-do) · **\*** stale — result predates current tesseract/harness source

**0 regression(s)** · **0 improvement(s)** · 0 other transition(s) · 60 new row(s) · 0 removed row(s) · score — → 0.96

### Added experiments

- `ns-3d-grid` · `forward/agreement`
- `ns-3d-grid` · `forward/baseline`
- `ns-3d-grid` · `forward/physical_laws/vs_N`
- `ns-3d-grid` · `forward/physical_laws/vs_nu`
- `ns-3d-grid` · `forward/physical_laws/vs_steps`
- `ns-3d-grid` · `cost/spatial_cost`
- `ns-3d-grid` · `cost/temporal_cost`
- `ns-3d-grid` · `cost/vjp_cost/by_N`
- `ns-3d-grid` · `cost/vjp_cost/by_steps`
- `ns-3d-grid` · `gradient/fd_check`
- `ns-3d-grid` · `gradient/horizon_sweep_limits`
- `ns-3d-grid` · `gradient/jacobian_svd`
- `ns-3d-grid` · `gradient/jacobian_svd_nu01`
- `ns-3d-grid` · `gradient/jacobian_svd_steps20`
- `ns-3d-grid` · `gradient/jacobian_svd_steps40`
- `ns-3d-grid` · `optimization/recovery_constant_ic_bfgs_proj`
- `ns-grid` · `forward/agreement/multimode`
- `ns-grid` · `forward/agreement/tgv`
- `ns-grid` · `forward/baseline`
- `ns-grid` · `forward/cylinder`
- `ns-grid` · `forward/physical_laws/vs_N`
- `ns-grid` · `forward/physical_laws/vs_nu`
- `ns-grid` · `forward/physical_laws/vs_steps`
- `ns-grid` · `forward/tgv_nu_sweep`
- `ns-grid` · `cost/spatial_cost`
- `ns-grid` · `cost/temporal_cost`
- `ns-grid` · `cost/vjp_cost/by_N`
- `ns-grid` · `cost/vjp_cost/by_steps`
- `ns-grid` · `gradient/fd_check`
- `ns-grid` · `gradient/horizon_sweep`
- `ns-grid` · `gradient/jacobian_svd`
- `ns-grid` · `gradient/jacobian_svd_nu01`
- `ns-grid` · `gradient/jacobian_svd_steps20`
- `ns-grid` · `gradient/jacobian_svd_steps40`
- `ns-grid` · `gradient/param_sweep`
- `ns-grid` · `optimization/drag_opt`
- `structural-mesh` · `forward/agreement`
- `structural-mesh` · `forward/baseline`
- `structural-mesh` · `forward/physical_laws`
- `structural-mesh` · `cost/spatial_cost`
- `structural-mesh` · `cost/temporal_cost`
- `structural-mesh` · `cost/vjp_cost/by_N`
- `structural-mesh` · `cost/vjp_cost/by_steps`
- `structural-mesh` · `gradient/fd_check`
- `structural-mesh` · `gradient/param_sweep`
- `structural-mesh` · `optimization/topopt`
- `thermal-mesh` · `forward/agreement`
- `thermal-mesh` · `forward/baseline`
- `thermal-mesh` · `forward/physical_laws`
- `thermal-mesh` · `forward/source_baseline`
- `thermal-mesh` · `forward/source_linearity`
- `thermal-mesh` · `cost/spatial_cost`
- `thermal-mesh` · `cost/temporal_cost`
- `thermal-mesh` · `cost/vjp_cost/by_N`
- `thermal-mesh` · `cost/vjp_cost/by_steps`
- `thermal-mesh` · `gradient/fd_check`
- `thermal-mesh` · `gradient/param_sweep`
- `thermal-mesh` · `gradient/source_fd_check`
- `thermal-mesh` · `gradient/source_width_sweep`
- `thermal-mesh` · `optimization/conductivity_recovery_bfgs`

## Mosaic status

**Legend** · ✅ ok · 🟠 anom · ❌ fail · · missing · 🚫 excluded (permanent — out of score denominator) · ⚪ excluded (work-to-do) · **\*** stale — result predates current tesseract/harness source

| problem | ok | anom | fail | missing | excl (work) | excl (perm) | stale | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ns-3d-grid` | 99 | 0 | 0 | 4 | 0 | 9 | 0 | 🟢 **0.97** |
| `ns-grid` | 103 | 17 | 0 | 4 | 0 | 16 | 0 | 🟢 **0.91** |
| `structural-mesh` | 45 | 0 | 0 | 0 | 0 | 5 | 0 | 🟢 **1.00** |
| `thermal-mesh` | 63 | 0 | 0 | 0 | 0 | 7 | 0 | 🟢 **1.00** |
| **overall** | **310** | **17** | **0** | **8** | **0** | **37** | **0** | 🟢 **0.96** |

### Failures & anomalies

- 🟠 `ns-grid` · `forward/agreement/multimode` · **INS.jl** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **jax-cfd** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **OpenFOAM** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **PhiFlow** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **PICT** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **Warp-NS** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/multimode` · **XLB** — error 1.03 at sweep=0.001 > max_error=0.5
- 🟠 `ns-grid` · `forward/agreement/tgv` · **jax-cfd** — error 0.0145 at sweep=0.001 is 6.0× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/agreement/tgv` · **PhiFlow** — phiflow's double CenteredGrid↔StaggeredGrid resampling gives 4.18% amplitude damping (ratio=0.9582); cosine=0.9999924 (p…
<details><summary>Full traceback</summary>

```
phiflow's double CenteredGrid↔StaggeredGrid resampling gives 4.18% amplitude damping (ratio=0.9582); cosine=0.9999924 (pattern correct); arithmetic-average output conversion fix worsened error 9×; upstream library change required
```

</details>

- 🟠 `ns-grid` · `forward/agreement/tgv` · **XLB** — error 0.274 at sweep=0.001 is 113.7× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/baseline` · **INS.jl** — staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) rou…
<details><summary>Full traceback</summary>

```
staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) round-trip error at all N; 35-40x above collocated peers
```

</details>

- 🟠 `ns-grid` · `forward/baseline` · **jax-cfd** — staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) rou…
<details><summary>Full traceback</summary>

```
staggered MAC grid double-interpolation: collocated TGV IC -> staggered faces -> collocated output gives sin^2(pi/N) round-trip error at all N; 35-40x above collocated peers
```

</details>

- 🟠 `ns-grid` · `forward/baseline` · **XLB** — error 0.00704 at sweep=128 is 11.7× peer median (0.000602); threshold k=3.0
- 🟠 `ns-grid` · `forward/cylinder` · **PICT** — error 0.672 at sweep=0.05 > max_error=0.5
- 🟠 `ns-grid` · `forward/tgv_nu_sweep` · **jax-cfd** — error 0.0145 at sweep=0.0001 is 6.0× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `forward/tgv_nu_sweep` · **XLB** — error 0.275 at sweep=0.0001 is 114.2× peer median (0.00241); threshold k=3.0
- 🟠 `ns-grid` · `cost/temporal_cost` · **PhiFlow** — median time 11.4s is 20× peer median (0.57s); threshold k=20.0

<details><summary>ns-3d-grid — 16 experiment(s)</summary>

| experiment | `Exponax` | `INS.jl` | `OpenFOAM` | `PhiFlow` | `PICT` | `Warp-NS` | `XLB` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/baseline` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_N` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_nu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_steps` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/horizon_sweep_limits` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd` | ✅ | · | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_nu01` | ✅ | · | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps20` | ✅ | · | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps40` | ✅ | · | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/recovery_constant_ic_bfgs_proj` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |

</details>

<details><summary>ns-grid — 20 experiment(s)</summary>

| experiment | `INS.jl` | `jax-cfd` | `OpenFOAM` | `PhiFlow` | `PICT` | `Warp-NS` | `XLB` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement/multimode` | 🟠 | 🟠 | 🟠 | 🟠 | 🟠 | 🟠 | 🟠 |
| `forward/agreement/tgv` | ✅ | 🟠 | ✅ | 🟠 | ✅ | ✅ | 🟠 |
| `forward/baseline` | 🟠 | 🟠 | ✅ | ✅ | ✅ | ✅ | 🟠 |
| `forward/cylinder` | 🚫 | 🚫 | ✅ | ✅ | 🟠 | 🚫 | ✅ |
| `forward/physical_laws/vs_N` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_nu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws/vs_steps` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/tgv_nu_sweep` | ✅ | 🟠 | ✅ | ✅ | ✅ | ✅ | 🟠 |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/horizon_sweep` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd` | · | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_nu01` | · | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps20` | · | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/jacobian_svd_steps40` | · | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/param_sweep` | ✅ | ✅ | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/drag_opt` | 🚫 | 🚫 | 🚫 | ✅ | ✅ | 🚫 | ✅ |

</details>

<details><summary>structural-mesh — 10 experiment(s)</summary>

| experiment | `deal.II` | `FEniCS` | `Firedrake` | `JAX-FEM` | `TopOpt.jl` |
|---|:---:|:---:|:---:|:---:|:---:|
| `forward/agreement` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/baseline` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `forward/physical_laws` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/spatial_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/temporal_cost` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_N` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `cost/vjp_cost/by_steps` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/fd_check` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `gradient/param_sweep` | 🚫 | ✅ | ✅ | ✅ | ✅ |
| `optimization/topopt` | 🚫 | ✅ | ✅ | ✅ | ✅ |

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

> **Note on wall-clock measurements:** Cost-suite timings are collected on dedicated CI runners
> with no concurrent benchmark workloads. Relative solver rankings within a single run are
> reliable; absolute wall times may vary ±10–15% across runs due to cloud VM variability.

