# Known Issues

## Cross-Cutting

### [CC-1] Canonical schemas have `Differentiable` baked in; solvers should use `make_differentiable` instead

**Files:** `mosaic/mosaic_shared/problems/structural_mesh/schemas.py`,
`mosaic/mosaic_shared/problems/thermal_mesh/schemas.py`,
`mosaic/mosaic_shared/problems/navier_stokes_grid/schemas.py`

All canonical `InputSchema` / `OutputSchema` definitions hard-code `Differentiable[...]`
on specific fields. This conflates the data contract (what fields exist) with the
differentiation contract (which fields support gradients), and forces every conforming
solver to advertise capabilities it may not have.

`make_differentiable` already exists at `mosaic/mosaic_shared/types.py:244` and is the
right tool: base schemas should carry plain array types, and each solver wraps only the
fields it can actually differentiate:

```python
# In a solver's tesseract_api.py
from mosaic_shared.problems.structural_mesh import OutputSchema as _Base
from mosaic_shared.types import make_differentiable

OutputSchema = make_differentiable(_Base, ["compliance"])
```

**Resolves SM-2 and TM-1** entirely, and makes SM-3 / JVP inconsistencies visible at
the schema level rather than at runtime.

---

## Structural Mesh — Tesseract Implementation

### [SM-1] OutputSchema has too many canonical outputs

**File:** `mosaic/mosaic_shared/problems/structural_mesh/schemas.py`

The canonical `OutputSchema` currently includes `compliance`, `von_mises_stress`, and
`displacement`. Not all structural mesh solvers compute von Mises stress or displacement —
these are solver-dependent extras, not universal canonicals. The schema should only mandate
what every conforming solver is guaranteed to produce.

**Impact:** Tesseracts that don't return `von_mises_stress` or `displacement` are forced to
either stub those fields or fail schema validation.

---

### [SM-2] All outputs are marked `Differentiable` regardless of solver support

**File:** `mosaic/mosaic_shared/problems/structural_mesh/schemas.py`

All three output fields (`compliance`, `von_mises_stress`, `displacement`) are typed as
`Differentiable[...]`. Several solvers cannot differentiate through von Mises stress or
displacement — only `compliance` has reliable gradient support across implementations.

**Impact:** Consumers that request gradients of non-differentiable outputs will silently get
wrong results or hard errors at runtime rather than a clear schema-level contract.

**Resolves with CC-1:** Remove `Differentiable` from the canonical schema entirely; each
solver calls `make_differentiable` on the fields it actually supports.

---

### [SM-3] JVP coverage is inconsistent across structural mesh solvers

`jax-fem` defines a `jacobian_vector_product` and asserts JVP support for all three outputs
(`compliance`, `von_mises_stress`, `displacement`):

- `mosaic/tesseracts/structural-mesh/jax-fem/tesseract_api.py:375`

`dealii` has no JVP implementation at all — forward-only.

- `mosaic/tesseracts/structural-mesh/dealii/tesseract_api.py` (uses `_CanonicalOutputSchema`, no `jacobian_vector_product`)

**Impact:** Consumers cannot rely on JVP being available for any canonical structural mesh
tesseract. The inconsistency is invisible at the schema level — callers only discover the
gap at runtime.

**Resolves with SM-1/CC-1:** If the canonical schema narrows to `compliance`-only and
solvers declare differentiability via `make_differentiable`, this inconsistency becomes
visible at the schema level and per-solver JVP extras are opt-in.

**Note:** The thermal mesh `jax-fem` solver has the same pattern (`mosaic/tesseracts/thermal-mesh/jax-fem/tesseract_api.py:339`) and should be reviewed together.

---

### [SM-4] topopt-jl duplicates von Mises FEA math in NumPy instead of Julia

**File:** `mosaic/tesseracts/structural-mesh/topopt-jl/tesseract_api.py`

`_von_mises_hex8`, `_von_mises_direct_grad`, and `_von_mises_adjoint_rhs` are a full
NumPy reimplementation of B-matrix assembly, constitutive law, and stress-sensitivity
math that belongs in Julia. The compliance and displacement VJP paths both delegate their
heavy lifting to Julia (`topopt_forward`, `topopt_general_vjp`); von Mises does not.

Notably, `Zygote` is imported in `topopt_solver.jl` but never used — the adjoint is
already manual there, so a Julia von Mises function with a matching hand-written adjoint
would be consistent with the existing pattern.

**Resolves with SM-1:** `compliance` is the only output actually consumed by the
optimization loop (verified in `mosaic/benchmarks/suites/optimization.py` and
`mosaic/benchmarks/plots/paper/physical_accuracy.py`). Once von Mises is removed from the
canonical schema, this code becomes dead weight and can simply be deleted.

---

## Thermal Mesh — Tesseract Implementation

### [TM-1] OutputSchema has too many canonical outputs / wrong `Differentiable` annotations

**File:** `mosaic/mosaic_shared/problems/thermal_mesh/schemas.py`

Mirrors SM-1 and SM-2. The canonical `OutputSchema` exposes three fields —
`thermal_compliance`, `temperature`, and `identification_error` — all marked
`Differentiable`. In practice:

- `thermal_compliance` is the objective for topology-optimisation runs.
- `identification_error` is the objective for source-identification runs.
- `temperature` is only read for visualization in `conductivity_recovery`
  (`mosaic/benchmarks/suites/optimization.py:1896`) — never in a gradient path.

`temperature` should be removed from the canonical schema (or demoted to a non-`Differentiable`
solver-specific extra). Only `thermal_compliance` and `identification_error` need to remain,
and only those two need the `Differentiable` annotation.

**Resolves with CC-1:** Use `make_differentiable` per solver instead of annotating the
canonical schema.

---

### [TM-2] dealii-heat VJP code is unnecessary — it is the reference solver

**File:** `mosaic/tesseracts/thermal-mesh/dealii-heat/tesseract_api.py`

`dealii-heat` implements a full VJP (`vector_jacobian_product`, `_compute_source_vjp`,
`_compute_compliance_source_vjp`) even though it is defined as the preferred reference
solver in `conductivity_recovery` (`mosaic/benchmarks/suites/optimization.py:1785`). Its
role is to produce ground-truth temperature fields for visualization — it is never the
solver being optimized. The VJP code exists only so `dealii-heat` qualifies as a
`_diff_solver` candidate and can be selected as reference; it is not exercised in any
meaningful gradient path.

**Suggestion:** Make `dealii-heat` forward-only and select the reference solver
independently of the differentiability requirement.

---

## Navier-Stokes Grid — Tesseract Implementation

### [NS-1] Lid-cavity boundary condition code should be removed from all solvers

Lid-cavity BC logic is scattered across three NS solvers and the benchmark suite:

- `mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py:321` — `_apply_lid_cavity_bc_3d` function and call sites
- `mosaic/tesseracts/navier-stokes-grid/warp-ns/tesseract_api.py:3016` — lid-cavity comment in BC fix
- `mosaic/tesseracts/navier-stokes-grid/pict/tesseract_api.py:447` — canonical lid-cavity IC comment
- `mosaic/benchmarks/problems/navier_stokes_3d_grid.py:123,217` — `recovery/lid_cavity` benchmark entries
- `mosaic/benchmarks/plots/paper/lid_cavity.py` — dedicated plot script

The canonical `InputSchema` in `mosaic/mosaic_shared/problems/navier_stokes_grid/schemas.py:90`
also carries a `lid_velocity` field that exists solely for this experiment and should be
removed along with all call sites in xlb, warp-ns, and incompressible-navier-stokes-jl.

**Suggestion:** Remove all lid-cavity-specific BC code from the solvers, drop `lid_velocity`
from the canonical `InputSchema`, and drop the `recovery/lid_cavity` benchmark entries.

---

### [NS-2] Dead code in phiflow — `staggered_to_collocated_ux` and `get_pressure` are never called

**File:** `mosaic/tesseracts/navier-stokes-grid/phiflow/tesseract_api.py:310`

Both inner functions are defined but have no call sites anywhere in the file.

---

### [NS-3] Extensive dead code in warp-ns — 11 unused kernels/helpers

**File:** `mosaic/tesseracts/navier-stokes-grid/warp-ns/tesseract_api.py`

The following `@wp.func` / `@wp.kernel` symbols are defined but never launched or called:

| Symbol | Line | Note |
|--------|------|------|
| `wrap_idx` | 38 | Docstring claims use in stencil kernels; kernels inline the modulo instead |
| `clamp_idx` | 49 | Intended for Neumann BCs; never referenced |
| `curl_to_vorticity_kernel` | 121 | Vorticity computation; never launched |
| `psi_to_velocity_kernel` | 140 | Streamfunction → velocity; never launched |
| `vorticity_rhs_kernel` | 158 | Vorticity RHS; never launched |
| `rk3_stage1_kernel` | 197 | SSP-RK3 stage 1; never launched |
| `rk3_stage2_kernel` | 209 | SSP-RK3 stage 2; never launched |
| `rk3_combine_kernel` | 227 | SSP-RK3 combine; never launched |
| `apply_obstacle_mask_kernel` | 242 | Volume penalization mask; never launched |
| `_apply_pressure_correction_kernel` | 781 | Periodic-BC pressure correction; wall variant is used instead |
| `apply_lid_bc_kernel` | 889 | Lid BC (superseded by `apply_lid_field_bc_kernel`); never launched |

Several of these (`rk3_*`, `vorticity_*`, `psi_to_velocity_kernel`) look like an abandoned vorticity-streamfunction solver path.
