# JAX-FEM: `umfpack_solver` option vs the default `jax_solver`

**Status:** Dion chose option 3 (2026-10-27). Done: the overhead was recompilation, the jitted solve removes it, draft #286 pending quiet-GPU timing.

Both JAX-FEM tesseracts pass `solver_options={"umfpack_solver": {}}` (forward and adjoint) to `ad_wrapper`, copied from the JAX-FEM topology-optimization demo. That calls SciPy's `spsolve` on the host. The library default is `jax_solver`, a Jacobi-preconditioned BiCGSTAB that runs on the GPU. I ran both on the A100 box, one image each, nothing else running.

**Accuracy:** identical. Every forward metric in both domains matches to all recorded digits, and `gradient/fd_check` (both) and `source_fd_check` (thermal) give the same errors.

**Cost** (mean of 3 trials, seconds):

|                            | nx   | `umfpack_solver` (now) | `jax_solver` (default) |
| -------------------------- | ---- | ---------------------- | ---------------------- |
| structural `spatial_cost`  | 4    | 0.19                   | 0.53                   |
|                            | 32   | 0.45                   | 0.75                   |
|                            | 128  | 13.7                   | 2.8                    |
| structural `vjp_cost/by_N` | 4    | 0.72                   | 1.29                   |
|                            | 64   | 4.65                   | 1.49                   |
|                            | 128  | 24.7                   | 3.18                   |
| thermal `spatial_cost`     | 16   | 0.25                   | 0.84                   |
|                            | 1024 | 66.2                   | 8.5                    |
| thermal `vjp_cost/by_N`    | 16   | 0.34                   | 2.13                   |
|                            | 1024 | 126                    | 17.2                   |

The default has a fixed cost of roughly 0.5 s per forward and 1 to 2 s per VJP, which looks like per-call compilation or setup rather than solve time (not investigated). Above a few thousand cells it is 5 to 8 times faster.

**What it means:**

- The cost columns at large sizes would improve a lot with the default.
- The optimization experiments run at small sizes (thermal conductivity recovery at 16 × 8, hundreds of VJP calls) and would slow down several-fold.
- `jax_solver` is the documented default, and the option has no measured reason in Mosaic. The demo is the only source.

**Options:**

1. Switch to the default (documented use). Cost at large N improves 5 to 8×; small-N calls and the optimization wall time get 2 to 6× slower.
2. Keep `umfpack_solver` and record the measurement as its reason (`too-slow` at small N, L0, tuned).
3. Look into the fixed overhead of `jax_solver` first. If it is re-tracing on every call, caching it could give the default's large-N speed without the small-N penalty. That would be a runtime adaptation under policy 2.

**Recommendation:** option 3, then option 1 if the overhead goes away. Otherwise option 2, which matches the evidence for this benchmark's mix of sizes. I am unsure how much the optimization experiments' wall time should weigh, since it is not a scored check as far as I can tell.

Images: `jax_fem_{structural,thermal}_mesh:jaxsolver` on the box (lost on reset); the change is deleting the two `solver_options` lines.

## Option 3, measured (2026-10-27)

The fixed cost is recompilation. `jax_fem.solver.jax_solve` calls `jax.scipy.sparse.linalg.bicgstab` eagerly, so every linear solve traces and compiles anew: 2 XLA compilations per forward call and 4 per VJP, measured in the container. Passing the same Jacobi-preconditioned BiCGSTAB, jitted once per matrix size, through JAX-FEM's `custom_solver` option brings that to 0 after the first call.

Accuracy over all suites equals umfpack to the recorded digits, except the thermal recovery final ratio (2.17e-8 → 2.4e-8). Cost (shared GPU, so rough):

|                            | N    | umfpack | `jax_solver` | jitted |
| -------------------------- | ---- | ------- | ------------ | ------ |
| structural `spatial_cost`  | 4    | 0.26    | 0.56         | 0.30   |
|                            | 128  | 14.0    | 2.56         | 2.46   |
| structural `vjp_cost/by_N` | 4    | 0.48    | 1.13         | 0.62   |
|                            | 128  | 24.5    | 5.62         | 4.50   |
| thermal `spatial_cost`     | 16   | 0.23    | 0.57         | 0.29   |
|                            | 1024 | 69.6    | 7.0          | 8.4    |
| thermal `vjp_cost/by_N`    | 16   | 0.58    | 1.2          | 0.64   |
|                            | 1024 | 137     | 14.3         | 14.0   |

Small-N calls stay 15 to 30% slower than umfpack. Optimization wall time: thermal recovery 1026 → 1012 s; structural `topopt` 2591 → 4241 s, measured with the two arms overlapping, so not yet trusted. Under policy 2 the jit cache is allowed (it depends only on matrix shapes).
