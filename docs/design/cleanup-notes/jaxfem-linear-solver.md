# JAX-FEM: `umfpack_solver` option vs the default `jax_solver`

**Status:** measured 2026-10-26, decision for Dion. No PR.

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
