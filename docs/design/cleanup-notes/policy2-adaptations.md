# Policy 2: runtime adaptations per solver

**Status:** 2026-10-29. Survey of `main` at 3d480d0 against the harness contract in `interventions.md` (items 1 to 8). This is the per-solver statement of policy 2 work item 4. Line numbers refer to `mosaic/tesseracts/` at that commit. Every tesseract is served with `num_workers=1` (`mosaic/benchmarks/core/runner.py`), so module-level state lives for the whole container.

## Per solver

| Tesseract        | Setup kept across calls                                                                                                                                                              | Process model          | Compiled programs reused         | Gaps                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Exponax          | none beyond the jit cache                                                                                                                                                            | in-process             | `eqx.filter_jit`                 | differentiable scalars static in `apply_jit`                                                                       |
| INS.jl           | `_SETUP_CACHE` `(Setup, spectral pressure solver)`, key `(n, L, ndim)`, one entry                                                                                                    | in-process (juliacall) | Julia method cache               | —                                                                                                                  |
| jax-cfd          | none beyond the jit cache                                                                                                                                                            | in-process             | `eqx.filter_jit` forward and VJP | differentiable scalars static, forward and VJP                                                                     |
| OpenFOAM         | none: `blockMesh` (and `decomposePar` for the channel) every call                                                                                                                    | subprocess per request | n/a                              | item 3 (whether a persistent case counts is open)                                                                  |
| PhiFlow          | none in the wrapper (phiml internals unchecked)                                                                                                                                      | in-process             | `eqx.filter_jit` forward and VJP | differentiable scalars static                                                                                      |
| PICT             | none: domain, channel blocks, wall geometry and transfer weights rebuilt per call                                                                                                    | in-process             | n/a                              | item 3; the `Domain` also holds viscosity and velocity                                                             |
| Warp-NS          | 11 `functools.cache` factories (FFT kernels, chirps, `1/λ` arrays) keyed by `(n, extent, device)`, never cleared; channel setup (masks, dense `eigh` basis, CG, CUDA graph) per call | in-process             | Warp modules per `n`             | item 6 (one entry per grid size, ~64 MiB per `1/λ` array at 256³); item 3 for the channel                          |
| XLB              | `_OPS` for 4 fixed configs; `_diff_apply` `functools.cache` keyed `(sub_k, dx)`, unbounded (closures only)                                                                           | in-process             | `eqx.filter_jit`                 | sub-step count `sub_k` is set from max\|v0\| or the inflow peak times `dt`, both differentiable, and is jit-static |
| deal.II (both)   | none                                                                                                                                                                                 | subprocess per request | n/a                              | item 3 and 7 (start-up in every call)                                                                              |
| FEniCS (both)    | `_SETUP_CACHE`: `ReducedFunctional` with tape, SHA-256 key over mesh, BCs and constants, one entry                                                                                   | in-process             | compiled UFL forms               | —                                                                                                                  |
| Firedrake (both) | as FEniCS; MUMPS symbolic factorization reused per the code comment                                                                                                                  | in-process             | TSFC kernels                     | —                                                                                                                  |
| JAX-FEM (both)   | `_SETUP_CACHE` `(Problem, fwd_pred)`, SHA-256 key, one entry                                                                                                                         | in-process             | `_bicgstab` jit per matrix size  | —                                                                                                                  |
| TopOpt.jl        | `_CACHE` `(InpStiffness, FEASolver, Compliance)`, one entry                                                                                                                          | in-process (juliacall) | Julia method cache               | key combines hashes with XOR, so equal or swapped values collide (unlikely here, weaker than SHA-256)              |
| torch-fem (both) | none: model, material and BCs rebuilt per call                                                                                                                                       | in-process             | none                             | item 3                                                                                                             |

Deliberate warm-up: only Firedrake (two replay and derivative sweeps on the first tape, inside the untimed first call).

## Item 2: no state that depends on a differentiable value

None found. Every warm start stays inside one call:

- PICT (channel only): each pressure corrector starts from the previous step's solution, and each adjoint solve from the previous adjoint solution. A new simulation object is created per call.
- Warp-NS channel: pressure and adjoint CG start from the previous step.
- PhiFlow: the pressure carry in the scan.

The FEniCS, Firedrake, JAX-FEM and TopOpt.jl caches hold objects that contain the last call's state. Each call overwrites that state before using it (tape replay, `set_params`, new densities). TopOpt.jl 0.11 and pyadjoint internals were not read.

## Open question: compiled programs keyed on differentiable values

Item 1 lets a cache depend on the shapes of differentiable inputs, not their values. Exponax, jax-cfd, PhiFlow and XLB convert differentiable scalars (`viscosity`, `dt` and others) to Python floats, so jit specializes on their values. The benchmark holds them fixed, so this costs nothing here, but an optimizer that varies them would recompile on every call. XLB goes further: the number of sub-steps per `dt` depends on the velocity magnitude, so the discretization changes with a differentiable input. Whether item 1 should say something about compiled programs is for Dion.

## This pass

- torch-fem: setup cache (model, unscaled stiffness or conductivity, BC tensors), one entry, SHA-256 key. Branch `perf/torch-fem-setup-cache`, A/B pending.
- deal.II: one persistent `--serve` process per tesseract (#297). Each request still builds its own mesh and system. The 47 non-cost cells are identical; small meshes lose the ~0.2 s start-up per call (structural 0.206 → 0.009 s at N = 4, thermal 0.220 → 0.013 s at N = 16). Large meshes are dominated by the solve.
- Not done: PICT, OpenFOAM, the Warp-NS channel and Warp-NS's unbounded caches.
