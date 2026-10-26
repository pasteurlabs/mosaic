# Policy 2: a harness contract for runtime adaptations

**Status:** draft for Dion. No code changed.

## The problem

Tesseract calls are stateless, and the timing protocol measures one unreported warm-up call followed by `n_trials` timed calls per sweep point (`core/harness.py`, `problems/shared/cost.py`). Several tesseracts work around this differently:

| Solver           | Adaptation                                                                                           | Measured effect                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| JAX-FEM (both)   | module-level setup cache (problem + `ad_wrapper`)                                                    | thermal, before #195: 0.56 s apply / 2.1 s VJP per-call floor; 3124 s vs ~80 s for 200 L-BFGS iterations |
| TopOpt.jl        | cache of INP problem, solver and `Compliance`                                                        | not measured                                                                                             |
| FEniCS (both)    | pyadjoint tape recorded once, replayed (`setup-tape-replay`)                                         | structural: negligible (0.4% of apply); heat: mixed with other #182 changes                              |
| Firedrake (both) | tape replay plus reusable solver                                                                     | #229: heat VJP 15.8 → 1.3 s at nx = 256, mixed with other changes                                        |
| Firedrake (both) | `tsfc-kernel-warmup`: replays twice after recording so a lazily compiled kernel lands in the warm-up | #229: first timed VJP trial otherwise inflates the mean ~10x at small sizes                              |
| deal.II (both)   | C++ binary launched per request                                                                      | every call pays process start, mesh and assembly; #245 removed 1.3 s of MPI start-up                     |
| torch-fem (both) | none: rebuilds the `Solid` model every call                                                          | not measured                                                                                             |

The effects are large where they were measured, and they are applied unevenly. Each one is defensible on its own, but together they make the cost column compare engineering effort as much as solver speed.

## Proposed contract

1. **What may be cached across calls:** anything that depends only on inputs the harness marks as setup-defining for that problem (mesh, boundary conditions, material constants, target fields) and not on the differentiable inputs. The cache key must cover all of them by content, not by shape (the JAX-FEM bug fixed in #261 is the cautionary example).
2. **What must not be cached:** forward solutions reused by the VJP. Each endpoint recomputes the forward pass, as the thermal JAX-FEM comment already states, so VJP cost stays comparable across solvers.
3. **Warm-up:** the harness promises exactly one untimed call per (solver, sweep point) with the same inputs as the timed calls, and nothing else. A tesseract may arrange for compilation to happen in that call (Firedrake's TSFC warm-up is then allowed, not an intervention), but it may not depend on more than one warm-up.
4. **Process model:** a tesseract may launch a process per request, but then it pays for it in the cost column, as deal.II does today. Starting a persistent worker inside the container is allowed under rule 1.
5. **Uniformity:** every solver whose library supports reuse gets it. torch-fem is the open case: its `Solid` model is rebuilt per call. Either add a cache under this contract or record why not.
6. **Disclosure:** each solver's results page states which of 1, 3 and 4 it uses, without counting them as interventions.

## Decisions needed

- Is the setup-defining input list per problem the right boundary, or should it be "everything except the differentiable inputs"? The two differ for the thermal target temperature, which FEniCS keys its tape on.
- Should a setup cache be required (so the cost column measures steady-state solve cost), or only permitted? Required is fairer but asks for work on torch-fem and possibly deal.II.
- Should the harness report the first (cold) call's time separately? That would make the effect of caching visible to practitioners instead of hiding it.

## Not measured here

I didn't measure torch-fem's per-call setup cost or TopOpt.jl's cache benefit. Both would take one A/B run of `cost/vjp_cost`.
