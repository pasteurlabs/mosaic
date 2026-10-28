# Policy 2: a harness contract for runtime adaptations

**Status:** decided 2026-10-27. Contract in `interventions.md` ("Harness contract for runtime adaptations"). Work item 1 is #285, item 2 is #292; 3 and 4 open.

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

## Reasoning behind the details

The question the cost column answers is what a call costs inside a loop that a practitioner would write: an optimizer, a sampler or a training loop. In that loop the setup is fixed and the differentiable inputs change on every call. Anyone using the library directly builds the setup once, so the steady-state cost is the fair headline, and the once-off cost is real too, so it is reported next to it rather than hidden.

- **The boundary is the schema's differentiability marking**, not a per-problem list. A tesseract cannot know which inputs a given experiment varies, so the only safe rule is that state derived from a differentiable input's _value_ never survives a call. State derived from its _shape_ (compiled kernels, sparsity, symbolic factorizations) may. The thermal target temperature is not differentiable, so FEniCS keying its tape on it is fine.
- **Warm starts across calls are out.** The timed trials repeat the warm-up's inputs, so an initial guess carried over from the previous call would start at the converged answer and make iterative solvers look faster than they are in a loop. A warm start within a call (PICT's channel pressure from the previous time step) is the solver's own business.
- **Required, but as an obligation on Mosaic.** A missing cache is a gap in Mosaic's wrapper, not a property of the solver, so it is disclosed in the cost entry and tracked under tuning checklist item 4 instead of failing anything.
- **Application-level solvers are not required to cache.** OpenFOAM is used as a process per case, so its per-request launch is how it is used.
- **One setup at a time.** An unbounded cache makes a container's memory grow with every sweep point, which eats the headroom of the largest size. It does not show in the recorded metrics: `ram_peak_mib` is the harness process's RSS and `vram_peak_mib` the whole-GPU change over one point (`core/hardware.py`), so neither sees a cache built at an earlier point.
- **First call is reported, not scored.** It mixes compilation, setup and first-touch effects that depend on the image and the host; it is useful to a practitioner deciding whether a few calls are worth it, but it is not a stable comparison.
- I considered perturbing the differentiable inputs between timed trials to make output caching impossible instead of forbidden. It would change iteration counts for iterative solvers and so change what is measured; review of the cache key is enough.

## Where each solver stands

| Solver                   | Contract status                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------- |
| FEniCS, Firedrake (both) | conforms: content-keyed, one entry (`_SETUP_CACHE.clear()` before insert)              |
| JAX-FEM (both)           | content-keyed (structural after #261); **unbounded** dict                              |
| TopOpt.jl                | content-keyed hash; **unbounded** `Dict`                                               |
| INS.jl                   | keyed on `(n, L, ndim)`; **unbounded** `Dict`                                          |
| Warp-NS, XLB             | compiled kernels / jitted functions cached per shape (`functools.cache`); allowed by 1 |
| torch-fem (both)         | **no cache**: `Solid` rebuilt per call                                                 |
| deal.II (both)           | **no cache**: binary per request                                                       |
| PICT                     | **no cache**: domain rebuilt per call                                                  |
| OpenFOAM                 | process per request; not required (application)                                        |
| harness                  | **first-call time not recorded** (warm-up return value and time discarded)             |

## Work items

1. Harness: record the warm-up call's wall time and report it as first-call cost (`core/harness.py`, `problems/shared/cost.py`, results page). **#285.**
2. Bound the JAX-FEM, TopOpt.jl and INS.jl caches to one entry. **#292** (four `clear()`/`empty!` calls). An earlier sandbox implementation kept every output bit-identical over the forward, gradient and cost suites. No recorded metric can show the effect (see "One setup at a time"), and none of these containers is near its memory limit. It was opened anyway because contract item 6 requires it, so the code says what the contract says.
3. Setup caches for torch-fem (both) and PICT; a persistent worker for deal.II (both).
4. Per-solver disclosure on the results pages, with the record format (rollout step 4).

Item 3 changes cost results only; 1 adds a metric.
