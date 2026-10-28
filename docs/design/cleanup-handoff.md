# Handoff: solver cleanup pass

**Status:** 2026-10-28, end of the fourth session on the A100 sandbox. Backlog items 1 to 6 are done, opened as pull requests or decided; what remains is listed under "Open". Notes, issue texts and diagnostic scripts are in [cleanup-notes/](cleanup-notes/).

## Read first

1. [interventions.md](interventions.md): the agreed design for recording interventions, the four policies, the harness contract for runtime adaptations, and the rollout.
2. [intervention-spike.md](intervention-spike.md): findings from annotating all 19 tesseracts. The backlog came from it.

## Pull requests

Merged: #259 to #265 (dead code, Julia solvers into their tesseracts, JAX-FEM cache key, firedrake-heat float64, OpenFOAM sub-stepping, Warp-NS comments, XLB warp pin), #266 (deal.II heat CG tolerance), #271 (deal.II structural threading), #278 (drop `shared_sources`) and #279 (pin `phiml`).

Open, all based on `main`:

| PR   | Change                                                                 | Label  | Verified on the A100 (before → after)                                                                                           |
| ---- | ---------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| #269 | OpenFOAM tutorial pressure tolerance, serial channel                   | solver | Cd within 6e-9, 82 → 53 s                                                                                                       |
| #270 | TopOpt.jl gradient from its Zygote rule                                | solver | bit-identical gradients, same VJP time                                                                                          |
| #280 | Exponax default 2/3-rule dealiasing, ns-3d-grid references rebuilt     | solver | forward errors grow (still lowest), adjoint no longer blows up past 160 steps, recovery 1.6e-2 → 1.0e-4                         |
| #281 | TopOpt.jl gets the consistent nodal loads of the traction              | domain | agreement error 5.5e-3 → 0, compliance 3.99731e-3 → 3.97532e-3, gradients bit-identical                                         |
| #282 | XLB: BGK collision everywhere, smoothed KBC removed                    | solver | every status `ok`; analytic_check 3.1e-2 → 1.9e-4; 3-D horizon 1280 → 160 steps (method limit)                                  |
| #283 | jax-cfd on the ns-3d-grid experiments                                  | domain | all suites run; excluded from forward agreement (staggered round trip)                                                          |
| #284 | no harness-side time-step rescaling for XLB in ns-3d-grid              | domain | statuses unchanged; N = 8 error 0.064 → 0.079; cost untimed                                                                     |
| #285 | record the cost warm-up call as `first_call_s`                         | none   | FEniCS structural: 18.4 s first call vs 0.031 s mean at N = 4; 3 new tests                                                      |
| #286 | draft: JAX-FEM BiCGSTAB compiled once per mesh                         | solver | removes the per-call recompilation; timings on a shared GPU, needs a quiet-GPU rerun                                            |
| #287 | PICT periodic linear-solve tolerance 1e-5 → 1e-6                       | solver | fd_check 8.8e-4 → 5.7e-5, forward unchanged; ns-3d-grid and cost not run locally                                                |
| #288 | PhiFlow: RK4 central differences (periodic), semi-Lagrangian (channel) | solver | periodic cost ×2.4–4.4 (quiet GPU); cylinder Cd error 0.054 → 0.011; stable at low ν in 2-D; `cleanup-notes/policy1-phiflow.md` |

#267, #268 and #272 to #275 belong to the Kolmogorov work, not to this pass. #268 also covers backlog item 6 for Exponax in 2D; I have not verified it.

Not opened: `refactor/one-entry-setup-caches` (policy 2 work item 2, see `cleanup-notes/policy2-runtime-adaptations.md`).

**Reading CI results.** Single-cell CI metric changes on cells a PR cannot touch are baseline drift: #259, #262 and #264 showed 18–40% shifts that local A/B runs disproved. Local timings from this session are unreliable: the GPU was shared with another session's jobs and, at times, up to four of mine, which slowed some experiments 17–28× against the reference run. The exception is #288's cost table, timed with one job on the GPU.

## Decisions taken 2026-10-27

1. `shared_sources`: drop it (#278).
2. Item 6: include jax-cfd in 3-D (#283); Exponax in 2-D via #268.
3. JAX-FEM linear solver: find the default's fixed overhead first. It was recompilation; #286.
4. Exponax dealiasing: measured, back to the default (#280).
5. jax-cfd's central advection with RK4: keep it and mark its tier. This amends policy 1: an optimization a practitioner would find natural is allowed when its tier is recorded.
6. PhiFlow: decided from first principles and measurement (#288, `policy1-phiflow.md`).
7. XLB: one entry, no duplicates (#282).
8. Policy 2: the contract in `interventions.md`; #285 is work item 1.
9. Policy 3: preference order in `policy3-setup-deviations.md`; #281, #284.
10. TopOpt.jl: re-tested; 0.14.0 fixes the parser bug, nothing to file.
11. PhiML: file the issue (not yet filed, see below) and pin (#279).
12. PICT: test `master`, then file. Master still fails (not yet filed).

## Open

- **Upstream issues Dion has to file himself:** the session's token is limited to pasteurlabs and gets HTTP 403 elsewhere. Ready: `cleanup-notes/upstream-phiml.md`, `cleanup-notes/upstream-pict.md` (attach the matrices from the sandbox). Not approved yet: `upstream-phiflow-rk4.md` (`incompressible_rk4` is third-order), the TopOpt.jl Julia 1.11 precompile issue (noted in `upstream-topopt-jl.md`), and `upstream-xlb.md` (optional, a release request).
- **TopOpt.jl 0.14.0 bump** would remove `inp-parser-hex-patch`; it needs Julia 1.12 and three renames (`upstream-topopt-jl.md`).
- **Quiet-GPU timing** for #284 and #286; the `benchmark:*` CI runs may answer it.
- **Kolmogorov flow and #288:** the Kolmogorov plan adds the forcing to PhiFlow's periodic step. After #288 it belongs in `momentum`, so every RK4 stage sees it.
- **Policy 2 work items 3 and 4:** setup caches for torch-fem and PICT, a persistent worker for deal.II, per-solver disclosure on the results pages.
- **Policy 4:** the tuning list in `cleanup-notes/policy4-tuning.md` is a draft.
- **PICT's advective outflow** is recorded as a deviation without a measured effect (`policy3-setup-deviations.md`).
- **After the cleanup:** the rollout in `interventions.md` (pilot the record format, annotate all solvers, show the records on the results pages, #203).

## Constraints

- **Benchmark solvers as they are.** Artefacts of the numerical method stay in the results and get an explained anomaly. Fixing genuine bugs and misconfiguration is in scope, and so is tuning.
- **Repository conventions:** public repository, no AI attribution in commits or pull requests; conventional-commit titles with the scopes `harness`, `solver`, `domain` or `deps` (CI rejects others); the pull request template; descriptions under about 300 words; a benchmark label on every pull request touching `mosaic/`. Use `benchmark:solver` or `benchmark:domain`, never `benchmark:all`.
- **Ask Dion first** before filing upstream issues and before any change decided by policy 1.

## Where things are

- **Branch `spike/intervention-annotations`** (this branch): design docs, notes and the spike annotations. **Not for merge.** Find an intervention's code with `grep -rn "mosaic-intervention: <id>" mosaic`; `python docs/design/intervention_report.py` summarizes them.
- **A/B runs** on the sandbox: `~/localfiles/runs/cleanup2/ab.sh` runs one solver into a copy of the b6eb311 full-run results so peers are present; outputs in `/nvme/scratch/ab/<arm>`. It relies on an uncommitted `MOSAIC_DIAG_TAGS` patch to `core/config.py` (`~/localfiles/runs/cleanup2/diag-tags.patch`; applied in the `mosaic-gpu2` worktree) that selects image tags per solver; never commit it.

## Useful commands

```bash
uv sync
mosaic run -p <problem> --suites <suite> -s <solver> --debug
mosaic run -p <problem> -s <solver> -e <suite>/<experiment>
mosaic status -p <problem> -f
mosaic status --format json          # before/after snapshots for tuning PRs
tesseract build -t <tag> mosaic/tesseracts/<domain>/<solver>
python docs/design/intervention_report.py
```
