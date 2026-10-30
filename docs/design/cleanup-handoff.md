# Handoff: solver cleanup pass

**Status:** 2026-10-30, end of the sixth A100 session. Every item of the backlog has been measured and either has a pull request or a recorded reason for not having one ("Done in the sixth session" below). What remains is reviewing and merging the open pull requests, the decisions listed under "For Dion", and the upstream issues. Notes, issue texts and diagnostic scripts are in [cleanup-notes/](cleanup-notes/); the sixth session's measurements are in [session6-measurements.md](cleanup-notes/session6-measurements.md).

## Read first

1. [interventions.md](interventions.md): the agreed design for recording interventions, the four policies, the harness contract for runtime adaptations, and the rollout.
2. [intervention-spike.md](intervention-spike.md): findings from annotating all 19 tesseracts. The backlog came from it.

## Pull requests

Merged: #259 to #266, #270, #271, #278 to #283, #285 to #294, #297.

Open from this pass, CI green on every head commit unless noted:

| PR   | Change                                                                  | Label  | Notes                                                                                                                                         |
| ---- | ----------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| #269 | OpenFOAM tutorial pressure tolerance, serial channel                    | solver | Cd within 6e-9, 82 → 53 s                                                                                                                     |
| #284 | No harness-side time-step rescaling for XLB in ns-3d-grid               | domain | Statuses unchanged; N = 8 error 0.064 → 0.079                                                                                                 |
| #296 | PhiFlow periodic CG at phiml's default 1e-5; explained 2-D cost anomaly | solver | All four PhiFlow ns-grid cost cells explained on the head commit. The bot's "regressions" compare against a PhiFlow baseline from before #288 |
| #299 | TopOpt.jl 0.14.0 on Julia 1.12.5, one Julia thread                      | solver | One thread fixes a segfault on the larger meshes (juliacall + multithreaded GC); every cost entry completes                                   |
| #300 | TopOpt.jl setup cache keyed on its inputs                               | solver | 22 non-cost cells identical                                                                                                                   |
| #301 | torch-fem GPU CG at 1e-6 instead of 1e-10                               | solver | Statuses unchanged; recovery endpoint varies with any tolerance                                                                               |
| #302 | Intervention records in the harness, piloted on four solvers            | none   | Rollout step 1                                                                                                                                |
| #303 | Intervention records for every remaining solver                         | none   | Rollout step 2. Records describe `main`; update them as #299 and #305 to #308 merge                                                           |
| #304 | Solver effort and setup deviations on the results pages                 | none   | Rollout step 3                                                                                                                                |
| #305 | deal.II heat CG preconditioned with AMG                                 | solver | 61.1 → 25.6 s at nx = 1024, forward metrics identical                                                                                         |
| #306 | OpenFOAM periodic pressure with GAMG                                    | solver | 2-D N = 256 37.4 → 16.4 s; decomposition measured, not adopted                                                                                |
| #307 | INS.jl on its CUDA backend                                              | solver | 3-D VJP 15–35× faster locally. New ceiling on the T4: the 3-D horizon sweep stops at 5120 steps (28 GB of device memory)                      |
| #308 | Channel pressure CG: Warp-NS 1e-3, PhiFlow 1e-2                         | solver | Every channel check unchanged; PhiFlow drag_opt 10 782 → 1 271 s. Bot's PhiFlow rows explained in a PR comment                                |
| #309 | JAX-FEM's `numerics` label and JVP docstring describe its BiCGSTAB      | none   | Docs only; PETSc solvers measured, not adopted                                                                                                |
| #310 | Forcing in every PhiFlow RK4 stage; merges `main` into #267's branch    | solver | Based on `feat/ns-grid-forcing`, not `main`. Forcing tests pass for PhiFlow and PICT; XLB's forcing path untested                             |
| #311 | Warp-NS setup caches hold the current grid only                         | solver | Statuses and forward errors unchanged, cost within 6%                                                                                         |

Conflicts when merging: #299 and #300 both edit TopOpt.jl's cache in `solver.jl`; #308 and #296 edit adjacent lines of PhiFlow's `tesseract_api.py`; #269 removes the OpenFOAM channel decomposition, near #306's periodic change. #303's records must follow whatever merges first.

#298 (Dion, open, `benchmark:domain`) scores ns-grid's multimode agreement against a checked-in consensus. It removes the subset effect behind the multimode shifts on #291 and #308; against all seven solvers, PhiFlow's published 4.7e-4 becomes 3.7e-3.

#267, #268 and #272 to #275 belong to the Kolmogorov work, not to this pass.

**Reading CI results.** Each pull request opened with a label starts two benchmark runs; the first is cancelled. Read the bot comment of the completed run on the head commit. Metric changes on cells a pull request cannot touch are baseline drift; compare with a local A/B before acting on them. A crashed sweep entry can show as "ok" locally when `ab.sh` has merged the reference results: check per-entry `metrics.status` (`compare.py` prints the failed entries).

## For Dion

1. **Firedrake heat with MPI:** the solve halves by 8 ranks, but a served tesseract would need a persistent MPI worker pool with pyadjoint's tape across ranks (`session6-measurements.md`). Worth doing?
2. **OpenFOAM decomposition:** faster on an idle machine, collapses under load (ranks busy-wait). Not adopted because benchmark runs share the machine with GPU jobs. Revisit if CPU jobs get dedicated cores.
3. **PhiFlow channel tolerance:** #308 uses 1e-2, the loosest at which every check holds. 1e-3 is the more conservative choice with the same checks, at about twice the drag_opt time.
4. **INS.jl threads:** with `JULIA_NUM_THREADS=auto` and juliacall's signal handling on, the CPU entry is 1.5–4.4× faster. Recorded only, since #307 moves INS.jl to the GPU.
5. **INS.jl horizon ceiling on the T4** (#307): reverse mode stores ~5.5 MB per 3-D N = 20 step on the device. Checkpointing would lift it but is engineering beyond the library's documented use; as it stands the ceiling is a result.
6. **PICT's ns-grid cost anomaly** (32× and 26× the GPU median in CI, unexplained on `main`): 90% of an apply is in `PISOtorch.SolveLinear`, about 32 ms per solve nearly independent of grid size. A documented-cause exclusion could cite this; not written.
7. **Unused explanations:** the bot reports the ns-3d-grid `forward/agreement` face-averaging explanation as unused for INS.jl, jax-cfd and PhiFlow. Remove them, or keep them for runs where the cells trip again?
8. **phiml `'cluster'` preconditioner issue:** drafted in `upstream-phiml-cluster.md`, not approved.

## Done in the sixth session

The fifth session's "Next" list, in its order:

1. **#296's run:** all four PhiFlow ns-grid cost cells are explained anomalies on the head commit. #291's multimode shifts: on the A100, `forward/agreement/multimode` with the old flags and with XLA's default autotuning is bit-identical for jax-cfd and XLB and within 6e-4 relative for PhiFlow, and `main`'s forward, gradient and cost suites for the three solvers have no failed entries beyond the b6eb311 reference run's, except XLB's 3-D VJP NaN at 320 steps, which is BGK's limit (recorded in #282) and reproduces with the old flags. This agrees with Dion's finding that the CI shifts came from scoring against the consensus of the solvers present in each CI job, which #298 fixes. #292: INS.jl cost unchanged locally (every cell within 3%), so the CI shift was drift; TopOpt.jl's structural forward error of exactly 0 is expected, since its float32 compliance matches the checked-in reference built from the same load (#281).
2. **Tolerances:** torch-fem's CG (#301); the channel CG of Warp-NS and PhiFlow (#308). PICT's periodic tolerance was set in #287.
3. **CPU parallelism and solvers:** deal.II heat AMG (#305); OpenFOAM periodic GAMG (#306), decomposition not adopted; FEniCS heat stays serial (slower with every rank); Firedrake and INS.jl threads are options (For Dion 1 and 4).
4. **INS.jl on the GPU:** #307.
5. **Policy 2 work items** (the per-solver statement is `policy2-adaptations.md`):
   - the deal.II worker (#297);
   - Warp-NS one-entry caches (#311);
   - torch-fem setup cache: measured and dropped (no gain);
   - PICT channel geometry cache: measured and dropped. Outputs are bit-identical and a repeat build of the same mesh drops from 14/50/199 ms to 5/6/40 ms (res 10/20/40), but the benchmark builds each channel mesh once (`forward/cylinder` runs two resolutions; PICT is excluded from `drag_opt`), so the cache never hits. Branch `perf/pict-channel-geometry-cache` stays unpushed.
6. **JAX-FEM:** PETSc CG with hypre or GAMG fails JAX-FEM's convergence check or is slower; `main`'s BiCGSTAB stays. #309 corrects the solver description.
7. **PICT periodic:** BiCG changes outputs and is slower in 3-D (within CG's run-to-run spread in 2-D), preconditioned BiCG breaks down at N = 256, `pressure_reuse_result` changes nothing. No pull request.
8. **TopOpt.jl 0.14.0:** #299, which also needed one Julia thread.
9. **PICT's advective outflow:** measured in the fifth session (`policy3-setup-deviations.md`).
10. **Kolmogorov forcing in RK4:** #310, into #267's branch.
11. **Rollout:** #302 to #304.

## Upstream issues

Dion files them, because the sandbox token gets HTTP 403 outside pasteurlabs. Ready: `upstream-phiml.md` and `upstream-pict.md` (attach the matrices from the sandbox). Not approved: `upstream-phiflow-rk4.md`, the TopOpt.jl Julia 1.11 precompile issue, `upstream-xlb.md`, `upstream-phiml-cluster.md`.

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

## Decisions taken 2026-10-28

1. **Policy 1 stays as written** in `interventions.md`: a non-default scheme is allowed when an experienced user of this kind of simulator would pick it and a measurement shows the gain. Evidence is not tied to what docs, tests or papers say, since those drift.
2. **Tuning rules** (tuning checklist in `interventions.md`): tolerances are the loosest at which every check of the setup is unchanged, in either direction; framework defaults are only switched off for a reason that reproduces on the benchmark hardware; within a device class each solver may use the machine's cores and devices as an experienced user would, picked by measurement; moving to a library's own GPU backend is tuning of the same entry.
3. **Cost checks compare within a device class** (#293), as the results pages do. A serial CPU solver is no longer flagged against GPU peers, so OpenFOAM's explained cost anomaly in ns-grid goes away. This exposes PhiFlow's 2D cost: 21–42× the GPU median on the current baseline.
4. **XLA autotuning:** restored (#291). If a failure reproduces on the T4 or the A100, the flag returns for that solver with the reason recorded.
5. The answers to the questions in `cleanup-notes/policy4-tuning.md` are recorded there.

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
