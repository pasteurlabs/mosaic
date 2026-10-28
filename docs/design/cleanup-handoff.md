# Handoff: solver cleanup pass

**Status:** 2026-10-28, after review. The fourth A100 session finished backlog items 1 to 6; the tuning questions are settled (below). What remains is listed under "Next", in order. Notes, issue texts and diagnostic scripts are in [cleanup-notes/](cleanup-notes/).

## Read first

1. [interventions.md](interventions.md): the agreed design for recording interventions, the four policies, the harness contract for runtime adaptations, and the rollout.
2. [intervention-spike.md](intervention-spike.md): findings from annotating all 19 tesseracts. The backlog came from it.

## Pull requests

Merged: #259 to #266, #271, #278 to #283.

Open, all based on `main`:

| PR   | Change                                                                         | Label  | Status                                                                                                                                  |
| ---- | ------------------------------------------------------------------------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| #269 | OpenFOAM tutorial pressure tolerance, serial channel                           | solver | Ready. Cd within 6e-9, 82 → 53 s                                                                                                        |
| #270 | TopOpt.jl gradient from its Zygote rule                                        | solver | Ready. Bit-identical gradients, same VJP time                                                                                           |
| #284 | No harness-side time-step rescaling for XLB in ns-3d-grid                      | domain | Ready. Statuses unchanged; N = 8 error 0.064 → 0.079                                                                                    |
| #285 | Record the cost warm-up call as `first_call_s`                                 | none   | Ready. 3 new tests                                                                                                                      |
| #286 | Draft: JAX-FEM BiCGSTAB compiled once per mesh                                 | solver | Needs a quiet-GPU timing run                                                                                                            |
| #287 | PICT periodic linear-solve tolerance 1e-5 → 1e-6                               | solver | Ready. fd_check 8.8e-4 → 5.7e-5, forward unchanged                                                                                      |
| #288 | PhiFlow: RK4 central differences (periodic), semi-Lagrangian (channel)         | solver | Benchmark run in progress. Merge before the Kolmogorov PhiFlow forcing (see Next)                                                       |
| #289 | PICT CUDA kernels for 7.5 (T4), 8.0 (A100) and 9.0+PTX                         | solver | The final list is unbuilt locally; the benchmark run builds it on the T4                                                                |
| #290 | CI: a run superseded by a newer run for the same commit passes the gate        | —      | Removes the red duplicate check on pull requests opened with a label                                                                    |
| #291 | XLA's default GEMM autotuning in jax-cfd, PhiFlow and XLB                      | solver | The T4 benchmark run is the test. Not checked on an A100                                                                                |
| #292 | One-entry setup caches in JAX-FEM, TopOpt.jl and INS.jl (policy 2 work item 2) | solver | Fresh implementation; the benchmark run checks that outputs are unchanged                                                               |
| #293 | Cost checks compare within a device class                                      | none   | On the baseline: OpenFOAM's two explained ns-grid cost anomalies become ok; PhiFlow gets three new ones (2D cost 21–42× the GPU median) |

#267, #268 and #272 to #275 belong to the Kolmogorov work, not to this pass. #268 also covers backlog item 6 for Exponax in 2D; not verified.

The sandbox's unpushed branch `refactor/one-entry-setup-caches` is superseded by #292, which uses the same branch name on GitHub; delete the local branch instead of pushing it.

**Reading CI results.** Each pull request opened with a label starts two benchmark runs; the first is cancelled and, until #290 merges, shows as a failed "Benchmark checks passed". Look at the run on the head commit that completed. Single-cell metric changes on cells a pull request cannot touch are baseline drift: #259, #262 and #264 showed 18–40% shifts that local A/B runs disproved. Local timings from the fourth session are unreliable because the GPU was shared, except #288's cost table.

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

## Next

In order. Items marked GPU need the A100.

1. **Watch the open benchmark runs:** #288, #292 (outputs unchanged), #289 (PICT on the T4 with the new arch list), #291 (any failure on the T4; PhiFlow's earlier out-of-memory error was on a 16 GiB V100, and the T4 also has 16 GiB). GPU: run #291's three solvers on the A100 too.
2. **Tolerances** (rule in the tuning checklist): torch-fem's CG at 1e-10 in both tesseracts, then the unmeasured iterative tolerances in `policy4-tuning.md` (Warp-NS channel CG 1e-5, PhiFlow channel 1e-5). GPU.
3. **CPU parallelism and solvers** (hardware rule): OpenFOAM periodic, GAMG against PCG/DIC and decomposition against serial at the benchmark sizes; INS.jl with `JULIA_NUM_THREADS` set to the runner's cores; FEniCS, Firedrake and deal.II heat with their documented parallelism. Keep what is faster, record what is used. deal.II heat: also compare SSOR with AMG or a direct solve.
4. **INS.jl on its GPU backend** (GPU). If Zygote differentiates through it, set `uses_gpu: true`; otherwise record why it stays on the CPU.
5. **Policy 2 work items 3 and 4** (item 2 is #292): setup caches for torch-fem and PICT, a persistent worker for deal.II, and a per-solver statement of which runtime adaptations it uses.
6. **JAX-FEM:** quiet-GPU timing for #286, then the rest of its tuning list (GPU).
7. **PhiFlow 2D cost** (GPU): once #293 merges it is flagged at 21–42× the GPU median, and #288 adds another 2.6–3×. Find out why before explaining it: CPU against GPU for 2D (#233 found the GPU about 3× slower), the unpreconditioned CG, and per-step host syncs. **PICT periodic:** BiCG with a preconditioner (constructor options, L1 if undocumented).
8. **TopOpt.jl 0.14.0 bump**, which removes `inp-parser-hex-patch`; needs Julia 1.12 and three renames (`upstream-topopt-jl.md`).
9. **PICT's advective outflow:** measure its effect with one OpenFOAM A/B run (p = 0 against advective outflow).
10. **Kolmogorov flow:** after #288, PhiFlow's forcing belongs in `momentum`, so every RK4 stage sees it.
11. **After the cleanup:** the rollout in `interventions.md` (pilot the record format on OpenFOAM, FEniCS, XLB and Warp-NS, annotate all solvers, show the records on the results pages, #203).

Dion files the upstream issues himself, because the sandbox token gets HTTP 403 outside pasteurlabs: `upstream-phiml.md` and `upstream-pict.md` are ready (attach the matrices from the sandbox). Not approved: `upstream-phiflow-rk4.md`, the TopOpt.jl Julia 1.11 precompile issue, `upstream-xlb.md`.

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
