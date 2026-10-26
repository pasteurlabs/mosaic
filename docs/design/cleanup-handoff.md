# Handoff: solver cleanup pass

**Status:** 2026-10-26, after a second session on the A100 sandbox. Backlog
items 1 to 3 are done or decided by measurement, item 4 is drafted, item 5 has
decision notes. The state of each item is recorded in place. Notes, issue
drafts and diagnostic scripts are in [cleanup-notes/](cleanup-notes/).

## Read first

1. [interventions.md](interventions.md): the agreed design for recording
   interventions, including the four policies and the rollout.
2. [intervention-spike.md](intervention-spike.md): findings from annotating all
   19 tesseracts. The backlog below comes from it.

## Where things stand

Open pull requests from this pass (all `benchmark:solver`):

| PR   | Item                                                                  | Verified                                                                  |
| ---- | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| #260 | 1: INS.jl and TopOpt.jl solver code into their tesseracts             | both images build, forward suites ok, `test_ins_jl_consistency.py` passes |
| #261 | 2: JAX-FEM structural setup cache key                                 | reproduced the stale-geometry bug; fixed                                  |
| #262 | 2: firedrake-heat misfit in float64                                   | source_fd_check 2.98e-4 → 5.18e-5; BFGS final loss 5737 → 194             |
| #263 | 3: OpenFOAM Courant sub-stepping removed                              | bit-identical                                                             |
| #264 | 3: Warp-NS no-op device scope, eigenvalue and fd_check comments       | fields bit-identical                                                      |
| #265 | 4: XLB `warp-lang==1.12.1` pin instead of the build-time source patch | 27/27 experiments bit-identical                                           |
| #266 | 3: deal.II heat CG tolerance 1e-12 → 1e-8                             | forward identical, −15% at nx = 1024                                      |
| #269 | 3: OpenFOAM tutorial pressure tolerance, serial run (stacked on #263) | Cd within 6e-9, 82 → 53 s                                                 |
| #270 | 3: TopOpt.jl gradient via Zygote (stacked on #260)                    | bit-identical gradients, same VJP time                                    |
| #271 | 3: deal.II structural default threading                               | forward identical, cost unchanged                                         |

**CI baseline drift.** PR #259 (dead-code removal) and #264 both show
ns-grid `gradient/fd_check` for Warp-NS at 3.42e-5 → 4.04e-5, and #260 shows
INS.jl's at +19%, although none of these PRs can affect that cell. Locally
main and #259 are indistinguishable (2.97–3.00e-5 over three runs each). The
ns-grid fd_check baseline probably predates #250, which changed that check;
not confirmed. #262's report also shows ±40% cost shifts on Firedrake
structural cells it does not touch. Read single-cell CI metric changes with
this in mind.

Waiting for Dion:

- removing the `shared_sources` mechanism (no tesseract uses it after #260);
- JAX-FEM's linear solver (`cleanup-notes/jaxfem-linear-solver.md`);
- the policy 1–4 decision notes and the upstream issue drafts in
  `cleanup-notes/`;
- item 6.

## Decisions and constraints

- **Benchmark solvers as they are.** Artefacts of the numerical method stay in
  the results and get an explained anomaly. Do not change a tesseract's glue
  just to shrink an error that comes from how the solver works. Fixing genuine
  bugs and misconfiguration is in scope.
- **Tuning is normal use.** Tuning tolerances, preconditioners and similar
  settings is fine, and every solver should receive some.
- **Deferred or dropped, do not reintroduce:** a static lint for L3 patterns
  (later), a maintainer-review step or badge (dropped).
- **Prefer adapting docs and CI to the code** over adding aliases or
  compatibility shims.
- **Repository conventions:**
  - The repository is public: no AI attribution in commits or pull requests.
  - Pull request titles follow conventional commits, e.g.
    `fix(solver): …`, `refactor(solver): …`.
  - Use `.github/PULL_REQUEST_TEMPLATE.md`. Descriptions are written for a
    reviewer who knows the codebase, under about 300 words, without hard-wrapped
    lines.
  - Every pull request touching `mosaic/` needs a benchmark label:
    `benchmark:none`, `benchmark:solver`, `benchmark:domain` or
    `benchmark:all`.
  - Branch from `main`. One pull request per solver for changes that affect
    results.
- **Ask Dion first** before filing upstream issues, before any change decided by
  policy 1 (choice of scheme), and on the open decision in item 6.

## Where things are

- **Branch `spike/intervention-annotations`** (this branch): the design docs and
  the spike annotations. **Not for merge.** The annotations work as an index:
  find any intervention's code with
  `grep -rn "mosaic-intervention: <id>" mosaic`, and its recorded reason,
  evidence and history in the `interventions:` list of the tesseract's
  `tesseract_config.yaml`.
- **Report script:** `python docs/design/intervention_report.py` prints a
  summary of all annotations.
- **Pull request #259** (`cleanup/solver-dead-code`): removes the unused Warp-NS
  adjoint gradient clipping and `mosaic_shared/.../grid_fluid.py`. Labelled
  `benchmark:solver`, so CI checks that Warp-NS results are unchanged.

Cleanup pull requests are based on `main`, not on this branch.

## Backlog

Intervention ids refer to the spike annotations.

### 1. Move Julia solver code into its tesseracts

**Done: #260.** Removing `shared_sources` is waiting for Dion.

No result change.

INS.jl's solver lives in `mosaic/mosaic_shared/problems/navier_stokes_grid/ns_solver.jl`,
and TopOpt.jl's in `mosaic/mosaic_shared/problems/structural_mesh/topopt_solver.jl`.
Each is used by one tesseract only. Move them into the tesseract directories and
ship them with `build_config.package_data`, as the WaterLily tesseract in #211
does.

References to update:

- `ins-jl/tesseract_api.py` and `topopt-jl/tesseract_api.py` (include path);
- the `shared_sources` entries in both `tesseract_config.yaml` files;
- `tests/test_ins_jl_consistency.py` (mounts the file into an existing image);
- `tests/test_detect_changed_solvers.py` (uses both paths as examples).

After the move no `shared_sources` entries remain. Ask Dion whether to remove
the mechanism from the changed-solver detection script and its tests.

Verify by building both images with `tesseract build` and running
`mosaic run -p ns-grid --suites forward -s INS.jl --debug` and the structural
equivalent for TopOpt.jl.

### 2. Bug fixes

**Done: #261 (JAX-FEM, latent, no result change expected), #262 (firedrake-heat,
results change).**

These may change results; use `benchmark:solver`.

- **JAX-FEM structural setup cache:** keyed on `pts.shape` instead of the
  coordinates (`structural-mesh/jax-fem/tesseract_api.py`, around line 178). No
  current experiment varies geometry at a fixed mesh size, so this is latent.
- **firedrake-heat:** computes the identification error in float32 (around lines
  475–505) but builds the gradient seed in float64 (`nodal-misfit-seed`). FEniCS
  uses float64 for both.

### 3. Check, then fix or remove

**Done.** Outcomes and numbers in `cleanup-notes/item3-measurements.md`:
TopOpt.jl #270; Warp-NS #264; OpenFOAM #263 and #269; deal.II #266 and #271;
XLB channel, FEniCS `mumps-adjoint-monkeypatch` and deal.II
`umfpack-direct-solve` measured as needed (keep, record the evidence); JAX-FEM
`umfpack-solver-option` is a size-dependent trade-off waiting for Dion.

GPU needed for the Warp-NS and XLB items.

- **TopOpt.jl `compliance-grad-field`:** reads an undocumented struct field.
  TopOpt.jl's own Zygote rule computes the same thing, so switch to the
  documented path and confirm the gradients are identical.
- **Warp-NS `channel-cg-scoped-device`:** probably a no-op, because the chosen
  device always equals Warp's default. Confirm on a GPU and remove.
- **OpenFOAM `courant-substepping`:** by the spike's arithmetic, Co ≈ 0.87 at
  both cylinder resolutions, so it never sub-steps. Confirm. If it would ever
  trigger, it changes the prescribed time step, which makes it a setup
  deviation.
- **Warp-NS `poisson-fd-eigenvalues-3d`:** the comment claims an exact inverse,
  but central-difference divergence and gradient compose to the wide-stencil
  Laplacian. A May 2026 comment in `navier_stokes_3d_grid/config.py` still
  reports a 1.7e-2 gradient bias. Check numerically and fix the comment.
- **XLB channel staying on BGK:** the docstring says KBC makes the drag
  non-smooth, but #194 measured KBC drag gradients matching finite differences
  (3e-5 to 1.3e-3) on an older channel setup. Measure on the current channel.
- **Measure or remove:** interventions with no recorded reason.
  - `mumps-adjoint-monkeypatch` in fenics-structural. Out-of-memory evidence
    exists only for the heat slab at nx = 1024 (#180).
  - OpenFOAM's channel settings from #239: `channel-momentum-tolerance`,
    `channel-pressure-gamg`, `channel-non-orthogonal-corrector`,
    `graded-channel-mesh`, `mpi-decomposition`.
  - deal.II `umfpack-direct-solve` and `cg-ssor-tight-tolerance`; JAX-FEM
    `umfpack-solver-option`.
- **deal.II threading:** the structural binary runs serially without MPI
  (`serial-without-mpi`, #245), while the heat binary never caps threads. Make
  them consistent.

### 4. Upstream issues

**Drafted, not filed:** `cleanup-notes/upstream-{topopt-jl,phiml,pict,xlb}.md`.
TopOpt.jl, PhiML and PICT are reproduced. TopOpt.jl 0.14.0 and PICT `master`
are newer than our pins and untested. XLB's fix is on its `main` but
unreleased; #265 pins Warp instead, so that issue is only a release request.

Draft them, but file only after Dion approves.

- **TopOpt.jl:** INP parser bug worked around by `inp-parser-hex-patch`; still
  present on TopOpt.jl master.
- **PhiFlow / PhiML:** initial-guess tracer leak under `lax.scan` with a VJP,
  worked around by `channel-pressure-warm-start`.
- **PICT:** linear solvers break down in the first step with
  `non_orthogonal=True` on the O-grid (`orthogonal-scheme-on-o-grid`, see #239).
- **XLB packaging** (`xlb-ibm-stepper-import-patch`): first try pinning
  `warp-lang<1.13` (Warp 1.13 removed the `warp.utils.ScopedTimer` fallback),
  which may remove the build-time patch entirely.

### 5. Apply the policies

**Notes written, no code changed:** `cleanup-notes/policy1-{xlb,jaxcfd,phiflow,exponax}.md`,
`policy2-runtime-adaptations.md`, `policy3-setup-deviations.md`,
`policy4-tuning.md`.

- **Policy 1, choice of scheme.** No GPU needed for the research. For each case,
  find what the library's own documentation and tutorials recommend for this
  kind of problem, and write a short decision note for Dion. Do not change code
  before he decides.
  - XLB: the automatic switch from BGK to KBC above ω = 1.8
    (`kbc-collision-low-viscosity`) and the smooth KBC stabiliser
    (`smooth-kbc-stabiliser`, a method change, #194).
  - jax-cfd: central advection with RK4 instead of the default
    (`central-advection-rk4`, #225).
  - PhiFlow: differential advection (`differential-advection`) and explicit
    Euler, although PhiFlow ships `incompressible_rk4`.
  - Exponax: dealiasing turned off (`no-dealiasing`). This also affects every
    other solver, because an Exponax run at N = 64 is the reference for the
    ns-3d-grid forward checks (`mosaic/benchmarks/core/reference.py`).
- **Policy 2, runtime adaptations.** Draft the harness contract for setup
  caches, compile warm-ups and per-request process launches. Current cases:
  setup caches in JAX-FEM (both), TopOpt.jl, FEniCS and Firedrake
  (`setup-tape-replay`, `setup-cache`); Firedrake's `tsfc-kernel-warmup`;
  deal.II launched as a subprocess per request. torch-fem has no setup cache.
- **Policy 3, setup deviations.** To be recorded: PICT's advective outflow
  instead of p = 0 (`convective-outflow`); TopOpt.jl's load, split evenly over
  face nodes in harness code (`mosaic/benchmarks/problems/structural_mesh/physics.py`,
  around lines 45–72); XLB's harness-side time-step scaling (`lbm_N_base` in
  `navier_stokes_3d_grid/config.py`), which only XLB gets.
- **Policy 4, tuning.** Go through the tuning checklist in `interventions.md`
  for each solver and list the solvers that have had little tuning.

### 6. Open decision for Dion

jax-cfd's and Exponax's tesseracts both implement 2D and 3D. Yet jax-cfd is
removed from the 3D suite (`navier_stokes_3d_grid/config.py`, around line 77)
and Exponax from the 2D suite (`navier_stokes_grid/config.py`, around line 91),
with comments calling them "2D-only" and "3D-only". They are removed from the
solver registry, so they do not even show as "not run". Including them would
change results.

### 7. After the cleanup

Follow the rollout in `interventions.md`: pilot the revised record format on
OpenFOAM, FEniCS, XLB and Warp-NS, then annotate all solvers starting from the
spike's records, then show the record in the results pages (#203).

## Useful commands

```bash
uv sync
mosaic run -p <problem> --suites <suite> -s <solver> --debug
mosaic status -p <problem> -f
mosaic status --format json          # before/after snapshots for tuning PRs
tesseract build mosaic/tesseracts/<domain>/<solver>
python docs/design/intervention_report.py
```
