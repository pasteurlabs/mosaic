# Recording interventions: the line between using and developing a solver

**Status:** design, revised after the annotation spike
([intervention-spike.md](intervention-spike.md))

## Problem

Almost any solver can pass a benchmark if someone works around its problems
hard enough: tuning undocumented settings, writing missing numerics,
monkeypatching internals, or getting direct help from its developers. Mosaic's
tesseracts already contain all of these, to very different degrees. A pass
that looks the same for a solver used as documented and for one whose
internals were rewritten tells the reader little, and it is unfair in both
directions. A heavily tuned entry looks better than its users will find it,
and a lightly tuned one looks worse than it could be.

There is no principled place to draw a single line between using a solver and
developing it. So instead of drawing that line, Mosaic records where each
entry sits relative to it, what was done, and **why it was necessary**.

## Goals

The results have two audiences:

- **Practitioners** choosing a solver want to know what works for their kind of
  problem, at what cost, and how much work it takes to get there.
- **Solver developers** want a fair comparison and actionable signal: what went
  wrong, where, and what would fix it.

Both are served by the same record. The practical cost of obtaining correct,
usable gradients from a solver is what Mosaic sets out to measure, so the
effort it took to make a solver pass is a result in its own right, not
something to hide.

The record answers three questions:

| Question                               | Record                                               | Main audience     |
| -------------------------------------- | ---------------------------------------------------- | ----------------- |
| Does it solve my problem as specified? | Setup deviations                                     | Practitioners     |
| What did it take?                      | Interventions                                        | Both              |
| Is the comparison fair?                | Tuning checklist, provenance, uniform harness policy | Solver developers |

Fairness here means the same rules for every solver, not equal effort.

## Principles already in place

- Results are reported per **setup** (one concrete problem statement, such as
  periodic Taylor–Green flow or channel flow past a cylinder), not per physical
  domain, and setups are not averaged.
- Each solver gets **checks** (pass, borderline, fail, not supported, not run)
  instead of scores or ranks.
- **Cost** is a multiple of the fastest solver on the same device class.
- **Artefacts of the method stay in the results.** They are explained (as
  explained anomalies), not engineered away. Fixing a genuine bug or
  misconfiguration in a tesseract is in scope; hiding an error that comes from
  how the solver works is not.

## Policies

1. **Choice of scheme.** A solver runs with its default scheme unless the
   library's own documentation or tutorials would tell a user to change it for
   this kind of problem. If they would, the change is allowed and recorded as
   an L0 intervention with evidence. If not, the default stays, and an error
   it causes is an explained anomaly.
2. **Runtime adaptations.** Adaptations that exist only because Tesseract
   calls are stateless or because of the timing protocol (setup caches,
   compile warm-ups, per-request process launches) follow one harness-wide
   contract that applies to every solver alike. They are not per-solver
   interventions.
3. **Setup deviations.** A solver that cannot express the prescribed setup
   exactly may run with a documented substitute (a different outflow condition,
   a different load discretization) if it still passes the setup's checks. The
   deviation is shown next to its results. Otherwise the setup is "not
   supported" for that solver.
4. **Tuning.** Tuning solver settings such as tolerances, preconditioners or
   precision is normal use, and every solver receives it. A common checklist
   (below) records how far each solver's tuning has gone, so uneven tuning is
   visible rather than hidden. Tuned documented settings are L0 interventions
   with the `tuned` flag.

## The intervention ladder

Every intervention in a tesseract, meaning anything beyond calling the solver
with its defaults, is assigned a level by its mechanism:

| Level  | Name                           | What it covers                                                                                                                                               |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **L0** | Documented use                 | Public API and documented options: choice of linear solver, tolerances, resolution, schemes and settings from the official docs or tutorials.                |
| **L1** | Undocumented configuration     | Settings found by reading the source, undocumented flags, values outside the documented range.                                                               |
| **L2** | User code on public primitives | Code written with the library's public API to express what the library does not provide: boundary conditions, objectives, missing physics, custom VJP rules. |
| **L3** | Changing the solver            | Monkeypatching, replacing or overriding internal components, private APIs, vendoring or patching solver source, including at build time.                     |
| **L4** | Unreleased upstream help       | Fixes that exist only on a maintainer's branch or in an unreleased version. Not accepted.                                                                    |

The level records _how_ something was done. Whether it was needed is a
separate field (`necessity`, below).

### Baseline depends on the abstraction level

Each solver declares its abstraction level, and the baseline (what counts as
ordinary use, not recorded) is defined relative to it:

| Abstraction level | Examples                                          | Baseline                                                                                                                                                                                                                                                                                |
| ----------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Application       | OpenFOAM                                          | Case files, dictionaries and function objects as in the tutorials.                                                                                                                                                                                                                      |
| Framework         | FEniCS, Firedrake, deal.II, PhiFlow, XLB, JAX-FEM | Expressing the prescribed PDE and setup through the library's API, at the level it is designed for: a UFL form for FEniCS, an assembly loop for deal.II. For libraries built on an AD framework, documented features of that framework (e.g. `jax.checkpoint`) count as documented use. |
| Kernel toolkit    | NVIDIA Warp                                       | The hand-written solver itself. Only interventions made for the benchmark inside it are recorded (numerics chosen to pass a check, workarounds, performance tuning).                                                                                                                    |

## What gets recorded

### Interventions

Each intervention is recorded with:

- **What** was done, in one or two sentences.
- **Level**, by mechanism.
- **Necessity:** `required` (the setup fails without it), `choice` (a library
  alternative exists and was not used), or `unknown`.
- **Tuned:** whether the value was found by measurement.
- **Why, per setup:** what a user who skips this would see on that setup.
  - one or more **consequences** (categories below);
  - the **symptom**, with a measured number where one exists;
  - the **evidence** (see "Evidence" below).
- **Upstream:** a link to the upstream issue, if the root cause is in the
  solver, or the recorded reason why there is none (for example, the upstream
  project is no longer maintained).
- **Provenance:** who wrote it (`mosaic-team`, `contributor`,
  `maintainer-authored`).
- **Active:** `false` for code that exists but no experiment exercises.
  Inactive interventions are removed in the next cleanup.

Consequence categories:

| Category             | Meaning without the intervention                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `cannot-express`     | The setup cannot be represented (missing boundary condition, geometry, physics or input).         |
| `crash`              | The solver raises an error or diverges.                                                           |
| `non-finite`         | The solver returns NaN or infinite values.                                                        |
| `inaccurate-forward` | The forward solution misses the setup's accuracy check.                                           |
| `no-gradient`        | Reverse-mode differentiation is unavailable or fails.                                             |
| `incorrect-gradient` | A gradient is returned but does not match the reported objective.                                 |
| `optimizer-stall`    | The optimization stalls, for example on a noisy loss.                                             |
| `too-slow`           | The experiment exceeds its time budget, or the solver is substantially slower (state the factor). |
| `out-of-memory`      | The experiment exceeds the memory available on the benchmark hardware.                            |
| `non-reproducible`   | Results vary from run to run.                                                                     |

An intervention is an exclusion that was resolved instead of recorded: without
it, the cell would carry an exclusion or a failure. The "why" should read like
that exclusion's reason, using the same vocabulary as `ExclusionCategory`.

### Setup deviations

Each deviation (policy 3) is recorded with what differs from the prescribed
setup, why (usually `cannot-express`), and its effect on the results where
known. Deviations introduced by harness code for one solver, such as a
solver-specific time step or load, are recorded on that solver.

### Not recorded

- Fixes to Mosaic's own code (for example removing redundant solves from a
  wrapper).
- Code that detects failures without changing what the solver computes.
- Runtime adaptations covered by the harness contract (policy 2).
- Build infrastructure with no possible effect on results (for example limiting
  compile parallelism).

### Evidence

The evidence required depends on the claim:

- `cannot-express` is often evident from the code and the library's
  documentation; a pointer to either is enough.
- Measurable consequences (`too-slow`, `inaccurate-forward`, `crash`, …) need a
  linked measurement: a benchmark run, a test, a log, or a before/after table
  in the PR. Where cheap, the intervention is switchable so the measurement can
  be repeated when the solver is updated.
- When the reason is not known, the record says `unknown`, the results show it
  as unknown, and an issue is opened to measure it or remove the intervention.

### Example record

```yaml
metadata:
  mosaic:
    abstraction: framework
    interventions:
      - id: dense-pressure-solver
        level: L3
        necessity: required
        what: >
          Replaces Oceananigans' FFTW-based pressure solve with an explicit
          separable cosine transform.
        why:
          buoyancy-closed-box:
            consequences: [no-gradient]
            symptom: >
              Enzyme has no rules for the FFTW calls in the native pressure
              solver, so reverse-mode differentiation through a time step fails.
            evidence: <link to failing run or test>
        upstream: <link to upstream issue>
        provenance: mosaic-team
    deviations: []
```

## Tuning checklist

Every solver goes through the same checklist on every setup it runs. The
results record which items have been done:

1. Resolution, time step and horizon as prescribed by the setup.
2. Solver tolerances chosen for the setup's accuracy needs.
3. Documented performance options tried: linear solver and preconditioner,
   precision, compilation or JIT options, device placement, checkpointing for
   long rollouts.
4. Runtime adaptations used through the harness contract (policy 2).

## Rules

1. **Released versions only.** Every entry runs on a released upstream version
   plus code in this repository. Where a project has no release process, a
   pinned commit is used and recorded as such.
2. **Upstream bugs are excluded by default, not worked around.** The default
   response to a solver bug is an `UPSTREAM_BUG` exclusion and an upstream
   issue. A workaround is allowed only if it is recorded and links to that
   issue, or records why no issue is possible. When an upstream release fixes
   the bug, the workaround is removed.
3. **No workarounds for artefacts of the method** (policy 1).
4. **Changes to the method create a separate entry.** If an intervention
   changes the discrete method rather than how it is used, the result is a
   different solver and gets its own entry, as `openfoam` and `openfoam-ad`
   already are.
5. **Every intervention above L0 states its necessity with evidence** at the
   level described above. An intervention shown to be unnecessary is removed.

## Code markers

Interventions at L2 and L3 carry marker comments around the implementing code,
so developers can find it:

```python
# mosaic-intervention: dense-pressure-solver
...
# mosaic-intervention-end: dense-pressure-solver
```

C and C++ use `//`. Markers go outside import blocks, because import sorting
splits a block at a comment. L0 and L1 interventions are usually a single
setting and need no markers. Line counts are not used as a metric.

## Procedure

- **The author annotates, someone else reviews.** Records are added or updated
  in the same pull request that adds or changes an intervention, and a second
  maintainer reviews the level, necessity, deviations and evidence. The pull
  request template asks for this on new-solver and solver-tuning pull requests.
- **A rubric settles recurring cases** (below). New ambiguous cases are added to
  it when they are decided.
- **A format check in CI** validates the records with the report script
  (`docs/design/intervention_report.py`).

### Rubric

Worked examples from the spike:

| Case                                                                   | Decision                                                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| deal.II's hand-written assembly loop                                   | Baseline: deal.II is designed for user-written assembly.                                       |
| Warp-NS's hand-written solver                                          | Baseline for a kernel toolkit. Benchmark-motivated numerics and tuning inside it are recorded. |
| XLB's checkpointing with `jax.checkpoint`                              | L0: a documented feature of the AD framework XLB is built on.                                  |
| PICT's advective outflow instead of p = 0                              | Setup deviation, not an intervention.                                                          |
| Setup caches (JAX-FEM, TopOpt.jl, FEniCS, Firedrake), TSFC warm-up     | Runtime adaptation under the harness contract.                                                 |
| Tolerance tuned by measurement (PhiFlow CG tolerance)                  | L0 with `tuned: true`.                                                                         |
| Removing redundant forward solves from a wrapper                       | Not recorded: a fix to Mosaic's code.                                                          |
| PICT's linear-solve failure detection                                  | Not recorded: detects failures without changing results.                                       |
| PICT's single-threaded CUDA extension build                            | Not recorded: build infrastructure.                                                            |
| Hand-written boundary conditions where the library ships them (XLB)    | L2 with `necessity: choice`, unless a reason is found.                                         |
| jax-cfd RK4, PhiFlow advection, Exponax dealiasing, XLB KBC stabiliser | Decided by policy 1 during the cleanup.                                                        |

## Presenting the record

The same data feeds two views:

- **Practitioner view (results overview).** Each setup's check table gets a
  deviations column (with the deviation on hover) and an effort column showing
  the highest level among _required_ interventions. A filter selects the
  highest level the reader is willing to do themselves. Cost always comes from
  the same configuration as the checks next to it.
- **Developer view (solver pages).** Per setup: interventions, deviations,
  upstream issues, tuning checklist status, and interventions whose reason is
  unknown. Each item is an invitation to contribute: an upstream fix or a better
  configuration moves the entry down the ladder.

## Mapping onto existing code

| Existing piece                                                           | Role                                                                                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `ExclusionCategory` (`UPSTREAM_BUG`, `NOT_IMPLEMENTED`, `INFEASIBLE`, …) | What a cell shows when an intervention was _not_ made. Shares its vocabulary with the consequence categories. |
| `ANOMALY_EXPLAINED`                                                      | Artefacts of the method (policy 1, rule 3).                                                                   |
| `SolverSpec` / `tesseract_config.yaml` metadata                          | Storage for interventions, deviations, abstraction level and tuning checklist.                                |
| `docs/scoreboard.py` check tables                                        | Gains the deviations and effort columns and the level filter.                                                 |

## Rollout

1. **Cleanup pass over all solver implementations**, using the spike's
   findings:
   - changes that do not affect results: dead and inactive code, no-op
     interventions, wrong or outdated comments, unused shared code, moving
     solver code that lives in shared directories into its tesseract;
   - bug fixes, one pull request per solver;
   - upstream issues for every workaround of an upstream bug;
   - applying policies 1–4: the method-artefact cases, the harness contract for
     runtime adaptations, solver-specific harness code, and a first tuning pass
     for solvers that have had little;
   - measuring or removing interventions whose reason is unknown.
2. **Pilot the record format and rubric** on four contrasting solvers: OpenFOAM
   (application), FEniCS (framework), XLB (framework on an AD framework, many
   interventions) and Warp-NS (kernel toolkit).
3. **Annotate all solvers**, starting from the spike's records.
4. **Show the record in the results pages** (practitioner and developer views).

A static lint for L3 patterns was prototyped in the spike and is deferred.

## Open questions

- **Default cut-off.** Is "up to L2" the right default for the practitioner
  view's level filter?
- **Shared harness numerics.** Harness code that defines the setup (for
  example the staircase cylinder mask) is part of the setup. Harness code that
  serves several solvers but changes what they compute (for example the Fourier
  lifting of corrections onto staggered faces in #121) is not yet classified.
