## Summary

- **Problems as a single declaration.** Replaces scattered module-level dicts (`MAKE_IC`, `EXPERIMENTS`, `EXCLUSIONS`, …) with one `Problem()` built via `add_experiment(...)` / `add_ic(...)` / `exclude(...)` — config, plot, checks, and exclusions live next to each other.

- **Per-suite loops in the framework.** Forward/gradient/cost runners were ~500 lines of near-duplicated loops per problem. Now a single generic driver + `@kernel` decorator; a problem supplies a one-step kernel and an aggregate.

- **Framework / problem separation.** Shared infra under `benchmarks/core/`; problem-specific code (optimisers, ICs, schemas) under each problem package or `tesseracts/`.

Supporting: `mosaic new-domain` scaffolds the new shape; tests rebuilt around kernel/harness units + a dummy-tesseract integration.

## CI & docs

- **PR docs previews.** `docs.yml` triggers on `pull_request`, builds the site from the PR head, and uploads `_site/` as a `docs-preview-pr-<N>` artifact (14-day retention). Pages deploy stays push-only.

- **PR-scoped preview data.** Docs build polls the GitHub API for the matching `benchmark.yml` run on the PR head SHA and overlays its `benchmark-results-<sha>` artifact into `mosaic-results/` before regenerating per-problem pages. Falls back to the `benchmark-results` branch when no PR benchmark ran.

- **Benchmark waits for tesseract builds.** `benchmark.yml` polls for the same-SHA `build-tesseracts` run before pulling images (curl + jq, no `gh` needed). Kills the race where the pull saw stale or missing `:latest`.

## Robustness fixes

- **Solver-start failures classify as `failed`.** Tesseract container import failures used to leave `by_solver[name]` unset → `not_run`. New `on_error` callback through `run_with_gpu_pool` / `per_solver_loop` records into `_solver_failures` on the result JSON; `_classify_result` promotes those `NOT_RUN` cells to `FAILED` with the exception message.

- **PICT `PISOtorch_simulation` import.** Top-level modules at the PICT repo root weren't on `sys.path`. Added a build step that writes a `.pth` file pointing at `/opt/pict`.

- **Slash-in-name experiments addressable via `-e`.** New `_resolve_experiment_target` matches literal multi-segment names (e.g. `physical_laws/vs_N`) against the per-suite registry before falling back to the `suite/exp/ic` interpretation.

## Type of change

- [ ] New solver backend
- [x] Solver tuning / improvement
- [ ] New benchmark domain
- [x] Harness / infrastructure change
- [x] Documentation
- [x] Bug fix

## Checklist

- [ ] `ruff check --fix && ruff format` passes
- [x] `pytest` passes (unit tests, no Docker required)

### For new solvers

- [ ] `tesseract_config.yaml` has a `metadata.mosaic:` block with at least `name` and `backend`
- [ ] `tesseract build mosaic/tesseracts/<domain>/<solver>` succeeds
- [ ] `mosaic run -p <domain> --suites forward -s <solver> --debug` completes
- [ ] `mosaic status -p <domain> -f` output pasted below
- [ ] Exclusions and explained anomalies documented (if any)

### For solver tuning

- [ ] Before/after `mosaic status --format json` snapshots compared
- [ ] No regressions to other solvers

## Status output

```

```

## Notes

`_solver_failures` is a new top-level key in `result.json`; older results without it classify as before.
