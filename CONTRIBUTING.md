# Contributing to Mosaic

Mosaic grows as the community adds solver backends, improves configurations, and extends to new domains. This guide covers the contribution paths and the local tooling you need.

## Adding a solver backend

A solver backend is a Tesseract wrapper that implements the forward map and (optionally) its VJP for an existing benchmark domain. The full walkthrough — directory layout, schemas, `apply` / `abstract_eval` / `vector_jacobian_product`, the `metadata.mosaic:` config block, and domain-specific overrides (exclusions, input_overrides, explained_anomalies) — is in the [Add a Solver tutorial](https://pasteurlabs.github.io/mosaic/tutorial-add-solver.html) (source: [`docs/tutorial-add-solver.qmd`](https://github.com/pasteurlabs/mosaic/blob/main/docs/tutorial-add-solver.qmd)). It uses a short pseudo-spectral solver on the ns-grid domain as a running example.

## Tuning an existing solver

Results reflect out-of-the-box configurations at the time of each tagged release. If you believe a solver can perform better with different settings, submit a PR with the improved configuration.

### What to change

Solver behavior can be adjusted at three levels:

- **Solver-internal parameters** in `tesseract_api.py` — sub-stepping counts, relaxation factors, convergence tolerances, time integration order, etc.
- **Input overrides** in the problem config (`mosaic/benchmarks/problems/<domain>.py`) — solver-specific values for shared schema fields (e.g. forcing a particular `dt` or `density`).
- **Schema defaults** in `tesseract_config.yaml` or the solver's `InputSchema` subclass — changing the default value of an exposed parameter.

### How to test

1. Snapshot the current state:

   ```bash
   mosaic status --format json > before.json
   ```

2. Make your changes and run the affected suites:

   ```bash
   # Quick iteration (small problem size)
   mosaic run -p <problem> --suites gradient,optimization -s <solver> --debug

   # Full run (production sizes)
   mosaic run -p <problem> --suites gradient,optimization -s <solver>
   ```

3. Compare against the baseline:
   ```bash
   mosaic status --format json > after.json
   mosaic status --format md --diff-against before.json
   ```

### Review expectations

- The change must not regress any other solver's results.
- CI runs the full evaluation suite when a maintainer adds the `benchmark` label to your PR.
- Include the diff output in your PR description so reviewers can see the impact at a glance.

## Adding a benchmark domain

A new domain bundles canonical schemas (`mosaic/mosaic_shared/problems/<domain>/schemas.py`), a `ProblemConfig` (`mosaic/benchmarks/problems/<domain>.py`), and at least one reference solver. The fastest way to start is `mosaic new-domain <name> --from-template <template>`, which scaffolds the file tree from a built-in template. The end-to-end walkthrough — schemas, suite defaults, IC generators, error functions, and a working reference solver — is in the [Add a Domain tutorial](https://pasteurlabs.github.io/mosaic/tutorial-add-domain.html) (source: [`docs/tutorial-add-domain.qmd`](https://github.com/pasteurlabs/mosaic/blob/main/docs/tutorial-add-domain.qmd)).

## Building the docs

The Quarto site at <https://pasteurlabs.github.io/mosaic> is generated from `docs/*.qmd` plus two auto-generated files: `docs/solvers.qmd` (gitignored) and `docs/results_*.qmd`.

**Install the Quarto CLI** — this is a standalone binary, _not_ a pip package. Grab it from <https://quarto.org/docs/get-started/> (the page autodetects your OS) or:

```bash
# Linux x86_64 example — check quarto.org for the latest version
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-amd64.deb
sudo dpkg -i quarto-1.5.57-linux-amd64.deb
quarto --version   # should print 1.5.x
```

Then, mirroring `.github/workflows/docs.yml`:

```bash
uv sync --extra dev

# Generate the solver reference page from tesseract_config.yaml metadata
uv run python docs/generate.py

# (Optional) Regenerate the per-domain results pages from mosaic-results/.
# Skip this if you have no local benchmark results; the tracked copies will be used.
uv run python docs/generate_results.py

# Render the full site to _site/
quarto render

# Or run a live-reloading preview
quarto preview
```

Quarto will error with a missing-file message if you run `quarto render` before `docs/generate.py` — `docs/solvers.qmd` is referenced in `_quarto.yml` but produced on the fly.

## Code style

- Python 3.10+, formatted with ruff (`ruff check --fix && ruff format`)
- Pre-commit hooks enforce formatting on every commit
- Run tests with `pytest`

## Pull request workflow

1. Fork the repository and create a feature branch
2. Make your changes, ensuring `ruff check` and `pytest` pass
3. Open a pull request with a clear description of what you're adding
4. CI runs lint, tests, and tesseract config validation on every PR. Full benchmark runs (which require GPU runners) are triggered by adding the `benchmark` label to the PR
