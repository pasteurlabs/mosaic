# Contributing to Mosaic

Mosaic grows as the community adds solver backends, improves configurations, and extends to new domains. This guide covers the contribution paths and the local tooling you need. Any contributions you make are greatly appreciated.

## Code of Conduct

Ensure your contributions adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Code

Mosaic is developed under the [Apache 2.0](LICENSE) license. By contributing to the Mosaic project you agree that your code contributions are governed by this license. We require you to sign our [Contributor License Agreement](https://github.com/pasteurlabs/pasteur-oss-cla/blob/main/README.md) to state so.

## Adding a solver backend

A solver backend is a Tesseract wrapper that implements the forward map and (optionally) its VJP for an existing benchmark domain. The full walkthrough — directory layout, schemas, `apply` / `abstract_eval` / `vector_jacobian_product`, the `metadata.mosaic:` config block, and domain-specific overrides (exclusions, input_overrides, explained_anomalies) — is in [Part A of the Add a Backend tutorial](https://docs.pasteurlabs.ai/projects/mosaic/latest/tutorial.html#part-a--add-a-solver-to-an-existing-domain) (source: [`docs/tutorial.qmd`](https://github.com/pasteurlabs/mosaic/blob/main/docs/tutorial.qmd)). It uses a short pseudo-spectral solver on the ns-grid domain as a running example.

## Tuning an existing solver

Results reflect out-of-the-box configurations at the time of each tagged release. If you believe a solver can perform better with different settings, submit a PR with the improved configuration.

### What to change

Solver behavior can be adjusted at three levels:

- **Solver-internal parameters** in `tesseract_api.py` — sub-stepping counts, relaxation factors, convergence tolerances, time integration order, etc.
- **Input overrides** in the problem config (`mosaic/benchmarks/problems/<domain>/config.py`, via `_SOLVERS["<solver>"].input_overrides = {...}`) — solver-specific values for shared schema fields (e.g. forcing a particular `dt` or `density`).
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

   After an edit, `mosaic run --only failed,stale -s <solver>` re-runs
   only the cells that aren't already fresh-ok, leaving the rest of the
   campaign untouched.

3. Compare against the baseline:
   ```bash
   mosaic status --format json > after.json
   mosaic status --format md --diff-against before.json
   ```

### Review expectations

- The change must not regress any other solver's results.
- CI requires a maintainer to add a benchmark label to any PR that touches `mosaic/` code. The options are:
  - `benchmark:none` — skip benchmarks (maintainer trusts no answer-changing code).
  - `benchmark:solver` — run benchmarks only on the modified solver (changes must be isolated to one solver).
  - `benchmark:all` — run the full benchmark suite from scratch.
- Include the diff output in your PR description so reviewers can see the impact at a glance.

## Adding a benchmark domain

A new domain bundles canonical schemas (`mosaic/mosaic_shared/problems/<domain>/schemas.py`), a `Problem` package (`mosaic/benchmarks/problems/<domain>/`, split into `config.py`, `ics.py`, `physics.py`, plus any per-problem `optimization.py` / `plots.py`), and at least one reference solver. The fastest way to start is `mosaic new-domain <name> --from-template <template>`, which scaffolds the file tree from a built-in template. The end-to-end walkthrough — schemas, suite defaults, IC generators, error functions, and a working reference solver — is in [Part B of the Add a Backend tutorial](https://docs.pasteurlabs.ai/projects/mosaic/latest/tutorial.html#part-b--add-a-new-benchmark-domain) (source: [`docs/tutorial.qmd`](https://github.com/pasteurlabs/mosaic/blob/main/docs/tutorial.qmd)).

## Building the docs

The Quarto site at <https://docs.pasteurlabs.ai/projects/mosaic/latest> is generated from `docs/*.qmd` plus two auto-generated files: `docs/solvers.qmd` (gitignored) and `docs/results_*.qmd`.

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

## Local development setup

Clone the repository and install the dependencies with [pre-commit](https://pre-commit.com/) hooks:

```console
$ git clone git@github.com:pasteurlabs/mosaic.git
$ cd mosaic
$ python -m venv venv
$ . venv/bin/activate
$ pip install -e .[dev]
$ pre-commit install
```

## Pull request workflow

1. Fork the repository and create a feature branch.
2. Make your changes, ensuring `ruff check` and `pytest` pass.
3. Open a pull request with a clear description of what you're adding.
4. CI runs lint, tests, and tesseract config validation on every PR. If your PR touches `mosaic/` code, a maintainer will add one of three benchmark labels (`benchmark:none`, `benchmark:solver`, or `benchmark:all`) to control whether and how benchmarks run.

## CI/CD

The project uses several GitHub Actions workflows that work together.

### Lint and test (`ci.yml`)

Runs on every PR: linting (ruff), tests (pytest), and Tesseract config validation. Must pass before merge.

### Benchmarks (`benchmark.yml`)

Runs on PRs that carry a benchmark label. The workflow has four stages:

1. **Plan** — detects which solvers changed, reads the benchmark label, and builds a (suite, problem, hardware) matrix.
2. **Build** — builds Docker images for any changed solvers and pushes them to GHCR.
3. **Run** — executes the benchmark matrix across CPU and GPU runners.
4. **Report** — merges CPU/GPU results, generates a status snapshot, renders a docs preview, posts a PR comment with the diff, and publishes results.

For dev PRs the report step overlays the PR's results on top of the current rolling baseline so the preview shows all solvers, not just the ones that ran. It then merges the PR's results into the `main/` directory on the `benchmark-results` branch.

For release PRs the report step writes results to a `_pending/` staging area instead.

### Docs (`docs.yml`)

Builds a docs preview on every PR. Fetches rolling benchmark results from the `benchmark-results` branch so the preview includes result plots even when benchmarks didn't run on this PR.

### Read the Docs (`.readthedocs.yaml`)

Builds the production documentation site. Tag builds (releases) fetch the matching versioned results directory; branch builds fetch the rolling `main/` results.

### Release (`release.yml`)

Handles the full release lifecycle (see [Release process](#release-process) below). When a release PR merges, the workflow creates a GitHub release and promotes the staged benchmark results: `_pending/` is copied to a permanent `{version}/` directory and `main/` is reset to that release baseline.

### PR preview cleanup (`cleanup-pr-previews.yml`)

Removes the PR's docs preview from `gh-pages` when the PR is closed.

### Benchmark results branch layout

The `benchmark-results` branch stores all published benchmark results:

- **`main/`** — rolling accumulation of results merged from dev PRs. Best-effort; different solvers may come from different commits.
- **`{version}/`** (e.g. `v0.4.0/`) — immutable release results. Produced by a full benchmark run on a release PR.
- **`_pending/`** — staging area for an in-progress release. Promoted to `{version}/` when the release PR merges.

Each directory contains per-domain subdirectories with `result.json`, `params.json`, plots (`.png`, `.gif`), plus top-level `snapshot.json` and `status-report.md` files.

## Commit and pull request message guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for all commits that reach the `main` branch. Each commit is crafted from a pull request that is squash-merged. The commit title and message comes from the pull request title and message, respectively. As such, they should be structured following the specification.

The title consists of a _type_, an optional _scope_, and a short _description_: `type[(scope)]: description`. The types we use are:

- `chore`: for changes that affect the build system, external dependencies, or general housekeeping.
- `ci`: for changes in the CI.
- `doc`: for documentation only changes.
- `feat`: for a new feature.
- `fix`: for fixing a bug.
- `perf`: for a code change that improves performance.
- `refactor`: for a code change that neither adds a feature nor fixes a bug.
- `security`: for a change that fixes a security issue.
- `test`: for adding new tests or fixing existing ones.

The scopes we use are:

- `harness`: for changes that affect the benchmark harness or CLI.
- `solver`: for changes to solver backends.
- `domain`: for changes to benchmark domains.
- `deps`: for changes in the dependencies.

In case there are breaking changes in your code, this should be indicated in the message either by appending an exclamation mark (`!`) after the type/scope or by adding a `BREAKING CHANGE:` trailer to the message.

## Versioning

Mosaic follows [semantic versioning](https://semver.org).

## Release process

(code owners only)

Releases are done via GitHub Actions, which automatically build the release artifacts and publish them to PyPI. To create a new release:

1. Make sure the code is in a good state, all tests pass, and the documentation is up to date.
2. Trigger the release workflow through the [GitHub UI](https://github.com/pasteurlabs/mosaic/actions/workflows/release.yml). This opens a new pull request with the release notes and the version number. Full benchmarks run automatically on the release PR, writing results to `_pending/` on the `benchmark-results` branch.
3. Add any additional release notes to the pull request message.
4. Once the pull request is ready, merge it into `main`.
5. GitHub Actions will then automatically release the new version, promote the staged benchmark results to a permanent `{version}/` directory, and reset the rolling `main/` baseline. Verify that the release artifacts are correctly built and published.
