# Contributing to Mosaic

Mosaic is designed to grow as the community adds solver backends, improves configurations, and extends to new domains. This guide covers the two main contribution types.

## Adding a solver backend

> **Full walkthrough:** see the [Add a Solver tutorial](docs/tutorial-add-solver.qmd) for a step-by-step guide with a complete working example (ns-grid domain, Lattice Boltzmann solver).

A solver backend is a Tesseract wrapper that implements the forward map and (optionally) its VJP for an existing benchmark domain.

### 1. Create the solver directory

```
mosaic/tesseracts/<domain>/<solver-name>/
  tesseract_config.yaml
  tesseract_requirements.txt
  tesseract_api.py
```

### 2. Implement the Tesseract API

Import the canonical schemas for your domain and subclass if extra fields are needed:

```python
from mosaic_shared.problems.<domain> import InputSchema as _Base, OutputSchema  # inside container
from pydantic import Field

class InputSchema(_Base):
    extra_param: int = Field(default=1, description="Solver-specific parameter")

def apply(inputs: InputSchema) -> OutputSchema:
    """Forward map: run the solver and return outputs."""
    ...

def abstract_eval(inputs: InputSchema) -> OutputSchema:
    """Return outputs with correct shapes but no computation (for JIT tracing)."""
    ...

def vector_jacobian_product(inputs: InputSchema, output_cotangents: OutputSchema) -> InputSchema:
    """Reverse-mode VJP. Optional for forward-only reference solvers."""
    ...
```

### 3. Write the config files

**tesseract_config.yaml** — include a `metadata.mosaic:` block so the harness discovers your solver automatically:

```yaml
name: my-solver
version: 0.1.0
description: >
  Short description of what this solver does and how it computes gradients.

metadata:
  mosaic:
    name: "My Solver" # display name (required)
    backend: jax # runtime: jax, pytorch, julia, cpp, warp, fenics, ...
    family: projection # solver family: projection, lbm, spectral, fd, fem, ...
    scheme: "MAC FD + projection" # numerical scheme label
    color: "#1f77b4" # hex colour for plots
    marker: "o" # matplotlib marker
    ad_strategy: autodiff # autodiff | adjoint | hybrid | null
    differentiable: true # explicit VJP flag
    uses_gpu: true # false for CPU-only solvers
    description: "One-sentence description for reference tables."
    doc_url: "https://..." # upstream docs link
```

**tesseract_requirements.txt:**

```
my-solver-package>=1.0
../../../mosaic_shared
```

The `metadata.mosaic:` block is all the harness needs to discover and run your solver. No Python registration step is required for basic operation.

### Domain-specific overrides (optional)

Most solvers work out of the box with no additional configuration. If your solver needs **domain-specific overrides**, add them in the problem config (`mosaic/benchmarks/problems/<domain>.py`):

- **Exclusions** — experiments your solver cannot run (e.g. periodic-only solver on a channel domain)
- **Input overrides** — solver-specific values for shared schema fields (e.g. forcing a particular `dt` or `density`)
- **Explained anomalies** — results that are correct but noticeably different from peers for known, method-intrinsic reasons

These overrides are merged on top of the auto-discovered `SolverSpec` from your `tesseract_config.yaml`. See existing solvers in the problem configs for examples.

### 4. Build and test

```bash
tesseract build mosaic/tesseracts/<domain>/<solver-name>
tesseract apply mosaic/tesseracts/<domain>/<solver-name> '{}'
mosaic run -p <domain> --suites forward -s my-solver
mosaic run -p <domain> --suites gradient -s my-solver -e fd_check
```

Once registered, the full evaluation suite (compatibility, accuracy, overhead, scaling, optimization convergence) runs automatically. No modifications to the evaluation code are needed.

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

A new domain requires interface schemas, at least one solver, and a problem configuration.

### 1. Define schemas

Create `mosaic/mosaic_shared/problems/<domain>/schemas.py` with `InputSchema` and `OutputSchema`. All fields must have defaults (Tesseract requirement). Re-export from `__init__.py`.

### 2. Create the problem config

Create `mosaic/benchmarks/problems/<domain>.py` with a `ProblemConfig` defining:

- `solvers` — use `discover_solvers(tesseract_dir)` to auto-populate from `tesseract_config.yaml` files, then merge domain-specific overrides (exclusions, input_overrides)
- `make_ic` — function that generates initial conditions
- `make_inputs` — function that builds solver inputs from IC and physics parameters
- `error_fn` — function that computes the objective from solver outputs
- Suite defaults (`forward_defaults`, `gradient_defaults`, `inverse_defaults`) specifying sweep parameters, FD settings, and optimizer configs

See `navier_stokes_grid.py` for a complete example.

### 3. Add at least one solver backend

Follow the solver contribution guide above.

## Code style

- Python 3.10+, formatted with ruff (`ruff check --fix && ruff format`)
- Pre-commit hooks enforce formatting on every commit
- Run tests with `pytest`

## Pull request workflow

1. Fork the repository and create a feature branch
2. Make your changes, ensuring `ruff check` and `pytest` pass
3. Open a pull request with a clear description of what you're adding
4. CI runs lint, tests, and tesseract config validation on every PR. Full benchmark runs (which require GPU runners) are triggered by adding the `benchmark` label to the PR
