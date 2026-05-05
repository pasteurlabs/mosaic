# Contributing to Mosaic

Mosaic is designed to grow as the community adds solver backends, improves configurations, and extends to new domains. This guide covers the two main contribution types.

## Adding a solver backend

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
from mosaic_shared.problems.<domain> import InputSchema as _Base, OutputSchema
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

**tesseract_config.yaml:**
```yaml
name: my-solver
version: 0.1.0
description: >
  Short description of what this solver does and how it computes gradients.
```

**tesseract_requirements.txt:**
```
my-solver-package>=1.0
../../../mosaic_shared
```

### 4. Register the solver

Add a `SolverSpec` entry in `mosaic/benchmarks/problems/<domain>.py`:

```python
"my-solver": SolverSpec(
    name="my-solver",
    backend="jax",           # or "pytorch", "julia", "cpp"
    ad_strategy="autodiff",  # or "adjoint", "hybrid"
    color="#1f77b4",
    has_forward=True,
    has_vjp=True,
),
```

### 5. Build and test

```bash
tesseract build mosaic/tesseracts/<domain>/<solver-name>
tesseract apply mosaic/tesseracts/<domain>/<solver-name> '{}'
mosaic forward -p <domain> -s my-solver
mosaic gradient -p <domain> -s my-solver -e fd_check
```

Once registered, the full evaluation suite (compatibility, accuracy, overhead, scaling, optimization convergence) runs automatically. No modifications to the evaluation code are needed.

## Adding a benchmark domain

A new domain requires interface schemas, at least one solver, and a problem configuration.

### 1. Define schemas

Create `mosaic/mosaic_shared/problems/<domain>/schemas.py` with `InputSchema` and `OutputSchema`. All fields must have defaults (Tesseract requirement). Re-export from `__init__.py`.

### 2. Create the problem config

Create `mosaic/benchmarks/problems/<domain>.py` with a `ProblemConfig` defining:

- `solvers` — dict of `SolverSpec` entries
- `make_ic` — function that generates initial conditions
- `make_inputs` — function that builds solver inputs from IC and physics parameters
- `error_fn` — function that computes the objective from solver outputs
- Suite defaults (`forward_defaults`, `gradient_defaults`, `inverse_defaults`) specifying sweep parameters, FD settings, and optimizer configs

See `navier_stokes_grid.py` for a complete example.

### 3. Register the domain

Add the import and registry entry in `mosaic/benchmarks/problems/__init__.py`.

### 4. Add at least one solver backend

Follow the solver contribution guide above.

## Code style

- Python 3.10+, formatted with ruff (`ruff check --fix && ruff format`)
- Pre-commit hooks enforce formatting on every commit
- Run tests with `pytest`

## Pull request workflow

1. Fork the repository and create a feature branch
2. Make your changes, ensuring `ruff check` and `pytest` pass
3. Open a pull request with a clear description of what you're adding
4. CI will build all affected solver containers and run the evaluation suite

## Solver tuning

Results in this benchmark reflect out-of-the-box configurations at the time of the tagged release. If you believe a solver can perform better with different settings, submit a PR with the improved configuration. CI re-runs the full evaluation suite automatically, and updated results are published with each release.
