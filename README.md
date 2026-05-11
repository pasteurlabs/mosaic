![logo](logo.png)

# Mosaic: a benchmark suite for differentiable physics solvers

Mosaic measures gradient quality, computational cost, and solver compatibility across 14 differentiable physics solvers in 4 domains. Each solver is packaged as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) container exposing a uniform `apply` / `vjp` interface, enabling cross-solver comparison regardless of language or AD backend.

| ID     | Domain                     | Optimization task              | Solvers                                                |
| :----- | :------------------------- | :----------------------------- | :----------------------------------------------------- |
| **H**  | Heat transfer              | Conductivity inversion         | deal.II, FEniCS, Firedrake, JAX-FEM, torch-fem         |
| **S**  | Structural mechanics       | Compliance minimization (SIMP) | deal.II, FEniCS, Firedrake, JAX-FEM, TopOpt.jl         |
| **F2** | Incompressible fluids (2D) | Inflow optimization (drag)     | JAX-CFD, PhiFlow, INS.jl, XLB, PICT, Warp-NS, OpenFOAM |
| **F3** | 3D Navier-Stokes           | Initial condition recovery     | PhiFlow, XLB, PICT, Warp-NS, Exponax, INS.jl, OpenFOAM |

---

**Jump to your use case:**

- [Reproduce paper results](#reproduce-paper-results) — reviewer or reader wanting to verify figures and claims
- [Use Tesseracts in your own code](#use-tesseracts-in-your-own-code) — researcher building on Mosaic solvers
- [Contribute](#contribute) — add a solver, tune a configuration, or extend to a new domain

---

## Reproduce paper results

Two paths, from fastest to most thorough. Both require Python >= 3.10.

### Regenerate figures from published data (no Docker, no GPU)

Download the result artifacts and regenerate every figure in the paper without re-running any solver.

```bash
git clone https://github.com/pasteurlabs/mosaic && cd mosaic

# Install (pick one)
uv sync              # uv (recommended)
pip install -e .     # pip

# Download the paper's benchmark results (Zenodo: https://zenodo.org/records/20067888, ~443 MB)
wget -qO- 'https://zenodo.org/records/20067888/files/mosaic-results.tar?download=1' | tar x

# Regenerate per-suite plots (PNG/PDF alongside each result.json)
mosaic run --plots-only

# Regenerate the publication figures used in the paper
mosaic paper-plots                          # writes to mosaic-results/figures/
mosaic paper-plots --only scaling,fd_check  # or just specific figures
```

### Re-run all experiments from scratch

Re-run every solver on every benchmark task. Requires Docker and (for GPU solvers) an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/pasteurlabs/mosaic && cd mosaic

# Install (pick one)
uv sync              # uv (recommended)
pip install -e .     # pip

# Build all solver containers and run the full suite
mosaic run                    # builds containers, runs experiments, generates plots

# Generate publication figures from fresh results
mosaic paper-plots
```

A full run builds 14 solver containers and executes 5 suites across 4 domains. Expect several hours on a machine with a modern GPU. CPU-only solvers can be run separately with `mosaic run --hardware cpu`.

> **Platform note:** x86-64 Linux with Docker Engine (not Docker Desktop) is strongly recommended. Docker Desktop adds a virtualisation layer that significantly increases overhead on macOS and Windows. Some solver images (notably the Julia-based solvers) do not build on ARM/Apple Silicon. For full reproducibility, an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required.

For an environment that exactly matches published results:

```bash
# With uv (recommended)
cp production.uv.lock uv.lock && uv sync --frozen

# Or with pip
pip install -r requirements.txt && pip install -e .
```

### Continuous benchmarking

After the initial paper release, all results are generated automatically by CI:

- **Every push to `main`** that touches a solver or the harness triggers a benchmark run on dedicated GPU and CPU runners. Results are published to the [`benchmark-results`](https://github.com/pasteurlabs/mosaic/tree/benchmark-results) branch along with a machine-readable snapshot and a diff against the previous baseline.
- **Tagged releases** (`v*`) run the full suite across all domains and attach `benchmark-results.tar.gz` to the GitHub release — same archive format used in the "regenerate figures" step above.
- **Pull requests** labelled `benchmark` get a full evaluation run; CI posts a status comparison as a PR comment.

This means published results stay up-to-date as solvers evolve and new backends land, without manual re-runs.

### Inspect results

```bash
mosaic status                        # per-experiment completion table
mosaic status -p ns-grid -f          # single domain with failure reasons
mosaic status --format md > report.md
mosaic status --format json > snap.json
```

---

## Use Tesseracts in your own code

Every solver in Mosaic is a standalone [Tesseract](https://github.com/pasteurlabs/tesseract-core) that you can call from your own research code — no benchmark harness required.

### Install

```bash
# Shared schemas (only deps: pydantic + tesseract-core)
pip install -e mosaic/mosaic_shared

# For containerised usage (recommended): also install tesseract-jax
pip install tesseract-core tesseract-jax jax
```

### Option A: Local (no Docker)

Fastest for prototyping. Requires the solver's native Python dependencies.

```python
import numpy as np
from tesseract_core import Tesseract
from mosaic_shared.problems.navier_stokes_grid.schemas import make_vortex_ic

ic = make_vortex_ic(N=64, seed=42)
inputs = {"v0": ic, "viscosity": np.array([0.01], dtype=np.float32), "steps": 50}

t = Tesseract.from_tesseract_api(
    "mosaic/tesseracts/navier-stokes-grid/exponax/tesseract_api.py"
)
outputs = t.apply(inputs)
```

### Option B: Via container (requires Docker, fully isolated)

Works for every solver regardless of language. Build the image once, then use it from JAX:

```bash
$ tesseract build mosaic/tesseracts/navier-stokes-grid/exponax
```

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from mosaic_shared.problems.navier_stokes_grid.schemas import make_vortex_ic

ic = make_vortex_ic(N=64, seed=42)
inputs = {"v0": ic, "viscosity": jnp.array([0.01]), "steps": 50}

with Tesseract.from_image("exponax_navier_stokes_grid:latest") as t:
    outputs = apply_tesseract(t, inputs)
    grad_v0 = jax.grad(lambda v0: jnp.mean(
        apply_tesseract(t, {**inputs, "v0": v0})["result"] ** 2
    ))(inputs["v0"])
```

See [Standalone Usage](docs/standalone.qmd) for the full guide (GPU usage, mesh-based solvers, common gotchas) and the [Solver Reference](docs/solvers.qmd) for the per-solver catalog with image names.

### Programmatic API

Mosaic also exposes a Python API for running evaluations without the CLI:

```python
from mosaic import get_config, gradient, PROBLEMS

cfg = get_config("ns-grid")           # ProblemConfig for 2-D Navier-Stokes
tags = {"exponax": "exponax_navier_stokes_grid:latest"}
results = gradient.run_fd_check(cfg, tags)
```

Available top-level imports: `PROBLEMS`, `get_config`, `ProblemConfig`, `SolverSpec`, `IcSpec`, and the suite modules `forward`, `gradient`, `cost`, `optimization`.

---

## Contribute

Mosaic is designed to grow with the community. There are three ways in, roughly ordered by scope:

- **Tune an existing solver** — improve an out-of-the-box configuration. Snapshot `mosaic status --format json` before/after and include the diff in your PR. See [CONTRIBUTING.md](CONTRIBUTING.md#tuning-an-existing-solver) for the full workflow.
- **Add a solver** to an existing domain — three files under `mosaic/tesseracts/<domain>/<solver-name>/`. Walkthrough: [Add a Solver tutorial](docs/tutorial-add-solver.qmd).
- **Add a benchmark domain** — scaffold with `mosaic new-domain <name> --from-template <template>`. Walkthrough: [Add a Domain tutorial](docs/tutorial-add-domain.qmd).

[CONTRIBUTING.md](CONTRIBUTING.md) covers code style, the PR workflow, and how to build the docs locally.

## Documentation

- [Getting Started](docs/getting-started.qmd) — prerequisites, installation, first benchmark
- [Standalone Usage](docs/standalone.qmd) — using individual Tesseracts in your own code
- [Architecture](docs/architecture.qmd) — Tesseract interface, data structures, evaluation protocol
- [Solver Reference](docs/solvers.qmd) — per-solver documentation with numerical methods, AD strategies, and known limitations
- [Add a Solver](docs/tutorial-add-solver.qmd) — step-by-step tutorial with a complete working example
- [Add a Domain](docs/tutorial-add-domain.qmd) — end-to-end walkthrough for a new physics domain

## Project structure

```
mosaic/
  benchmarks/           # evaluation harness (Python package: mosaic.benchmarks)
    cli.py              # command-line interface
    core/               # runner, config, hardware detection, solver auto-discovery
    suites/             # forward, gradient, cost, optimization
    problems/           # per-domain configs (ns-grid, structural-mesh, etc.)
    plots/              # paper figure generation
  mosaic_shared/        # shared Tesseract interface schemas (also pip-installable)
    problems/           # per-domain input/output schemas
    utils/              # comparison metrics, plotting utilities
  templates/            # task templates for scaffolding new domains
  tesseracts/           # solver backends (each is a Tesseract container)
    navier-stokes-grid/ # JAX-CFD, PhiFlow, XLB, PICT, Warp-NS, etc.
    structural-mesh/    # deal.II, FEniCS, Firedrake, JAX-FEM, TopOpt.jl
    thermal-mesh/       # deal.II, FEniCS, Firedrake, JAX-FEM, torch-fem
  tests/                # unit tests (run with pytest)
docs/                   # Quarto documentation site
```

## License

Apache 2.0. Individual solver backends retain their upstream licenses, documented per solver in the repository.
