![logo](logo.png)

# Mosaic: a benchmark suite for differentiable physics solvers

Mosaic measures gradient quality, computational cost, and solver compatibility across 14 differentiable physics solvers in 4 domains. Each solver is packaged as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) container exposing a uniform `apply` / `vjp` interface, enabling cross-solver comparison regardless of language or AD backend.

### Why Mosaic?

Differentiable physics solvers unlock gradient-based optimization for topology optimization, aerodynamic design, optimal control, and solver-in-the-loop ML. But the practical cost of obtaining correct gradients is largely undocumented: solvers span multiple languages and AD frameworks, runtime overhead varies by orders of magnitude, and subtle numerical issues (ill-conditioned Jacobians, chaotic divergence, floating-point truncation) can silently corrupt gradients. Mosaic provides a standardized, solver-agnostic evaluation that surfaces these practically relevant differences so practitioners can make informed choices.

<p align="center"><img src="visual_abstract.png" width="100%" alt="Overview of Mosaic: diverse solver backends are wrapped behind a uniform containerized interface (Tesseract), enabling cross-solver comparison on shared benchmark tasks across different physical domains."></p>

| ID     | Domain                     | Optimization task              | Solvers                                                |
| :----- | :------------------------- | :----------------------------- | :----------------------------------------------------- |
| **H**  | Heat transfer              | Conductivity inversion         | deal.II, FEniCS, Firedrake, JAX-FEM, torch-fem         |
| **S**  | Structural mechanics       | Compliance minimization (SIMP) | deal.II, FEniCS, Firedrake, JAX-FEM, TopOpt.jl         |
| **F2** | Incompressible fluids (2D) | Inflow optimization (drag)     | JAX-CFD, PhiFlow, INS.jl, XLB, PICT, Warp-NS, OpenFOAM |
| **F3** | 3D Navier-Stokes           | Initial condition recovery     | PhiFlow, XLB, PICT, Warp-NS, Exponax, INS.jl, OpenFOAM |

### 📊 [Browse the latest benchmark results →](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_ns_grid.html)

Per-domain pages with every plot, solver rankings, and the full evaluation protocol, refreshed on each release:
[Navier–Stokes 2D](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_ns_grid.html) ·
[Navier–Stokes 3D](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_ns_3d_grid.html) ·
[Structural mechanics](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_structural_mesh.html) ·
[Heat transfer](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_thermal_mesh.html)

---

> **Paper reproduction:** if you're here to reproduce the results from [our paper](https://arxiv.org/abs/XXXX.XXXXX), see the [`v0.1+paper-repro`](https://github.com/pasteurlabs/mosaic/tree/v0.1+paper-repro) tag which contains the figure-generation code, pinned dependencies, and step-by-step instructions.

**Jump to your use case:**

- [Browse the results](https://docs.pasteurlabs.ai/projects/mosaic/latest/docs/results_ns_grid.html) — see how the solvers compare, no setup required
- [Run the benchmarks](#run-the-benchmarks) — run solvers and inspect results
- [Use Tesseracts in your own code](#use-tesseracts-in-your-own-code) — researcher building on Mosaic solvers
- [Contribute](#contribute) — add a solver, tune a configuration, or extend to a new domain

---

## Run the benchmarks

Requires Python >= 3.10, Docker, and (for GPU solvers) the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

> **Platform note:** We strongly recommend **Linux with Docker Engine** for the best experience. Docker Desktop on macOS and Windows runs containers inside a VM, which adds significant overhead to solver execution and can surface ARM-related compatibility issues on Apple Silicon. If you're on macOS or Windows, consider using a Linux VM or WSL 2 with Docker Engine installed natively inside it.

```bash
git clone https://github.com/pasteurlabs/mosaic && cd mosaic
uv sync          # or: pip install -e .
mosaic run       # builds containers, runs experiments, generates plots
```

A single-problem run with `--debug` (reduced grid sizes) finishes in minutes and is a good way to verify your setup:

```bash
$ mosaic run -p thermal-mesh --suites forward --debug
──────────────────────────── problem: thermal-mesh ─────────────────────────────
──────────────────────────────────── build ─────────────────────────────────────
  deal.II          → dealii_heat_thermal_mesh:latest     (3.6s)
  FEniCS           → fenics_heat_thermal_mesh:latest     (3.2s)
  Firedrake        → firedrake_heat_thermal_mesh:latest  (2.4s)
  JAX-FEM          → jax_fem_thermal_mesh:latest         (5.1s)
  torch-fem        → torch_fem_thermal_mesh:latest       (4.8s)
───────────────────────────────   suite: forward ───────────────────────────────
  5 experiment(s) queued, 5 solver(s) registered
────────────────────────── experiment: baseline [1/5] ──────────────────────────
  deal.II done in 0.8s
  FEniCS done in 1.2s
  Firedrake done in 1.5s
  JAX-FEM done in 2.3s
  torch-fem done in 1.1s
...
─────────────────────────────────── summary ────────────────────────────────────
┏━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ problem      ┃ forward ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ thermal-mesh │   ok    │
└──────────────┴─────────┘
```

### Inspect results

```bash
mosaic status                        # per-experiment completion table
mosaic status -p ns-grid -f          # single domain with failure reasons
mosaic status --format md > report.md
mosaic status --format json > snap.json
```

### Pick which solvers run

`-s` (alias `--solvers`) takes either a flat CSV applied as a union
across every problem, or a per-problem map for finer control:

```bash
# Flat CSV — each problem keeps only the listed solvers that exist
# there; problems with zero matches are skipped.
mosaic run -s OpenFOAM,XLB,deal.II,JAX-FEM

# Per-problem map — explicit picks per domain.
mosaic run -s "ns-grid=XLB,jax-cfd;structural-mesh=Firedrake,JAX-FEM"
```

### Re-run a subset

After an initial pass, `mosaic run --only <state[,…]>` re-executes only
the cells currently in the given state and leaves fresh-ok cells alone.
Useful for iterating on a single solver or recovering from a partial
failure without redoing everything.

```bash
mosaic run --only failed              # re-run only failed cells
mosaic run --only failed,stale        # plus anything the harness/source has invalidated
mosaic run --only missing             # first-time runs only
mosaic run -s PhiFlow --only excluded # re-check after dropping an exclusion
```

States: `failed`, `anom`, `missing`, `stale`, `excluded`. Combinable
with `-p / --suites / -e / -s` for finer scoping.

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
from mosaic import get_config, PROBLEMS

cfg = get_config("ns-grid")           # Problem for 2-D Navier-Stokes
print(cfg.solver_names)               # available solver backends

# Each (suite, experiment) is registered on the Problem as an Experiment
# closure. Invoke one directly with a {solver_name: image_tag} mapping:
tags = {s.name: s.image_tag for s in cfg.solvers}
results = cfg.experiments["gradient/fd_check"].fn(cfg, tags)
```

Available top-level imports: `PROBLEMS`, `get_config`, `Problem`, `SolverSpec`, `IcSpec`, and the shared suite-kernel modules `forward`, `gradient`, `cost`, `optimization` (from `mosaic.benchmarks.problems.shared`).

---

## Contribute

Mosaic is designed to grow with the community. There are three ways in, roughly ordered by scope:

- **Tune an existing solver** — improve an out-of-the-box configuration. Snapshot `mosaic status --format json` before/after and include the diff in your PR. See [CONTRIBUTING.md](CONTRIBUTING.md#tuning-an-existing-solver) for the full workflow.
- **Add a solver** to an existing domain — three files under `mosaic/tesseracts/<domain>/<solver-name>/`. Walkthrough: [Add a Solver tutorial](docs/tutorial-add-solver.qmd).
- **Add a benchmark domain** — scaffold with `mosaic new-domain <name> --from-template <template>`. Walkthrough: [Add a Domain tutorial](docs/tutorial-add-domain.qmd).

[CONTRIBUTING.md](CONTRIBUTING.md) covers code style, the PR workflow, and how to build the docs locally. For questions and support, visit the [Tesseract Forum](https://si-tesseract.discourse.group/).

## Documentation

- [Getting Started](docs/getting-started.qmd) — prerequisites, installation, first benchmark
- [Use Mosaic solvers elsewhere](docs/standalone.qmd) — using individual Tesseracts in your own code
- [Architecture](docs/architecture.qmd) — Tesseract interface, data structures, evaluation protocol
- [Solver Reference](docs/solvers.qmd) — per-solver documentation with numerical methods, AD strategies, and known limitations
- [Add a Solver](docs/tutorial-add-solver.qmd) — step-by-step tutorial with a complete working example
- [Add a Domain](docs/tutorial-add-domain.qmd) — end-to-end walkthrough for a new physics domain

## Project structure

```
mosaic/
  benchmarks/             # evaluation harness (Python package: mosaic.benchmarks)
    cli.py                # command-line interface
    core/                 # runner, config, hardware detection, solver auto-discovery
    problems/             # per-domain packages (ns-grid, ns-3d-grid, structural-mesh, thermal-mesh)
      shared/             # cross-domain suite kernels (forward, gradient, cost, optimization) + plots
    plots/                # plotting infrastructure
  templates/              # task templates for scaffolding new domains
  tesseracts/             # solver backends (each is a Tesseract container)
    mosaic_shared/     # shared Tesseract interface schemas (also pip-installable)
      problems/           # per-domain input/output schemas
      utils/              # comparison metrics, plotting utilities
    navier-stokes-grid/   # JAX-CFD, PhiFlow, XLB, PICT, Warp-NS, etc.
    structural-mesh/      # deal.II, FEniCS, Firedrake, JAX-FEM, TopOpt.jl
    thermal-mesh/         # deal.II, FEniCS, Firedrake, JAX-FEM, torch-fem
  tests/                  # unit tests (run with pytest)
docs/                     # Quarto documentation site
```

## License

Apache 2.0. Individual solver backends retain their upstream licenses, documented per solver in the repository.
