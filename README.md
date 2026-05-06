![logo](logo.png)

# Mosaic: a benchmark suite for differentiable physics solvers

Mosaic is an extensible benchmark framework that measures gradient quality, computational cost, and solver compatibility across differentiable physics solvers. It treats gradient accuracy as a first-class criterion alongside forward accuracy and throughput.

Each solver is packaged as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) container exposing a uniform `apply` / `vjp` interface regardless of language or AD strategy. The benchmark harness calls these containers as native JAX functions via [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax), enabling comparison across incompatible AD backends (JAX, PyTorch, Julia Zygote, C++) without shared dependencies.

## Benchmark domains

| ID     | Domain                     | Optimization task                 | Control dim. | Backends                                               |
| :----- | :------------------------- | :-------------------------------- | :----------- | :----------------------------------------------------- |
| **H**  | Heat transfer              | Conductivity inversion            | 128          | deal.II, FEniCS, Firedrake, JAX-FEM, torch-fem         |
| **S**  | Structural mechanics       | Compliance minimization (SIMP)    | 256          | deal.II, FEniCS, Firedrake, JAX-FEM, TopOpt.jl         |
| **F2** | Incompressible fluids (2D) | Inflow optimization for drag min. | 32           | JAX-CFD, PhiFlow, INS.jl, XLB, PICT, Warp-NS, OpenFOAM |
| **F3** | 3D Navier-Stokes           | Initial condition recovery        | 12k          | PhiFlow, XLB, PICT, Warp-NS, Exponax, INS.jl, OpenFOAM |

## Evaluation protocol

All metrics are computed uniformly across solvers through the Tesseract interface:

- **Setup compatibility.** Whether a solver produces a usable gradient, fails numerically, or is structurally excluded.
- **Gradient accuracy.** Central finite differences through the Tesseract interface serve as the solver-agnostic reference. Cosine similarity and relative error across multiple random perturbation directions.
- **Performance.** Forward and VJP wall-clock time, their ratio, and peak memory, measured across multiple problem sizes.
- **Forward accuracy.** Resolution sweep against a reference solver and adherence to physical laws.
- **Optimization convergence.** Adam with each solver's gradients for a fixed iteration budget; success is reaching within 1% of the best solution across all solvers.

## Quick start

**Prerequisites:** Python >= 3.10 and Docker.

```bash
git clone https://github.com/pasteurlabs/mosaic
cd mosaic
pip install -e ".[dev]"
pre-commit install
```

## Running benchmarks

```bash
mosaic run-all                                    # all suites, all problems
mosaic run-all --problems ns-grid,structural-mesh # filter problems
mosaic run-all --suites forward,gradient          # filter suites
mosaic run-all --no-build                         # skip container builds
mosaic run-all --plots-only                       # regenerate plots from existing results

mosaic ics      -p <problem>                      # visualize initial conditions
mosaic forward  -p <problem>                      # forward accuracy
mosaic cost     -p <problem>                      # wall-clock scaling
mosaic gradient -p <problem>                      # gradient quality (FD check, Jacobian SVD)
mosaic recovery -p <problem>                      # optimization convergence

# Useful flags
mosaic gradient -p ns-grid -e fd_check --debug          # small problem for quick smoke-test
mosaic gradient -p ns-grid -e fd_check --ics tgv        # run only specific IC
mosaic forward  -p ns-grid -s exponax                   # single solver
mosaic forward  -p ns-grid --gpus 0,1,2                 # multi-GPU parallel
```

Results land in `mosaic/benchmarks/results/<problem>/<suite>/` as JSON, NPZ, and PNG/PDF plots.

## Coverage status

```bash
mosaic status                             # full per-problem tables
mosaic status -p ns-grid -f               # single problem with failure reasons
mosaic status --format md > report.md     # markdown report
mosaic status --format json > snap.json   # machine-readable snapshot
```

## Documentation

- [Getting Started](docs/getting-started.qmd) — prerequisites, installation, first benchmark
- [Architecture](docs/architecture.qmd) — Tesseract interface, data structures, evaluation protocol
- [Solver Reference](docs/solvers.qmd) — per-solver documentation with numerical methods, AD strategies, and known limitations

## Programmatic API

Mosaic exposes a Python API for running evaluations without the CLI:

```python
from mosaic import get_config, gradient, PROBLEMS

cfg = get_config("ns-grid")           # ProblemConfig for 2-D Navier-Stokes
tags = {"exponax": "exponax:latest"}   # solver image tags (after `tesseract build`)
results = gradient.run_fd_check(cfg, tags)
```

Available top-level imports: `PROBLEMS`, `get_config`, `ProblemConfig`, `SolverSpec`, `IcSpec`, and the suite modules `forward`, `gradient`, `cost`, `optimization`.

## Extending to new domains

Mosaic ships with task templates — validated starting configurations for common physics patterns. Use them to scaffold a new benchmark domain:

```bash
mosaic templates                                           # list available templates
mosaic validate-template ns-periodic                       # check a template
mosaic new-domain my-flow --from-template ns-periodic      # scaffold a new domain
```

This generates schema stubs, a problem config with suite defaults, and the tesseract directory. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

### Adding a solver

New solvers are auto-discovered from a `mosaic:` metadata block in `tesseract_config.yaml` — no Python registration step is required:

```yaml
# mosaic/tesseracts/<domain>/<solver-name>/tesseract_config.yaml
name: my-solver
version: 0.1.0
description: Short description.

mosaic:
  name: "My Solver"
  backend: jax
  scheme: "FEM HEX8"
  color: "#1f77b4"
  ad_strategy: autodiff
  differentiable: true
  uses_gpu: true
```

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new solver backend or benchmark domain.

Contributions are validated automatically: CI runs the full evaluation suite on every pull request, and updated results are published with each release.

## Reproducing paper results

Benchmark results from the paper are available as downloadable artifacts attached to each [tagged release](https://github.com/pasteurlabs/mosaic/releases). To reproduce figures from the paper:

1. Download the release artifacts and extract into `mosaic/benchmarks/results/`
2. Run `mosaic run-all --plots-only` to regenerate all plots

To re-run benchmarks from scratch, ensure Docker is running and execute `mosaic run-all`. This requires building all solver containers and may take several hours depending on hardware.

## License

Apache 2.0. Individual solver backends retain their upstream licenses, documented per solver in the repository.
