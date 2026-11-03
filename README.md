<p align="center">
  <img src="docs/logo.webp" width="560" alt="Mosaic">
</p>

# Mosaic

Mosaic is a benchmark for differentiable solvers. It measures whether they are fit for purpose and at what cost, in compute and in development effort. Because every solver runs the same setups under the same protocol, Mosaic also allows researchers to investigate patterns across solvers, for example between solvers that share numerical characteristics, or between gradient quality and optimization performance. [Our paper](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/paper.html) is one such investigation.

<a href="https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/results.html"><img src="https://img.shields.io/badge/%F0%9F%93%8A_results-browse-3b6fb6" alt="Results"></a>
<a href="#run-the-benchmark"><img src="https://img.shields.io/badge/%E2%96%B6_run-the_benchmark-555" alt="Run"></a>
<a href="#use-solvers-in-your-own-code"><img src="https://img.shields.io/badge/%F0%9F%A7%A9_use-in_your_code-555" alt="Use"></a>
<a href="#contribute"><img src="https://img.shields.io/badge/%F0%9F%A4%9D_contribute-add_a_solver-555" alt="Contribute"></a>
<a href="https://docs.pasteurlabs.ai/projects/mosaic/stable/"><img src="https://img.shields.io/badge/docs-stable-3b6fb6" alt="Stable docs"></a>
<a href="https://arxiv.org/abs/2606.27895"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b" alt="Paper"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>

![Overview of Mosaic: diverse solvers are wrapped behind a uniform containerized interface (Tesseract), enabling cross-solver comparison on shared benchmark tasks across different physical domains.](docs/visual_abstract.png)

## What Mosaic measures

If you optimize or train _through_ a simulation, the solver has to return a correct forward prediction and a usable gradient (the vector–Jacobian product, VJP). Mosaic checks both. For every solver and setup, it reports:

- **Fit for purpose**: whether the forward solution matches a trusted reference, whether the gradient agrees with finite differences, and whether a gradient-based optimizer makes progress through the solver. Reliability counts how many experiments ran cleanly, and deviations record where a solver runs a substitute for part of the prescribed setup.
- **Cost**: wall-clock time and peak memory of the forward and VJP calls, and the development effort it took to make the solver run the setup, from L0 (nothing beyond expected tweaks) to L3 (changing the solver itself).

Each solver is packaged as a [Tesseract](https://github.com/pasteurlabs/tesseract-core) container with a uniform `apply` / `vjp` interface, so a single harness can compare solvers across languages and AD frameworks (JAX, PyTorch, Julia, hand-written C++ adjoints). Every solver, its numerics and its intervention record are listed in the [solver reference](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/solvers.html).

## 📊 Results

**[Browse the benchmark results →](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/results.html)**

The results overview shows the checks, cost, effort and deviations of every solver on every setup, refreshed on each release. Each setup has its own page with every plot. For findings drawn from these results, see the [paper page](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/paper.html).

## 📖 Documentation

Two versions are published. You most likely want [stable](https://docs.pasteurlabs.ai/projects/mosaic/stable/), which tracks the latest release, where all solvers were benchmarked in the same run. [Latest](https://docs.pasteurlabs.ai/projects/mosaic/latest/) tracks the `main` branch and may combine results from different runs.

- [Overview](https://docs.pasteurlabs.ai/projects/mosaic/stable/)
- [Results](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/results.html)
- [Solvers](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/solvers.html)
- [Run the benchmark](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/getting-started.html)
- [Use a solver in your code](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/standalone.html)
- [Contributing guide](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/contributing.html)
- [Paper](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/paper.html)

> [!TIP]
> **Reproducing our paper?** See the [`v0.1+paper-repro`](https://github.com/pasteurlabs/mosaic/tree/v0.1+paper-repro) tag for figure-generation code, pinned dependencies, and step-by-step instructions.

---

## Run the benchmark

Mosaic needs Python ≥ 3.11 and Docker. Linux with Docker Engine, an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) runs every solver. Without a GPU, `--gpus none` restricts a run to the CPU-only solvers. macOS and Windows are not supported, because Docker Desktop can't pass an NVIDIA GPU through to containers and not every solver image is known to build on ARM hosts.

```bash
git clone https://github.com/pasteurlabs/mosaic && cd mosaic
uv sync && source .venv/bin/activate   # or: pip install -e .
mosaic run -p thermal-mesh --suites forward --gpus none --debug
```

The last command builds the CPU-only heat-transfer solvers and runs a reduced forward check, which works without a GPU. [Run the benchmark](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/getting-started.html) covers the full run, choosing what runs, resuming, reading the output and troubleshooting.

## Use solvers in your own code

Every solver is a standalone Tesseract that you can call from your own JAX or PyTorch code without the benchmark harness. Build the image once, then differentiate through it with [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax):

```bash
pip install -e mosaic/mosaic_shared tesseract-jax
tesseract build mosaic/tesseracts/navier-stokes-grid/exponax
```

```python
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from mosaic_shared.problems.navier_stokes_grid.schemas import make_vortex_ic

inputs = {"v0": make_vortex_ic(N=64, seed=42), "viscosity": jnp.array([0.01]), "steps": 50}

with Tesseract.from_image("exponax_navier_stokes_grid:latest") as t:
    def loss(v0):
        return jnp.mean(apply_tesseract(t, {**inputs, "v0": v0})["result"] ** 2)

    grad_v0 = jax.grad(loss)(inputs["v0"])
```

[Use a solver in your code](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/standalone.html) explains the inputs and outputs, PyTorch, running without Docker, mesh solvers and GPUs.

## Contribute

Mosaic is designed to grow with its community. There are three main ways in, roughly by scope:

- **Tune an existing solver** and record what you changed in its intervention record. See [tuning an existing solver](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/contributing.html#tuning-an-existing-solver).
- **Add a solver** to an existing domain, with three files under `mosaic/tesseracts/<domain>/<solver-name>/`. See [Add a solver](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/add-solver.html).
- **Add a domain**, scaffolded with `mosaic new-domain <name> --from-template <template>`. See [Add a domain](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/add-domain.html).

The [contributing guide](https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/contributing.html) covers the development setup, the pull request workflow and the conventions. [CONTRIBUTING.md](CONTRIBUTING.md) has the contributor license agreement and the code of conduct. For questions, [open an issue](https://github.com/pasteurlabs/mosaic/issues) or visit the [Tesseract Forum](https://si-tesseract.discourse.group/).

### How Mosaic is built

Mosaic's code is written by coding agents, and curated and reviewed by the [people who maintain it](https://github.com/pasteurlabs/mosaic/graphs/contributors). The results stay accountable because every intervention a solver needs and every deviation from a setup is recorded with its reason and evidence, and linked from the results it affects. Issues and forum posts are answered by people.

## Project structure

```
mosaic/
  benchmarks/             # evaluation harness (Python package: mosaic.benchmarks)
    cli/                  # command-line interface, one module per subcommand
    core/                 # runner, config, hardware detection, solver discovery, intervention records
    problems/             # one package per problem (ns-grid, ns-3d-grid, structural-mesh, thermal-mesh)
      shared/             # suite kernels shared across problems (forward, gradient, cost, optimization, ics)
        plots/            # plotting infrastructure
  mosaic_shared/          # canonical schemas and shared types for the solvers (also pip-installable)
  templates/              # templates for scaffolding new domains
  tesseracts/             # solvers, one Tesseract per directory
    navier-stokes-grid/   # incompressible flow on grids, 2D and 3D
    structural-mesh/      # linear elasticity on hexahedral meshes
    thermal-mesh/         # steady heat conduction on hexahedral meshes
tests/                    # unit and integration tests (run with pytest)
docs/                     # Quarto documentation site
  index.qmd               # overview
  results.qmd             # results overview (per-setup pages are generated)
  getting-started.qmd     # run the benchmark
  standalone.qmd          # use a solver in your code
  contributing.qmd        # contributing guide
  add-solver.qmd          # add a solver
  add-domain.qmd          # add a domain
  architecture.qmd        # how the harness fits together
  paper.qmd               # the paper's findings, dated
  generate.py             # generates the solver reference (solvers.qmd)
  generate_results.py     # generates the results pages
```

## Citation

If you use Mosaic in your research, please cite the paper ([arXiv:2606.27895](https://arxiv.org/abs/2606.27895)):

```bibtex
@misc{rehmann2026mosaic,
  title         = {Mosaic: A Benchmark Suite for Differentiable Physics Solvers},
  author        = {Rehmann, Andrin and Zimmermann, Heiko and H{\"a}fner, Dion},
  year          = {2026},
  eprint        = {2606.27895},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2606.27895}
}
```

## License

Apache 2.0. Individual solvers retain their upstream licenses, documented per solver in the repository.
