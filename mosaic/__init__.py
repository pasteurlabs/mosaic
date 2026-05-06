"""Mosaic: a benchmark suite for differentiable physics solvers.

Programmatic API
----------------

Quick start::

    from mosaic import get_config, PROBLEMS

    cfg = get_config("ns-grid")         # ProblemConfig for 2-D Navier-Stokes
    print(cfg.solvers.keys())           # available solver backends

    # Run a single evaluation (requires built Tesseract images):
    from mosaic import forward, gradient, cost, optimization
    tags = {"exponax": "exponax:latest", ...}
    results = gradient.run_fd_check(cfg, tags)

See the ``suites`` modules for the full list of ``run_*`` functions:
:mod:`mosaic.forward`, :mod:`mosaic.gradient`, :mod:`mosaic.cost`,
:mod:`mosaic.optimization`.
"""

from mosaic.benchmarks.core.config import IcSpec, ProblemConfig, SolverSpec
from mosaic.benchmarks.problems import PROBLEMS, get_config
from mosaic.benchmarks.suites import cost, forward, gradient, optimization

__all__ = [
    # Problem registry
    "PROBLEMS",
    "get_config",
    # Config dataclasses
    "ProblemConfig",
    "SolverSpec",
    "IcSpec",
    # Evaluation suites
    "forward",
    "gradient",
    "cost",
    "optimization",
]
