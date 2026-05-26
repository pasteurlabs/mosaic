# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mosaic: a benchmark suite for differentiable physics solvers.

Programmatic API
----------------

Quick start::

    from mosaic import get_config

    cfg = get_config("ns-grid")        # Problem for 2-D Navier-Stokes
    print(cfg.solver_names)            # available solver backends

    # Run a single registered experiment (requires built Tesseract images):
    tags = {"exponax": "exponax:latest", ...}
    results = cfg.experiments["gradient/fd_check"].fn(cfg, tags)

Experiments are written as small *kernels* decorated with
:func:`mosaic.benchmarks.core.experiment.experiment` and registered on a
:class:`Problem` via :meth:`Problem.add_experiment`. Per-suite kernels
(``fd_check``, ``param_sweep``, ``horizon_sweep_limits``, ``jacobian_svd``,
``run_agreement``, …) live under ``mosaic.benchmarks.problems.shared``.
"""

from mosaic.benchmarks.core.config import IcSpec, Problem, SolverSpec
from mosaic.benchmarks.problems import PROBLEMS, get_config
from mosaic.benchmarks.problems.shared import cost, forward, gradient, optimization

__all__ = [
    # Problem registry
    "PROBLEMS",
    "IcSpec",
    # Config dataclasses
    "Problem",
    "SolverSpec",
    "cost",
    # Evaluation suites
    "forward",
    "get_config",
    "gradient",
    "optimization",
]
