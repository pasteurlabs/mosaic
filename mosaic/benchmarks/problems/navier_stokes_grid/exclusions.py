# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-solver exclusions for ns-grid.

Two layers:

* Reusable :class:`Exclusion` constants for failure modes that recur across
  multiple experiments (e.g. OpenFOAM's missing VJP excludes it from every
  gradient and VJP-cost experiment).
* :func:`register` — applies every ``problem.exclude(...)`` call, so
  ``config.py`` keeps a single ``_register_exclusions(problem)`` line and the
  long reason strings live here instead of cluttering the assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem


# ── Reusable constants ───────────────────────────────────────────────────────

JAX_CFD_NO_OBSTACLE = Exclusion(
    ExclusionCategory.NOT_IMPLEMENTED,
    "this tesseract implements only periodic flows; jax-cfd itself has "
    "non-periodic pressure solvers for channel flows, which are not wired "
    "up here",
)
INS_JL_NO_OBSTACLE = Exclusion(
    ExclusionCategory.NOT_IMPLEMENTED,
    "this tesseract implements only periodic flows; INS.jl itself supports "
    "channel BCs (Dirichlet inflow, pressure outflow) and obstacles via "
    "Brinkman forcing, which are not wired up here",
)
OPENFOAM_NO_VJP = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "OpenFOAM has no VJP to benchmark",
)
OPENFOAM_NON_DIFFERENTIABLE_GRAD = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "OpenFOAM is non-differentiable (C++, no AD path); "
    "DAFoam/OpenFOAM-AD exist but are not deployed in this tesseract",
)
OPENFOAM_NON_DIFFERENTIABLE_OPT = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "OpenFOAM is a non-differentiable, forward-only solver",
)
PICT_DRAG_OPT_TOO_SLOW = Exclusion(
    ExclusionCategory.INFEASIBLE,
    "each drag_opt iteration runs a full forward and VJP through PICT's "
    "unpreconditioned pressure solves on the body-fitted channel mesh, so the "
    "experiment takes hours on the benchmark GPU",
)
PICT_LATENCY_BOUND_SOLVES = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "PICT's linear solves are bound by synchronisation, not by grid size. "
    "Each step runs three of them (the momentum predictor and two pressure "
    "correctors). Every solve synchronises the whole device about 20 times "
    "and frees device memory 10 times, and its CG copies scalars back to the "
    "host about three times per iteration. A 64-times larger grid (N = 32 to "
    "256) makes a solve only about 4 times slower, so at ns-grid's sizes the "
    "GPU mostly waits",
)
PICT_PISO_SPLITTING_FLOOR = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "PICT runs PISO with its default two pressure correctors, and its "
    "momentum predictor carries no pressure gradient. The splitting error "
    "this leaves scales with ν·dt/dx², which grows as the grid is refined at "
    "fixed CFL, so the error levels off on the finest grids and the observed "
    "order drops. More correctors remove the floor at a multiple of the wall "
    "time and reverse-mode memory",
)
_OBSTACLE_GATE = {
    "jax_cfd": JAX_CFD_NO_OBSTACLE,
    "ins_jl": INS_JL_NO_OBSTACLE,
}


def register(problem: Problem) -> None:
    """Apply every ns-grid exclusion via :meth:`Problem.exclude`."""
    # Forward
    problem.exclude("forward/baseline", {"pict": PICT_PISO_SPLITTING_FLOOR})
    problem.exclude("forward/cylinder", _OBSTACLE_GATE)

    # Cost
    for key in ("cost/spatial_cost", "cost/temporal_cost"):
        problem.exclude(key, {"pict": PICT_LATENCY_BOUND_SOLVES})
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})

    # Gradient — suite-level, covers every gradient/* below
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})

    # Optimization — suite-level + per-experiment obstacle gates
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
    problem.exclude("optimization/drag_opt", _OBSTACLE_GATE)
    problem.exclude("optimization/drag_opt", {"pict": PICT_DRAG_OPT_TOO_SLOW})
