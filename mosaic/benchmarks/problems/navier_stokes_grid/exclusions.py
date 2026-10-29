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
PICT_PISO_SPLITTING_FLOOR = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "PICT runs PISO with its default two pressure correctors, and its "
    "momentum predictor carries no pressure gradient. The splitting error "
    "this leaves scales with ν·dt/dx², which grows as the grid is refined at "
    "fixed CFL, so the error levels off on the finest grids and the observed "
    "order drops. More correctors remove the floor at a multiple of the wall "
    "time and reverse-mode memory",
)
PHIFLOW_SPARSE_CG_COST = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "PhiFlow solves each pressure projection with phiml's conjugate gradient "
    "on an explicit sparse matrix, as phiml does for every linear solve on a "
    "backend with sparse tensors, JAX included. JAX runs the sparse "
    "matrix-vector product as a gather and a scatter-add, about 160 of the "
    "184 µs per CG iteration at 256², against 14 µs for the same Laplacian "
    "as a stencil. A projection takes about 300 iterations at 256², and each "
    "RK4 step four projections",
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
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})
    for key in (
        "cost/spatial_cost",
        "cost/temporal_cost",
        "cost/vjp_cost/by_N",
        "cost/vjp_cost/by_steps",
    ):
        problem.exclude(key, {"phiflow": PHIFLOW_SPARSE_CG_COST})

    # Gradient — suite-level, covers every gradient/* below
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})

    # Optimization — suite-level + per-experiment obstacle gates
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
    problem.exclude("optimization/drag_opt", _OBSTACLE_GATE)
    problem.exclude("optimization/drag_opt", {"pict": PICT_DRAG_OPT_TOO_SLOW})
