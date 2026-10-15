# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-solver exclusions for ns-grid.

Two layers:

* Reusable :class:`Exclusion` constants for failure modes that recur across
  multiple experiments (e.g. the staggered MAC double-interpolation bias hits
  both jax_cfd and ins_jl on forward/baseline).
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
    ExclusionCategory.CATEGORICAL,
    "periodic FFT pressure solve + IBM volume penalization is incompatible "
    "with cylinder obstacle channel BCs",
)
INS_JL_NO_OBSTACLE = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "this tesseract represents the cylinder by hard-mask Brinkman "
    "penalization (velocity zeroed after each projection), which is "
    "projection-inconsistent and diverges; INS.jl itself supports the "
    "obstacle via a finite Brinkman forcing term with its native channel "
    "BCs (Dirichlet inflow, pressure outflow, LU/DCT pressure solvers "
    "handle non-periodic BCs), but that route is not implemented here",
)
WARP_NS_NO_OBSTACLE = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "warp-ns is periodic-only; obstacle/inflow flows are not supported",
)
OPENFOAM_NO_VJP = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "standard icoFoam has no VJP to benchmark",
)
OPENFOAM_NON_DIFFERENTIABLE_GRAD = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "standard icoFoam is non-differentiable (C++, no AD path); "
    "DAFoam/OpenFOAM-AD exist but are not deployed in this tesseract",
)
OPENFOAM_NON_DIFFERENTIABLE_OPT = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "standard icoFoam is non-differentiable forward-only solver",
)

_OBSTACLE_GATE = {
    "jax_cfd": JAX_CFD_NO_OBSTACLE,
    "ins_jl": INS_JL_NO_OBSTACLE,
    "warp_ns": WARP_NS_NO_OBSTACLE,
}


def register(problem: Problem) -> None:
    """Apply every ns-grid exclusion via :meth:`Problem.exclude`."""
    # Forward
    problem.exclude("forward/cylinder", _OBSTACLE_GATE)

    # Cost
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})

    # Gradient — suite-level, covers every gradient/* below
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})

    # Optimization — suite-level + per-experiment obstacle gates
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
    problem.exclude("optimization/drag_opt", _OBSTACLE_GATE)
    problem.exclude("optimization/drag_opt_bfgs", _OBSTACLE_GATE)
