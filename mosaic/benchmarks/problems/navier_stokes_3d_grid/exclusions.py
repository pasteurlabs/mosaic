# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-solver exclusions for ns-3d-grid.

See the ns-grid sibling module for the rationale — one ``register`` call
keeps every long reason string out of ``config.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem


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
XLB_2D_SURROGATE_FIXED_TASK = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "trained only for the fixed N=32, Re=20 2D cylinder-flow drag-optimization "
    "task; every 3D benchmark cell is out of contract",
)
XLB_3D_SURROGATE_FIXED_TASK = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "trained only for the fixed N=16, nu=0.01, dt=0.02, 100-step periodic "
    "3D initial-condition recovery task",
)


def register(problem: Problem) -> None:
    """Apply every ns-3d-grid exclusion via :meth:`Problem.exclude`."""
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})

    # The 2D cylinder surrogate shares this tesseract family but has no valid
    # 3D benchmark cell.
    _surrogate_2d_only = {"xlb_surrogate": XLB_2D_SURROGATE_FIXED_TASK}
    problem.exclude("forward", _surrogate_2d_only)
    problem.exclude("cost", _surrogate_2d_only)
    problem.exclude("gradient", _surrogate_2d_only)
    problem.exclude("optimization", _surrogate_2d_only)

    # The 3D surrogate is admitted only to the exact recovery cell registered
    # in this suite.  All general forward, cost, and gradient sweeps vary its
    # fixed resolution, horizon, or physical parameters.
    _surrogate_3d_only = {
        "xlb_3d_surrogate": XLB_3D_SURROGATE_FIXED_TASK,
    }
    problem.exclude("forward", _surrogate_3d_only)
    problem.exclude("cost", _surrogate_3d_only)
    problem.exclude("gradient", _surrogate_3d_only)
