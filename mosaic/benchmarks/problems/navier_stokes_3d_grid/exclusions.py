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


def register(problem: Problem) -> None:
    """Apply every ns-3d-grid exclusion via :meth:`Problem.exclude`."""
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})

    # Fail closed: new experiments are out of contract until explicitly admitted.
    admitted = (
        "forward/recovery_teacher",
        "gradient/recovery_fd_check",
        "cost/recovery_forward",
        "cost/recovery_vjp",
        "optimization/recovery_constant_ic_bfgs",
        "optimization/recovery_constant_ic_bfgs_proj",
    )
    fixed_task = Exclusion(
        ExclusionCategory.CATEGORICAL,
        "surrogate trained only for N=16, nu=0.01, dt=0.02, 100-step periodic "
        "3D recovery physics; use the matched recovery experiments",
    )
    for key in problem.experiments:
        if not key.startswith("ics/") and not any(
            key == prefix or key.startswith(prefix + "/") for prefix in admitted
        ):
            problem.exclude(key, {"xlb_3d_surrogate": fixed_task})
    problem.exclude("cost/recovery_vjp", {"openfoam": OPENFOAM_NO_VJP})
