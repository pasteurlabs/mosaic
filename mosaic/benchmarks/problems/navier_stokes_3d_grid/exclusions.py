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
    "standard icoFoam is a non-differentiable, forward-only solver",
)
STAGGERED_ROUND_TRIP = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "the solver stores velocity on cell faces, so the cell-centered initial "
    "condition is averaged onto the faces and the result averaged back. Each "
    "average scales a mode of wavenumber k by cos(πk/N), so the round trip "
    "alone leaves an error that does not shrink with viscosity. On this "
    "coarse grid it exceeds the peers' errors, which fall as viscosity rises",
)


def register(problem: Problem) -> None:
    """Apply every ns-3d-grid exclusion via :meth:`Problem.exclude`."""
    problem.exclude(
        "forward/agreement",
        {
            "ins_jl": STAGGERED_ROUND_TRIP,
            "jax_cfd": STAGGERED_ROUND_TRIP,
            "phiflow": STAGGERED_ROUND_TRIP,
        },
    )
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
