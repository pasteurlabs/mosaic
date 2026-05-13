"""Per-solver exclusions for ns-3d-grid.

See the ns-grid sibling module for the rationale — one ``register`` call
keeps every long reason string out of ``config.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem


PHIFLOW_3D_CUDA_OOM = Exclusion(
    ExclusionCategory.INFEASIBLE,
    "CUDA OOM during JAX CUDA graph profiling in 3D cost benchmark "
    "(allocate 16.09MiB failed); xla_gpu_autotune_level=0 fix deployed "
    "— pending re-run to confirm resolved",
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


def register(problem: Problem) -> None:
    """Apply every ns-3d-grid exclusion via :meth:`Problem.exclude`."""
    problem.exclude("cost/temporal_cost", {"phiflow": PHIFLOW_3D_CUDA_OOM})
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
