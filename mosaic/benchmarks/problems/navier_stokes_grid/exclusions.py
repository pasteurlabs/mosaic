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

STAGGERED_MAC_BIAS = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "staggered MAC grid double-interpolation: collocated TGV IC -> "
    "staggered faces -> collocated output gives sin^2(pi/N) round-trip "
    "error at all N; 35-40x above collocated peers",
)
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
XLB_MA_FLOOR = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "irreducible O(Ma²) LBM compressibility error floor: at fixed "
    "dt=0.01, Ma=u·dt/dx grows with N; at N=128 Ma~0.2 giving ~0.007 "
    "error floor (230× peers); anomalous at all N",
)
XLB_DX2_FLOOR = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "same root cause as forward/agreement/tgv — automatic k=9 sub-stepping reduces Ma 0.88→0.098 "
    "but residual O(dx²) LBM spatial discretization gives 11-24× peer errors "
    "at all nu values (0.0001–0.05); 0.0309 at nu=0.05 is 12.0× peer median; "
    "not reducible by further sub-stepping (tested k=9..27); valid=True",
)
XLB_TGV_LBM_FLOOR = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "automatic k=9 sub-steps reduce Ma 0.88→0.098 (81× Ma² reduction); "
    "errors drop from 0.216-0.278 → 0.026-0.031 (11-24× peers); "
    "remaining floor is O(dx²) LBM spatial discretization at N=64, not reducible "
    "by further sub-stepping (tested k=9..27); valid=True",
)
XLB_3D_SURROGATE_FIXED_TASK = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "trained only for the fixed N=16, nu=0.01, dt=0.02, 100-step periodic "
    "3D initial-condition recovery task; this 2D benchmark is out of contract",
)

_OBSTACLE_GATE = {
    "jax_cfd": JAX_CFD_NO_OBSTACLE,
    "ins_jl": INS_JL_NO_OBSTACLE,
    "warp_ns": WARP_NS_NO_OBSTACLE,
}


def register(problem: Problem) -> None:
    """Apply every ns-grid exclusion via :meth:`Problem.exclude`."""
    # Forward
    problem.exclude(
        "forward/baseline",
        {
            "jax_cfd": STAGGERED_MAC_BIAS,
            "ins_jl": STAGGERED_MAC_BIAS,
            "xlb": XLB_MA_FLOOR,
        },
    )
    problem.exclude(
        "forward/agreement/tgv",
        {"xlb": XLB_TGV_LBM_FLOOR},
    )
    problem.exclude("forward/tgv_nu_sweep", {"xlb": XLB_DX2_FLOOR})
    problem.exclude("forward/cylinder", _OBSTACLE_GATE)

    # Cost
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})

    # Gradient — suite-level, covers every gradient/* below
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})

    # Optimization — suite-level + per-experiment obstacle gates
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
    problem.exclude("optimization/drag_opt", _OBSTACLE_GATE)
    problem.exclude("optimization/drag_opt_bfgs", _OBSTACLE_GATE)
    # The 3D recovery surrogate shares this tesseract family but has no valid
    # 2D benchmark cell.
    _surrogate_3d_only = {
        "xlb_3d_surrogate": XLB_3D_SURROGATE_FIXED_TASK,
    }
    problem.exclude("forward", _surrogate_3d_only)
    problem.exclude("cost", _surrogate_3d_only)
    problem.exclude("gradient", _surrogate_3d_only)
    problem.exclude("optimization", _surrogate_3d_only)
