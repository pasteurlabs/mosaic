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
EXPONAX_PERIODIC_ONLY = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "a Fourier pseudo-spectral method represents only periodic fields, so it "
    "cannot express a channel with walls, an inflow and an outflow",
)
EXPONAX_EXACT_TGV = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "the Taylor–Green vortex is a single Fourier mode, which a spectral method "
    "represents exactly on every grid, so Exponax's error is float32 round-off "
    "from the first step. Round-off grows with the number of grid points and "
    "steps rather than shrinking with the grid spacing, so no positive order "
    "can be observed",
)
OPENFOAM_NO_SOURCE_TERM = Exclusion(
    ExclusionCategory.NOT_IMPLEMENTED,
    "this tesseract runs icoFoam, which has no momentum source term, so it "
    "cannot apply the body force or the drag; OpenFOAM supports both through "
    "the foamRun incompressibleFluid solver with an fvModel, which is not "
    "wired up here",
)
EXPLICIT_CENTRAL_UNSTABLE = Exclusion(
    ExclusionCategory.ANOMALY_EXPLAINED,
    "the solver advances with explicit Euler and central differences for "
    "advection, which is linearly unstable unless the timestep is below about "
    "4ν/u², a bound that does not shrink with the grid spacing. The turbulent "
    "Kolmogorov flow's speeds put that bound far below the timestep of every "
    "grid, so the solver diverges on all of them",
)
EXPLICIT_CENTRAL_UNSTABLE_CHAIN = (
    "the solver advances with explicit Euler and central differences for "
    "advection, which is linearly unstable at the timestep of the chain (see "
    "forward/kolmogorov). Over the chain's calls the unstable modes grow, so "
    "they dominate both the gradient and the finite differences, which then "
    "do not agree at any ε a float32 run resolves; where they grow fast "
    "enough, the chain from the ground-truth state diverges"
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
_OBSTACLE_GATE = {
    "jax_cfd": JAX_CFD_NO_OBSTACLE,
    "ins_jl": INS_JL_NO_OBSTACLE,
    "exponax": EXPONAX_PERIODIC_ONLY,
}


def register(problem: Problem) -> None:
    """Apply every ns-grid exclusion via :meth:`Problem.exclude`."""
    # Forward
    problem.exclude(
        "forward/baseline",
        {"pict": PICT_PISO_SPLITTING_FLOOR, "exponax": EXPONAX_EXACT_TGV},
    )
    problem.exclude("forward/cylinder", _OBSTACLE_GATE)
    problem.exclude("forward/kolmogorov_laminar", {"openfoam": OPENFOAM_NO_SOURCE_TERM})
    problem.exclude(
        "forward/kolmogorov",
        {
            "openfoam": OPENFOAM_NO_SOURCE_TERM,
            "phiflow": EXPLICIT_CENTRAL_UNSTABLE,
            "warp_ns": EXPLICIT_CENTRAL_UNSTABLE,
        },
    )

    # Cost
    problem.exclude("cost/vjp_cost", {"openfoam": OPENFOAM_NO_VJP})

    # Gradient — suite-level, covers every gradient/* below
    problem.exclude("gradient", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_GRAD})
    problem.exclude(
        "gradient/chain_fd_check",
        {
            "phiflow": Exclusion(
                ExclusionCategory.ANOMALY_EXPLAINED, EXPLICIT_CENTRAL_UNSTABLE_CHAIN
            ),
            "warp_ns": Exclusion(
                ExclusionCategory.UNSTABLE, EXPLICIT_CENTRAL_UNSTABLE_CHAIN
            ),
        },
    )

    # Optimization — suite-level + per-experiment obstacle gates
    problem.exclude("optimization", {"openfoam": OPENFOAM_NON_DIFFERENTIABLE_OPT})
    problem.exclude("optimization/drag_opt", _OBSTACLE_GATE)
    problem.exclude("optimization/drag_opt", {"pict": PICT_DRAG_OPT_TOO_SLOW})
