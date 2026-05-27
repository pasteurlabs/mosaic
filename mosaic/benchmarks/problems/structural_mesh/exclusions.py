# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-solver exclusions for structural-mesh.

Mirrors the ns-grid layout (:mod:`mosaic.benchmarks.problems.navier_stokes_grid.exclusions`):

* Reusable :class:`Exclusion` constants for failure modes that recur across
  multiple experiments.
* :func:`register` — applies every ``problem.exclude(...)`` call, so
  ``config.py`` keeps a single ``_register_exclusions(problem)`` line and the
  long reason strings live here.

For structural-mesh the bulk of the exclusion mass is the deal.II
tesseract: it implements the forward FEM solve cleanly but ships no
VJP/AD path, so every gradient-/VJP-using experiment must mark it
categorically excluded (rather than ``not_run``) so the status score
denominator correctly omits the cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem


# ── Reusable constants ───────────────────────────────────────────────────────

DEALII_NO_VJP = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "deal.II structural tesseract is forward-only (C++/dealii-adjoint not "
    "wired); no VJP endpoint, so cost/vjp and gradient experiments are "
    "categorically out of scope.",
)
DEALII_NON_DIFFERENTIABLE_OPT = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "deal.II structural tesseract has no AD path; topology-optimisation "
    "(compliance gradient-based ρ updates) requires a differentiable "
    "compliance output.",
)


# ── Registration entry point ─────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Attach every per-solver exclusion to ``problem``.

    Called from ``config.py`` after all experiments have been registered;
    :func:`mosaic.benchmarks.core.utils.exclusion_lookup` does
    longest-prefix matching, so suite-level keys (e.g. ``"gradient"``)
    cover every sub-experiment registered under that suite.
    """
    # Cost — only the VJP variants need exclusion. Forward timing (spatial
    # / temporal) works fine on the deal.II tesseract.
    problem.exclude("cost/vjp_cost", {"dealii_structural": DEALII_NO_VJP})

    # Gradient — suite-level, covers every gradient/* below
    # (fd_check, jacobian_svd, param_sweep, …).
    problem.exclude("gradient", {"dealii_structural": DEALII_NO_VJP})

    # Optimization — topopt requires a differentiable compliance gradient.
    problem.exclude(
        "optimization", {"dealii_structural": DEALII_NON_DIFFERENTIABLE_OPT}
    )
