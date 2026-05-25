"""Per-solver exclusions for thermal-mesh.

Mirrors the structural-mesh / ns-grid layout: reusable
:class:`Exclusion` constants plus a :func:`register` entry point called
once from ``config.py``.

The deal.II thermal tesseract is forward-only — the C++ solver ships no
AD path, so every gradient-/VJP-using experiment marks it categorically
excluded (rather than letting it surface as ``not_run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mosaic.benchmarks.core.config import Exclusion, ExclusionCategory

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem


# ── Reusable constants ───────────────────────────────────────────────────────

DEALII_NO_VJP = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "deal.II thermal tesseract is forward-only (no AD path in the C++ "
    "solver); cost/vjp and gradient experiments are categorically out of "
    "scope.",
)
DEALII_NON_DIFFERENTIABLE_OPT = Exclusion(
    ExclusionCategory.CATEGORICAL,
    "deal.II thermal tesseract has no AD path; conductivity-recovery "
    "(gradient-based ρ updates against a target temperature field) "
    "requires a differentiable thermal-compliance output.",
)


# ── Registration entry point ─────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Attach per-solver exclusions; called once from ``config.py``.

    Uses suite-level keys (e.g. ``"gradient"``) so a single
    ``problem.exclude(...)`` covers every sub-experiment via
    :func:`mosaic.benchmarks.core.utils.exclusion_lookup`'s
    longest-prefix matching.
    """
    problem.exclude("cost/vjp_cost", {"dealii_heat": DEALII_NO_VJP})
    problem.exclude("gradient", {"dealii_heat": DEALII_NO_VJP})
    problem.exclude("optimization", {"dealii_heat": DEALII_NON_DIFFERENTIABLE_OPT})
