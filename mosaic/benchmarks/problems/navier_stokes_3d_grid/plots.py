# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""3-D Navier–Stokes plots: initial-condition recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import animations, optimization
from mosaic.benchmarks.problems.shared.plots.charts.common import render


def _resolve_recovery_out_dir(base_dir: Path, ic: str | None) -> Path:
    """Resolve the experiment directory: root-level, explicit IC, or auto-detected."""
    root_result = base_dir / "result.json"
    if root_result.exists() and ic is None:
        return base_dir
    if ic is not None:
        return base_dir / ic
    # Auto-detect: look for IC subdirectories with a result.json
    ic_dirs = sorted(
        p.parent for p in base_dir.glob("*/result.json") if p.parent != base_dir
    )
    if not ic_dirs:
        raise FileNotFoundError(
            f"No result.json found in {base_dir} or its subdirectories."
        )
    return ic_dirs[0]


def plot_recovery(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    ic: str | None = None,
    exp_key: str = "optimization",
    **_kw: Any,
) -> Any:
    """IC recovery: loss convergence curves and the reconstruction animation.

    Writes ``convergence_curves.png`` (loss per sweep value, when the run
    recorded loss histories) and the animated ``recovery_evolution``.

    When results live in per-IC subdirectories (from
    ``--experiments <suite>/<exp>/<ic>`` runs), pass ``ic`` to select one; if
    the root-level ``result.json`` is missing, the first IC subdirectory is
    used.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    out_dir = _resolve_recovery_out_dir(base_dir, ic)
    out = render(
        cfg,
        out_dir,
        {
            "convergence_curves": optimization.convergence_curves,
            "recovery_evolution": animations.recovery_evolution,
        },
        suite="optimization",
        experiment=str(out_dir.relative_to(results_dir() / cfg.name / "optimization")),
        save=save,
    )
    return out.get("recovery_evolution")
