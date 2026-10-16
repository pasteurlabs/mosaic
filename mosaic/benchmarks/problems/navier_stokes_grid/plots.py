# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-D Navier–Stokes plots: inflow optimization for drag minimisation."""

from __future__ import annotations

from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import animations, optimization
from mosaic.benchmarks.problems.shared.plots.charts.common import render


def plot_drag_opt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "drag_opt",
    **_kw: Any,
) -> list:
    """Drag optimization: convergence, flow fields and inflow-profile animation.

    Writes ``drag_opt.png`` (reduction, optimised profile, profile history),
    ``drag_opt_fields.png`` (initial vs optimised flow) and the animated
    ``drag_opt_evolution`` (inflow profile per snapshot).
    """
    exp = f"{exp_key}{suffix}"
    out = render(
        cfg,
        results_dir() / cfg.name / "optimization" / exp,
        {
            "drag_opt": optimization.drag_opt,
            "drag_opt_fields": optimization.drag_opt_fields,
            "drag_opt_evolution": animations.drag_opt_evolution,
        },
        suite="optimization",
        experiment=exp,
        save=save,
    )
    return list(out.values())
