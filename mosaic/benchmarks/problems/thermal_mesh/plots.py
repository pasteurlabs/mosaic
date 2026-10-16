# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heat-transfer plots: conductivity recovery."""

from __future__ import annotations

from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import optimization
from mosaic.benchmarks.problems.shared.plots.charts.common import render


def plot_conductivity_recovery(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "conductivity_recovery",
    **_kw: Any,
) -> Any:
    """``conductivity_recovery_convergence.png``: identification error vs iteration."""
    exp = f"{exp_key}{suffix}"
    out = render(
        cfg,
        results_dir() / cfg.name / "optimization" / exp,
        {
            "conductivity_recovery_convergence": optimization.conductivity_recovery_convergence
        },
        suite="optimization",
        experiment=exp,
        save=save,
    )
    return out.get("conductivity_recovery_convergence")
