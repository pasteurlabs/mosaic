# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost suite plot: wall-clock time and peak memory vs problem size."""

from __future__ import annotations

from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import cost as charts
from mosaic.benchmarks.problems.shared.plots.charts.common import render

_SUITE = "cost"


def plot_cost(cfg: Problem, *, save: bool = True, suffix: str = "", **_kw: Any) -> Any:
    """Suite-level ``cost/cost.png``: forward/VJP time and memory vs N and steps.

    Registered on every cost experiment; each call redraws the same overview
    from all cost results present.
    """
    out = render(
        cfg,
        results_dir() / cfg.name / _SUITE,
        {f"cost{suffix}": charts.build_cost},
        suite=_SUITE,
        save=save,
    )
    return out.get(f"cost{suffix}")
