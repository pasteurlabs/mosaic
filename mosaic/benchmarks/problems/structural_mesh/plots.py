# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural-mechanics plots: topology optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import animations, optimization
from mosaic.benchmarks.problems.shared.plots.charts.common import render


def plot_topopt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "topopt",
    **_kw: Any,
) -> Any:
    """Topology optimization: compliance, densities and density evolution.

    Writes ``topopt.png`` (compliance vs iteration), ``topopt_fields.png``
    (initial and final densities), ``topopt_3d.png`` (final densities with
    BCs, slice view) and one animated ``topopt_evolution_<solver>`` per solver
    with a density history.
    """
    exp = f"{exp_key}{suffix}"
    out_dir = results_dir() / cfg.name / "optimization" / exp
    builders = {
        "topopt": optimization.topopt,
        "topopt_fields": optimization.topopt_fields,
        "topopt_3d": optimization.topopt_3d,
    }
    fields = out_dir / "topopt_fields.npz"
    if fields.exists():
        with np.load(fields, allow_pickle=False) as npz:
            names = (
                [str(n) for n in npz["solver_names"]]
                if "solver_names" in npz.files
                else []
            )
        for name in dict.fromkeys(names):
            builders[f"topopt_evolution_{name}"] = animations.topopt_evolution
    out = render(
        cfg, out_dir, builders, suite="optimization", experiment=exp, save=save
    )
    return out.get("topopt")
