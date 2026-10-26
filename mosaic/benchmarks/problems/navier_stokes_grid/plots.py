# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-D Navier–Stokes plots: drag optimization and the Kolmogorov-flow forward check."""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import results_dir
from mosaic.benchmarks.problems.shared.plots.charts import animations, optimization
from mosaic.benchmarks.problems.shared.plots.charts.common import (
    Ctx,
    legend_pick,
    log_scale,
    render,
    solver_encodings,
    solver_lines,
)


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


def _kolmogorov_forward(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Median error and correlation vs time, coarse and reference configurations."""
    res = ctx.result()
    if res is None:
        return None
    rows = []
    for entry in res.get("results", []):
        m = entry.get("metrics") or {}
        if not m.get("valid"):
            continue
        for config in ("coarse", "reference"):
            per_seed = (m.get(config) or {}).get("per_seed") or []
            if not per_seed:
                continue
            # result.json stores NaN as null.
            errors = np.median(
                np.array([s["errors"] for s in per_seed], dtype=float), axis=0
            )
            corr = np.median(
                np.array([s["correlations"] for s in per_seed], dtype=float), axis=0
            )
            for t, e, c in zip(m["times"], errors, corr, strict=True):
                rows.append(
                    {
                        "solver": entry["solver"],
                        "configuration": config,
                        "time": t,
                        "error": float(e),
                        "correlation": float(c),
                    }
                )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    enc = solver_encodings(df["solver"].unique())
    df = enc.relabel(df)
    pick = legend_pick()
    x = alt.X("time:Q", title="time")
    panels = []
    for col, title, scale in (
        ("error", "relative error", log_scale()),
        ("correlation", "vorticity correlation", alt.Scale()),
    ):
        panels.append(
            solver_lines(
                None,
                enc,
                x=x,
                y=alt.Y(f"{col}:Q", title=title, scale=scale),
                tooltip=["solver:N", "configuration:N", "time:Q", f"{col}:Q"],
                pick=pick,
            )
            .properties(width=220, height=180)
            .facet(column=alt.Column("configuration:N", title=None), data=df)
        )
    return alt.vconcat(*panels).add_params(pick)


def plot_kolmogorov(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "kolmogorov",
    **_kw: Any,
) -> list:
    """Coarse and reference configurations against the Kolmogorov ground truth."""
    exp = f"{exp_key}{suffix}"
    out = render(
        cfg,
        results_dir() / cfg.name / "forward" / exp,
        {"kolmogorov": _kolmogorov_forward},
        suite="forward",
        experiment=exp,
        save=save,
    )
    return list(out.values())
