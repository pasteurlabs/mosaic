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
    """Error at the last time against wall time, and error per time on each grid."""
    res = ctx.result()
    if res is None:
        return None
    frontier, history, last = [], [], None
    for entry in res.get("results", []):
        m = entry.get("metrics") or {}
        if not m.get("valid"):
            continue
        last = m["times"][-1]
        for run in m.get("runs") or []:
            # result.json stores NaN as null.
            errors = np.array(run["median_errors"], dtype=float)
            seconds = run.get("seconds_per_time")
            if seconds is not None and not run["diverged"]:
                frontier.append(
                    {
                        "solver": entry["solver"],
                        "grid": run["grid"],
                        "seconds": float(seconds),
                        "error": float(errors[-1]),
                    }
                )
            for t, e in zip(m["times"], errors, strict=True):
                history.append(
                    {
                        "solver": entry["solver"],
                        "grid": f"{run['grid']}²",
                        "order": run["grid"],
                        "time": t,
                        "error": float(e),
                    }
                )
    if not history:
        return None
    enc = solver_encodings(pd.unique(pd.Series([r["solver"] for r in history])))
    pick = legend_pick()
    panels = []
    if frontier:
        df = enc.relabel(pd.DataFrame(frontier))
        panels.append(
            solver_lines(
                df,
                enc,
                x=alt.X(
                    "seconds:Q",
                    title="wall time per time unit (s)",
                    scale=log_scale(),
                ),
                y=alt.Y(
                    "error:Q",
                    title=f"relative error at t = {last:g}",
                    scale=log_scale(),
                ),
                tooltip=["solver:N", "grid:Q", "seconds:Q", "error:Q"],
                pick=pick,
            ).properties(width=360, height=240)
        )
    df = enc.relabel(pd.DataFrame(history))
    panels.append(
        solver_lines(
            None,
            enc,
            x=alt.X("time:Q", title="time"),
            y=alt.Y("error:Q", title="relative error", scale=log_scale()),
            tooltip=["solver:N", "grid:N", "time:Q", "error:Q"],
            pick=pick,
        )
        .properties(width=160, height=160)
        .facet(
            column=alt.Column("grid:N", title=None, sort=alt.SortField("order")),
            data=df,
        )
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
    """Each solver's grids against the Kolmogorov ground truth."""
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


# ── Kolmogorov solver in the loop ────────────────────────────────────────────

_ARMS = {
    "supervised": ("supervised", "#4c78a8"),
    "stopped": ("stopped gradient", "#f58518"),
    "sil": ("solver in the loop", "#54a24b"),
}


def _sil_cost(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Wall time per training update, per solver and arm."""
    res = ctx.result()
    rows = []
    for e in (res or {}).get("results", []):
        m = e.get("metrics") or {}
        if not m.get("eligible"):
            continue
        for key, cost in m["cost"].items():
            K, arm = key.split("/")
            if cost.get("update_seconds") is None:
                continue
            rows.append(
                {
                    "solver": e["solver"],
                    "unroll": int(K[1:]),
                    "arm": _ARMS[arm][0],
                    "seconds": cost["update_seconds"],
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    enc = solver_encodings(df["solver"].unique())
    df = enc.relabel(df)
    arms = [label for label, _ in _ARMS.values()]
    return (
        alt.Chart(df)
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X("seconds:Q", title="seconds per update", scale=log_scale()),
            y=alt.Y("solver:N", title=None, sort=enc.labels),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=arms, range=[c for _, c in _ARMS.values()]),
                sort=arms,
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["solver:N", "arm:N", "unroll:Q", "seconds:Q"],
        )
        .properties(width=360, height=alt.Step(28))
    )


def plot_kolmogorov_sil(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    **_kw: Any,
) -> list:
    """Cost per training update of the solver-in-the-loop arms."""
    exp = f"kolmogorov_sil{suffix}"
    out = render(
        cfg,
        results_dir() / cfg.name / "optimization" / exp,
        {"kolmogorov_sil_cost": _sil_cost},
        suite="optimization",
        experiment=exp,
        save=save,
    )
    return list(out.values())
