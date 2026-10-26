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


# ── Kolmogorov solver in the loop ────────────────────────────────────────────

_ARMS = {
    "uncorrected": ("uncorrected", "#8c8c8c"),
    "supervised": ("supervised", "#4c78a8"),
    "stopped": ("stopped gradient", "#f58518"),
    "sil": ("solver in the loop", "#54a24b"),
}


def _arm_color() -> alt.Color:
    labels = [label for label, _ in _ARMS.values()]
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=labels, range=[c for _, c in _ARMS.values()]),
        sort=labels,
        legend=alt.Legend(orient="bottom", title=None),
    )


def _sil_entries(ctx: Ctx) -> list[tuple[str, dict]]:
    res = ctx.result()
    if res is None:
        return []
    return [
        (e["solver"], e["metrics"])
        for e in res.get("results", [])
        if (e.get("metrics") or {}).get("eligible")
    ]


def _sil_errors(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Median rollout error against the solver's reference, per arm and unroll."""
    rows = []
    for solver, m in _sil_entries(ctx):
        interval = m["interval"]
        unc = np.asarray(m["evaluation"]["uncorrected"]["errors"])
        for K in m["config"]["unrolls"]:
            curves = {"uncorrected": unc.reshape(-1, unc.shape[-1])}
            for arm in ("supervised", "stopped", "sil"):
                e = np.asarray(m["evaluation"][f"K{K}/{arm}"]["errors"])
                curves[arm] = e.reshape(-1, e.shape[-1])
            for arm, e in curves.items():
                for i, (med, lo, hi) in enumerate(
                    zip(np.median(e, 0), e.min(0), e.max(0), strict=True)
                ):
                    if i == 0:
                        continue
                    rows.append(
                        {
                            "solver": solver,
                            "unroll": f"K = {K}",
                            "arm": _ARMS[arm][0],
                            "time": i * interval,
                            "error": float(med),
                            "low": float(lo),
                            "high": float(hi),
                        }
                    )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(x=alt.X("time:Q", title="time"), color=_arm_color())
    band = base.mark_area(opacity=0.15).encode(
        y=alt.Y("low:Q", scale=log_scale(), title="relative error"), y2="high:Q"
    )
    line = base.mark_line(strokeWidth=2).encode(
        y="error:Q", tooltip=["solver:N", "unroll:N", "arm:N", "time:Q", "error:Q"]
    )
    return (
        alt.layer(band, line)
        .properties(width=200, height=150)
        .facet(
            column=alt.Column("solver:N", title=None),
            row=alt.Row("unroll:N", title=None),
        )
    )


def _sil_paired(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Gap closed per arm and value of the solver gradient, per seed and test state."""
    rows = []
    for solver, m in _sil_entries(ctx):
        for key, values in m["paired"].items():
            kind, K, *arm = key.split("/")
            label = (
                f"gap closed: {_ARMS[arm[0]][0]}"
                if kind == "gap_closed"
                else "value of the gradient"
            )
            rows += [
                {"solver": solver, "unroll": K, "measure": label, "value": v}
                for v in np.ravel(values)
            ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_tick(thickness=2, size=14)
        .encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("unroll:N", title=None),
            color=alt.Color("unroll:N", legend=None),
            tooltip=["solver:N", "unroll:N", "measure:N", "value:Q"],
        )
        .properties(width=180, height=70)
        .facet(
            column=alt.Column("measure:N", title=None),
            row=alt.Row("solver:N", title=None),
        )
        .resolve_scale(x="independent")
    )


def _sil_cost(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Wall time per update against the unroll, per arm."""
    rows = []
    for solver, m in _sil_entries(ctx):
        for key, cost in m["cost"].items():
            K, arm = key.split("/")
            if cost.get("update_seconds") is None:
                continue
            rows.append(
                {
                    "solver": solver,
                    "K": int(K[1:]),
                    "arm": _ARMS[arm][0],
                    "seconds": cost["update_seconds"],
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="unroll K", scale=alt.Scale(type="log", base=2)),
            y=alt.Y("seconds:Q", title="seconds per update", scale=log_scale()),
            color=_arm_color(),
            tooltip=["solver:N", "arm:N", "K:Q", "seconds:Q"],
        )
        .properties(width=180, height=150)
        .facet(column=alt.Column("solver:N", title=None))
    )


def plot_kolmogorov_sil(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "kolmogorov_sil",
    **_kw: Any,
) -> list:
    """Rollout errors, paired summaries and cost of the solver-in-the-loop arms."""
    exp = f"{exp_key}{suffix}"
    out = render(
        cfg,
        results_dir() / cfg.name / "optimization" / exp,
        {
            "kolmogorov_sil": _sil_errors,
            "kolmogorov_sil_paired": _sil_paired,
            "kolmogorov_sil_cost": _sil_cost,
        },
        suite="optimization",
        experiment=exp,
        save=save,
    )
    return list(out.values())
