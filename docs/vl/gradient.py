# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-suite charts: finite-difference check and gradient fields.

Mirror ``_fd_check_figure`` and the field part of ``plot_fd_check`` in
``mosaic/benchmarks/problems/shared/plots/gradient.py``.
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd

from .common import (
    Ctx,
    cell_outline,
    cell_px,
    field_cells,
    field_xy,
    legend_pick,
    linked_cell,
    log_axis,
    log_scale,
    solver_encodings,
    solver_lines,
)

# Same exclusions as the matplotlib figure.
_FD_CHECK_BLACKLIST = {"fenics_ns", "su2"}

METRIC_ERR = "relative FD error"
METRIC_COS = "1 − cos(∇AD, ∇FD)"


def fd_check(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Rel. FD error and 1 − cos vs ε, one panel each, log–log."""
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    data = ctx.legacy()
    if data is None:
        return None
    rows = []
    for solver, sdata in data["by_solver"].items():
        alias = resolve_solver_alias(solver)
        if solver in _FD_CHECK_BLACKLIST or alias in _FD_CHECK_BLACKLIST:
            continue
        for eps, entry in (sdata.get("eps_sweep") or {}).items():
            re = entry["rel_error"]
            vals = [
                v for v in (re if isinstance(re, list) else [re]) if math.isfinite(v)
            ]
            if vals:
                rows.append((solver, float(eps), METRIC_ERR, float(np.mean(vals))))
            rows.append(
                (solver, float(eps), METRIC_COS, max(1 - float(entry["cosine"]), 1e-9))
            )
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "eps", "metric", "value"])
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    pick = legend_pick()
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    lines = solver_lines(
        None,
        enc,
        x=alt.X("eps:Q", scale=log_scale(), axis=log_axis("perturbation size ε")),
        y=alt.Y("value:Q", scale=log_scale(), axis=log_axis(None)),
        tooltip=[
            alt.Tooltip("solver:N"),
            alt.Tooltip("eps:Q", title="ε", format=".0e"),
            alt.Tooltip("metric:N"),
            alt.Tooltip("value:Q", format=".3e"),
        ],
        pick=pick,
    )
    return (
        alt.layer(lines, data=df)
        .add_params(pick, zoom)
        .properties(width=300, height=220)
        .facet(column=alt.Column("metric:N", sort=[METRIC_ERR, METRIC_COS], title=None))
        .resolve_scale(y="independent")
        .properties(title="Gradient check vs finite differences (valley = accurate)")
    )


def gradient_fields(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Initial condition next to each solver's gradient magnitude ∂L/∂IC.

    Same transforms as ``plot_fd_check`` with its defaults: vorticity of the
    IC (velocity fields) and ``grad_magnitude_2d`` of each gradient. Each
    gradient panel is shown relative to its own maximum (the PNG uses one
    colourbar per panel); hover shows absolute values.
    """
    from mosaic.benchmarks.problems.shared.plots.style import (
        grad_magnitude_2d,
        vorticity_2d,
    )

    npz = ctx.npz("gradient_fields.npz")
    if npz is None or "ic" not in npz:
        return None
    ic = npz["ic"]
    if ic.ndim != 4:
        return None  # only velocity fields have the IC panel in the PNG
    names = [str(n) for n in npz["solver_names"]]
    frames = []
    for j, name in enumerate(names):
        key = f"grad_{j}"
        if key not in npz:
            continue
        g = grad_magnitude_2d(npz[key])
        vmax = float(np.max(g)) or 1.0
        cells = field_cells(g, solver=name)
        cells["rel"] = cells["value"] / vmax
        frames.append(cells)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    w = vorticity_2d(ic)
    ic_df = field_cells(w)
    n = int(ic_df["i"].max()) + 1
    size = cell_px(n)
    cell = linked_cell()
    if float(w.min()) >= -1e-6:
        ic_scale = alt.Scale(scheme="greys", domain=[0, max(float(w.max()), 1.0)])
    else:
        vm = float(np.abs(w).max()) or 1.0
        ic_scale = alt.Scale(scheme="redblue", reverse=True, domain=[-vm, vm])

    ic_panel = (
        alt.Chart(ic_df, title=alt.Title("initial condition (vorticity)", fontSize=12))
        .mark_rect()
        .encode(
            **field_xy(),
            color=alt.Color(
                "value:Q",
                scale=ic_scale,
                legend=alt.Legend(title="ω", orient="bottom", gradientLength=size - 20),
            ),
            **cell_outline(cell, "#222222"),
            tooltip=[alt.Tooltip("value:Q", title="ω", format=".3e")],
        )
        .properties(width=size, height=size)
    )
    grads = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            **field_xy(),
            color=alt.Color(
                "rel:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                legend=alt.Legend(
                    title="|∂L/∂IC| / panel max", orient="bottom", gradientLength=220
                ),
            ),
            **cell_outline(cell),
            tooltip=[
                alt.Tooltip("solver:N"),
                alt.Tooltip("value:Q", title="|∂L/∂IC|", format=".3e"),
            ],
        )
        .add_params(cell)
        .properties(width=size, height=size)
        .facet(facet=alt.Facet("solver:N", sort=enc.labels, title=None), columns=3)
    )
    return (
        alt.hconcat(ic_panel, grads, spacing=28)
        .resolve_scale(color="independent")
        .properties(title="Gradient magnitude ∂L/∂IC per solver")
    )


BUILDERS = {
    "fd_check": fd_check,
    "gradient_fields": gradient_fields,
}
