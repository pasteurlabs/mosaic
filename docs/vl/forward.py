# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-suite charts: agreement, scalar-output curves, physical laws, raw fields.

Mirror ``_agreement_figure``, ``_agreement_plot_scalar``, ``_pa_plot_fem_single``
and ``_agreement_raw_fields`` in
``mosaic/benchmarks/problems/shared/plots/forward.py``.
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

# Sweep key → axis title (plain-Unicode versions of the matplotlib labels).
_GREEK = {"nu": "ν", "mu": "μ", "rho": "ρ", "sigma": "σ", "alpha": "α"}
_SCALAR_X = {"N": "Mesh resolution N", "n_elements": "Elements per side N"}
_SCALAR_Y = {
    "output": ("Scalar output", "Solver output vs resolution"),
    "thermal_compliance": ("Thermal compliance C", "Thermal compliance vs resolution"),
    "compliance": (
        "Structural compliance C = F·u",
        "Structural compliance vs resolution",
    ),
}
_REF_DESC = {"analytic": "analytic", "converged": "converged reference"}

# Same per-problem specs as ``_PA_FEM_SPECS``.
_PA_FEM = {
    "structural-mesh": ("compliance", "F_total", "Compliance", "Structural"),
    "thermal-mesh": ("thermal_compliance", "Q_total", "Thermal compliance", "Thermal"),
}

_REF_GREY = "#aaaaaa"

# Longest side of a raw-field panel (16 panels per chart; see ``fields_raw``).
_GRID_FIELD = 32


def _npz_solvers(npz) -> list[str]:
    """``solver_names`` deduplicated by alias, as ``_dedup_solver_names``."""
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    seen, out = set(), []
    for n in (str(s) for s in npz["solver_names"]):
        key = resolve_solver_alias(n) or n
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _tooltip(x: str, x_title: str, y_title: str) -> list:
    return [
        alt.Tooltip("solver:N"),
        alt.Tooltip(f"{x}:Q", title=x_title, format=".4g"),
        alt.Tooltip("value:Q", title=y_title, format=".4e"),
    ]


# ── agreement: error vs sweep parameter ──────────────────────────────────────


def agreement(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Per-solver error vs the sweep parameter, log–log.

    Like ``_agreement_figure``: solvers in the problem's canonical order, only
    finite positive numeric errors; a linear y axis when every error lies
    within one decade.
    """
    from mosaic.benchmarks.problems.shared.plots.style import (
        resolve_solver_alias,
        solver_order_for_problem,
    )

    data = ctx.legacy()
    if data is None or not data.get("by_param"):
        return None
    by_param = data["by_param"]
    sweep_key = data.get("sweep_key", "param")
    ref_desc = _REF_DESC.get(data.get("reference_label", "consensus"), "consensus")
    order = solver_order_for_problem(ctx.config.name)
    rows = []
    for p in sorted(by_param, key=float):
        for key, entry in by_param[p].items():
            if (resolve_solver_alias(key, prefer=order) or key) not in order:
                continue
            err = entry.get("error") if isinstance(entry, dict) else None
            if isinstance(err, int | float) and math.isfinite(err) and err > 0:
                rows.append((key, float(p), float(err)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "x", "value"])
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    x_title = _GREEK.get(sweep_key, sweep_key)
    y_title = f"Error vs {ref_desc}"
    if df["value"].max() / df["value"].min() < 10:
        y = alt.Y("value:Q", scale=alt.Scale(zero=False), title=y_title)
    else:
        y = alt.Y("value:Q", scale=log_scale(), axis=log_axis(y_title))
    pick = legend_pick()
    cfg = ctx.config
    return (
        solver_lines(
            df,
            enc,
            x=alt.X(
                "x:Q", scale=log_scale(nice=False, padding=10), axis=log_axis(x_title)
            ),
            y=y,
            tooltip=_tooltip("x", x_title, "error"),
            pick=pick,
        )
        .add_params(pick)
        .properties(
            width=340,
            height=220,
            title=f"{cfg.category_label or cfg.name} — vs {sweep_key}",
        )
    )


# ── curves: scalar output vs sweep parameter ─────────────────────────────────


def curves(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Scalar solver output vs the sweep parameter (``_agreement_plot_scalar``).

    Linear x; log y when the positive outputs span more than a decade.
    """
    npz = ctx.npz("fields.npz")
    data = ctx.legacy()
    if npz is None or data is None or "sweep_values" not in npz:
        return None
    if "consensus_0" not in npz or npz["consensus_0"].ndim != 0:
        return None
    sweep = [float(v) for v in npz["sweep_values"]]
    sweep_key = data.get("sweep_key", "param")
    rows = [
        (name, x, float(npz[f"{name}_{i}"]))
        for name in _npz_solvers(npz)
        for i, x in enumerate(sweep)
        if f"{name}_{i}" in npz
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "x", "value"])
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    output_key = getattr(ctx.config, "output_key", None) or "output"
    y_title, title = _SCALAR_Y.get(
        output_key,
        (output_key.replace("_", " "), output_key.replace("_", " ").capitalize()),
    )
    x_title = _SCALAR_X.get(sweep_key, sweep_key)
    pos = df["value"][(df["value"] > 0) & np.isfinite(df["value"])]
    if len(pos) and pos.max() / pos.min() > 10:
        y = alt.Y("value:Q", scale=log_scale(), axis=log_axis(y_title))
    else:
        y = alt.Y("value:Q", scale=alt.Scale(zero=False), title=y_title)
    pick = legend_pick()
    return (
        solver_lines(
            df,
            enc,
            x=alt.X("x:Q", scale=alt.Scale(zero=False), title=x_title),
            y=y,
            tooltip=_tooltip("x", x_title, y_title),
            pick=pick,
        )
        .add_params(pick)
        .properties(width=520, height=220, title=title)
    )


# ── physical_accuracy: FEM metric vs total load ──────────────────────────────


def physical_accuracy(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Compliance vs total load, log–log, with a slope-2 reference line.

    FEM branch of ``plot_physical_laws``: the reference is calibrated on the
    mean over solvers at the first load. x ticks sit on (at most four of) the
    swept loads, as in ``_pa_set_axis_ticks``.
    """
    from mosaic.benchmarks.core.io import legacy_by_param

    spec = _PA_FEM.get(ctx.problem)
    data = ctx.legacy()
    if spec is None or data is None:
        return None
    metric, x_title, y_title, title = spec
    by_param = legacy_by_param(data)
    if not by_param:
        return None
    params = sorted(by_param, key=float)
    rows = [
        (solver, float(p), float(entry[metric]))
        for p in params
        for solver, entry in by_param[p].items()
        if isinstance(entry, dict) and entry.get(metric) is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "x", "value"])
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    x0 = float(params[0])
    c0 = float(df.loc[df["x"] == x0, "value"].mean())
    ref = pd.DataFrame({"x": [float(p) for p in params]})
    ref["value"] = c0 * (ref["x"] / x0) ** 2

    ticks = sorted(set(ref["x"]))
    if len(ticks) > 4:
        ticks = [
            ticks[i] for i in np.round(np.linspace(0, len(ticks) - 1, 4)).astype(int)
        ]
    x = alt.X(
        "x:Q",
        scale=log_scale(nice=False, padding=12),
        axis=alt.Axis(values=ticks, format=".2f", labelAngle=-40, grid=True),
        title=x_title,
    )
    y = alt.Y("value:Q", scale=log_scale(), axis=log_axis(y_title))
    reference = (
        alt.Chart(ref)
        .mark_line(color=_REF_GREY, strokeDash=[6, 3], strokeWidth=1.2)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("x:Q", title=x_title, format=".4g"),
                alt.Tooltip("value:Q", title="slope 2 reference", format=".4e"),
            ],
        )
    )
    pick = legend_pick()
    lines = solver_lines(
        df,
        enc,
        x=x,
        y=y,
        tooltip=_tooltip("x", x_title, y_title),
        pick=pick,
        legend=alt.Legend(orient="top-left", title=None, fillColor="white", padding=4),
    )
    return (
        alt.layer(reference, lines)
        .add_params(pick)
        .properties(width=240, height=260, title=title)
    )


# ── fields_raw: vorticity grid (solvers × sweep values) ──────────────────────


def fields_raw(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Vorticity of each solver's final field, rows = solvers, cols = sweep.

    ``field_grid`` gives every panel its own symmetric colour scale (RdBu_r,
    ±max|ω|); here each panel is shown relative to its own max|ω| and hover
    shows the absolute vorticity, outlining the same cell in every panel.

    Sixteen 64² panels would embed ~65k cells, so fields are subsampled to
    :data:`_GRID_FIELD` per side and the cells ship as inline CSV keyed by
    panel; solver, sweep value and max|ω| are joined back by lookup.
    """
    from mosaic.benchmarks.problems.shared.plots.style import vorticity_2d

    npz = ctx.npz("fields.npz")
    data = ctx.legacy()
    if npz is None or data is None or "sweep_values" not in npz:
        return None
    sweep_key = data.get("sweep_key", "param")
    sweep_label = _GREEK.get(sweep_key, sweep_key)
    cols = [f"{sweep_label} = {float(v):.3g}" for v in npz["sweep_values"]]
    panels, frames = [], []
    for name in _npz_solvers(npz):
        for i, col in enumerate(cols):
            key = f"{name}_{i}"
            if key not in npz or npz[key].ndim < 2:
                continue
            w = vorticity_2d(npz[key])
            step = max(1, math.ceil(max(w.shape) / _GRID_FIELD))
            vmax = float(np.abs(w).max()) or 1.0
            cells = field_cells(w[::step, ::step] / vmax, p=len(panels))
            panels.append((len(panels), name, col, vmax))
            frames.append(cells)
    if not frames:
        return None
    cells = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(panels, columns=["p", "solver", "sweep", "vmax"])
    enc = solver_encodings(meta["solver"])
    meta = enc.relabel(meta)
    csv = cells[["p", "i", "j", "value"]].to_csv(index=False, float_format="%.3f")
    source = alt.InlineData(
        values=csv,
        format=alt.DataFormat(
            type="csv",
            parse={"p": "number", "i": "number", "j": "number", "value": "number"},
        ),
    )

    n = int(cells["i"].max()) + 1
    size = cell_px(n, total=160)
    cell = linked_cell()
    return (
        alt.Chart(source)
        .transform_lookup(
            lookup="p", from_=alt.LookupData(meta, "p", ["solver", "sweep", "vmax"])
        )
        .transform_calculate(omega="datum.value * datum.vmax")
        .mark_rect()
        .encode(
            **field_xy(),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", reverse=True, domain=[-1, 1]),
                legend=alt.Legend(
                    title="ω / panel max|ω|", orient="bottom", gradientLength=240
                ),
            ),
            **cell_outline(cell, "#222222"),
            tooltip=[
                alt.Tooltip("solver:N"),
                alt.Tooltip("sweep:N", title=sweep_key),
                alt.Tooltip("omega:Q", title="ω", format=".3e"),
                alt.Tooltip("value:Q", title="ω / panel max|ω|", format=".2f"),
            ],
        )
        .add_params(cell)
        .properties(width=size, height=size)
        .facet(
            row=alt.Row("solver:N", sort=enc.labels, title=None),
            column=alt.Column("sweep:N", sort=cols, title=None),
            spacing=10,
        )
        .properties(title=f"{ctx.config.name} — solver fields")
    )


BUILDERS = {
    "agreement": agreement,
    "curves": curves,
    "physical_accuracy": physical_accuracy,
    "fields_raw": fields_raw,
}
