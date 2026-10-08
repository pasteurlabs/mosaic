# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-suite overview: wall-clock time and peak (V)RAM vs problem size.

Mirrors ``plot_cost`` in ``mosaic/benchmarks/problems/shared/plots/cost.py``
(the function that writes ``<problem>/cost/cost.png``). The data reduction is
taken from that module's helpers so the two cannot drift apart: mean trial
time, peak VRAM if above 50 MiB else RAM, the first failed sweep point drawn
as a marker at the last good value, steady-state problems without the "vs
steps" columns, and empty columns / an empty memory row dropped.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .common import (
    _SHAPE,
    Ctx,
    SolverEnc,
    faded,
    legend_pick,
    log_axis,
    log_scale,
    solver_encodings,
)

TITLE = "Wall-clock time and peak memory vs problem size"
ROW_TIME = "Wall-clock time (s)"
ROW_MEM = "Peak (V)RAM (MiB)"

# Failure legend label → Vega shape (``_FAILURE_LABEL`` / ``_FAILURE_MARKER``).
_FAILURE_SHAPE = {
    "OOM (VRAM)": _SHAPE["v"],
    "NaN gradient": _SHAPE["X"],
    "error": _SHAPE["D"],
    "timeout": _SHAPE["s"],
}

_W, _H = 140, 120


def _load(ctx: Ctx) -> tuple[tuple[dict, dict, dict], str] | None:
    """``_load_cost_inputs`` reading through ``ctx`` instead of ``results_dir()``."""
    from mosaic.benchmarks.problems.shared.plots.cost import (
        _hardware_str,
        _norm_resolution_sweep,
    )

    spatial = ctx.legacy("spatial_cost")
    temporal = ctx.legacy("temporal_cost")
    vjp_n = ctx.legacy("vjp_cost/by_N")
    vjp_s = ctx.legacy("vjp_cost/by_steps")
    if spatial is None and temporal is None and vjp_n is None and vjp_s is None:
        return None
    spatial = _norm_resolution_sweep(spatial or {})
    temporal = temporal or {}
    vjp_n = _norm_resolution_sweep(vjp_n or {})
    vjp_s = vjp_s or {}
    vjp: dict = {}
    if vjp_n.get("by_N"):
        vjp["by_N"] = vjp_n["by_N"]
        vjp.setdefault("hardware", vjp_n.get("hardware"))
    if vjp_s.get("by_steps"):
        vjp["by_steps"] = vjp_s["by_steps"]
        vjp.setdefault("hardware", vjp_s.get("hardware"))
    hw = next((_hardware_str(d) for d in (spatial, temporal, vjp) if d), "")
    return (spatial, temporal, vjp), hw


def _series_rows(col: str, row: str, xs: list[float], ys: list[float], fail) -> tuple:
    """Line rows (split at NaN like matplotlib) plus the failure connector/marker."""
    lines, seg = [], 0
    for x, y in zip(xs, ys, strict=True):
        if np.isfinite(y):
            lines.append({"col": col, "row": row, "x": x, "y": float(y), "seg": seg})
        else:
            seg += 1
    fails = []
    if fail is not None and lines:
        fx, ft = fail  # ``_draw_failure``: last finite point → (fail x, same y)
        last = lines[-1]
        fails = [
            {
                "col": col,
                "row": row,
                "x": last["x"],
                "y": last["y"],
                "ft": ft,
                "end": False,
            },
            {"col": col, "row": row, "x": fx, "y": last["y"], "ft": ft, "end": True},
        ]
    return lines, fails


def _tidy(ctx: Ctx, columns: list, mem_row: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mosaic.benchmarks.problems.shared.plots.cost import (
        _first_failure,
        _mem_vals,
        _time_vals,
    )

    lines, fails = [], []
    for _pid, by_data, _xlabel, title, keys in columns:
        xs = [float(int(k)) for k in keys]
        for spec in ctx.config.solvers:
            data = by_data.get(spec.name) or {}
            fk, ft = _first_failure(data, keys)
            fail = (xs[list(keys).index(fk)], ft) if fk is not None else None
            series = [(ROW_TIME, _time_vals(data, keys))]
            if mem_row:
                series.append((ROW_MEM, _mem_vals(data, keys)))
            for row, ys in series:
                ln, fl = _series_rows(title, row, xs, ys, fail)
                for r in ln + fl:
                    r["solver"] = spec.name
                lines += ln
                fails += fl
    return pd.DataFrame(lines), pd.DataFrame(fails)


def _device_labels(ctx: Ctx, enc: SolverEnc) -> SolverEnc:
    """Append the "(GPU)"/"(CPU)" tag of the matplotlib legend; rebuild the scales."""
    labels = [
        f"{enc.label(k)} ({'GPU' if ctx.config.solver(k).uses_gpu else 'CPU'})"
        for k in enc.keys
    ]
    return SolverEnc(
        keys=enc.keys,
        labels=labels,
        color=alt.Scale(domain=labels, range=enc.color.range),
        dash=alt.Scale(domain=labels, range=enc.dash.range),
        shape=alt.Scale(domain=labels, range=enc.shape.range),
    )


def _x_axis(xs: list[float], title: str) -> alt.Axis:
    """Decade ticks, or the sweep values when the sweep spans under two decades.

    matplotlib falls back to minor-tick labels (``2×10¹``) in that case; plain
    sweep values say the same and avoid an unlabelled axis.
    """
    lo, hi = min(xs), max(xs)
    if sum(lo <= 10.0**e <= hi for e in range(-16, 9)) >= 2:
        return log_axis(title)
    return alt.Axis(values=sorted(set(xs)), format="d", title=title)


def _panel(
    col: str,
    xs: list[float],
    xlabel: str,
    row: str,
    *,
    first_col: bool,
    top: bool,
    enc: SolverEnc,
    marks: alt.Scale,
    fail_legend: list[str],
    pick: alt.Parameter,
) -> alt.LayerChart:
    """One log–log panel: solver lines, markers, failure connectors and markers."""
    unit = "s" if row == ROW_TIME else "MiB"
    metric = "time" if row == ROW_TIME else "memory"
    x = alt.X("x:Q", scale=log_scale(nice=False, padding=8), axis=_x_axis(xs, xlabel))
    y = alt.Y("y:Q", scale=log_scale(), axis=log_axis(row if first_col else None))
    color = alt.Color(
        "solver:N",
        scale=enc.color,
        sort=enc.labels,
        legend=alt.Legend(
            orient="bottom", direction="horizontal", columns=4, title=None
        ),
    )
    dash = alt.StrokeDash("solver:N", scale=enc.dash, legend=None)
    shape_legend = (
        alt.Legend(
            orient="bottom",
            title=None,
            values=fail_legend,
            symbolFillColor="#666666",
            symbolStrokeColor="#666666",
        )
        if fail_legend
        else None
    )
    shape = alt.Shape("mark:N", scale=marks, legend=shape_legend)
    xtip = alt.Tooltip("x:Q", title=xlabel, format="d")
    here = alt.datum.col == col
    base = alt.Chart().encode(x=x, y=y, color=color, opacity=faded(pick))
    ok = base.transform_filter(here & (alt.datum.row == row) & (alt.datum.kind == "ok"))
    fl = base.transform_filter(
        here & (alt.datum.row == row) & (alt.datum.kind == "fail")
    )
    layers = [
        ok.mark_line(strokeWidth=2).encode(strokeDash=dash, detail="seg:N"),
        fl.mark_line(strokeWidth=1).encode(strokeDash=dash, detail="solver:N"),
        ok.mark_point(filled=True, size=40, opacity=1).encode(
            shape=shape,
            tooltip=[
                alt.Tooltip("solver:N"),
                xtip,
                alt.Tooltip("y:Q", title=f"{metric} ({unit})", format=".3~g"),
            ],
        ),
        fl.transform_filter(alt.datum.end)
        .mark_point(
            filled=True, size=130, stroke="white", strokeWidth=1.2, strokeOpacity=1
        )
        .encode(
            shape=shape,
            tooltip=[
                alt.Tooltip("solver:N"),
                xtip,
                alt.Tooltip("mark:N", title="failed"),
            ],
        ),
    ]
    chart = alt.layer(*layers).properties(width=_W, height=_H)
    return chart.properties(title=col) if top else chart


def build_cost(ctx: Ctx) -> alt.TopLevelMixin | None:
    """2 × n grid (time / memory × Forward·N, Forward·steps, VJP·N, VJP·steps)."""
    from mosaic.benchmarks.problems.shared.plots.cost import (
        _FAILURE_LABEL,
        _NON_TRANSIENT_PROBLEMS,
        _build_columns,
        _column_has_data,
    )

    loaded = _load(ctx)
    if loaded is None:
        return None
    (spatial, temporal, vjp), hw = loaded
    columns = _build_columns(
        spatial, temporal, vjp, "N", drop_steps=ctx.problem in _NON_TRANSIENT_PROBLEMS
    )
    has = [_column_has_data(c[1], c[4], ctx.config) for c in columns]
    columns = [c for c, (t, m) in zip(columns, has, strict=True) if t or m]
    if not columns:
        return None
    mem_row = any(m for t, m in has if t or m)

    lines, fails = _tidy(ctx, columns, mem_row)
    if lines.empty:
        return None
    enc = _device_labels(ctx, solver_encodings(lines["solver"]))
    lines = enc.relabel(lines).assign(kind="ok")
    lines["mark"] = lines["solver"]
    # Legend entries for failure types come from the time row, as in matplotlib.
    seen: set[str] = set()
    if not fails.empty:
        fails = enc.relabel(fails).assign(kind="fail")
        fails["mark"] = fails["ft"].map(lambda ft: _FAILURE_LABEL.get(ft, "error"))
        seen = set(fails.loc[fails["row"] == ROW_TIME, "ft"])
    df = pd.concat([lines, fails], ignore_index=True)
    df = df.drop(columns="ft", errors="ignore")
    df["end"] = df["end"].fillna(False).astype(bool) if "end" in df else False

    # One shape scale for solver markers and failure markers, so every panel
    # shares it and the failure legend lists only the failure types seen.
    marks = alt.Scale(
        domain=enc.labels + list(_FAILURE_SHAPE),
        range=list(enc.shape.range) + list(_FAILURE_SHAPE.values()),
    )
    fail_legend = [
        _FAILURE_LABEL[ft] for ft in ("OOM", "nan", "error", "timeout") if ft in seen
    ]

    pick = legend_pick()
    rows = [ROW_TIME] + ([ROW_MEM] if mem_row else [])
    grid = alt.vconcat(
        *(
            alt.hconcat(
                *(
                    _panel(
                        title,
                        [float(int(k)) for k in keys],
                        xlabel,
                        row,
                        first_col=i == 0,
                        top=row == ROW_TIME,
                        enc=enc,
                        marks=marks,
                        fail_legend=fail_legend,
                        pick=pick,
                    )
                    for i, (_pid, _by, xlabel, title, keys) in enumerate(columns)
                ),
                spacing=14,
            )
            for row in rows
        ),
        data=df,
        spacing=10,
    )
    title = alt.Title(TITLE, subtitle=f"Runner: {hw}" if hw else [], anchor="middle")
    return grid.add_params(pick).properties(title=title)


BUILDERS = {"cost": build_cost}
