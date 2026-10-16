# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost-suite overview: wall-clock time and peak (V)RAM vs problem size.

Written to ``<problem>/cost/cost.png`` by ``plot_cost``. Data reduction: mean
trial time, peak VRAM if above 50 MiB else RAM, the first failed sweep point drawn
as a marker at the last good value, steady-state problems without the "vs
steps" columns, and empty columns / an empty memory row dropped.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.hardware import hardware_by_solver

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

# ── Cost-table helpers (cost table → plot rows) ──

# Steady-state problems have no time integration, so a "vs steps" cost sweep is
# meaningless — those columns are dropped from their cost figure.
_NON_TRANSIENT_PROBLEMS = {"structural-mesh", "thermal-mesh"}


_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
    "container_died": "error",
}


def _hardware_str(results: list[dict], cfg: Problem) -> str:
    """Describe the hardware behind the cost plot for its subtitle.

    GPU-capable solvers are described by the GPU they ran on, CPU-only ones
    (OpenFOAM, deal.II, FEniCS, ...) by their CPU, and the per-curve
    (GPU)/(CPU) legend tags say which is which. In CI the two kinds run on
    different machines, so each solver's own hardware record is used.
    """
    hw_by_solver: dict[str, dict] = {}
    for result in results:
        hw_by_solver.update(hardware_by_solver(result))
    gpus: list[str] = []
    cpus: list[str] = []
    for spec in cfg.solvers:
        hw = hw_by_solver.get(spec.name, {})
        if spec.uses_gpu and hw.get("gpus"):
            gpus.append(hw["gpus"][0].split(",")[0].strip())
        elif hw.get("cpu"):
            cpus.append(_short_cpu(hw["cpu"]))
    parts = []
    if gpus:
        parts.append("GPU " + " / ".join(dict.fromkeys(gpus)))
    if cpus:
        parts.append("CPU " + " / ".join(dict.fromkeys(cpus)))
    return " · ".join(parts)


def _short_cpu(cpu: str) -> str:
    """Trim a verbose CPU model string so the subtitle fits the figure width.

    ``"AMD EPYC 7V12 64-Core Processor"`` → ``"AMD EPYC 7V12"``.
    """
    for noise in ("64-Core Processor", "Processor", "CPU", "(R)", "(TM)"):
        cpu = cpu.replace(noise, "")
    return " ".join(cpu.split())


def _time_vals(row: dict, keys: list) -> list[float]:
    return [(row.get(k) or {}).get("mean", np.nan) for k in keys]


def _mem_vals(row: dict, keys: list) -> list[float]:
    """Peak (V)RAM per key — VRAM if non-zero, else RAM, else NaN."""
    out = []
    for k in keys:
        v = row.get(k) or {}
        if not isinstance(v, dict):
            out.append(np.nan)
            continue
        vram = v.get("vram_peak_mib") or 0.0
        ram = v.get("ram_peak_mib") or 0.0
        mem = vram if vram > 50 else ram
        out.append(float(mem) if mem > 0 else np.nan)
    return out


def _first_failure(row: dict, keys: list) -> tuple[str | None, str | None]:
    """Return (key, failure_type) of the first failed entry, or (None, None)."""
    for k in keys:
        v = row.get(k)
        if isinstance(v, dict) and v.get("status") == "failed":
            return k, v.get("failure_type", "error")
    return None, None


def _has_inner_data(top: dict) -> bool:
    if not isinstance(top, dict):
        return False
    return any(isinstance(v, dict) and v for v in top.values())


def _first_nonempty_keys(top: dict) -> list[str]:
    for v in top.values():
        if isinstance(v, dict) and v:
            return sorted(v.keys(), key=int)
    return []


# Resolution-sweep keys other than the canonical "N"/"n_elements" that
# v1_to_legacy already routes to ``by_N``. Mesh domains sweep their mesh
# parameter (e.g. ``nx``), which v1_to_legacy leaves under ``by_solver``; the
# cost column builder only reads ``by_N``, so without this alias the mesh cost
# plots came out empty and the suite looked NS-only.
_RESOLUTION_SWEEP_KEYS = {"nx", "ny", "nz", "n", "resolution"}


def _norm_resolution_sweep(data: dict) -> dict:
    """Alias a resolution sweep stored under ``by_solver`` to ``by_N``.

    The ``by_solver`` shape (solver → {sweep_value → metrics}) is identical to
    ``by_N``, so a shallow alias is enough. No-op when ``by_N`` already exists
    or the sweep isn't a resolution sweep.
    """
    if not isinstance(data, dict) or "by_N" in data:
        return data
    if data.get("sweep_key") in _RESOLUTION_SWEEP_KEYS and "by_solver" in data:
        data = dict(data)
        data["by_N"] = data["by_solver"]
    return data


def _build_columns(
    spatial_data: dict,
    temporal_data: dict,
    vjp_data: dict,
    res_key: str,
    *,
    drop_steps: bool = False,
) -> list[tuple[str, dict, str, str, list]]:
    """Assemble the column specs in display order, skipping empty ones.

    When ``drop_steps`` is set (steady-state problems), the time-integration
    "vs steps" columns are omitted entirely, leaving only the N / size sweep.
    """
    columns: list[tuple[str, dict, str, str, list]] = []

    if _has_inner_data(spatial_data.get("by_N", {})):
        columns.append(
            (
                "spatial_N",
                spatial_data["by_N"],
                res_key,
                "Forward · N",
                _first_nonempty_keys(spatial_data["by_N"]),
            )
        )

    if not drop_steps and _has_inner_data(temporal_data.get("by_steps", {})):
        columns.append(
            (
                "temporal_steps",
                temporal_data["by_steps"],
                "steps",
                "Forward · steps",
                _first_nonempty_keys(temporal_data["by_steps"]),
            )
        )

    if _has_inner_data(vjp_data.get("by_N", {})):
        columns.append(
            (
                "vjp_N",
                vjp_data["by_N"],
                res_key,
                "VJP · N",
                _first_nonempty_keys(vjp_data["by_N"]),
            )
        )

    if not drop_steps and _has_inner_data(vjp_data.get("by_steps", {})):
        columns.append(
            (
                "vjp_steps",
                vjp_data["by_steps"],
                "steps",
                "VJP · steps",
                _first_nonempty_keys(vjp_data["by_steps"]),
            )
        )

    return columns


def _column_has_data(by_data: dict, keys: list, cfg: Problem) -> tuple[bool, bool]:
    """Return ``(has_time, has_mem)`` for a column across all solvers."""
    has_time = False
    has_mem = False
    for spec in cfg.solvers:
        row = by_data.get(spec.name) or {}
        if any(np.isfinite(v) for v in _time_vals(row, keys)):
            has_time = True
        if any(np.isfinite(v) for v in _mem_vals(row, keys)):
            has_mem = True
    return has_time, has_mem


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
    if vjp_s.get("by_steps"):
        vjp["by_steps"] = vjp_s["by_steps"]
    subs = ("spatial_cost", "temporal_cost", "vjp_cost/by_N", "vjp_cost/by_steps")
    results = [r for r in map(ctx.result, subs) if r is not None]
    return (spatial, temporal, vjp), _hardware_str(results, ctx.config)


def _series_rows(
    col: str, row: str, xs: list[float], ys: list[float], fail: Any
) -> tuple:
    """Line rows (split at NaN) plus the failure connector/marker."""
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
    """Append the "(GPU)"/"(CPU)" hardware tag to the legend; rebuild the scales."""
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

    Plain sweep values avoid an unlabelled axis on a narrow log range.
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
    # Legend entries for failure types come from the time row, so each type appears once.
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
    title = alt.Title(TITLE, subtitle=f"Hardware: {hw}" if hw else [], anchor="middle")
    return grid.add_params(pick).properties(title=title)


BUILDERS = {"cost": build_cost}
