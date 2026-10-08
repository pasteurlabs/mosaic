# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-suite sweep charts: parameter, horizon and Jacobian-spectrum sweeps.

Mirror ``plot_param_sweep`` / ``_plot_best_eps_overlay``,
``_horizon_sweep_figure`` and ``plot_jacobian_svd_comparison`` in
``mosaic/benchmarks/problems/shared/plots/gradient.py``, and
``_plot_horizon_sweep_limits`` in
``mosaic/benchmarks/problems/navier_stokes_3d_grid/extras.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .common import (
    Ctx,
    SolverEnc,
    faded,
    legend_pick,
    log_axis,
    log_scale,
    solver_encodings,
    solver_legend,
    solver_lines,
)

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# matplotlib's "X" marker (failure markers), as a Vega SVG path.
# Thicker arms than common's "X" so the white edge doesn't swallow it.
_X_PATH = "M-1,-.6L-.6,-1L0,-.4L.6,-1L1,-.6L.4,0L1,.6L.6,1L0,.4L-.6,1L-1,.6L-.4,0Z"

# ``_SWEEP_MATH_LABELS`` in Unicode. ``rho_0`` is added: the matplotlib map
# only knows ``rho0``, so the FEM density sweeps fall back to the raw key.
_SWEEP_LABELS = {
    "nu": "ν  (viscosity)",
    "rho0": "ρ₀  (density)",
    "rho_0": "ρ₀  (density)",
    "rho": "ρ  (density)",
    "density": "ρ₀  (density)",
    "source_width": "σ  (source width)",
    "sigma": "σ",
    "steps": "rollout steps T",
}

ROLLOUT = "rollout steps T"


def _finite(v: Any) -> float:
    """``_finite_or_nan``: ``None`` / non-numeric / inf → nan."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _best_eps(eps_sweep: dict, key: str, pick: Any) -> tuple[float, float]:
    """``(best value, its ε)`` of ``key`` over the ε sweep (``pick`` = min/max)."""
    vals = [(_finite(e.get(key)), float(eps)) for eps, e in eps_sweep.items()]
    vals = [(v, eps) for v, eps in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return pick(vals, key=lambda t: t[0])


def _log_axis_for(values, title: str | None, **kw: Any) -> alt.Axis:
    """:func:`log_axis` when the data spans ≥ 2 decade ticks, else ``m×10ⁿ`` ticks.

    matplotlib labels minor ticks on narrow log ranges; decade-only ticks would
    leave such an axis unlabelled.
    """
    v = np.asarray([x for x in values if x > 0 and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return log_axis(title, **kw)
    lo, hi = float(v.min()), float(v.max())
    e_lo, e_hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    if sum(lo <= 10.0**e <= hi for e in range(e_lo, e_hi + 1)) >= 2:
        return log_axis(title, lo=e_lo - 1, hi=e_hi + 1, **kw)
    ticks = []
    for e in range(e_lo - 1, e_hi + 1):
        for m in (1, 2, 3, 4, 6, 8):
            t = m * 10.0**e
            if lo / 1.15 <= t <= hi * 1.15:
                ticks.append((t, m, e))
    sup = {e: str(e).translate(_SUPERSCRIPT) for _, _, e in ticks}
    expr = " : ".join(
        f"datum.value == {t!r} ? '{'' if m == 1 else f'{m}×'}10{sup[e]}'"
        for t, m, e in ticks
    )
    return alt.Axis(
        values=[t for t, _, _ in ticks], labelExpr=f"{expr} : ''", title=title, **kw
    )


def _pad_log(lo: float, hi: float, frac: float = 0.06) -> list[float]:
    """Log-axis limits padded by ``frac`` of the log span (matplotlib margins)."""
    span = max(math.log10(hi) - math.log10(lo), 0.3)
    return [lo / 10 ** (frac * span), hi * 10 ** (frac * span)]


def _log_y(field: str, values, title: str | None) -> alt.Y:
    """Log y over the data range plus 5 % margins (no rounding out to decades)."""
    v = [float(x) for x in values if x > 0 and math.isfinite(x)]
    dom = _pad_log(min(v), max(v), 0.05) if v else None
    scale = log_scale(domain=dom, nice=False) if dom else log_scale()
    return alt.Y(field, scale=scale, axis=_log_axis_for(v, title))


# ── best_eps_vs_param ────────────────────────────────────────────────────────


def best_eps_vs_param(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Best-ε relative FD error vs the sweep parameter, all solvers, log–log.

    Same reduction as ``_plot_best_eps_overlay``: per sweep value the minimum
    finite ``rel_error_mean`` over the ε sweep; sweep values are the first
    solver's keys. The x axis is ticked at the sweep values.
    """
    data = ctx.legacy()
    by_solver = (data or {}).get("by_solver") or {}
    if not by_solver:
        return None
    sweep_key = data.get("sweep_key", "param")
    x_label = _SWEEP_LABELS.get(sweep_key, sweep_key)
    param_vals = sorted(next(iter(by_solver.values())).keys(), key=float)

    rows = []
    for name, results in by_solver.items():
        for k in param_vals:
            r = results.get(k)
            if not (isinstance(r, dict) and "eps_sweep" in r):
                continue
            err, eps = _best_eps(r["eps_sweep"], "rel_error_mean", min)
            if math.isfinite(err):
                rows.append((name, float(k), err, eps))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "param", "err", "eps"])
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)

    xs = sorted(df["param"].unique())
    pick = legend_pick()
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    lines = solver_lines(
        df,
        enc,
        x=alt.X(
            "param:Q",
            scale=log_scale(domain=_pad_log(xs[0], xs[-1], 0.05)),
            axis=alt.Axis(values=xs, format="~g", title=x_label, grid=False),
        ),
        y=_log_y("err:Q", df["err"], "relative FD error (best ε)"),
        tooltip=[
            alt.Tooltip("solver:N"),
            alt.Tooltip("param:Q", title=x_label.split()[0], format="~g"),
            alt.Tooltip("err:Q", title="rel. FD error", format=".3e"),
            alt.Tooltip("eps:Q", title="best ε", format=".0e"),
        ],
        pick=pick,
    )
    return lines.add_params(pick, zoom).properties(
        width=560,
        height=220,
        title=f"Gradient accuracy vs {x_label}: lower is better",
    )


# ── horizon_sweep ────────────────────────────────────────────────────────────

# Same exclusions / jitter as the matplotlib figure.
_HORIZON_EXCLUDED = {"fenics_ns", "su2", "openfoam"}
_HORIZON_JITTER_LOG = 0.04
_NAN_GRADIENT = "NaN gradient"


def _jitter(fail_at_step: dict[int, list[str]]) -> dict[tuple[str, int], float]:
    """Spread coincident failure markers by ±0.04 decades (matplotlib jitter)."""
    out: dict[tuple[str, int], float] = {}
    for step, here in fail_at_step.items():
        n = len(here)
        for i, s in enumerate(here):
            off = 0.0 if n == 1 else (2 * i / (n - 1) - 1) * _HORIZON_JITTER_LOG
            out[(s, step)] = step * 10**off
    return out


def _circles(base: alt.Chart, enc: SolverEnc, pick: alt.Parameter) -> alt.Chart:
    """Round markers in solver colour (the sweep figures use ``o`` for all)."""
    return base.mark_point(filled=True, size=30, shape="circle").encode(
        color=alt.Color("solver:N", scale=enc.color, sort=enc.labels, legend=None),
        opacity=faded(pick),
    )


def horizon_sweep(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Gradient norm, best-ε FD error and 1 − cosine vs rollout steps T.

    ``_horizon_plot_curves``: per T the min ``rel_error_mean`` and max
    ``cosine_mean`` over ε; T values with a non-finite / non-positive gradient
    norm or FD error become × markers at the solver's last valid value.
    """
    from mosaic.benchmarks.core.io import legacy_by_solver
    from mosaic.benchmarks.problems.shared.plots.style import (
        NS_ORDER,
        resolve_solver_alias,
    )

    data = ctx.legacy()
    by_solver = legacy_by_solver(data) if data else {}
    if not by_solver:
        return None
    alias_to_display = {}
    for name in by_solver:
        a = resolve_solver_alias(name)
        if a is not None:
            alias_to_display[a] = name
    ordered = [
        a for a in NS_ORDER if a in alias_to_display and a not in _HORIZON_EXCLUDED
    ]

    fail_at_step: dict[int, list[str]] = defaultdict(list)
    for a in ordered:
        for k, v in by_solver[alias_to_display[a]].items():
            gn = _finite(v.get("grad_norm", 1.0))
            if not math.isfinite(gn) or gn <= 0:
                fail_at_step[int(k)].append(a)
    jitter_x = _jitter(fail_at_step)

    ok_rows, fail_rows = [], []
    for a in ordered:
        sv = by_solver[alias_to_display[a]]
        ok, fails = [], []
        for k in sorted(sv, key=int):
            v = sv[k]
            gn = _finite(v.get("grad_norm", float("nan")))
            eps_sweep = v.get("eps_sweep", {})
            err, err_eps = _best_eps(eps_sweep, "rel_error_mean", min)
            cos, cos_eps = _best_eps(eps_sweep, "cosine_mean", max)
            if math.isfinite(gn) and gn > 0 and math.isfinite(err) and err > 0:
                ok.append(
                    (a, int(k), int(k), gn, err, err_eps, max(1 - cos, 1e-12), cos_eps)
                )
            else:
                fails.append(int(k))
        ok_rows += ok
        if ok:
            last = ok[-1]
            for fs in fails:
                fail_rows.append((a, jitter_x.get((a, fs), float(fs)), fs, *last[3:]))
    if not ok_rows:
        return None
    cols = ["solver", "x", "T", "gn", "err", "err_eps", "cos", "cos_eps"]
    ok_df = pd.DataFrame(ok_rows, columns=cols)
    fail_df = pd.DataFrame(fail_rows, columns=cols).assign(failure=_NAN_GRADIENT)
    enc = solver_encodings(ok_df["solver"])
    ok_df, fail_df = enc.relabel(ok_df), enc.relabel(fail_df)

    pick = legend_pick()
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    x = alt.X(
        "x:Q",
        scale=log_scale(domain=_pad_log(ok_df["T"].min(), ok_df["T"].max(), 0.05)),
        axis=log_axis(ROLLOUT, lo=0, hi=5),
    )
    t_tip = alt.Tooltip("T:Q", title="T")
    panels = [
        (
            "Gradient norm",
            "gn",
            "‖∇ℒ‖",
            [alt.Tooltip("gn:Q", title="‖∇ℒ‖", format=".4g")],
        ),
        (
            "FD relative error (best ε)",
            "err",
            "relative FD error",
            [
                alt.Tooltip("err:Q", title="rel. FD error", format=".3e"),
                alt.Tooltip("err_eps:Q", title="best ε", format=".0e"),
            ],
        ),
        (
            "Cosine similarity (best ε)",
            "cos",
            "1 − cosine",
            [
                alt.Tooltip("cos:Q", title="1 − cosine", format=".3e"),
                alt.Tooltip("cos_eps:Q", title="best ε", format=".0e"),
            ],
        ),
    ]
    charts = []
    for i, (title, field, ytitle, tips) in enumerate(panels):
        vals = pd.concat([ok_df[field], fail_df[field]])
        y = _log_y(f"{field}:Q", vals, ytitle)
        tooltip = [alt.Tooltip("solver:N"), t_tip, *tips]
        lines = solver_lines(
            None,
            enc,
            x=x,
            y=y,
            tooltip=tooltip,
            pick=pick,
            points=False,
            legend=solver_legend(enc),
        )
        layers = [
            lines,
            _circles(alt.Chart(), enc, pick).encode(x=x, y=y, tooltip=tooltip),
        ]
        if not fail_df.empty:
            layers.append(
                alt.Chart(fail_df)
                .mark_point(filled=True, size=150, stroke="white", strokeWidth=0.6)
                .encode(
                    x=x,
                    y=y,
                    color=alt.Color(
                        "solver:N", scale=enc.color, sort=enc.labels, legend=None
                    ),
                    shape=alt.Shape(
                        "failure:N",
                        scale=alt.Scale(domain=[_NAN_GRADIENT], range=[_X_PATH]),
                        legend=alt.Legend(
                            title=None,
                            orient="bottom",
                            symbolFillColor="#666666",
                            symbolStrokeWidth=0,
                        ),
                    ),
                    opacity=faded(pick),
                    tooltip=[alt.Tooltip("solver:N"), t_tip, alt.Tooltip("failure:N")],
                )
            )
        chart = alt.layer(*layers, data=ok_df).properties(
            width=190, height=170, title=title
        )
        if i == 0:
            chart = chart.add_params(pick, zoom)
        charts.append(chart)
    return alt.hconcat(*charts, spacing=30).resolve_scale(x="shared")


# ── horizon_sweep_limits (ns-3d-grid) ────────────────────────────────────────

_HSL_SOLVER_ORDER = ["phiflow", "xlb", "pict", "warp_ns", "exponax", "ins_jl"]
_HSL_EXCLUDED = {"fenics_ns", "fenics_ns_3d", "su2"}
_HSL_FAILURE_SHAPE = {
    "OOM": "triangle-down",
    "nan": _X_PATH,
    "error": "diamond",
    "timeout": "square",
}
_HSL_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
}
_HSL_VRAM_LIMIT_MIB = 16_384
_HSL_PANELS = [
    ("vram", "Peak (V)RAM", "MiB"),
    ("wall", "Wall time", "Seconds"),
    ("gn", "Gradient norm", "‖∇ℒ‖"),
]


def _hsl_parse(step_results: dict) -> dict:
    """``_hsl_parse_one_solver``: ok steps plus the first failed step."""
    ok_steps, ok_vram, ok_wall, ok_gn = [], [], [], []
    fail = None
    for k in sorted(step_results, key=int):
        r = step_results[k]
        if r.get("status") == "ok":
            ok_steps.append(int(k))
            ok_vram.append(r.get("vram_peak_mib") or 0.0)
            ok_wall.append(r.get("wall_time_s") or 0.0)
            ok_gn.append(r.get("grad_norm") or 1.0)
        elif r.get("status") == "failed" and fail is None:
            fail = {
                "step": int(k),
                "vram": r.get("vram_peak_mib") or 1.0,
                "wall": r.get("wall_time_s") or 0.0,
                "ram": r.get("ram_peak_mib") or 1.0,
                "ft": r.get("failure_type") or "error",
            }
    ok_ram = [step_results[str(s)].get("ram_peak_mib") for s in ok_steps]
    return {
        "steps": ok_steps,
        "vram": ok_vram,
        "wall": ok_wall,
        "gn": ok_gn,
        "ram": ok_ram,
        "cpu_only": bool(ok_vram) and all(v == 0.0 for v in ok_vram),
        "has_ram": any(r is not None and r > 0 for r in ok_ram),
        "fail": fail,
    }


def _hsl_rows(solver: str, d: dict, jx: float | None) -> list[tuple]:
    """Tidy rows ``(panel, solver, x, T, value, kind, failure)`` for one solver.

    ``kind`` is ``ok`` (line + dots), ``tail`` (undotted segment from the last
    ok step to the failed one) or ``fail`` (failure marker at the jittered x).
    """
    rows: list[tuple] = []
    steps, fail = d["steps"], d["fail"]

    def add(panel: str, ys: list[float], tail_y: float | None) -> None:
        rows.extend(
            (panel, solver, s, s, y, "ok", None) for s, y in zip(steps, ys, strict=True)
        )
        if fail and tail_y is not None:
            rows.append((panel, solver, steps[-1], steps[-1], ys[-1], "tail", None))
            rows.append(
                (panel, solver, fail["step"], fail["step"], tail_y, "tail", None)
            )

    ft = _HSL_FAILURE_LABEL.get(fail["ft"], fail["ft"]) if fail else None
    if steps:
        # Peak (V)RAM: GPU memory, or host RAM for CPU-only solvers.
        if not d["cpu_only"]:
            add(
                "vram",
                [max(v, 1) for v in d["vram"]],
                max(fail["vram"], 1) if fail else None,
            )
        elif d["has_ram"]:
            tail = max(fail["ram"], 1) if fail and fail["ram"] else None
            add("vram", [max(r, 1) for r in d["ram"]], tail)
        add(
            "wall",
            [max(t, 1e-10) for t in d["wall"]],
            max(fail["wall"], 1e-10) if fail else None,
        )
        add(
            "gn",
            [max(g, 1e-30) for g in d["gn"]],
            max(d["gn"][-1], 1e-30) if fail else None,
        )
        if fail:
            rows.append(
                ("gn", solver, jx, fail["step"], max(d["gn"][-1], 1e-30), "fail", ft)
            )
    if fail:
        if not d["cpu_only"]:
            rows.append(
                ("vram", solver, jx, fail["step"], max(fail["vram"], 1), "fail", ft)
            )
        rows.append(
            ("wall", solver, jx, fail["step"], max(fail["wall"], 1e-10), "fail", ft)
        )
    return rows


def horizon_sweep_limits(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Peak (V)RAM, wall time and gradient norm of the VJP vs rollout steps T.

    Same data as ``_plot_horizon_sweep_limits``: ok steps as lines, an undotted
    segment to the first failed step, and a failure-type marker there. The
    matplotlib figure compresses its axes with piecewise "function" scales
    (log x squeezed above T = 10^2.5; log y with stretched/squeezed bands);
    here every axis is a plain log scale, which keeps the decade spacing
    honest and is zoomable.
    """
    from mosaic.benchmarks.core.io import legacy_by_solver
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    data = ctx.result()
    raw = legacy_by_solver(data) if data else {}
    if not raw:
        return None
    by_solver = {(resolve_solver_alias(n) or n): sv for n, sv in raw.items()}
    present = set(by_solver)
    ordered = [s for s in _HSL_SOLVER_ORDER if s in present] + [
        s for s in present if s not in _HSL_SOLVER_ORDER and s not in _HSL_EXCLUDED
    ]
    parsed = {s: _hsl_parse(by_solver[s]) for s in ordered}

    fail_at_step: dict[int, list[str]] = defaultdict(list)
    for s in ordered:
        if parsed[s]["fail"]:
            fail_at_step[parsed[s]["fail"]["step"]].append(s)
    jitter_x = _jitter(fail_at_step)

    rows = []
    for s in ordered:
        fail = parsed[s]["fail"]
        rows += _hsl_rows(
            s, parsed[s], jitter_x.get((s, fail["step"])) if fail else None
        )
    if not rows:
        return None
    df = pd.DataFrame(
        rows, columns=["panel", "solver", "x", "T", "value", "kind", "failure"]
    )
    enc = solver_encodings(ordered)
    df = enc.relabel(df)
    df["seg"] = df["solver"] + "/" + df["kind"]

    all_steps = sorted({int(k) for sv in raw.values() for k in sv})
    x_dom = _pad_log(all_steps[0], all_steps[-1])
    fts = [
        _HSL_FAILURE_LABEL[f]
        for f in _HSL_FAILURE_SHAPE
        if _HSL_FAILURE_LABEL[f] in set(df["failure"])
    ]
    shape_of = {_HSL_FAILURE_LABEL[f]: s for f, s in _HSL_FAILURE_SHAPE.items()}

    pick = legend_pick()
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    x = alt.X(
        "x:Q",
        scale=log_scale(domain=x_dom, nice=False),
        axis=log_axis(ROLLOUT, lo=0, hi=5),
    )
    color = alt.Color(
        "solver:N", scale=enc.color, sort=enc.labels, legend=solver_legend(enc, 3)
    )
    nocolor = alt.Color("solver:N", scale=enc.color, sort=enc.labels, legend=None)
    t_tip = alt.Tooltip("T:Q", title="T")

    charts = []
    for i, (panel, title, ytitle) in enumerate(_HSL_PANELS):
        sub = df[df["panel"] == panel]
        tip = [
            alt.Tooltip("solver:N"),
            t_tip,
            alt.Tooltip("value:Q", title=ytitle, format=".4g"),
        ]
        vals = [*sub["value"], *([_HSL_VRAM_LIMIT_MIB] if panel == "vram" else [])]
        y = _log_y("value:Q", vals, ytitle)
        base = alt.Chart().encode(x=x, y=y, opacity=faded(pick))
        layers = [
            base.transform_filter("datum.kind != 'fail'")
            .mark_line(strokeWidth=2)
            .encode(
                color=color,
                detail="seg:N",
                strokeDash=alt.StrokeDash("solver:N", scale=enc.dash, legend=None),
                tooltip=tip,
            ),
            base.transform_filter("datum.kind == 'ok'")
            .mark_point(filled=True, size=30, shape="circle")
            .encode(color=nocolor, tooltip=tip),
        ]
        if fts:
            layers.append(
                base.transform_filter("datum.kind == 'fail'")
                .mark_point(filled=True, size=150, stroke="white", strokeWidth=0.6)
                .encode(
                    color=nocolor,
                    shape=alt.Shape(
                        "failure:N",
                        scale=alt.Scale(domain=fts, range=[shape_of[f] for f in fts]),
                        legend=alt.Legend(
                            title=None,
                            orient="bottom",
                            symbolFillColor="#666666",
                            symbolStrokeWidth=0,
                        ),
                    ),
                    tooltip=[alt.Tooltip("solver:N"), t_tip, alt.Tooltip("failure:N")],
                )
            )
        if panel == "vram":
            limit = pd.DataFrame({"y": [_HSL_VRAM_LIMIT_MIB], "label": ["16 GiB"]})
            rule = alt.Chart(limit).encode(y="y:Q")
            layers += [
                rule.mark_rule(color="#595959", strokeDash=[5, 3], strokeWidth=1),
                rule.mark_text(
                    align="left",
                    baseline="bottom",
                    dx=4,
                    dy=-2,
                    fontSize=9,
                    color="#595959",
                ).encode(x=alt.datum(x_dom[0]), text="label:N"),
            ]
        chart = alt.layer(*layers, data=sub).properties(
            width=180, height=160, title=title
        )
        if i == 0:
            chart = chart.add_params(pick, zoom)
        charts.append(chart)
    return alt.hconcat(*charts, spacing=30).resolve_scale(x="shared")


# ── jacobian_svd_comparison (suite level) ────────────────────────────────────

_SVD_EXP_KEYS = [
    "jacobian_svd",
    "jacobian_svd_nu01",
    "jacobian_svd_steps20",
    "jacobian_svd_steps40",
]
_SVD_VARIANT_LABELS = {
    "jacobian_svd": "ν=0.001  T=0.5s",
    "jacobian_svd_nu01": "ν=0.01   T=0.5s",
    "jacobian_svd_steps20": "ν=0.001  T=1.0s",
    "jacobian_svd_steps40": "ν=0.001  T=2.0s",
}
# The comparison figure styles lines by variant (tab10 colours), not by solver.
_SVD_VARIANT_STYLES = [
    ("#1f77b4", "-"),
    ("#ff7f0e", "--"),
    ("#2ca02c", "-."),
    ("#d62728", ":"),
]
_SVD_DASH = {"-": [1, 0], "--": [6, 3], "-.": [6, 3, 1, 3], ":": [1, 3]}
# Spectra longer than this are thinned for display (see :func:`_thin`).
_SVD_MAX_POINTS = 256


def _thin(sv: np.ndarray) -> np.ndarray:
    """Indices to keep: every ``n/256``-th mode plus any 0.02-decade change.

    Keeps plateaus cheap and the steep spectral tail exact; the first and last
    modes are always kept.
    """
    n = len(sv)
    if n <= _SVD_MAX_POINTS:
        return np.arange(n)
    step = math.ceil(n / _SVD_MAX_POINTS)
    logs = np.log10(np.maximum(sv, 1e-300))
    keep, last = [0], 0
    for i in range(1, n - 1):
        if i - last >= step or abs(logs[i] - logs[last]) > 0.02:
            keep.append(i)
            last = i
    keep.append(n - 1)
    return np.asarray(keep)


def jacobian_svd_comparison(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Per-solver singular-value spectra σᵢ/σ₁, one line per jacobian_svd variant.

    Reads ``per_solver_spectra`` (already normalised by σ₁) from each
    variant's ``result.json``, like ``plot_jacobian_svd_comparison``; panels
    are the union of the variants' ``solver_names``. Long spectra are thinned
    for display (steep tails kept exactly).
    """
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    variants = []
    for key in _SVD_EXP_KEYS:
        data = ctx.legacy(key) if (ctx.dir / key / "result.json").exists() else None
        if data and data.get("per_solver_spectra"):
            variants.append((key, data))
    if not variants:
        return None

    solvers, seen = [], set()
    for _, data in variants:
        for s in data.get("solver_names", []):
            a = resolve_solver_alias(s) or s
            if a not in seen:
                seen.add(a)
                solvers.append(s)
    if not solvers:
        return None
    enc = solver_encodings(solvers)

    labels = [_SVD_VARIANT_LABELS.get(k, k) for k, _ in variants]
    styles = [
        _SVD_VARIANT_STYLES[i % len(_SVD_VARIANT_STYLES)] for i in range(len(variants))
    ]
    frames = []
    for solver in solvers:
        for (_, data), label in zip(variants, labels, strict=True):
            spec = data["per_solver_spectra"].get(solver)
            if spec is None:
                continue
            sv = np.asarray(spec, dtype=float)
            idx = _thin(sv)
            frames.append(
                pd.DataFrame(
                    {
                        "solver": enc.label(solver),
                        "variant": label,
                        "mode": idx + 1,
                        # 4 significant digits keep the embedded data lean.
                        "sigma": [float(f"{v:.4g}") for v in sv[idx]],
                    }
                )
            )
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    pick = alt.selection_point(fields=["variant"], bind="legend")
    vcolor = alt.Color(
        "variant:N",
        scale=alt.Scale(domain=labels, range=[c for c, _ in styles]),
        sort=labels,
        legend=alt.Legend(
            orient="bottom", direction="horizontal", columns=4, title=None
        ),
    )
    vdash = alt.StrokeDash(
        "variant:N",
        scale=alt.Scale(domain=labels, range=[_SVD_DASH[ls] for _, ls in styles]),
        legend=None,
    )
    tip = [
        alt.Tooltip("solver:N"),
        alt.Tooltip("variant:N"),
        alt.Tooltip("mode:Q", title="mode i"),
        alt.Tooltip("sigma:Q", title="σᵢ/σ₁", format=".3e"),
    ]
    charts = []
    for i, solver in enumerate(solvers):
        label = enc.label(solver)
        sub = df[df["solver"] == label]
        if sub.empty:
            continue
        zoom = alt.selection_interval(bind="scales", encodings=["x"], name=f"zoom_{i}")
        base = alt.Chart(sub).encode(
            x=alt.X(
                "mode:Q", title="mode index i", scale=alt.Scale(nice=False, padding=8)
            ),
            y=_log_y("sigma:Q", sub["sigma"], "σᵢ / σ₁"),
            color=vcolor,
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.12)),
            tooltip=tip,
        )
        layers = [base.mark_line(strokeWidth=1.8).encode(strokeDash=vdash)]
        if int(sub["mode"].max()) <= 32:  # matplotlib: markers for ≤ 32 modes
            layers.append(base.mark_point(filled=True, size=20))
        color = enc.color.range[enc.labels.index(label)]
        chart = alt.layer(*layers).properties(
            width=260, height=170, title=alt.Title(label, color=color, anchor="middle")
        )
        chart = chart.add_params(zoom, pick) if i == 0 else chart.add_params(zoom)
        charts.append(chart)
    return alt.concat(*charts, columns=3, spacing=30).properties(
        title=alt.Title(
            "Jacobian singular-value spectra (flatter = better-conditioned gradient)",
            anchor="middle",
        )
    )


BUILDERS = {
    "best_eps_vs_param": best_eps_vs_param,
    "horizon_sweep": horizon_sweep,
    "horizon_sweep_limits": horizon_sweep_limits,
    "jacobian_svd_comparison": jacobian_svd_comparison,
}
