# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization-suite charts: drag optimisation, topology optimisation, recovery.

Mirror ``_drag_opt_figure`` / ``_plot_drag_opt_fields`` in
``navier_stokes_grid/plots.py``, ``_plot_topopt_figure`` / ``plot_topopt`` /
``_plot_topopt_3d`` in ``structural_mesh/plots.py`` and
``plot_conductivity_recovery`` in ``thermal_mesh/plots.py``.

The matplotlib topopt figures draw 3-D voxel renders. Here they become 2-D
slices of the same thresholded density volume (the plane spanned by the two
largest mesh dimensions) with a slider over the slice index along the thinnest
one, plus the fixed face and load arrow of ``_add_bcs`` drawn in that plane.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd

from .common import (
    Ctx,
    cell_outline,
    cell_px,
    faded,
    field_cells,
    field_xy,
    legend_pick,
    linked_cell,
    log_axis,
    log_scale,
    solver_encodings,
    solver_legend,
    solver_lines,
)

# ── Drag optimisation (ns-grid) ──────────────────────────────────────────────

# Solvers in the paper panels, in display order (aliases).
_DRAG_OPT_SOLVER_ORDER = ["xlb", "phiflow", "pict"]
_INITIAL = "Initial"


def _alias_map(names) -> dict[str, str]:
    """alias → raw (display) name, first occurrence wins."""
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    out: dict[str, str] = {}
    for n in names:
        alias = resolve_solver_alias(n)
        if alias is not None:
            out.setdefault(alias, n)
    return out


def _with_initial(enc) -> dict:
    """Solver scales with a grey dashed "Initial" entry first (as in the PNG)."""
    return {
        "color": alt.Scale(
            domain=[_INITIAL, *enc.labels], range=["#999999", *enc.color.range]
        ),
        "dash": alt.Scale(
            domain=[_INITIAL, *enc.labels], range=[[6, 3], *enc.dash.range]
        ),
    }


def drag_opt(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Drag reduction vs iteration, optimised inlet profile, profile history.

    Same reduction as ``_drag_opt_figure``: drag reduction (d₀ − dᵢ)/d₀ in %
    on ≤ ~50 subsampled iterations, initial + final inlet profiles u_x(y), and
    one ``profile_history`` heatmap (snapshot × y, viridis, own colour range)
    per solver in XLB, PhiFlow, PICT order.
    """
    data = ctx.legacy()
    if data is None or not data.get("by_solver"):
        return None
    by_solver = data["by_solver"]
    prof = ctx.npz("profiles.npz")
    pkeys = set(prof.files) if prof is not None else set()

    aliases = _alias_map(
        list(by_solver)
        + [k[len("final_") :] for k in pkeys if k.startswith("final_")]
        + [
            k[len("profile_history_") :]
            for k in pkeys
            if k.startswith("profile_history_")
        ]
    )

    drag_rows = []
    for alias in _DRAG_OPT_SOLVER_ORDER:
        name = aliases.get(alias)
        drags = (by_solver.get(name) or {}).get("drags") or []
        if not drags or not drags[0] or np.isnan(drags[0]):
            continue
        d0 = drags[0]
        step = max(1, len(drags) // 50)
        idx = list(range(0, len(drags), step))
        if idx[-1] != len(drags) - 1:
            idx.append(len(drags) - 1)
        drag_rows += [(name, i, (d0 - drags[i]) / d0 * 100, drags[i]) for i in idx]
    prof_rows = []
    if "initial" in pkeys:
        init = prof["initial"]
        y = np.linspace(0, 1, init.shape[0])
        prof_rows += [
            (_INITIAL, float(yy), float(u)) for yy, u in zip(y, init, strict=True)
        ]
        for alias in _DRAG_OPT_SOLVER_ORDER:
            name = aliases.get(alias)
            if f"final_{name}" in pkeys:
                u = prof[f"final_{name}"]
                prof_rows += [
                    (name, float(yy), float(v)) for yy, v in zip(y, u, strict=True)
                ]
    if not drag_rows and not prof_rows:
        return None
    drag_df = pd.DataFrame(
        drag_rows, columns=["solver", "iteration", "reduction", "drag"]
    )
    prof_df = pd.DataFrame(prof_rows, columns=["solver", "y", "ux"])
    solvers = set(drag_df["solver"]) | (set(prof_df["solver"]) - {_INITIAL})
    enc = solver_encodings(solvers)
    drag_df = enc.relabel(drag_df)
    prof_df = prof_df.assign(
        solver=prof_df["solver"].map(lambda s: s if s == _INITIAL else enc.label(s))
    )
    scales = _with_initial(enc)
    legend = solver_legend(enc)
    pick = legend_pick()

    def lines(df, **kw) -> alt.Chart:
        return (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                color=alt.Color("solver:N", scale=scales["color"], legend=legend),
                strokeDash=alt.StrokeDash(
                    "solver:N", scale=scales["dash"], legend=None
                ),
                opacity=faded(pick),
                **kw,
            )
            .properties(width=205, height=240)
        )

    reduction = lines(
        drag_df,
        x=alt.X("iteration:Q", title="Iteration"),
        y=alt.Y(
            "reduction:Q", title="Drag reduction (%)", scale=alt.Scale(domainMin=0)
        ),
        tooltip=[
            alt.Tooltip("solver:N"),
            alt.Tooltip("iteration:Q"),
            alt.Tooltip("reduction:Q", title="drag reduction (%)", format=".2f"),
            alt.Tooltip("drag:Q", format=".4g"),
        ],
    ).properties(title="Drag reduction")
    profile = lines(
        prof_df,
        x=alt.X("ux:Q", title="u_x", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="y"),
        order=alt.Order("y:Q"),
        tooltip=[
            alt.Tooltip("solver:N"),
            alt.Tooltip("y:Q", format=".3f"),
            alt.Tooltip("ux:Q", title="u_x", format=".4f"),
        ],
    ).properties(title="Optimised profile", width=150)
    curves = alt.hconcat(reduction.add_params(pick), profile, spacing=40)

    history = _profile_history(prof, pkeys, aliases, by_solver, enc)
    if history is None:
        return curves
    return alt.hconcat(curves, history, spacing=40).resolve_scale(color="independent")


def _profile_history(prof, pkeys, aliases, by_solver, enc) -> alt.VConcatChart | None:
    """One ``imshow(hist.T, origin='lower', aspect='auto')`` panel per solver."""
    from mosaic.benchmarks.problems.shared.plots.style import solver_props

    cell = linked_cell()
    panels = []
    for alias in _DRAG_OPT_SOLVER_ORDER:
        name = aliases.get(alias)
        key = f"profile_history_{name}"
        if key not in pkeys:
            continue
        hist = prof[key]
        n_snaps = hist.shape[0]
        n_iters = len((by_solver.get(name) or {}).get("drags", [1]))
        snap_step = n_iters / max(n_snaps - 1, 1)
        df = field_cells(hist, solver=enc.label(name))
        df["iteration"] = (df["i"] * snap_step).astype(int)
        df["y"] = df["j"] / max(hist.shape[1] - 1, 1)
        ticks = [0, n_snaps // 2, n_snaps - 1]
        label, color, _ls, _mk = solver_props(alias)
        last = (
            len(panels)
            == sum(
                f"profile_history_{aliases.get(a)}" in pkeys
                for a in _DRAG_OPT_SOLVER_ORDER
            )
            - 1
        )
        panels.append(
            alt.Chart(
                df, title=alt.Title(label, color=color, fontSize=10, anchor="start")
            )
            .mark_rect()
            .encode(
                x=alt.X(
                    "i:O",
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                    axis=alt.Axis(
                        values=ticks,
                        labelExpr=f"floor(datum.value * {snap_step!r})",
                        labelAngle=0,
                        title="Iteration" if last else None,
                        labels=last,
                        grid=False,
                    ),
                ),
                y=alt.Y(
                    "j:O",
                    axis=None,
                    sort="descending",
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                ),
                color=alt.Color(
                    "value:Q", scale=alt.Scale(scheme="viridis"), legend=None
                ),
                **cell_outline(cell),
                tooltip=[
                    alt.Tooltip("solver:N"),
                    alt.Tooltip("iteration:Q", title="iteration ≈"),
                    alt.Tooltip("y:Q", format=".3f"),
                    alt.Tooltip("value:Q", title="u_x", format=".4f"),
                ],
            )
            .add_params(cell)
            .properties(width=200, height=72)
        )
    if not panels:
        return None
    return (
        alt.vconcat(*panels, spacing=6)
        .resolve_scale(color="independent")
        .properties(title=alt.Title("Profile history", anchor="middle"))
    )


def _vel_components(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u_x, u_y) from a (N, N, 1, 2) or (N, N, 2) field."""
    if v.ndim == 4:
        v = v[:, :, 0, :]
    return v[..., 0], v[..., 1]


def drag_opt_fields(ctx: Ctx) -> alt.TopLevelMixin | None:
    """u_x and u_y of the initial and each solver's optimised flow.

    Rows follow ``by_solver`` (deduplicated by alias) after the initial flow;
    both columns use a symmetric RdBu_r range set by the 99th percentile of
    |u| of the initial flow, as in ``_plot_drag_opt_fields``. Row labels carry
    the final drag and the convergence flag; hover links cells across panels.
    """
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    data = ctx.legacy()
    npz = ctx.npz("flow_fields.npz")
    if data is None or npz is None:
        return None
    by_solver = data.get("by_solver", {})
    names, seen = [], set()
    for s in by_solver:
        alias = resolve_solver_alias(s) or s
        if f"flow_final_{s}" in npz.files and alias not in seen:
            seen.add(alias)
            names.append(s)
    if not names:
        return None
    enc = solver_encodings(names)

    init = npz["flow_initial"] if "flow_initial" in npz.files else None
    if init is not None:
        ux0, uy0 = _vel_components(init)
        vmax = {
            "u_x": float(np.percentile(np.abs(ux0), 99)) or 1.0,
            "u_y": float(np.percentile(np.abs(uy0), 99)) or 1.0,
        }
    else:
        vmax = {"u_x": 1.0, "u_y": 0.5}

    rows = []
    if init is not None:
        rows.append(("Initial flow", init, None, None))
    for s in names:
        info = by_solver.get(s, {})
        rows.append(
            (s, npz[f"flow_final_{s}"], info.get("final_drag"), info.get("converged"))
        )

    frames, order, header = [], [], {}
    for name, field, drag, conv in rows:
        key = name if name == "Initial flow" else enc.label(name)
        lines = [key]
        if drag is not None:
            lines.append(f"drag={drag:.4g}")
        if conv is False:
            lines.append("[NOT CONVERGED]")
        order.append(key)
        header[key] = lines
        ux, uy = _vel_components(field)
        for comp, arr in (("u_x", ux), ("u_y", uy)):
            frames.append(field_cells(arr, row=key, component=comp))
    df = pd.concat(frames, ignore_index=True)
    df["value"] = df["value"].round(5)
    # Multi-line row labels (name, final drag, convergence) looked up in the
    # header instead of repeated on every cell.
    label_expr = f"{json.dumps(header, ensure_ascii=False)}[datum.value]"
    n = int(df["i"].max()) + 1
    size = cell_px(n)
    cell = linked_cell()

    def column(comp: str) -> alt.FacetChart:
        vm = vmax[comp]
        return (
            alt.Chart(df[df["component"] == comp].drop(columns="component"))
            .mark_rect()
            .encode(
                **field_xy(),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(
                        scheme="redblue", reverse=True, domain=[-vm, vm], clamp=True
                    ),
                    legend=alt.Legend(
                        title=comp, orient="bottom", gradientLength=size - 20
                    ),
                ),
                **cell_outline(cell, "#222222"),
                tooltip=[
                    alt.Tooltip("row:N", title="flow"),
                    alt.Tooltip("i:Q"),
                    alt.Tooltip("j:Q"),
                    alt.Tooltip("value:Q", title=comp, format=".4f"),
                ],
            )
            .add_params(cell)
            .properties(width=size, height=size)
            .facet(
                row=alt.Row(
                    "row:N",
                    sort=order,
                    title=None,
                    header=alt.Header(
                        labelAngle=0,
                        labelAlign="left",
                        labelBaseline="middle",
                        labelPadding=10,
                        labelFontWeight="normal",
                        labelExpr=label_expr,
                        labelOrient="left" if comp == "u_x" else "right",
                        labels=comp == "u_x",
                    ),
                ),
                spacing=16,
            )
            .properties(title=alt.Title(comp, anchor="middle"))
        )

    return (
        alt.hconcat(column("u_x"), column("u_y"), spacing=24)
        .resolve_scale(color="independent")
        .properties(title="Optimised flow fields")
    )


# ── Topology optimisation (structural-mesh) ──────────────────────────────────


def _thin(n: int, dense: int = 100, target: int = 500) -> list[int]:
    """Iterations kept for long curves: all of the first ``dense``, then every
    k-th (≤ ~``target`` points), always including the last one."""
    step = max(1, (n - dense) // target)
    idx = list(range(min(n, dense))) + list(range(dense, n, step))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


_THRESH = 0.35  # voxel threshold of the matplotlib renders
_CLR_FIXED = "#888888"
_CLR_LOAD = "#FF1744"


def _topopt_inputs(ctx: Ctx):
    """(legacy result, fields npz, physics params, field solver names)."""
    data = ctx.legacy()
    if data is None:
        return None
    npz = ctx.npz("topopt_fields.npz")
    p_path = ctx.dir / "params.json"
    params = json.loads(p_path.read_text(encoding="utf-8")) if p_path.exists() else None
    ph = (params or data.get("params", {}) or {}).get("physics", {}) or {}
    names: list[str] = []
    if npz is not None:
        n_fields = sum(k.startswith("rho_final_") for k in npz.files)
        stored = (
            [str(s) for s in npz["solver_names"]] if "solver_names" in npz.files else []
        )
        bys = list(data.get("by_solver", {}))
        if len(stored) >= n_fields > 0:
            names = stored[:n_fields]
        elif len(bys) >= n_fields > 0:
            names = bys[:n_fields]
        else:
            names = (stored or bys) + [
                f"solver_{k}" for k in range(len(stored or bys), n_fields)
            ]
    return data, npz, ph, names


def _volume_dims(ph: dict, n_cells: int) -> tuple[int, int, int]:
    """(nx, ny, nz) from params, else the quasi-2-D fallback of ``_plot_topopt_3d``."""
    nx, ny, nz = (int(ph.get(k, 0)) for k in ("nx", "ny", "nz"))
    if nx * ny * nz == n_cells and nx * ny * nz > 0:
        return nx, ny, nz
    ny = max(1, round((n_cells / 2) ** 0.5))
    return max(1, n_cells // ny), ny, 1


def _density_slices(
    data: dict, npz, ph: dict, names: list[str], enc, *, dedupe: bool, width: int = 150
) -> tuple[alt.HConcatChart, alt.Parameter] | None:
    """Per-solver slice views of the thresholded final density volume.

    Plane = the two largest of (nx, ny, nz) (x–z for the 16×2×8 cantilever);
    the slider walks the remaining axis, starting on the slice through the
    loaded corner. Voxels with ρ > 0.35 are drawn in the solver
    colour, blended towards white for low ρ exactly like
    ``_voxel_facecolors`` (opacity 0.55 → 1 over ρ ∈ [0.35, 1]). The grey bar
    is the fixed face x = 0, the red arrow the corner load of ``_add_bcs``
    (solid on the slice that contains the loaded corner, faint elsewhere).
    """
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    panels_idx, seen = [], set()
    for j, name in enumerate(names):
        if f"rho_final_{j}" not in npz.files:
            continue
        key = resolve_solver_alias(name) or name
        if dedupe and key in seen:
            continue
        seen.add(key)
        panels_idx.append((j, name))
    if not panels_idx:
        return None
    nx, ny, nz = _volume_dims(ph, len(npz[f"rho_final_{panels_idx[0][0]}"]))
    dims = {"x": nx, "y": ny, "z": nz}
    # In-plane axes: x horizontal, then the larger of y/z vertical.
    vert = "z" if nz >= ny else "y"
    depth = "y" if vert == "z" else "z"
    nd, nv = dims[depth], dims[vert]

    # Load corner / direction (``_add_bcs``).
    load_axis = ph.get("load_axis", "z")
    corner = {
        "y": (ny - 1) if ph.get("corner_y_high") else 0,
        "z": (nz - 1) if ph.get("corner_z_high") else 0,
    }
    high = {"y": bool(ph.get("corner_y_high")), "z": bool(ph.get("corner_z_high"))}
    # About a third of the panel height, so the arrow reads as a load marker
    # without dwarfing the slice.
    arrow_len = max(2.0, 0.35 * nv)

    # Start on the slice through the loaded corner (all slices if the load
    # runs along the slicing axis).
    slider = alt.param(
        name="slice_idx",
        value=corner[depth] if load_axis != depth else nd // 2,
        bind=alt.binding_range(min=0, max=nd - 1, step=1, name=f"{depth} index "),
    )

    px = max(2, round(width / nx))
    pad = 1.0
    lo_v = -pad
    hi_v = nv + pad
    if load_axis == vert:
        tip = corner[vert] + (1.0 if high[vert] else 0.0)
        sign = -1 if high[vert] else 1
        tail = tip - sign * arrow_len
        lo_v, hi_v = min(lo_v, tail - pad), max(hi_v, tail + pad)
    x_scale = alt.Scale(domain=[-pad - 0.6, nx + pad + 0.6], nice=False, zero=False)
    v_scale = alt.Scale(domain=[lo_v, hi_v], nice=False, zero=False)
    w = round((nx + 2 * pad + 1.2) * px)
    h = round((hi_v - lo_v) * px)

    charts = []
    for j, name in panels_idx:
        rho = np.asarray(npz[f"rho_final_{j}"], dtype=float).reshape(nz, ny, nx)
        kk, jj, ii = np.indices(rho.shape)
        df = pd.DataFrame(
            {"x": ii.ravel(), "y": jj.ravel(), "z": kk.ravel(), "rho": rho.ravel()}
        )
        df = df[df["rho"] > _THRESH]
        df = df.assign(
            d=df[depth],
            v=df[vert],
            solver=enc.label(name),
            x2=df["x"] + 1,
            v2=df[vert] + 1,
        )
        comp = (data.get("by_solver", {}).get(name) or {}).get("final_compliance")
        title = enc.label(name) + (f"\nC = {comp:.3e}" if comp is not None else "")
        x_enc = alt.X("x:Q", scale=x_scale, axis=None)
        v_enc = alt.Y("v:Q", scale=v_scale, axis=None)
        frame = (
            alt.Chart(pd.DataFrame({"x": [0], "x2": [nx], "v": [0], "v2": [nv]}))
            .mark_rect(fill=None, stroke="#cccccc", strokeWidth=1)
            .encode(x=x_enc, x2="x2:Q", y=v_enc, y2="v2:Q")
        )
        voxels = (
            alt.Chart(df)
            .transform_filter(alt.datum.d == slider)
            .mark_rect(stroke="white", strokeWidth=0.4)
            .encode(
                x=x_enc,
                x2="x2:Q",
                y=v_enc,
                y2="v2:Q",
                color=alt.Color("solver:N", scale=enc.color, legend=None),
                opacity=alt.Opacity(
                    "rho:Q",
                    scale=alt.Scale(
                        domain=[_THRESH, 1.0], range=[0.55, 1.0], clamp=True
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("solver:N"),
                    alt.Tooltip("x:Q"),
                    alt.Tooltip("y:Q"),
                    alt.Tooltip("z:Q"),
                    alt.Tooltip("rho:Q", title="ρ", format=".3f"),
                ],
            )
        )
        wall = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [-0.6],
                        "x2": [0.0],
                        "v": [0],
                        "v2": [nv],
                        "what": ["fixed face x = 0"],
                    }
                )
            )
            .mark_rect(color=_CLR_FIXED, opacity=0.7)
            .encode(
                x=x_enc,
                x2="x2:Q",
                y=v_enc,
                y2="v2:Q",
                tooltip=[alt.Tooltip("what:N", title="BC")],
            )
        )
        layers = [frame, wall, voxels]
        load_here = (
            alt.condition(
                f"slice_idx == {corner[depth]}", alt.value(1.0), alt.value(0.2)
            )
            if load_axis != depth
            else alt.value(1.0)
        )
        if load_axis == vert:
            tip = corner[vert] + (1.0 if high[vert] else 0.0)
            sign = -1 if high[vert] else 1
            arr = pd.DataFrame(
                {
                    "x": [nx + 0.0],
                    "v": [tip - sign * arrow_len],
                    "v2": [tip - sign * 0.6 * 1.0],
                    "what": [
                        f"load at corner ({nx}, {corner['y']}, {corner['z']}), along {load_axis}"
                    ],
                }
            )
            head = arr.assign(v=[tip - sign * 0.5])
            layers += [
                alt.Chart(arr)
                .mark_rule(color=_CLR_LOAD, strokeWidth=3)
                .encode(
                    x=x_enc,
                    y=v_enc,
                    y2="v2:Q",
                    opacity=load_here,
                    tooltip=[alt.Tooltip("what:N", title="BC")],
                ),
                alt.Chart(head)
                .mark_point(
                    shape="triangle-up" if sign > 0 else "triangle-down",
                    filled=True,
                    size=90,
                    color=_CLR_LOAD,
                )
                .encode(
                    x=x_enc,
                    y=v_enc,
                    opacity=load_here,
                    tooltip=[alt.Tooltip("what:N", title="BC")],
                ),
            ]
        else:
            # Load normal to the plane: mark the loaded corner.
            mark = pd.DataFrame(
                {
                    "x": [nx + 0.0],
                    "v": [corner[vert] + 0.5],
                    "what": [
                        f"load at corner ({nx}, {corner['y']}, {corner['z']}), along {load_axis}"
                    ],
                }
            )
            layers.append(
                alt.Chart(mark)
                .mark_point(shape="circle", filled=True, size=90, color=_CLR_LOAD)
                .encode(
                    x=x_enc,
                    y=v_enc,
                    opacity=load_here,
                    tooltip=[alt.Tooltip("what:N", title="BC")],
                )
            )
        charts.append(
            alt.layer(*layers).properties(
                width=w,
                height=h,
                title=alt.Title(title.split("\n"), anchor="middle", fontSize=11),
            )
        )
    return alt.hconcat(*charts, spacing=12), slider


def topopt(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Compliance vs iteration (log y), as in ``_plot_topopt_figure``.

    Every solver in ``by_solver``. The figure's 3-D voxel row is left out: the
    final densities are the ``topopt_3d`` chart on the same page.
    """
    inputs = _topopt_inputs(ctx)
    if inputs is None:
        return None
    data, _npz, _ph, names = inputs
    rows = []
    for solver, sdata in data.get("by_solver", {}).items():
        comp = sdata.get("compliances") or []
        rows += [(solver, it, float(comp[it])) for it in _thin(len(comp))]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "iteration", "compliance"])
    # Lowest decade below the data, as matplotlib's semilogy shows it.
    c_lo = 10.0 ** np.floor(np.log10(df["compliance"].min()))
    enc = solver_encodings(set(df["solver"]) | set(names))
    df = enc.relabel(df)
    pick = legend_pick()
    label = ctx.config.category_label or ctx.problem
    curves = (
        solver_lines(
            df,
            enc,
            x=alt.X("iteration:Q", title="Iteration"),
            y=alt.Y(
                "compliance:Q",
                title="Compliance",
                scale=log_scale(nice=False, domainMin=c_lo),
                axis=alt.Axis(format=".0e"),
            ),
            tooltip=[
                alt.Tooltip("solver:N"),
                alt.Tooltip("iteration:Q"),
                alt.Tooltip("compliance:Q", format=".4e"),
            ],
            pick=pick,
            points=False,
        )
        .add_params(pick)
        .properties(width=620, height=170, title=f"Compliance — {label}")
    )
    # The PNG's density row duplicates topopt_3d, which the same page shows as
    # an interactive slice view, so the web chart keeps only the curves.
    return curves


def topopt_fields(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Initial ρ and each solver's final density, greys on [0, 1].

    ``_rho_to_2d`` gives the mid-y (nz, nx) cross-section of the volume, drawn
    with ``imshow(origin='lower')`` (x right, z up); one shared colour bar.
    """
    from mosaic.benchmarks.problems.shared.plots.optimization import _rho_to_2d

    inputs = _topopt_inputs(ctx)
    if inputs is None:
        return None
    _data, npz, ph, names = inputs
    if npz is None or not names:
        return None
    enc = solver_encodings(names)
    panels = []
    if "rho_init" in npz.files:
        panels.append(("Initial ρ", npz["rho_init"]))
    panels += [
        (enc.label(n), npz[f"rho_final_{j}"])
        for j, n in enumerate(names)
        if f"rho_final_{j}" in npz.files
    ]
    frames = [field_cells(_rho_to_2d(rho, ph).T, panel=title) for title, rho in panels]
    df = pd.concat(frames, ignore_index=True)
    ni, nj = int(df["i"].max()) + 1, int(df["j"].max()) + 1
    px = max(4, round(105 / max(ni, nj)))
    cell = linked_cell()
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            **field_xy(),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(domain=[0, 1], range=["white", "black"]),
                legend=alt.Legend(title="ρ", gradientLength=nj * px),
            ),
            **cell_outline(cell, "#e4572e"),
            tooltip=[
                alt.Tooltip("panel:N", title="field"),
                alt.Tooltip("i:Q", title="x index"),
                alt.Tooltip("j:Q", title="z index"),
                alt.Tooltip("value:Q", title="ρ", format=".3f"),
            ],
        )
        .add_params(cell)
        .properties(width=ni * px, height=nj * px)
        .facet(
            column=alt.Column(
                "panel:N",
                sort=[t for t, _ in panels],
                title=None,
                header=alt.Header(labelFontWeight="bold", labelFontSize=11),
            ),
            spacing=14,
        )
        .properties(title=alt.Title("Optimised density fields", anchor="middle"))
    )


def topopt_3d(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Thresholded final densities with the BCs, one slice view per solver.

    Replaces the voxel grid of ``_plot_topopt_3d`` (solvers deduplicated by
    alias, titles with the final compliance) by :func:`_density_slices`.
    """
    inputs = _topopt_inputs(ctx)
    if inputs is None:
        return None
    data, npz, ph, names = inputs
    if npz is None or not names:
        return None
    enc = solver_encodings(names)
    sl = _density_slices(data, npz, ph, names, enc, dedupe=True, width=110)
    if sl is None:
        return None
    slices, slider = sl
    return slices.add_params(slider).properties(
        title=alt.Title(
            f"Optimised density (ρ > {_THRESH}) with load and fixed-face BCs",
            anchor="middle",
        )
    )


# ── Conductivity recovery (thermal-mesh) ─────────────────────────────────────


def conductivity_recovery_convergence(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Identification error ‖T(k) − T_obs‖² vs optimizer iteration, log y."""
    data = ctx.legacy()
    if data is None:
        return None
    rows = [
        (solver, it, float(e))
        for solver, res in data.get("by_solver", {}).items()
        for it, e in enumerate(res.get("errors") or [])
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["solver", "iteration", "error"])
    df = df[df["error"] > 0]
    enc = solver_encodings(df["solver"])
    df = enc.relabel(df)
    pick = legend_pick()
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return (
        solver_lines(
            df,
            enc,
            x=alt.X("iteration:Q", title="Optimizer iteration"),
            y=alt.Y(
                "error:Q",
                scale=log_scale(),
                axis=log_axis("Identification error ‖T(k) − T_obs‖²", lo=-16, hi=16),
            ),
            tooltip=[
                alt.Tooltip("solver:N"),
                alt.Tooltip("iteration:Q"),
                alt.Tooltip("error:Q", title="‖T(k) − T_obs‖²", format=".4e"),
            ],
            pick=pick,
            points=False,
        )
        .add_params(pick, zoom)
        .properties(
            width=560,
            height=220,
            title="Conductivity recovery: error vs iteration (lower is better)",
        )
    )


BUILDERS = {
    "drag_opt": drag_opt,
    "drag_opt_fields": drag_opt_fields,
    "topopt": topopt,
    "topopt_fields": topopt_fields,
    "topopt_3d": topopt_3d,
    "conductivity_recovery_convergence": conductivity_recovery_convergence,
}
