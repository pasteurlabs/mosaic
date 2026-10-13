# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Animated charts replacing the optimisation GIFs.

Mirror ``_render_topopt_evolution_gifs`` in ``structural_mesh/plots.py``,
``_render_drag_opt_evolution_gifs`` in ``navier_stokes_grid/plots.py`` and
``_render_recovery_evolution_gifs`` in ``navier_stokes_3d_grid/plots.py``.

Animation contract with the docs page: every animated row carries an integer
``frame`` column (0..N-1), one unit chart owns the ``frame`` parameter (a range
slider), animated layers filter on it, and the top-level ``usermeta.animation``
tells the page how many frames to step through and at which rate.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .common import (
    CELLS,
    Ctx,
    cell_outline,
    field_cells,
    linked_cell,
    solver_encodings,
)

FPS = 4  # PillowWriter(fps=4) of the GIFs
MAX_FRAMES = 40


def _keep(n: int, k: int = MAX_FRAMES) -> list[int]:
    """≤ ``k`` evenly spaced snapshot indices of ``n``, first and last included."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.rint(np.linspace(0, n - 1, k)).astype(int).tolist()))


def _sig4(a) -> np.ndarray:
    """Round to 4 significant digits."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a == 0, 0, np.floor(np.log10(np.abs(a))))
    scale = 10.0 ** (3 - mag)
    return np.round(a * scale) / scale


def _frame_param(n: int) -> alt.Parameter:
    return alt.param(
        name="frame",
        value=0,
        bind=alt.binding_range(min=0, max=n - 1, step=1, name="frame "),
    )


def _animated(chart: alt.TopLevelMixin, n: int) -> alt.TopLevelMixin:
    return chart.properties(
        usermeta={"animation": {"param": "frame", "frames": n, "fps": FPS}}
    )


def _frame_label(labels: list[str], width: int, **text_kw) -> alt.Chart:
    """One line of text per frame, showing the current one."""
    df = pd.DataFrame({"frame": range(len(labels)), "label": labels})
    return (
        alt.Chart(df)
        .transform_filter("datum.frame == frame")
        .mark_text(align="center", baseline="middle", **text_kw)
        .encode(x=alt.value(width / 2), y=alt.value(9), text="label:N")
        .properties(width=width, height=18)
    )


# ── Topology optimisation density (structural-mesh) ──────────────────────────


def topopt_evolution(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Density ρ of one solver over the optimiser snapshots.

    Same view as the GIF: ``_rho_to_2d`` of each ``rho_history_<j>`` row (the
    mid-y (nz, nx) cross-section, x right, z up), greys on [0, 1], titled
    ``<solver> — iter k / N`` where k counts snapshots. A second line gives the
    optimiser iteration (snapshot × ``snap_interval``) and the compliance there.
    """
    from mosaic.benchmarks.problems.shared.plots.optimization import _rho_to_2d

    from .optimization import _topopt_inputs

    inputs = _topopt_inputs(ctx)
    if inputs is None:
        return None
    data, npz, ph, names = inputs
    solver = ctx.png.stem.removeprefix("topopt_evolution_")
    if npz is None or solver not in names:
        return None
    j = names.index(solver)
    key = f"rho_history_{j}"
    if key not in npz.files:
        return None
    hist = np.asarray(npz[key], dtype=float)
    if hist.ndim != 2 or hist.shape[0] == 0:
        return None
    n_snaps = hist.shape[0]
    idx = _keep(n_snaps)
    label = solver_encodings([solver]).labels[0]

    sdata = (data.get("by_solver") or {}).get(solver) or {}
    comps = sdata.get("compliances") or []
    n_iters = len(comps) or sdata.get("n_iters")
    snap = int(ph.get("snap_interval") or 0)

    frames, titles, meta = [], [], []
    for f, s in enumerate(idx):
        frames.append(field_cells(_rho_to_2d(hist[s], ph).T, frame=f))
        line2, row = [], {"frame": f, "snapshot": s + 1}
        if snap:
            row["iteration"] = it = s * snap
            line2.append(f"iteration ≈ {it}" + (f" / {n_iters}" if n_iters else ""))
            if comps:
                c = float(comps[min(it, len(comps) - 1)])
                row["compliance"] = float(_sig4(c))
                line2.append(f"C = {c:.4e}")
        meta.append(row)
        titles.append((f"{label} — iter {s + 1} / {n_snaps}", " · ".join(line2)))
    df = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta)
    df["value"] = _sig4(df["value"])
    ni, nj = int(df["i"].max()) + 1, int(df["j"].max()) + 1
    px = max(4, round(360 / max(ni, nj)))
    w, h = ni * px, nj * px

    n = len(idx)
    frame = _frame_param(n)
    cell = linked_cell()
    tooltip = [
        alt.Tooltip("i:Q", title="x index"),
        alt.Tooltip("j:Q", title="z index"),
        alt.Tooltip("value:Q", title="ρ", format=".3f"),
        alt.Tooltip("snapshot:Q"),
    ]
    if "iteration" in meta:
        tooltip.append(alt.Tooltip("iteration:Q", title="iteration ≈"))
    if "compliance" in meta:
        tooltip.append(alt.Tooltip("compliance:Q", title="C", format=".4e"))
    field = (
        alt.Chart(df)
        .transform_filter("datum.frame == frame")
        .transform_lookup(
            lookup="frame",
            from_=alt.LookupData(meta, "frame", [c for c in meta if c != "frame"]),
        )
        .mark_rect()
        .encode(
            x=alt.X("i:O", axis=None, scale=CELLS),
            y=alt.Y("j:O", axis=None, sort="descending", scale=CELLS),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(domain=[0, 1], range=["white", "black"]),
                legend=alt.Legend(title="ρ", gradientLength=h),
            ),
            **cell_outline(cell, "#e4572e"),
            tooltip=tooltip,
        )
        .add_params(frame, cell)
        .properties(width=w, height=h)
    )
    head = alt.vconcat(
        _frame_label([t[0] for t in titles], w, fontSize=13, fontWeight="bold"),
        _frame_label([t[1] for t in titles], w, fontSize=11),
        spacing=0,
    )
    return _animated(alt.vconcat(head, field, spacing=6), n)


# ── Inflow-profile evolution (ns-grid drag optimisation) ─────────────────────

_INITIAL = "initial"


def drag_opt_evolution(ctx: Ctx) -> alt.TopLevelMixin | None:
    """u_x(y) of each solver's inflow profile over the BFGS snapshots.

    One panel per solver with a ``profile_history_<name>`` (``by_solver``
    order, deduplicated by alias), the initial profile dashed black, x range
    per panel from the history and initial extrema (+5 %), y ∈ [0, 1] at cell
    centres. Solvers with fewer snapshots hold their last one. The frame line
    adds the iteration (snapshot × ``snap_interval``); the tooltip the drag
    reduction (d₀ − dᵢ)/d₀ at that iteration.
    """
    from mosaic.benchmarks.problems.shared.plots.style import resolve_solver_alias

    data = ctx.legacy()
    prof = ctx.npz("profiles.npz")
    if data is None or prof is None or "initial" not in prof.files:
        return None
    by_solver = data.get("by_solver") or {}
    initial = np.asarray(prof["initial"], dtype=float)
    n_y = initial.size
    y = np.linspace(0, 1, n_y, endpoint=False) + 0.5 / n_y
    snap = int((data.get("params") or {}).get("optim", {}).get("snap_interval") or 0)

    panels, seen = [], set()
    for name in by_solver:
        key = f"profile_history_{name}"
        if key not in prof.files:
            continue
        hist = np.asarray(prof[key], dtype=float)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        dedup = resolve_solver_alias(name) or name
        if dedup in seen:
            continue
        seen.add(dedup)
        panels.append((name, hist))
    if not panels:
        return None
    n = max(h.shape[0] for _, h in panels)
    enc = solver_encodings([p[0] for p in panels])
    color = alt.Scale(domain=[*enc.labels, _INITIAL], range=[*enc.color.range, "black"])
    # The GIF draws the animated profiles solid and the initial one dashed.
    dash = alt.Scale(
        domain=[*enc.labels, _INITIAL], range=[*([[1, 0]] * len(enc.labels)), [6, 3]]
    )
    legend = alt.Legend(orient="bottom", direction="horizontal", title=None)
    frame = _frame_param(n)

    charts = []
    for k, (name, hist) in enumerate(panels):
        label = enc.label(name)
        drags = (by_solver.get(name) or {}).get("drags") or []
        rows = []
        for f in range(n):
            s = min(f, hist.shape[0] - 1)
            it = s * snap if snap else None
            red = None
            if drags and it is not None and drags[0]:
                d = drags[min(it, len(drags) - 1)]
                red = (drags[0] - d) / drags[0] * 100
            rows += [
                (label, f, s + 1, it, red, float(yy), float(u))
                for yy, u in zip(y, hist[s], strict=True)
            ]
        df = pd.DataFrame(
            rows,
            columns=[
                "solver",
                "frame",
                "snapshot",
                "iteration",
                "reduction",
                "y",
                "ux",
            ],
        )
        df["ux"] = _sig4(df["ux"])
        df["y"] = _sig4(df["y"])
        df["reduction"] = _sig4(df["reduction"].astype(float))
        init_df = pd.DataFrame(
            {"solver": _INITIAL, "y": _sig4(y), "ux": _sig4(initial)}
        )

        lo = min(hist.min(), initial.min())
        hi = max(hist.max(), initial.max())
        pad = 0.05 * (hi - lo + 1e-12)
        x = alt.X(
            "ux:Q",
            title="u_x",
            scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False, zero=False),
        )
        yenc = alt.Y("y:Q", title="y", scale=alt.Scale(domain=[0, 1], nice=False))
        enc_kw = {
            "color": alt.Color(
                "solver:N", scale=color, legend=legend if k == 0 else None
            ),
            "strokeDash": alt.StrokeDash("solver:N", scale=dash, legend=None),
            "order": alt.Order("y:Q"),
        }
        init_line = (
            alt.Chart(init_df)
            .mark_line(strokeWidth=1.4)
            .encode(
                x=x,
                y=yenc,
                **enc_kw,
                tooltip=[
                    alt.Tooltip("solver:N", title="profile"),
                    alt.Tooltip("y:Q", format=".3f"),
                    alt.Tooltip("ux:Q", title="u_x", format=".4f"),
                ],
            )
        )
        tooltip = [
            alt.Tooltip("solver:N"),
            alt.Tooltip("snapshot:Q"),
            alt.Tooltip("y:Q", format=".3f"),
            alt.Tooltip("ux:Q", title="u_x", format=".4f"),
        ]
        if snap:
            tooltip.insert(2, alt.Tooltip("iteration:Q", title="iteration ≈"))
        if drags:
            tooltip.append(
                alt.Tooltip("reduction:Q", title="drag reduction (%)", format=".2f")
            )
        line = (
            alt.Chart(df)
            .transform_filter("datum.frame == frame")
            .mark_line(strokeWidth=2)
            .encode(x=x, y=yenc, **enc_kw, tooltip=tooltip)
        )
        if k == 0:
            line = line.add_params(frame)
        charts.append(
            alt.layer(init_line, line).properties(
                width=200,
                height=200,
                title=alt.Title(label, fontSize=12, fontWeight="bold", anchor="middle"),
            )
        )
    row = alt.hconcat(*charts, spacing=30).resolve_scale(x="independent")
    width = len(charts) * 230
    n_iters = max(len((v or {}).get("drags") or []) for v in by_solver.values())
    labels = [
        f"snapshot {f + 1} / {n}"
        + (f" · iteration ≈ {f * snap} / {n_iters}" if snap else "")
        for f in range(n)
    ]
    out = alt.vconcat(_frame_label(labels, width, fontSize=11), row, spacing=4)
    return _animated(
        out.properties(title=alt.Title("Inflow profile evolution", anchor="middle")), n
    )


# ── IC recovery evolution (ns-3d-grid) ───────────────────────────────────────


def recovery_evolution(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Vorticity view of each solver's recovered IC over optimiser snapshots.

    Same mapping as the GIF (``vorticity_2d``: ω_z of the z = 0 plane), one
    panel per solver with an ``ic_history_<j>`` (deduplicated by alias),
    RdBu_r on a symmetric range fixed per panel over all its snapshots.
    Solvers with fewer snapshots hold their last one (the filter clamps).
    """
    from mosaic.benchmarks.problems.shared.plots.style import (
        resolve_solver_alias,
        vorticity_2d,
    )

    npz = ctx.npz("recovery_fields.npz")
    if npz is None or "solver_names" not in npz.files:
        return None
    data = ctx.legacy() or {}
    sweep_key = data.get("sweep_key", "steps")
    rep_key = next((k for k in ("rep_val", "rep_horizon") if k in npz.files), None)
    rep_val = float(npz[rep_key][0]) if rep_key else None
    snap = int((data.get("params") or {}).get("optim", {}).get("snap_interval") or 0)

    panels, seen = [], set()
    for j, name in enumerate(str(s) for s in npz["solver_names"]):
        key = f"ic_history_{j}"
        if key not in npz.files:
            continue
        hist = np.asarray(npz[key])
        if hist.ndim < 2 or hist.shape[0] == 0:
            continue
        dedup = resolve_solver_alias(name) or name
        if dedup in seen:
            continue
        seen.add(dedup)
        fields = [vorticity_2d(h) for h in hist]
        vmax = float(max(np.abs(f).max() for f in fields)) or 1.0
        panels.append((name, fields, vmax))
    if not panels:
        return None
    n = max(len(f) for _, f, _ in panels)
    enc = solver_encodings([p[0] for p in panels])
    frame = _frame_param(n)
    cell = linked_cell()

    charts = []
    for k, (name, fields, vmax) in enumerate(panels):
        label = enc.label(name)
        df = pd.concat(
            [field_cells(f, frame=s) for s, f in enumerate(fields)],
            ignore_index=True,
        )
        df["value"] = _sig4(df["value"])
        last = len(fields) - 1
        n_cells = int(df["i"].max()) + 1
        size = n_cells * max(4, round(112 / n_cells))
        tooltip = [
            alt.Tooltip("snapshot:Q", title=f"{label} snapshot"),
            alt.Tooltip("i:Q"),
            alt.Tooltip("j:Q"),
            alt.Tooltip("value:Q", title="ω_z", format=".4f"),
        ]
        if snap:
            tooltip.insert(1, alt.Tooltip("iteration:Q", title="iteration ≈"))
        chart = (
            alt.Chart(df)
            .transform_filter(f"datum.frame == min(frame, {last})")
            .transform_calculate(
                snapshot="datum.frame + 1", iteration=f"datum.frame * {snap}"
            )
            .mark_rect()
            .encode(
                x=alt.X("i:O", axis=None, scale=CELLS),
                y=alt.Y("j:O", axis=None, sort="descending", scale=CELLS),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(
                        scheme="redblue", reverse=True, domain=[-vmax, vmax]
                    ),
                    legend=None,
                ),
                **cell_outline(cell, "#222222"),
                tooltip=tooltip,
            )
            .properties(
                width=size,
                height=size,
                title=alt.Title(
                    label,
                    anchor="middle",
                    color=enc.color.range[enc.labels.index(label)],
                    fontSize=12,
                    fontWeight="bold",
                ),
            )
        )
        if k == 0:
            chart = chart.add_params(frame, cell)
        else:
            chart = chart.add_params(cell)
        charts.append(chart)
    row = alt.hconcat(*charts, spacing=20).resolve_scale(color="independent")
    width = len(charts) * (charts[0].width + 20) - 20
    tail = ""
    if rep_val is not None:
        rv = int(rep_val) if rep_val.is_integer() else rep_val
        tail = f"  ({sweep_key}={rv})"
    labels = [f"IC recovery evolution — snapshot {f + 1} / {n}{tail}" for f in range(n)]
    out = alt.vconcat(_frame_label(labels, width, fontSize=13), row, spacing=6)
    return _animated(out, n)


BUILDERS = {
    **{
        f"topopt_evolution_{s}": topopt_evolution
        for s in ("Firedrake", "torch-fem", "JAX-FEM", "TopOpt.jl", "FEniCS")
    },
    "drag_opt_evolution": drag_opt_evolution,
    "recovery_evolution": recovery_evolution,
}
