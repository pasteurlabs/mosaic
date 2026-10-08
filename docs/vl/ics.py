# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-condition overview: one panel per registered IC of a problem.

Mirrors ``plot_ic`` in ``mosaic/benchmarks/problems/shared/plots/ics.py``. The
ICs are not stored in ``mosaic-results/``: like the matplotlib figure, each one
is regenerated from its ``IcSpec`` and ``plot_params`` and projected to 2-D by
the same helper (vorticity for velocity fields, a hex-mesh slice for flat
scalar fields).
"""

from __future__ import annotations

import math

import altair as alt
import numpy as np

from .common import CELLS, Ctx, cell_outline, field_cells, linked_cell

# Same grid width as the matplotlib figure.
_NCOLS = 4


def _panels(cfg) -> list[tuple[str, np.ndarray, float, float]]:
    """``(name, arr2d, vmin, vmax)`` per IC, exactly as ``plot_ic`` builds them."""
    from mosaic.benchmarks.problems.shared.plots.ics import _ic_to_2d
    from mosaic.benchmarks.problems.shared.plots.style import vorticity_2d

    specs = dict(cfg.make_ic)
    # Sibling mesh ratio for flat scalar ICs with incomplete dims.
    mesh_ratio = None
    for spec in specs.values():
        pp = getattr(spec, "plot_params", {}) or {}
        ny_p, nz_p = int(pp.get("ny", 0)), int(pp.get("nz", 0))
        if ny_p > 0 and nz_p > 0:
            mesh_ratio = (ny_p, nz_p)
            break
    out = []
    for name in sorted(specs):
        spec = specs[name]
        params = dict(getattr(spec, "plot_params", {}) or {})
        try:
            arr = spec(**params)
        except Exception:  # noqa: S112 - plot_ic skips specs that fail too
            continue
        projected = _ic_to_2d(
            np.asarray(arr),
            to_2d=vorticity_2d,
            field_symmetric=True,
            plot_params=params,
            mesh_ratio=mesh_ratio,
        )
        if projected is not None:
            out.append((name, *projected))
    return out


def _panel(name: str, arr: np.ndarray, vmin: float, vmax: float, cell) -> alt.Chart:
    df = field_cells(arr, ic=name)
    df["value"] = [float(f"{v:.4g}") for v in df["value"]]  # lean embedded data
    ni, nj = int(df["i"].max()) + 1, int(df["j"].max()) + 1
    # One cell size per panel so non-square meshes keep their aspect ratio.
    px = max(2, round(150 / max(ni, nj, 1)))
    return (
        alt.Chart(df, title=alt.Title(name, anchor="middle", fontSize=12))
        .mark_rect()
        .encode(
            x=alt.X("i:O", axis=None, scale=CELLS),
            y=alt.Y("j:O", axis=None, sort="descending", scale=CELLS),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", reverse=True, domain=[vmin, vmax]),
                legend=alt.Legend(title=None, gradientLength=max(nj * px - 10, 40)),
            ),
            **cell_outline(cell, "#222222"),
            tooltip=[
                alt.Tooltip("ic:N", title="IC"),
                alt.Tooltip("i:Q", title="i"),
                alt.Tooltip("j:Q", title="j"),
                alt.Tooltip("value:Q", format=".3e"),
            ],
        )
        .add_params(cell)
        .properties(width=ni * px, height=nj * px)
    )


def ics(ctx: Ctx) -> alt.TopLevelMixin | None:
    """Every registered IC of the problem, one RdBu_r panel each.

    Each panel keeps its own symmetric colour range (the PNG has one colourbar
    per panel). Hovering a cell outlines the same (i, j) in every panel.
    """
    if ctx.experiment:
        return None
    panels = _panels(ctx.config)
    if not panels:
        return None
    cell = linked_cell()
    charts = [_panel(*p, cell) for p in panels]
    ncols = min(len(charts), _NCOLS)
    rows = [
        alt.hconcat(*charts[r * ncols : (r + 1) * ncols], spacing=24).resolve_scale(
            color="independent"
        )
        for r in range(math.ceil(len(charts) / ncols))
    ]
    chart = rows[0] if len(rows) == 1 else alt.vconcat(*rows, spacing=24)
    return chart.resolve_scale(color="independent")


BUILDERS = {"ics": ics}
