# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-condition overview: one panel per registered IC of a problem.

ICs are not stored in ``mosaic-results/``: each one is regenerated
from its ``IcSpec`` and ``plot_params`` and projected to 2-D (vorticity for
velocity fields, a hex-mesh slice for flat scalar fields).
"""

from __future__ import annotations

import math
from typing import Any

import altair as alt
import numpy as np

from .common import CELLS, Ctx, cell_outline, field_cells, linked_cell


def _flat_field_2d(
    ic: np.ndarray,
    plot_params: dict,
    mesh_ratio: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Reshape a flat ``(n_cells,)`` hex-mesh field to a 2-D mid-y slice.

    Fallback for scalar IC fields (structural/thermal density / source) that the
    primary ``to_2d`` transform (tuned for 2-D/3-D velocity) can't handle. The
    (nx, ny, nz) geometry is resolved, in order of preference:

      1. explicit ``plot_params`` dims when their product matches ``n_cells``;
      2. ``mesh_ratio = (ny, nz)`` from a sibling IC that *does* carry full dims
         (so e.g. ``uniform`` / ``random`` reuse ``two_density_bumps``'s mesh);
      3. a near-square ``ny`` factorisation of ``n_cells`` with ``nz = 1``.

    Returns the (nx, nz) mid-y slice, or ``None`` if the array is not a 1-D
    field or no consistent reshape exists.
    """
    arr = np.asarray(ic)
    if arr.ndim != 1:
        return None
    n_cells = int(arr.shape[0])
    nx_ = int(plot_params.get("nx", 0))
    ny_ = int(plot_params.get("ny", 0))
    nz_ = int(plot_params.get("nz", 0))
    if not (nx_ * ny_ * nz_ == n_cells and nx_ * ny_ * nz_ > 0):
        if mesh_ratio is not None:
            ny_, nz_ = mesh_ratio
            nx_ = n_cells // (ny_ * nz_) if ny_ * nz_ else 0
        else:
            ny_ = max(1, round(n_cells**0.5))
            while ny_ > 1 and n_cells % ny_:
                ny_ -= 1
            nx_, nz_ = n_cells // ny_, 1
    if nx_ <= 0 or ny_ <= 0 or nz_ <= 0 or nx_ * ny_ * nz_ != n_cells:
        return None
    xyz = arr.reshape(nz_, ny_, nx_).transpose(2, 1, 0)  # (nx, ny, nz)
    # Display the plane spanning the two largest dims (slice the thinnest axis at
    # its midpoint): for quasi-2D meshes (nz=1) this keeps the full (nx, ny)
    # plane; for thick 3-D meshes it takes the mid-y (nx, nz) cross-section.
    slice_axis = int(np.argmin(xyz.shape))
    plane = np.take(xyz, xyz.shape[slice_axis] // 2, axis=slice_axis)
    return np.asarray(plane, dtype=np.float32)


def _ic_to_2d(
    ic: np.ndarray,
    *,
    to_2d: Any,
    field_symmetric: bool,
    plot_params: dict,
    mesh_ratio: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Project one IC to a 2-D array + (vmin, vmax), or ``None`` if not plottable.

    Keeps the original field→2D transform and symmetric/auto colour-range logic.
    When that transform fails or yields a non-2-D result (e.g. flat scalar
    density / source fields), falls back to :func:`_flat_field_2d`. Returns
    ``None`` only when no projection is possible (e.g. 1-D inflow profiles), so
    the caller can skip ICs with no plottable field.
    """
    arr2d: np.ndarray | None
    try:
        candidate = np.asarray(to_2d(ic), dtype=np.float32)
        arr2d = candidate if candidate.ndim == 2 else None
    except Exception:
        arr2d = None
    if arr2d is None:
        arr2d = _flat_field_2d(ic, plot_params, mesh_ratio)
    if arr2d is None:
        return None
    if field_symmetric:
        vmax = float(np.abs(arr2d).max()) or 1.0
        return arr2d, -vmax, vmax
    vmin_v = float(arr2d.min())
    vmax_v = float(arr2d.max())
    if vmin_v == vmax_v:
        vmax_v = vmin_v + 1.0
    return arr2d, vmin_v, vmax_v


# Panels per row.
_NCOLS = 4


def _panels(
    cfg: Any, *, to_2d: Any = None, field_symmetric: bool = True
) -> list[tuple[str, np.ndarray, float, float]]:
    """``(name, arr2d, vmin, vmax)`` per IC, exactly as ``plot_ic`` builds them."""
    from mosaic.benchmarks.problems.shared.plots.style import vorticity_2d

    to_2d = vorticity_2d if to_2d is None else to_2d
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
            to_2d=to_2d,
            field_symmetric=field_symmetric,
            plot_params=params,
            mesh_ratio=mesh_ratio,
        )
        if projected is not None:
            out.append((name, *projected))
    return out


def _panel(
    name: str,
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cell: alt.Parameter,
    field_cmap: str = "RdBu_r",
) -> alt.Chart:
    df = field_cells(arr, ic=name)
    df["value"] = [float(f"{v:.4g}") for v in df["value"]]  # lean embedded data
    ni, nj = int(df["i"].max()) + 1, int(df["j"].max()) + 1
    # One cell size per panel so non-square meshes keep their aspect ratio.
    px = max(2, round(115 / max(ni, nj, 1)))
    return (
        alt.Chart(df, title=alt.Title(name, anchor="middle", fontSize=12))
        .mark_rect()
        .encode(
            x=alt.X("i:O", axis=None, scale=CELLS),
            y=alt.Y("j:O", axis=None, sort="descending", scale=CELLS),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(
                    scheme="redblue" if field_cmap == "RdBu_r" else field_cmap,
                    reverse=field_cmap == "RdBu_r",
                    domain=[vmin, vmax],
                ),
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


def ics(
    ctx: Ctx,
    *,
    to_2d: Any = None,
    field_cmap: str = "RdBu_r",
    field_symmetric: bool = True,
) -> alt.TopLevelMixin | None:
    """Every registered IC of the problem, one RdBu_r panel each.

    Each panel keeps its own symmetric colour range (the PNG has one colourbar
    per panel). Hovering a cell outlines the same (i, j) in every panel.
    """
    if ctx.experiment:
        return None
    panels = _panels(ctx.config, to_2d=to_2d, field_symmetric=field_symmetric)
    if not panels:
        return None
    cell = linked_cell()
    charts = [_panel(*p, cell, field_cmap=field_cmap) for p in panels]
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
