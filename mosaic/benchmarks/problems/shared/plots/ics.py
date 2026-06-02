# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IC visualisation: one combined PNG per problem, saved to results/{problem}/ics/."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.problems.shared.plots.style import (
    RCPARAMS,
    apply_style,
    imshow_with_cbar,
    paper_image_grid,
    save_fig,
    vorticity_2d,
)

apply_style()


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


def plot_ic(
    cfg: Problem,
    ic_name: str,
    ic: np.ndarray,
    out_dir: Path,
    *,
    make_ic: Any = None,
    field_to_2d: Any = None,
    ic_to_2d: Any = None,
    field_cmap: str = "RdBu_r",
    field_symmetric: bool = True,
    **_kw: Any,
) -> None:
    """Save ONE combined IC figure for the whole problem as ``ics.png``.

    ``plot_ic`` is registered per IC experiment and so is invoked once per IC,
    but it aggregates *every* IC of the problem into a single grid figure and
    writes it to the shared ICs directory (``<results>/<problem>/ics/ics.png``).
    The write is idempotent: each per-IC invocation rebuilds and overwrites the
    same combined figure rather than emitting a separate ``ic.png`` per IC.

    Each registered IC is regenerated from ``make_ic`` (its ``IcSpec.fn`` +
    ``plot_params``), projected to 2-D, and drawn as one panel titled with the
    concise IC name. ICs with no 2-D projection (e.g. 1-D profiles) are skipped.

    Uses ``ic_to_2d`` if set, else ``field_to_2d``, else ``vorticity_2d``
    (appropriate for 2-D velocity / force fields); colormap and symmetric-range
    logic are unchanged.
    """
    plt.rcParams.update(RCPARAMS)

    to_2d = ic_to_2d or field_to_2d or vorticity_2d

    # The combined figure lives one level up from the per-IC out_dir, i.e. the
    # shared ICs directory <results>/<problem>/ics/.
    ics_dir = Path(out_dir).parent
    ics_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate every registered IC of this problem (make_ic carries them all).
    # Fall back to just the IC handed in if make_ic is unavailable.
    if make_ic:
        specs = dict(make_ic)
    else:
        specs = {ic_name: None}

    # Derive a fallback (ny, nz) mesh ratio from any IC that carries full
    # (nx, ny, nz) dims, so flat scalar ICs with incomplete dims (e.g. uniform /
    # random density) reuse a sibling's hex-mesh geometry rather than being skipped.
    mesh_ratio: tuple[int, int] | None = None
    for spec in specs.values():
        pp = getattr(spec, "plot_params", {}) if spec else {}
        ny_p, nz_p = int(pp.get("ny", 0)), int(pp.get("nz", 0))
        if ny_p > 0 and nz_p > 0:
            mesh_ratio = (ny_p, nz_p)
            break

    panels: list[tuple[str, np.ndarray, float, float]] = []
    for name in sorted(specs):
        spec = specs[name]
        plot_params = dict(getattr(spec, "plot_params", {})) if spec else {}
        if spec is None:
            arr = ic
        else:
            try:
                arr = spec(**plot_params)
            except Exception:
                continue
        projected = _ic_to_2d(
            arr,
            to_2d=to_2d,
            field_symmetric=field_symmetric,
            plot_params=plot_params,
            mesh_ratio=mesh_ratio,
        )
        if projected is None:
            continue
        arr2d, vmin_v, vmax_v = projected
        panels.append((name, arr2d, vmin_v, vmax_v))

    if not panels:
        return

    n = len(panels)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = paper_image_grid(nrows, ncols, squeeze=False)
    axes_flat = list(axes.flat)

    for ax, (name, arr2d, vmin_v, vmax_v) in zip(axes_flat, panels, strict=False):
        imshow_with_cbar(
            ax,
            fig,
            arr2d.T,
            origin="lower",
            cmap=field_cmap,
            vmin=vmin_v,
            vmax=vmax_v,
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    save_fig(fig, "ics", ics_dir)
