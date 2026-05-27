# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plots for the recovery suite (R1, R2, R3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.console import print_saved
from mosaic.benchmarks.problems.shared.plots.style import apply_style

# ── Evolution-GIF helper ──────────────────────────────────────────────────────


def _save_animation(
    anim: manimation.FuncAnimation,
    stem: str,
    out_dir: Path,
    *,
    fps: int = 4,
) -> None:
    """Write *anim* to ``out_dir/<stem>.gif`` using Pillow, then close the figure.

    Wraps ``FuncAnimation.save`` with ``PillowWriter`` so callers do not have to
    import matplotlib.animation directly. Mirrors :func:`save_fig`'s close-on-
    save convention so each helper can build an animation and forget the figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{stem}.gif"
    writer = manimation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    print_saved(str(gif_path))
    plt.close(anim._fig)


apply_style()


def _rho_to_2d(
    rho: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """Per-cell density → 2D slice for visualisation.

    If ``params`` provides ``nx, ny, nz`` whose product equals ``n_cells``:
      - For ``nz == 1`` (quasi-2D slab), returns the full (ny, nx) top-down view.
      - For true 3-D (``nz > 1``), returns the mid-``y`` ``(nz, nx)`` cross-section.
    Otherwise falls back to the legacy heuristic assuming
    ``nx = 2·ny, nz = 1, n_cells = 2·ny²`` → returns ``(ny, nx)``.
    """
    n_cells = len(rho)
    if params is not None:
        nx = int(params.get("nx", 0))
        ny = int(params.get("ny", 0))
        nz = int(params.get("nz", 0))
        if nx * ny * nz == n_cells and nx > 0 and ny > 0 and nz > 0:
            # Storage convention matches structural-mesh _plot_topopt_3d:
            # flat layout is (nz, ny, nx).
            rho_xyz = rho.reshape(nz, ny, nx)
            if nz == 1:
                # Quasi-2D slab: return (ny, nx) top-down view.
                return rho_xyz[0]
            # True 3-D: mid-y cross-section → (nz, nx).
            return rho_xyz[:, ny // 2, :]
    ny_ = max(1, round((n_cells / 2) ** 0.5))
    nx_ = max(1, n_cells // ny_)
    return rho.reshape(ny_, nx_)
