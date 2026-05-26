# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IC visualisation: one PNG per initial condition, saved to results/{problem}/ics/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.problems.shared.plots.style import (
    apply_style,
    imshow_with_cbar,
    save_fig,
    vorticity_2d,
)

apply_style()


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
    """Save one IC visualisation as ic.png (and ic.pdf) under out_dir.

    Uses ``ic_to_2d`` if set, else ``field_to_2d``, else falls back to
    vorticity_2d (appropriate for 2-D velocity/force fields).  If 2-D
    conversion raises an exception the raw array shape is shown instead.

    The IC description is read off ``make_ic[ic_name].description`` when
    ``make_ic`` is provided (else falls back to ``""``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_2d = ic_to_2d or field_to_2d or vorticity_2d

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))

    try:
        arr2d = np.asarray(to_2d(ic), dtype=np.float32)
        if field_symmetric:
            vmax = float(np.abs(arr2d).max()) or 1.0
            vmin_v, vmax_v = -vmax, vmax
        else:
            vmin_v = float(arr2d.min())
            vmax_v = float(arr2d.max())
            if vmin_v == vmax_v:
                vmax_v = vmin_v + 1.0
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
        ax.axis("off")
    except Exception:
        ax.text(
            0.5,
            0.5,
            f"shape: {ic.shape}\n(no 2D projection)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

    desc = ""
    if make_ic is not None and ic_name in make_ic:
        desc = getattr(make_ic[ic_name], "description", "") or ""
    title_parts = [ic_name]
    if desc:
        # Wrap long descriptions onto a second line
        title_parts.append(desc)
    ax.set_title("\n".join(title_parts), fontsize=9, wrap=True)

    save_fig(fig, "ic", out_dir)
