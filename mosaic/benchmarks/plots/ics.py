"""IC visualisation: one PNG per initial condition, saved to results/{problem}/ics/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.core.config import ProblemConfig
from benchmarks.plots.style import apply_style, imshow_with_cbar, save_fig, vorticity_2d

apply_style()

_RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_ic(
    cfg: ProblemConfig,
    ic_name: str,
    ic: np.ndarray,
    out_dir: Path,
) -> None:
    """Save one IC visualisation as ic.png (and ic.pdf) under out_dir.

    Uses cfg.ic_to_2d if set, else cfg.field_to_2d, else falls back to
    vorticity_2d (appropriate for 2-D velocity/force fields).  If 2-D
    conversion raises an exception the raw array shape is shown instead.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_2d = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0))

    try:
        arr2d = np.asarray(to_2d(ic), dtype=np.float32)
        cmap = cfg.field_cmap
        if cfg.field_symmetric:
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
            cmap=cmap,
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

    desc = cfg.get_ic_description(ic_name)
    title_parts = [ic_name]
    if desc:
        # Wrap long descriptions onto a second line
        title_parts.append(desc)
    ax.set_title("\n".join(title_parts), fontsize=9, wrap=True)

    save_fig(fig, "ic", out_dir)
