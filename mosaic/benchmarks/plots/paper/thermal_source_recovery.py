"""2D heat source inversion: GT vs per-solver recovered source fields.

Loads from the source_recovery experiment results (source_fields.npz).
Shows GT source, per-solver recovered source, and pointwise error as 2D heatmaps.

Output: thermal_source_recovery.pdf / .png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES, THERMAL_ORDER

RESULTS = Path(__file__).parent.parent.parent / "results"
_EXP_DIR = RESULTS / "thermal-mesh" / "optimization" / "source_recovery"


def generate(out_dir: Path) -> None:
    npz_path = _EXP_DIR / "source_fields.npz"
    params_path = _EXP_DIR / "params.json"
    result_path = _EXP_DIR / "result.json"

    if not npz_path.exists():
        print(
            f"[thermal_source_recovery] {npz_path} not found — run: "
            "mosaic recovery -p thermal-mesh -e source_recovery"
        )
        return

    data = np.load(npz_path, allow_pickle=True)
    params = json.loads(params_path.read_text())
    result = json.loads(result_path.read_text())
    phys = params["physics"]
    nx, ny = phys["nx"], phys["ny"]

    solvers = [s for s in THERMAL_ORDER if f"source_final_{s}" in data]
    n_solvers = len(solvers)

    Q_gt = data["source_truth"].reshape(ny, nx)
    Lx, Ly = phys.get("Lx", 2.0), phys.get("Ly", 1.0)
    extent = [0, Lx, 0, Ly]

    vmax_Q = float(Q_gt.max()) * 1.05

    # Error reduction stats for titles
    by_solver = result.get("by_solver", {})

    # Layout: rows = [GT row, solver rows...], cols = [source | error | colorbar]
    # Row 0: GT source (spans all solver cols) + colorbar
    # Rows 1..N: per-solver recovered + error
    n_rows = 1 + n_solvers
    n_cols = 3  # GT/rec | error | cbar

    with plt.rc_context(RCPARAMS):
        fig_w = TEXTWIDTH
        fig_h = fig_w * (n_rows / n_cols) * 0.85
        fig = plt.figure(figsize=(fig_w, fig_h))

        gs = mgridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            hspace=0.35,
            wspace=0.06,
            left=0.01,
            right=0.88,
            top=0.93,
            bottom=0.04,
            width_ratios=[1, 1, 0.06],
        )

        def _imshow(ax, data_2d, cmap, vmin, vmax, title=None, ylabel=None):
            im = ax.imshow(
                data_2d,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title, fontsize=7, pad=2)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
            return im

        # Row 0: GT source + empty error panel (labelled "Init")
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_init = fig.add_subplot(gs[0, 1])
        ax_cb0 = fig.add_subplot(gs[0, 2])

        im_gt = _imshow(
            ax_gt,
            Q_gt,
            "inferno",
            0,
            vmax_Q,
            title="GT source $Q$",
            ylabel="Ground truth",
        )
        Q_init = data["source_init"].reshape(ny, nx)
        _imshow(ax_init, Q_init, "inferno", 0, vmax_Q, title="Init $Q_0$")
        cb0 = fig.colorbar(im_gt, cax=ax_cb0)
        cb0.ax.tick_params(labelsize=5.5)

        # Per-solver rows
        err_lim_global = (
            max(
                float(np.abs(data[f"source_final_{s}"].reshape(ny, nx) - Q_gt).max())
                for s in solvers
            )
            * 1.05
        )

        for row_i, solver in enumerate(solvers, start=1):
            label, color, _, _ = SOLVER_STYLES.get(
                solver, (solver, "#888888", "-", "o")
            )
            Q_rec = data[f"source_final_{solver}"].reshape(ny, nx)
            err = Q_rec - Q_gt

            stats = by_solver.get(solver, {})
            reduction = stats.get("error_reduction_pct", float("nan"))
            title_rec = (
                f"Recovered  ({reduction:.0f}% ↓)"
                if np.isfinite(reduction)
                else "Recovered"
            )

            ax_rec = fig.add_subplot(gs[row_i, 0])
            ax_err = fig.add_subplot(gs[row_i, 1])
            ax_cb = fig.add_subplot(gs[row_i, 2])

            _imshow(
                ax_rec,
                Q_rec,
                "inferno",
                0,
                vmax_Q,
                title=title_rec if row_i == 1 else None,
                ylabel=label,
            )
            im_err = _imshow(
                ax_err,
                err,
                "RdBu_r",
                -err_lim_global,
                err_lim_global,
                title="Error $\\hat{Q}-Q$" if row_i == 1 else None,
            )
            cb = fig.colorbar(im_err, cax=ax_cb)
            cb.ax.tick_params(labelsize=5.5)

        fig.suptitle("Thermal heat source inversion", fontsize=8, y=0.97)

        for ext in ("pdf", "png"):
            out = out_dir / f"thermal_source_recovery.{ext}"
            fig.savefig(out)
            print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parents[4]))
    out = Path(__file__).parents[4] / "paper" / "figures"
    out.mkdir(exist_ok=True)
    generate(out)
