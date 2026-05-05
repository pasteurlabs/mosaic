"""2D conductivity inversion: GT vs per-solver recovered rho fields.

Loads from the conductivity_recovery experiment results (rho_fields.npz).
Shows GT conductivity, per-solver recovered conductivity, and pointwise error
as 2D heatmaps.

Output: thermal_conductivity_recovery.pdf / .png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES, THERMAL_ORDER

RESULTS = Path(__file__).parent.parent.parent / "results"
_EXP_DIR = RESULTS / "thermal-mesh" / "optimization" / "conductivity_recovery"


def generate(out_dir: Path) -> None:
    npz_path = _EXP_DIR / "rho_fields.npz"
    params_path = _EXP_DIR / "params.json"
    result_path = _EXP_DIR / "result.json"

    if not npz_path.exists():
        print(f"[thermal_conductivity_recovery] {npz_path} not found — run: "
              "mosaic recovery -p thermal-mesh -e conductivity_recovery")
        return

    data = np.load(npz_path, allow_pickle=True)
    params = json.loads(params_path.read_text())
    result = json.loads(result_path.read_text())
    phys = params["physics"]
    nx, ny = phys["nx"], phys["ny"]

    solvers = [s for s in THERMAL_ORDER if f"rho_final_{s}" in data]
    n_solvers = len(solvers)

    rho_gt = data["rho_truth"].reshape(ny, nx)
    Lx, Ly = phys.get("Lx", 2.0), phys.get("Ly", 1.0)
    extent = [0, Lx, 0, Ly]

    vmin_rho, vmax_rho = 0.0, float(rho_gt.max()) * 1.05

    by_solver = result.get("by_solver", {})

    n_rows = 1 + n_solvers
    n_cols = 3  # GT/rec | error | cbar

    with plt.rc_context(RCPARAMS):
        fig_w = TEXTWIDTH
        fig_h = fig_w * (n_rows / n_cols) * 0.85
        fig = plt.figure(figsize=(fig_w, fig_h))

        gs = mgridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            hspace=0.35, wspace=0.06,
            left=0.01, right=0.88, top=0.93, bottom=0.04,
            width_ratios=[1, 1, 0.06],
        )

        def _imshow(ax, data_2d, cmap, vmin, vmax, title=None, ylabel=None):
            im = ax.imshow(
                data_2d, origin="lower", extent=extent,
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="bilinear", aspect="auto",
            )
            ax.set_xticks([]); ax.set_yticks([])
            if title:
                ax.set_title(title, fontsize=7, pad=2)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
            return im

        # Row 0: GT conductivity + initial conductivity
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_init = fig.add_subplot(gs[0, 1])
        ax_cb0 = fig.add_subplot(gs[0, 2])

        im_gt = _imshow(ax_gt, rho_gt, "viridis", vmin_rho, vmax_rho,
                        title=r"GT conductivity $\rho$", ylabel="Ground truth")
        rho_init = data["rho_init"].reshape(ny, nx)
        _imshow(ax_init, rho_init, "viridis", vmin_rho, vmax_rho,
                title=r"Init $\rho_0$")
        cb0 = fig.colorbar(im_gt, cax=ax_cb0)
        cb0.ax.tick_params(labelsize=5.5)

        # Per-solver rows
        err_lim_global = max(
            float(np.abs(data[f"rho_final_{s}"].reshape(ny, nx) - rho_gt).max())
            for s in solvers
        ) * 1.05

        for row_i, solver in enumerate(solvers, start=1):
            label, color, _, _ = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
            rho_rec = data[f"rho_final_{solver}"].reshape(ny, nx)
            err = rho_rec - rho_gt

            stats = by_solver.get(solver, {})
            reduction = stats.get("error_reduction_pct", float("nan"))
            title_rec = f"Recovered  ({reduction:.0f}% ↓)" if np.isfinite(reduction) else "Recovered"

            ax_rec = fig.add_subplot(gs[row_i, 0])
            ax_err = fig.add_subplot(gs[row_i, 1])
            ax_cb = fig.add_subplot(gs[row_i, 2])

            _imshow(ax_rec, rho_rec, "viridis", vmin_rho, vmax_rho,
                    title=title_rec if row_i == 1 else None,
                    ylabel=label)
            im_err = _imshow(ax_err, err, "RdBu_r", -err_lim_global, err_lim_global,
                             title=r"Error $\hat{\rho}-\rho$" if row_i == 1 else None)
            cb = fig.colorbar(im_err, cax=ax_cb)
            cb.ax.tick_params(labelsize=5.5)

        fig.suptitle("Thermal conductivity inversion", fontsize=8, y=0.97)

        for ext in ("pdf", "png"):
            out = out_dir / f"thermal_conductivity_recovery.{ext}"
            fig.savefig(out)
            print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[4]))
    out = Path(__file__).parents[4] / "paper" / "figures"
    out.mkdir(exist_ok=True)
    generate(out)
