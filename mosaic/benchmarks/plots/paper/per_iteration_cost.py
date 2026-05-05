"""Generate Figure: Per-iteration cost for gradient-based vs gradient-free (placeholder)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH

RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        np.random.seed(42)

        # ---------- placeholder data ----------
        ctrl_dims = np.array([10, 20, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5])

        grad_solvers = {
            "JAX-CFD (source transf.)": {
                "overhead": 2.5,
                "color": "#2ecc71",
                "marker": "o",
            },
            "PhiFlow (source transf.)": {
                "overhead": 3.2,
                "color": "#3498db",
                "marker": "s",
            },
            "FEniCS (tape-based adj.)": {
                "overhead": 5.0,
                "color": "#e74c3c",
                "marker": "^",
            },
            "XLB (source transf.)": {
                "overhead": 1.8,
                "color": "#e67e22",
                "marker": "D",
            },
        }

        # Gradient-free: CMA-ES population size ~ 4 + 3 ln(N)
        cmaes_pop = 4 + 3 * np.log(ctrl_dims)
        cmaes_cost = cmaes_pop  # normalized to forward solve = 1

        # ---------- figure ----------
        fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.55), dpi=150)

        ax.loglog(
            ctrl_dims,
            cmaes_cost,
            color="#7f8c8d",
            linewidth=2.5,
            linestyle="--",
            marker="x",
            markersize=7,
            label="CMA-ES (gradient-free)",
            zorder=5,
        )

        for name, s in grad_solvers.items():
            cost = np.full_like(ctrl_dims, 1.0 + s["overhead"])
            ax.loglog(
                ctrl_dims,
                cost,
                color=s["color"],
                linewidth=1.8,
                marker=s["marker"],
                markersize=5,
                alpha=0.85,
                label=name,
            )

        # Crossover region
        ax.axvspan(10, 30, alpha=0.08, color="#7f8c8d", zorder=0)
        ax.text(
            18,
            1.3,
            "comparable\ncost",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#7f8c8d",
            fontstyle="italic",
        )

        ax.set_xlabel("Control dimension $N$", fontsize=8)
        ax.set_ylabel("Cost per iteration (forward solves)", fontsize=8)
        ax.set_xlim(ctrl_dims[0] * 0.7, ctrl_dims[-1] * 1.5)
        ax.set_ylim(1, 100)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="upper left")

        # Watermark
        ax.text(
            0.5,
            0.5,
            "PLACEHOLDER",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=24,
            color="#cccccc",
            fontweight="bold",
            fontstyle="italic",
            alpha=0.5,
            rotation=25,
        )

        fig.savefig(
            out_dir / "per_iteration_cost.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        print(f"Saved {out_dir / 'per_iteration_cost.png'}")
