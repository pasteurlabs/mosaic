"""Generate Figure: Lid-cavity 3D convergence for ns-3d-grid.

1×3 panels, one per sweep val (U_x_true in {0.5, 1.0, 2.0}).
semilogy loss vs iteration per solver.
Output: lid_cavity_convergence.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

SOLVER_ORDER = ["ins_jl", "exponax", "phiflow", "xlb", "warp_ns", "pict"]

SWEEP_VALS = ["0.5", "1.0", "2.0"]


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        path = (
            results_dir() / "ns-3d-grid" / "optimization" / "lid_cavity" / "result.json"
        )
        data = json.loads(path.read_text())
        by_sweep = data["by_sweep"]

        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
        fig.subplots_adjust(bottom=0.22, wspace=0.35)

        present: set[str] = set()

        for col, sv in enumerate(SWEEP_VALS):
            ax = axes[col]

            for solver, sweep_data in by_sweep.items():
                if solver in {"fenics_ns", "su2"}:
                    continue
                if sv not in sweep_data:
                    continue
                label, color, ls, mk = SOLVER_STYLES.get(
                    solver, (solver, "#888888", "-", "o")
                )

                losses = sweep_data[sv]["losses"]
                iters = list(range(len(losses)))
                kw = dict(color=color, linestyle=ls, marker="", linewidth=1.6)
                ax.semilogy(iters, losses, **kw)
                present.add(solver)

            ax.set_title(f"$U_x^\\mathrm{{true}} = {sv}$")
            ax.set_xlabel("Iteration")
            if col == 0:
                ax.set_ylabel("Loss")

        handles = [
            mlines.Line2D(
                [],
                [],
                color=SOLVER_STYLES[s][1],
                linestyle=SOLVER_STYLES[s][2],
                linewidth=1.6,
                label=SOLVER_STYLES[s][0],
            )
            for s in SOLVER_ORDER
            if s in present
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            fontsize=7.5,
            framealpha=0.7,
            handlelength=2.0,
        )

        out = out_dir / "lid_cavity_convergence.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
