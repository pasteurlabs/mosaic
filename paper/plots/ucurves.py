"""Generate Figure: FD U-curves (relative error vs ε) for F2 (ns-grid).

One panel per rollout length, relative FD error vs perturbation size ε for each
solver, from the per-horizon ε sweeps of ``gradient/horizon_sweep``.

Output: appendix_ucurves_f2.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

_NCOLS = 4


def generate(out_dir: Path) -> None:
    data = by_solver(load_result("ns-grid", "gradient/horizon_sweep"))
    solvers = [s for s in NS_ORDER if s in data]
    all_steps = sorted({int(k) for sv in data.values() for k in sv})
    nrows = int(np.ceil(len(all_steps) / _NCOLS))

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(
            nrows,
            _NCOLS,
            figsize=(TEXTWIDTH, nrows * TEXTWIDTH / _NCOLS * 0.95),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, top=0.95, bottom=0.10, hspace=0.45, wspace=0.12
        )

        for ax, steps in zip(axes.flat, all_steps, strict=False):
            for solver in solvers:
                m = data[solver].get(str(steps))
                if m is None or not is_ok(m):
                    continue
                eps = sorted(m["eps_sweep"], key=float)
                _, color, ls, mk = solver_props(solver)
                ax.loglog(
                    [float(e) for e in eps],
                    [m["eps_sweep"][e]["rel_error_mean"] for e in eps],
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=3.5,
                    markeredgewidth=0,
                    linewidth=1.4,
                )
            ax.set_title(f"{steps} steps", fontweight="normal")

        # The lowest panel of each column carries the x-axis labels.
        for ax in axes.flat[len(all_steps) - _NCOLS : len(all_steps)]:
            ax.set_xlabel(r"$\varepsilon$")
            ax.xaxis.set_tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Relative FD error")

        # The legend fills the unused panel slots.
        for ax in axes.flat[len(all_steps) :]:
            ax.axis("off")
        legend_ax = axes.flat[-1]
        legend_ax.legend(
            handles=[make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),
            frameon=False,
            handlelength=2.6,
        )

        out = out_dir / "appendix_ucurves_f2.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
