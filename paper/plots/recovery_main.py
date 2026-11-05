"""Generate Figure: F3 initial-condition recovery with L-BFGS + projection.

Left: optimization loss vs L-BFGS iteration. Middle: IC error vs iteration.
Both curves are for the first IC seed, which is the only one whose history
the result stores. Right: final IC error per solver over all IC seeds (dot:
mean, bar: min to max).

Output: recovery_main.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

_EXPERIMENT = "optimization/recovery_constant_ic_bfgs_proj"


def generate(out_dir: Path) -> None:
    result = load_result("ns-3d-grid", _EXPERIMENT)
    snap = result["params"]["optim"]["snap_interval"]
    n_seeds = len(result["params"]["optim"]["ic_seeds"])
    runs = {s: next(iter(m.values())) for s, m in by_solver(result).items()}
    solvers = [s for s in NS_ORDER if s in runs and is_ok(runs[s])]

    with plt.rc_context(RCPARAMS):
        fig, (ax_loss, ax_ic, ax_final) = plt.subplots(
            1,
            3,
            figsize=(TEXTWIDTH, TEXTWIDTH * 0.36),
            gridspec_kw={"width_ratios": [1, 1, 0.85]},
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.36, wspace=0.42)

        for solver in solvers:
            m = runs[solver]
            _, color, ls, _ = solver_props(solver)
            kw = {"color": color, "linestyle": ls, "linewidth": 1.4}
            ax_loss.semilogy(np.arange(len(m["errors"])), m["errors"], **kw)
            # IC error: initial guess, every snap_interval-th iterate, final iterate.
            n_iters = len(m["errors"]) - 1
            hist = m["ic_error_history"]
            xs = [0, *range(snap, snap * len(hist) + 1, snap), n_iters]
            ys = [m["ic_error_init"], *hist, m["final_ic_error_trials"][0]]
            ax_ic.semilogy(xs, np.asarray(ys) * 100, **kw)

        ax_loss.set_title("Optimization loss")
        ax_loss.set_xlabel("L-BFGS iteration")
        ax_ic.set_title("IC error (%)")
        ax_ic.set_xlabel("L-BFGS iteration")
        for ax in (ax_loss, ax_ic):
            ax.set_xlim(left=0)

        # ── Final IC error over seeds, sorted ────────────────────────────────
        order = sorted(solvers, key=lambda s: runs[s]["final_ic_error"])
        for i, solver in enumerate(order):
            trials = np.asarray(runs[solver]["final_ic_error_trials"]) * 100
            _, color, _, mk = solver_props(solver)
            ax_final.plot(
                [trials.min(), trials.max()], [i, i], color=color, linewidth=1.4
            )
            ax_final.plot(
                trials.mean(),
                i,
                marker=mk,
                color=color,
                markersize=5,
                markeredgewidth=0,
            )
        ax_final.set_xscale("log")
        ax_final.set_yticks(range(len(order)), [solver_props(s)[0] for s in order])
        ax_final.tick_params(axis="y", length=0)
        ax_final.set_ylim(len(order) - 0.5, -0.5)
        ax_final.grid(axis="y", visible=False)
        ax_final.set_title(f"Final IC error (%), {n_seeds} seeds")

        fig.legend(
            handles=[make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(solvers),
            handlelength=2.0,
            columnspacing=1.2,
        )

        out = out_dir / "recovery_main.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
