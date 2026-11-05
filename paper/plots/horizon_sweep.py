"""Generate Figure: Horizon sweep for ns-grid (2D NS).

1×3 panels against rollout steps (log x-axis):
  - gradient norm
  - FD relative error at the best ε
  - 1 − cosine similarity at that ε
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props


def _best_eps(metrics: dict) -> dict:
    """Entry of the FD ε sweep with the lowest mean relative error."""
    return min(metrics["eps_sweep"].values(), key=lambda e: e["rel_error_mean"])


def generate(out_dir: Path) -> None:
    data = by_solver(load_result("ns-grid", "gradient/horizon_sweep"))
    solvers = [s for s in NS_ORDER if s in data]

    with plt.rc_context(RCPARAMS):
        fig, (ax_gn, ax_err, ax_cos) = plt.subplots(
            1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42)
        )
        fig.subplots_adjust(bottom=0.36, wspace=0.58, left=0.09, right=0.98, top=0.91)

        for solver in solvers:
            ok = {int(k): m for k, m in data[solver].items() if is_ok(m)}
            steps = sorted(ok)
            best = [_best_eps(ok[s]) for s in steps]
            _, color, ls, mk = solver_props(solver)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 3.5,
                "markeredgewidth": 0,
            }
            ax_gn.semilogx(steps, [ok[s]["grad_norm"] for s in steps], **kw)
            ax_err.loglog(steps, [b["rel_error_mean"] for b in best], **kw)
            ax_cos.loglog(steps, [1.0 - b["cosine_mean"] for b in best], **kw)

        ax_gn.set_title("Gradient norm")
        ax_gn.set_ylabel(r"$\|\nabla\mathcal{L}\|$")
        ax_err.set_title(r"FD error (best $\varepsilon$)")
        ax_err.set_ylabel("Relative FD error")
        ax_cos.set_title(r"Cosine (best $\varepsilon$)")
        ax_cos.set_ylabel(r"$1 -$ cosine similarity")
        for ax in (ax_gn, ax_err, ax_cos):
            ax.set_xlabel("Rollout steps")

        fig.legend(
            handles=[make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            handlelength=2.6,
        )

        out = out_dir / "horizon_sweep.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
