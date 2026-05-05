"""Generate Figure: Horizon sweep for ns-grid.

1×3 panels:
  - semilogy  grad_norm vs steps
  - semilogy  best FD rel_error vs steps
  - plot      best FD cosine vs steps
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

RESULTS = Path(__file__).parent.parent.parent / "results"

SOLVER_ORDER = ["jax_cfd", "phiflow", "ins_jl", "pict", "xlb", "warp_ns"]


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        path = RESULTS / "ns-grid" / "gradient" / "horizon_sweep" / "result.json"
        data = json.loads(path.read_text())

        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42))
        fig.subplots_adjust(bottom=0.30, wspace=0.40)

        ax_gn, ax_err, ax_cos = axes

        present: set[str] = set()

        for solver, h_results in data["by_solver"].items():
            label, color, ls, mk = SOLVER_STYLES.get(
                solver, (solver, "#888888", "-", "o"))

            step_keys = sorted(h_results.keys(), key=int)
            steps = [int(k) for k in step_keys]

            grad_norms = [float(h_results[k]["grad_norm"]) for k in step_keys]

            # Best FD rel_error and cosine across eps at each step
            rel_errors = []
            cosines = []
            for k in step_keys:
                eps_sweep = h_results[k]["eps_sweep"]
                best_rel = min(float(eps_sweep[e]["rel_error_mean"])
                               for e in eps_sweep)
                best_cos = max(float(eps_sweep[e]["cosine_mean"])
                               for e in eps_sweep)
                rel_errors.append(best_rel)
                cosines.append(best_cos)

            kw = dict(color=color, linestyle=ls, marker=mk, markersize=4,
                      markeredgewidth=0, linewidth=1.6, label=label)

            ax_gn.semilogy(steps, grad_norms, **kw)
            ax_err.semilogy(steps, rel_errors, **kw)
            ax_cos.plot(steps, cosines, **kw)

            present.add(solver)

        ax_gn.set_title("Gradient norm")
        ax_gn.set_xlabel("Steps $T$")
        ax_gn.set_ylabel(r"$\|\nabla J\|$")

        ax_err.set_title("FD relative error (best $\epsilon$)")
        ax_err.set_xlabel("Steps $T$")
        ax_err.set_ylabel("Relative FD error")

        ax_cos.set_title("Cosine similarity (best $\epsilon$)")
        ax_cos.set_xlabel("Steps $T$")
        ax_cos.set_ylabel("Cosine similarity")
        ax_cos.ticklabel_format(axis="y", useOffset=False)

        handles = [
            mlines.Line2D([], [],
                          color=SOLVER_STYLES[s][1],
                          linestyle=SOLVER_STYLES[s][2],
                          marker=SOLVER_STYLES[s][3],
                          markersize=5, markeredgewidth=0, linewidth=1.6,
                          label=SOLVER_STYLES[s][0])
            for s in SOLVER_ORDER if s in present
        ]

        fig.legend(handles=handles,
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=3,
                   fontsize=7.5,
                   framealpha=0.7,
                   handlelength=2.0)

        out = out_dir / "horizon_sweep.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
