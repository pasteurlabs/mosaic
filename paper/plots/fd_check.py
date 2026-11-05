"""Generate Figure: FD gradient verification across all four benchmark domains.

2-row × 4-column grid: relative FD error and 1 - cosine similarity against epsilon (log-log).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import (
    NS_ORDER,
    RCPARAMS,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)

DOMAINS = [
    ("2D NS", "ns-grid", NS_ORDER),
    ("3D NS", "ns-3d-grid", NS_ORDER),
    ("Structural", "structural-mesh", STRUCTURAL_ORDER),
    ("Thermal", "thermal-mesh", THERMAL_ORDER),
]


def _plot_fd_check(ax_err, ax_cos, problem: str, order: list[str]) -> list[str]:
    """Plot one problem's FD sweep into the two axes; return the plotted solvers."""
    data = {
        solver: metrics
        for solver, cells in by_solver(
            load_result(problem, "gradient/fd_check")
        ).items()
        for metrics in cells.values()
        if is_ok(metrics)
    }
    solvers = [s for s in order if s in data]

    for solver in solvers:
        eps_sweep = data[solver]["eps_sweep"]
        epsilons = sorted(eps_sweep.keys(), key=float)
        eps_f = [float(e) for e in epsilons]

        rel_mean = [float(np.mean(eps_sweep[e]["rel_error"])) for e in epsilons]
        cos_vals = [max(1 - float(eps_sweep[e]["cosine"]), 1e-9) for e in epsilons]

        _label, color, ls, mk = solver_props(solver)
        kw = {
            "color": color,
            "linestyle": ls,
            "marker": mk,
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.6,
        }
        ax_err.loglog(eps_f, rel_mean, **kw)
        ax_cos.loglog(eps_f, cos_vals, **kw)

    for ax in (ax_err, ax_cos):
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    return solvers


def generate_main(out_dir: Path) -> None:
    """Generate the focused FD check figure (ns-grid only, 1×2 layout)."""
    plt.rcParams.update(RCPARAMS)

    fig, (ax_err, ax_cos) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.3), dpi=300
    )
    fig.subplots_adjust(bottom=0.38, wspace=0.45)

    solvers = _plot_fd_check(ax_err, ax_cos, "ns-grid", NS_ORDER)

    ax_err.set_title("Relative error (2D NS)")
    ax_err.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_err.set_ylabel("Rel. FD error")

    ax_cos.set_title("Cosine similarity (2D NS)")
    ax_cos.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_cos.set_ylabel(r"$1 - \cos(\nabla_\mathrm{AD},\, \nabla_\mathrm{FD})$")

    fig.legend(
        handles=[make_handle(s) for s in solvers],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(solvers),
        fontsize=6.0,
        framealpha=0.7,
        handlelength=2.0,
    )

    out = out_dir / "fd_check.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.60), sharex="col")
    fig.subplots_adjust(bottom=0.34, wspace=0.30, hspace=0.45)

    handles = []
    for col, (domain_label, problem, order) in enumerate(DOMAINS):
        ax_err, ax_cos = axes[0, col], axes[1, col]
        solvers = _plot_fd_check(ax_err, ax_cos, problem, order)
        handles += [make_handle(s) for s in solvers]

        ax_err.set_title(domain_label)
        ax_err.set_ylabel("Relative FD error" if col == 0 else "")
        ax_cos.set_xlabel(r"$\varepsilon$")
        ax_cos.set_ylabel("1 - cos. sim." if col == 0 else "")
        ax_cos.xaxis.set_major_locator(mticker.LogLocator(base=100))

    fig.legend(
        handles=dedup_handles(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    out = out_dir / "appendix_fd_check.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    generate_main(out_dir)
