"""Generate Figure: FD gradient verification across all four benchmark domains.

2-row × 4-column grid: relative FD error (log-log) and cosine similarity (semilog-x).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    dedup_handles,
    make_handle,
    solver_props,
)

DOMAINS = [
    ("2D NS", "ns-grid", "fd_check"),
    ("3D NS", "ns-3d-grid", "fd_check"),
    ("Structural", "structural-mesh", "fd_check"),
    ("Thermal", "thermal-mesh", "fd_check"),
]


def generate_main(out_dir: Path) -> None:
    """Generate the main-paper FD check figure (ns-grid only, 1×2 layout)."""
    plt.rcParams.update(RCPARAMS)

    path = results_dir() / "ns-grid" / "gradient" / "fd_check" / "result.json"
    data = load_json(path)

    fig, (ax_err, ax_cos) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.3), dpi=300
    )
    fig.subplots_adjust(bottom=0.38, wspace=0.45)

    all_cos: list[float] = []

    for solver, sdata in data["by_solver"].items():
        if solver in {"fenics_ns", "su2"}:
            continue
        eps_sweep = sdata["eps_sweep"]
        epsilons = sorted(eps_sweep.keys(), key=float)
        eps_f = [float(e) for e in epsilons]

        rel_mean = [float(np.mean(eps_sweep[e]["rel_error"])) for e in epsilons]
        cos_vals = [max(1 - float(eps_sweep[e]["cosine"]), 1e-9) for e in epsilons]
        all_cos.extend(cos_vals)

        label, color, ls, mk = solver_props(solver)
        kw = {
            "color": color,
            "linestyle": ls,
            "marker": mk,
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.6,
            "label": label,
        }

        ax_err.loglog(eps_f, rel_mean, **kw)
        ax_cos.loglog(eps_f, cos_vals, **kw)

    ax_err.set_title("Relative error (2D NS)")
    ax_err.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_err.set_ylabel("Rel. FD error")
    ax_err.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_err.yaxis.set_minor_locator(mticker.NullLocator())

    ax_cos.set_title("Cosine similarity (2D NS)")
    ax_cos.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_cos.set_ylabel(r"$1 - \cos(\nabla_\mathrm{AD},\, \nabla_\mathrm{FD})$")
    ax_cos.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_cos.yaxis.set_minor_locator(mticker.NullLocator())

    present = set(data["by_solver"].keys()) - {"fenics_ns", "su2"}
    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in present and s in SOLVER_STYLES]
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=7,
        fontsize=6.0,
        framealpha=0.7,
        handlelength=2.0,
    )

    out = out_dir / "fd_check.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
    return fig


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.60), sharex="col")
    fig.subplots_adjust(bottom=0.34, wspace=0.30, hspace=0.45)

    ns_seen: set[str] = set()
    fem_seen: set[str] = set()

    for col, (domain_label, subdir, exp_key) in enumerate(DOMAINS):
        path = results_dir() / subdir / "gradient" / exp_key / "result.json"
        data = load_json(path)

        ax_err = axes[0, col]
        ax_cos = axes[1, col]

        all_cos: list[float] = []

        for solver, sdata in data["by_solver"].items():
            if solver in {"fenics_ns", "su2"}:
                continue
            eps_sweep = sdata["eps_sweep"]
            epsilons = sorted(eps_sweep.keys(), key=float)
            eps_f = [float(e) for e in epsilons]

            rel_mean = [float(np.mean(eps_sweep[e]["rel_error"])) for e in epsilons]
            cos_vals = [max(1 - float(eps_sweep[e]["cosine"]), 1e-9) for e in epsilons]
            all_cos.extend(cos_vals)

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

            if solver in NS_ORDER:
                ns_seen.add(solver)
            if solver in FEM_ORDER:
                fem_seen.add(solver)

        ax_err.set_title(domain_label)
        ax_err.set_ylabel("Relative FD error" if col == 0 else "")
        ax_cos.set_xlabel(r"$\varepsilon$")
        ax_cos.set_ylabel("Cosine similarity" if col == 0 else "")

        ax_err.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax_err.yaxis.set_minor_locator(mticker.NullLocator())
        ax_cos.set_ylabel("1 - cos. sim." if col == 0 else "")
        ax_cos.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax_cos.yaxis.set_minor_locator(mticker.NullLocator())

    all_handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in ns_seen]
        + [make_handle(s) for s in FEM_ORDER if s in fem_seen]
    )
    fig.legend(
        handles=all_handles,
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

    return generate_main(out_dir)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
