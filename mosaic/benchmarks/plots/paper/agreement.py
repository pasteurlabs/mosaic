"""Generate Figure: Forward agreement errors across all four benchmark domains.

1×4 panel figure with a shared two-group legend below (NS solvers | FEM solvers).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    dedup_handles,
    make_handle,
)

RESULTS = Path(__file__).parent.parent.parent / "results"

DOMAINS = [
    ("2D NS (TGV)", "ns-grid", "forward/agreement/tgv", r"$\nu$"),
    ("3D NS", "ns-3d-grid", "forward/agreement", r"$\nu$"),
    ("Structural", "structural-mesh", "forward/agreement", r"$\rho_0$"),
    ("Thermal", "thermal-mesh", "forward/agreement", r"$\rho_0$"),
]


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.50), sharey=False)
    fig.subplots_adjust(bottom=0.34, wspace=0.35)

    ns_seen: set[str] = set()
    fem_seen: set[str] = set()

    for col, (domain_label, subdir, exp_path, xlabel) in enumerate(DOMAINS):
        path = RESULTS / subdir / exp_path / "result.json"
        with open(path) as f:
            d = json.load(f)

        by_param = d["by_param"]
        params = sorted(by_param.keys(), key=float)
        x_vals = [float(p) for p in params]
        ax = axes[col]

        all_solvers: list[str] = []
        for pdata in by_param.values():
            for s in pdata:
                if s not in all_solvers:
                    all_solvers.append(s)

        for solver in all_solvers:
            style = SOLVER_STYLES.get(solver)
            if style is None:
                continue
            label, color, ls, mk = style

            ys, xs_valid = [], []
            for px, p in zip(x_vals, params):
                entry = by_param[p].get(solver, {})
                err = entry.get("error")
                valid = entry.get("valid", False)
                if err is not None and valid:
                    ys.append(float(err))
                    xs_valid.append(px)

            if ys:
                ax.semilogy(
                    xs_valid,
                    ys,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=4,
                    markeredgewidth=0,
                    linewidth=1.6,
                )
                if solver in NS_ORDER:
                    ns_seen.add(solver)
                if solver in FEM_ORDER:
                    fem_seen.add(solver)

        ax.set_title(domain_label)
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel("Relative error vs. consensus")

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

    out = out_dir / "appendix_agreement.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
