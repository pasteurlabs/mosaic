"""Generate Figure: Source / IC recovery convergence across all four domains.

2×2 grid:
  top-left:  2D NS IC recovery
  top-right: 3D NS IC recovery
  bot-left:  Structural load recovery
  bot-right: Thermal conductivity recovery
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)

# Default perturb_sigma to show for NS IC recovery
_NS_SIGMA = "0.1"


def _plot_ns_recovery(
    ax, subdir: str, solver_order: list[str], title: str, seen: set[str]
) -> None:
    path = results_dir() / subdir / "optimization" / "optimization" / "result.json"
    if not path.exists():
        ax.set_title(title)
        return
    data = json.loads(path.read_text())
    by_sweep = data["by_sweep"]

    for solver in solver_order:
        if solver not in by_sweep:
            continue
        sweep = by_sweep[solver]
        if not sweep:
            continue
        sigma_key = (
            _NS_SIGMA if _NS_SIGMA in sweep else sorted(sweep.keys())[len(sweep) // 2]
        )
        errors = sweep[sigma_key].get("errors") or sweep[sigma_key].get(
            "ic_error_history", []
        )
        if not errors:
            continue
        label, color, ls, mk = solver_props(solver)
        ax.semilogy(
            range(len(errors)), errors, color=color, linestyle=ls, linewidth=1.6
        )
        seen.add(solver)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("IC error")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _plot_fem_recovery(
    ax,
    result_path: Path,
    error_key: str,
    solver_order: list[str],
    title: str,
    ylabel: str,
    seen: set[str],
) -> None:
    if not result_path.exists():
        ax.set_title(title)
        return
    data = json.loads(result_path.read_text())
    by_solver = data["by_solver"]

    for solver in solver_order:
        if solver not in by_solver:
            continue
        vals = by_solver[solver].get(error_key, [])
        if not vals:
            continue
        label, color, ls, mk = solver_props(solver)
        ax.semilogy(range(len(vals)), vals, color=color, linestyle=ls, linewidth=1.6)
        seen.add(solver)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.80))
        fig.subplots_adjust(hspace=0.50, wspace=0.38, bottom=0.20)

        ns_seen: set[str] = set()
        fem_seen: set[str] = set()

        _plot_ns_recovery(
            axes[0, 0], "ns-grid", NS_ORDER, "2D NS — IC recovery", ns_seen
        )
        _plot_ns_recovery(
            axes[0, 1], "ns-3d-grid", NS_ORDER, "3D NS — IC recovery", ns_seen
        )
        _plot_fem_recovery(
            axes[1, 0],
            results_dir()
            / "structural-mesh"
            / "optimization"
            / "load_recovery"
            / "result.json",
            error_key="losses",
            solver_order=STRUCTURAL_ORDER,
            title="Structural — load recovery",
            ylabel="Loss",
            seen=fem_seen,
        )
        _plot_fem_recovery(
            axes[1, 1],
            results_dir()
            / "thermal-mesh"
            / "optimization"
            / "conductivity_recovery"
            / "result.json",
            error_key="errors",
            solver_order=THERMAL_ORDER,
            title="Thermal — conductivity recovery",
            ylabel="Error",
            seen=fem_seen,
        )

        ns_handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in ns_seen])
        fem_handles = dedup_handles(
            [make_handle(s) for s in FEM_ORDER if s in fem_seen]
        )

        legend_kw = dict(fontsize=7.5, framealpha=0.7, handlelength=2.0)
        fig.legend(
            handles=ns_handles,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.01),
            ncol=max(1, len(ns_handles) // 3),
            **legend_kw,
        )
        fig.legend(
            handles=fem_handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.01),
            ncol=max(1, len(fem_handles) // 3),
            **legend_kw,
        )

        out = out_dir / "recovery_convergence.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
