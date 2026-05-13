"""Source / IC recovery single-experiment + cross-domain paper figure.

Two public entry points:

  * :func:`plot_experiment(cfg, *, exp_key, suffix, save)` — the canonical
    single-experiment paper-styled IC-recovery figure: loss / IC-error
    convergence curves per solver. Reads
    ``<results>/<cfg.name>/optimization/<exp_key><suffix>/result.json``
    and writes a paper-quality PDF in the same experiment directory.
    Used both as the per-experiment plot delegate (called from
    :func:`mosaic.benchmarks.problems.navier_stokes_3d_grid.plots.plot_recovery`)
    and as the source figure for the paper-output pipeline.
  * :func:`generate(out_dir)` — paper-output entry point. Produces the
    2×2 cross-domain ``recovery_convergence.pdf`` figure and the
    canonical single-experiment figure for ns-3d-grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import experiment_dir, load_json, results_dir
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

# Default perturb_sigma to highlight on the single-experiment NS figure.
_NS_SIGMA = "0.1"


def _solver_order_for(cfg_name: str) -> list[str]:
    """Heuristic solver-ordering pick by problem name."""
    if "ns" in cfg_name:
        return NS_ORDER
    if "structural" in cfg_name:
        return STRUCTURAL_ORDER
    if "thermal" in cfg_name:
        return THERMAL_ORDER
    return NS_ORDER


def _plot_ns_recovery(
    ax, data: dict, solver_order: list[str], title: str, seen: set[str]
) -> None:
    """Draw NS-style by_sweep convergence into *ax* from ``data``."""
    by_sweep = data.get("by_sweep") or data.get("by_horizon", {})

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
        _label, color, ls, _mk = solver_props(solver)
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
    data = load_json(result_path)
    by_solver = data["by_solver"]

    for solver in solver_order:
        if solver not in by_solver:
            continue
        vals = by_solver[solver].get(error_key, [])
        if not vals:
            continue
        _label, color, ls, _mk = solver_props(solver)
        ax.semilogy(range(len(vals)), vals, color=color, linestyle=ls, linewidth=1.6)
        seen.add(solver)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _ns_path_for(subdir: str) -> Path:
    """Resolve the canonical NS recovery result path (or fall back)."""
    primary = (
        results_dir() / subdir / "optimization" / "recovery_constant_ic" / "result.json"
    )
    if primary.exists():
        return primary
    return results_dir() / subdir / "optimization" / "optimization" / "result.json"


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "recovery_constant_ic",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure | None:
    """Single-experiment paper-styled IC-recovery convergence figure.

    Draws per-solver IC-error vs iteration on a log scale (using the
    representative sigma slice, see ``_NS_SIGMA``). Reads ``result.json``
    from the experiment directory and writes ``<exp_key>.pdf`` next to it
    when ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[recovery] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)
    data = load_json(result_path)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.45), dpi=300)
    fig.subplots_adjust(bottom=0.36, top=0.90, left=0.13, right=0.96)

    seen: set[str] = set()
    solver_order = _solver_order_for(cfg.name)

    if "by_sweep" in data or "by_horizon" in data:
        _plot_ns_recovery(
            ax,
            data,
            solver_order,
            f"IC recovery — {cfg.category_label or cfg.name}",
            seen,
        )
    else:
        # FEM-style by_solver layout (errors / losses per solver).
        by_solver = data.get("by_solver", {})
        error_key = (
            "errors" if any("errors" in v for v in by_solver.values()) else "losses"
        )
        for solver in solver_order:
            sdata = by_solver.get(solver)
            if not sdata:
                continue
            vals = sdata.get(error_key, [])
            if not vals:
                continue
            _label, color, ls, _mk = solver_props(solver)
            ax.semilogy(
                range(len(vals)), vals, color=color, linestyle=ls, linewidth=1.6
            )
            seen.add(solver)
        ax.set_title(f"Recovery — {cfg.category_label or cfg.name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error" if error_key == "errors" else "Loss")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    handles = dedup_handles([make_handle(s) for s in solver_order if s in seen])
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=7.5,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / f"{exp_key}.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def generate(out_dir: Path) -> None:
    """Paper-output entry point: 2×2 cross-domain figure + ns-3d-grid canonical figure."""
    from mosaic.benchmarks.problems import get_config

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.80))
        fig.subplots_adjust(hspace=0.50, wspace=0.38, bottom=0.20)

        ns_seen: set[str] = set()
        fem_seen: set[str] = set()

        for ax, subdir, title in [
            (axes[0, 0], "ns-grid", "2D NS — IC recovery"),
            (axes[0, 1], "ns-3d-grid", "3D NS — IC recovery"),
        ]:
            path = _ns_path_for(subdir)
            if path.exists():
                _plot_ns_recovery(ax, load_json(path), NS_ORDER, title, ns_seen)
            else:
                ax.set_title(title)

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

        legend_kw = {"fontsize": 7.5, "framealpha": 0.7, "handlelength": 2.0}
        if ns_handles:
            fig.legend(
                handles=ns_handles,
                loc="lower left",
                bbox_to_anchor=(0.02, 0.01),
                ncol=max(1, len(ns_handles) // 3),
                **legend_kw,
            )
        if fem_handles:
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

        # Canonical single-experiment figure (ns-3d-grid).
        try:
            cfg = get_config("ns-3d-grid")
        except Exception:
            return
        sub_fig = plot_experiment(
            cfg, exp_key="recovery_constant_ic", suffix="", save=False
        )
        if sub_fig is not None:
            plt.close(sub_fig)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
