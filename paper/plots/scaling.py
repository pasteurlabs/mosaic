"""Generate Figure: Solver scaling — forward time, VJP time, gradient overhead ratio vs problem size.

Two figures, each with a shared legend:
  scaling_ns_combined.pdf  — 2D NS + 3D NS (x-axis: grid cells)
  scaling_fem_combined.pdf — Structural + Thermal (x-axis: hex elements)

Each domain gets one row of 3 log-log panels: forward | VJP | ratio.
Solver labels carry (G) / (C) for GPU vs. CPU execution. A failed run is
marked with an x at the top edge of its panel.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result, solver_metadata
from .style import (
    NS_ORDER,
    RCPARAMS,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    make_handle,
    solver_props,
)

# (row label, problem, sweep value -> number of cells/elements); mesh sizes
# follow the canonical geometries in mosaic/benchmarks/problems/*/physics.py.
NS_ROWS = [
    ("2D NS", "ns-grid", lambda n: n**2),
    ("3D NS", "ns-3d-grid", lambda n: n**3),
]
FEM_ROWS = [
    ("Structural", "structural-mesh", lambda nx: nx * 2 * max(1, nx // 2)),
    ("Thermal", "thermal-mesh", lambda nx: nx * max(1, nx // 2)),
]

Row = tuple[str, str, Callable[[int], int]]


def _fmt_size(x: float, _pos=None) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{round(x / 1e3):.0f}k"
    return str(int(x))


def _times(metrics: dict[str, dict], size: Callable[[int], int]):
    """Split one solver's sweep into ({size: mean time}, [sizes of failed runs])."""
    ok = {size(int(v)): m["mean"] for v, m in metrics.items() if is_ok(m)}
    failed = sorted(size(int(v)) for v, m in metrics.items() if not is_ok(m))
    return dict(sorted(ok.items())), failed


def _combined_fig(rows: list[Row], order: list[str], xlabel: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.60), dpi=300)
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.28, top=0.94, wspace=0.40, hspace=0.55
    )

    gpu: dict[str, bool] = {}
    for (row_label, problem, size), (ax_fwd, ax_vjp, ax_ratio) in zip(
        rows, axes, strict=True
    ):
        fwd = by_solver(load_result(problem, "cost/spatial_cost"))
        vjp = by_solver(load_result(problem, "cost/vjp_cost/by_N"))
        meta = solver_metadata(problem)
        solvers = sorted(set(fwd) | set(vjp), key=order.index)
        sizes: set[int] = set()

        for solver in solvers:
            _label, color, ls, mk = solver_props(solver)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 4,
                "markeredgewidth": 0,
                "linewidth": 1.5,
            }
            f_ok, f_fail = _times(fwd.get(solver, {}), size)
            v_ok, v_fail = _times(vjp.get(solver, {}), size)
            common = [n for n in f_ok if n in v_ok]
            for ax, xs, ys, failed in (
                (ax_fwd, list(f_ok), list(f_ok.values()), f_fail),
                (ax_vjp, list(v_ok), list(v_ok.values()), v_fail),
                (ax_ratio, common, [v_ok[n] / f_ok[n] for n in common], []),
            ):
                ax.loglog(xs, ys, **kw)
                ax.plot(
                    failed,
                    [1.0] * len(failed),
                    "x",
                    color=color,
                    markersize=5,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                )
                sizes.update(xs, failed)
            gpu.setdefault(solver, meta[solver]["uses_gpu"])

        ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)
        ax_ratio.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax_ratio.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax_fwd.set_ylabel(f"{row_label}\nTime (s)", fontsize=7.5)
        ax_vjp.set_xlabel(xlabel, fontsize=7.5)

        # x-ticks: the sweep sizes closest to 4 log-spaced targets
        logs = np.log(sorted(sizes))
        targets = np.linspace(logs[0], logs[-1], 4)
        ticks = sorted(
            {round(np.exp(logs[np.abs(logs - t).argmin()])) for t in targets}
        )
        for ax in (ax_fwd, ax_vjp):
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        for ax in (ax_fwd, ax_vjp, ax_ratio):
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_size))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.tick_params(axis="y", labelsize=7)

    for ax, title in zip(
        axes[0], ("Forward time", "VJP time", "VJP / forward"), strict=True
    ):
        ax.set_title(title)

    handles = []
    for s in sorted(gpu, key=order.index):
        h = make_handle(s)
        h.set_label(h.get_label() + (" (G)" if gpu[s] else " (C)"))
        handles.append(h)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )
    return fig


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)
    fem_order = list(dict.fromkeys(STRUCTURAL_ORDER + THERMAL_ORDER))
    for name, rows, order, xlabel in (
        ("scaling_ns_combined", NS_ROWS, NS_ORDER, "Grid cells"),
        ("scaling_fem_combined", FEM_ROWS, fem_order, "Elements"),
    ):
        fig = _combined_fig(rows, order, xlabel)
        for ext in ("pdf", "png"):
            out = out_dir / f"{name}.{ext}"
            fig.savefig(out)
            print(f"Saved {out}")
        plt.close(fig)
