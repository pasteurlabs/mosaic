"""Generate Figure: Physical accuracy of the 2D Navier-Stokes solvers on the periodic TGV.

Produces appendix_pa_ns2d.pdf, a 3×3 grid for the 2D NS domain.

Rows sweep N, ν and rollout steps (``forward/physical_laws/{vs_N,vs_nu,vs_steps}``);
columns show the error against the TGV reference, divergence RMS and kinetic energy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

# (run, row label); every sweep is log-spaced, so all x-axes are log.
SWEEPS = [("vs_N", "vs $N$"), ("vs_nu", r"vs $\nu$"), ("vs_steps", "vs steps")]
# (metric, log y)
METRICS = [
    ("analytic_error", True),
    ("divergence_rms", True),
    ("kinetic_energy", False),
]


def _ke_tgv2d(sweep_key: str, xs: list[float], phys: dict) -> list[float]:
    """Exact 2D TGV kinetic energy, KE(t) = 1/4 exp(-4 nu t), at each sweep value."""
    nu, steps = phys["nu"], phys["steps"]
    if sweep_key == "vs_nu":
        return [0.25 * np.exp(-4.0 * x * steps * phys["dt"]) for x in xs]
    if sweep_key == "vs_steps":
        return [0.25 * np.exp(-4.0 * nu * x * phys["dt"]) for x in xs]
    return [0.25 * np.exp(-4.0 * nu * steps * phys["dt"])] * len(xs)


def _set_axis_ticks(ax, xs: list[float], log_y: bool) -> None:
    ticks = sorted(set(xs))
    if len(ticks) > 4:
        ticks = [
            ticks[i] for i in np.round(np.linspace(0, len(ticks) - 1, 4)).astype(int)
        ]
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", labelsize=7.5, rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    if log_y and np.log10(ax.get_ylim()[1] / ax.get_ylim()[0]) < 1.5:
        # under ~one decade, decade ticks alone would leave the axis unlabeled
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    elif log_y:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=7.5)


def _plot_ns_domain(
    problem: str, title: str, col_titles: list[str], out_path: Path
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.85))
    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22, wspace=0.35, hspace=0.55)

    seen: set[str] = set()
    for row, (sweep_key, row_label) in enumerate(SWEEPS):
        result = load_result(problem, f"forward/physical_laws/{sweep_key}")
        data = by_solver(result)
        solvers = sorted(data, key=NS_ORDER.index)
        seen.update(solvers)

        for col, (metric, log_y) in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_yscale("log" if log_y else "linear")
            x_all: list[float] = []
            for solver in solvers:
                _label, color, ls, mk = solver_props(solver)
                pts = sorted(
                    (float(v), m[metric]) for v, m in data[solver].items() if is_ok(m)
                )
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=4,
                    markeredgewidth=0,
                    linewidth=1.6,
                )
                x_all.extend(xs)

            if metric == "kinetic_energy":
                xs = sorted(set(x_all))
                ax.plot(
                    xs,
                    _ke_tgv2d(sweep_key, xs, result["params"]["physics"]),
                    color="#aaaaaa",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=0,
                )

            if row == 0:
                ax.set_title(col_titles[col])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)
            if row == len(SWEEPS) - 1:
                ax.set_xlabel("Steps")
            _set_axis_ticks(ax, x_all, log_y)

    handles = [make_handle(s) for s in sorted(seen, key=NS_ORDER.index)]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)
    _plot_ns_domain(
        "ns-grid",
        "2D NS (periodic TGV) — physical accuracy",
        ["Analytic TGV error", "Divergence RMS", "Kinetic energy"],
        out_dir / "appendix_pa_ns2d.pdf",
    )
