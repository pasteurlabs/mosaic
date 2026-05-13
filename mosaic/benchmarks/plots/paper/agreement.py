"""Cross-solver agreement figures — single-experiment + cross-domain.

Three public entry points (mirrors the :mod:`fd_check` layout):

  * :func:`plot_experiment(cfg, exp_key)` — single-experiment 1-panel
    figure: per-solver error vs sweep parameter, in paper styling.
    Used both as the canonical mosaic experiment plot (delegated to by
    :func:`mosaic.benchmarks.problems.shared.plots.forward.plot_agreement`)
    and as one panel of the cross-domain figure.
  * :func:`generate(out_dir)` — the appendix figure: 2×3 grid covering
    multiple ``(problem, exp_key)`` pairs across F2/F3.

Sharing the per-experiment helper between the experiment-plot registry
and the paper-figure registry means the styling stays in lockstep —
edits to the polished paper version land automatically on the
``mosaic run --plots-only`` output and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import experiment_dir, load_json, results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    dedup_handles,
    make_handle,
    solver_props,
)

# Cross-domain panel registry: (row, col, problem, exp_key, x_label, log_x,
# title, y_label) — drives the 2×3 appendix figure.
_PAPER_PANELS = [
    (
        0,
        0,
        "ns-grid",
        "tgv_nu_sweep",
        r"$\nu$",
        True,
        "F2 — TGV agreement vs $\\nu$",
        "TGV analytic error",
    ),
    (
        0,
        1,
        "ns-grid",
        "cylinder",
        r"$\nu$",
        True,
        "F2 — cylinder flow vs $\\nu$",
        "Consensus error",
    ),
    (
        0,
        2,
        "ns-grid",
        "baseline",
        "$N$",
        True,
        "F2 — convergence vs $N$",
        "TGV analytic error",
    ),
    (
        1,
        0,
        "ns-3d-grid",
        "agreement",
        r"$\nu$",
        True,
        "F3 — TGV agreement vs $\\nu$",
        "TGV analytic error",
    ),
    (
        1,
        2,
        "ns-3d-grid",
        "baseline",
        "$N$",
        True,
        "F3 — convergence vs $N$",
        "TGV analytic error",
    ),
]


# Greek-letter sweep keys → LaTeX math; everything else is wrapped in $…$.
_MATH_LABELS = {
    "nu": r"$\nu$",
    "mu": r"$\mu$",
    "rho": r"$\rho$",
    "sigma": r"$\sigma$",
    "alpha": r"$\alpha$",
}


def _math_label(sweep_key: str) -> str:
    """Render *sweep_key* as a math-mode axis label."""
    if not sweep_key:
        return ""
    return _MATH_LABELS.get(sweep_key, f"${sweep_key}$")


def _plot_curves(ax, data: dict, seen: set[str]) -> None:
    """Plot per-solver error-vs-sweep curves onto ``ax``.

    ``data`` is the parsed forward-suite ``result.json``. Walks
    :data:`NS_ORDER`, drawing every solver that produced at least one valid
    finite positive error. Updates ``seen`` so the caller can build a
    legend covering only solvers that actually appear.
    """
    by_param = data.get("by_param", {})
    if not by_param:
        return
    params = sorted(by_param.keys(), key=float)

    for solver in NS_ORDER:
        _label, color, ls, mk = solver_props(solver)
        xs, ys = [], []
        for p in params:
            entry = by_param[p].get(solver)
            if isinstance(entry, dict):
                err = entry.get("error")
                if (
                    err is not None
                    and isinstance(err, float)
                    and np.isfinite(err)
                    and err > 0
                ):
                    xs.append(float(p))
                    ys.append(err)
        if not xs:
            continue
        ax.semilogy(
            xs,
            ys,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4,
            markeredgewidth=0,
            linewidth=1.6,
        )
        seen.add(solver)


def _style_axis(
    ax,
    *,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool,
    has_data: bool = True,
) -> None:
    """Apply consistent axis labels / ticks to one panel.

    ``has_data=False`` (no solver produced a finite-positive error)
    bypasses the log-scale locators and prints a placeholder annotation
    instead — matplotlib's ``LogLocator`` raises
    ``"Data has no positive values"`` the moment a log-scaled axis is
    drawn empty, so we avoid setting log scale in that case.
    """
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if not has_data:
        ax.text(
            0.5,
            0.5,
            "no positive data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
            fontsize=7,
        )
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        return
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if log_x:
        ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "agreement",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure:
    """Single-experiment agreement figure for ``(cfg, exp_key)``.

    Reads
    ``<results>/<cfg.name>/forward/<exp_key>{suffix}/result.json``
    and writes ``agreement.pdf`` next to it. The single axis plots
    per-solver error vs the sweep parameter (sweep_key inferred from
    the result file), styled with the paper palette + rcParams.
    """
    plt.rcParams.update(RCPARAMS)

    out_dir = experiment_dir(results_dir(), cfg.name, "forward", exp_key + suffix)
    data = load_json(out_dir / "result.json")
    sweep_key = data.get("sweep_key", "param")
    reference_label = data.get("reference_label", "consensus")
    ref_desc = "analytic" if reference_label == "analytic" else "consensus"

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, TEXTWIDTH * 0.4), dpi=300)
    fig.subplots_adjust(bottom=0.30, left=0.18, right=0.95, top=0.88)

    seen: set[str] = set()
    _plot_curves(ax, data, seen)

    # x-axis: log when any value is > 0 (sweep is over a physical scale).
    x_label = _math_label(sweep_key)
    _style_axis(
        ax,
        title=f"{cfg.category_label or cfg.name} — vs {sweep_key}",
        x_label=x_label,
        y_label=f"Error vs {ref_desc}",
        log_x=True,
        has_data=bool(seen),
    )

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen and s in SOLVER_STYLES]
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=6.5,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / "agreement.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def generate(out_dir: Path) -> None:
    """Appendix figure: 2×3 grid over ``_PAPER_PANELS`` plus a shared legend."""
    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(TEXTWIDTH, TEXTWIDTH * 0.72),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.50, wspace=0.38, bottom=0.20)

        # Hide panels that have no registered entry (e.g. F3/middle column).
        used: set[tuple[int, int]] = set()

        seen: set[str] = set()
        R = results_dir()

        for row, col, problem, exp_key, x_label, log_x, title, y_label in _PAPER_PANELS:
            ax = axes[row][col]
            used.add((row, col))
            path = R / problem / "forward" / exp_key / "result.json"

            if not path.exists():
                ax.set_visible(False)
                continue

            data = load_json(path)
            panel_seen: set[str] = set()
            _plot_curves(ax, data, panel_seen)
            seen.update(panel_seen)
            _style_axis(
                ax,
                title=title,
                x_label=x_label,
                y_label=y_label,
                log_x=log_x,
                has_data=bool(panel_seen),
            )

        # Hide any axes not addressed by ``_PAPER_PANELS``.
        for r in range(2):
            for c in range(3):
                if (r, c) not in used:
                    axes[r][c].set_visible(False)

        handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=7.5,
            framealpha=0.9,
            edgecolor="0.8",
            handlelength=2.0,
        )

        out = out_dir / "appendix_agreement.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
