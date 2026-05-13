"""FD gradient verification figures — single-experiment + cross-domain.

Three public entry points:

  * :func:`plot_experiment(cfg, exp_key)` — single-experiment 1×2 panel
    (relative FD error + cosine similarity). Used both as the
    canonical mosaic experiment plot (registered via ``plot=`` on
    :meth:`Problem.add_experiment`) and as one column of the
    cross-domain figure.
  * :func:`generate_main(out_dir)` — the main-paper figure (ns-grid only,
    1×2 layout). Thin wrapper that calls ``plot_experiment`` on the
    ns-grid cfg.
  * :func:`generate(out_dir)` — the appendix figure: 2×4 grid covering
    all four domains, plus a call to ``generate_main``.

Sharing the per-experiment helper between the experiment-plot registry
and the paper-figure registry means the styling stays in lockstep —
edits to the polished paper version land automatically on the
``mosaic run --plots-only`` output and vice versa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import experiment_dir, load_json, results_dir
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

# Domains that the cross-paper figure aggregates. Each entry maps a
# display label to the on-disk ``(problem, suite, experiment_key)`` triple.
_PAPER_DOMAINS = [
    ("2D NS", "ns-grid", "fd_check"),
    ("3D NS", "ns-3d-grid", "fd_check"),
    ("Structural", "structural-mesh", "fd_check"),
    ("Thermal", "thermal-mesh", "fd_check"),
]

# Solvers blacklisted from FD-check figures (e.g. variants that don't
# implement VJP at all and would just clutter the legend).
_BLACKLIST = {"fenics_ns", "su2"}


def _plot_curves(ax_err, ax_cos, data: dict, seen: dict[str, set]) -> None:
    """Plot per-solver rel-error + cosine curves into ``ax_err``/``ax_cos``.

    ``data`` is the parsed ``result.json`` for a single fd_check experiment.
    ``seen`` is a ``{"ns": set, "fem": set}`` dict the caller updates so the
    cross-domain figure can build a single shared legend across all panels.
    """
    for solver, sdata in data["by_solver"].items():
        if solver in _BLACKLIST:
            continue
        eps_sweep = sdata.get("eps_sweep") or {}
        if not eps_sweep:
            continue
        epsilons = sorted(eps_sweep.keys(), key=float)
        eps_f = [float(e) for e in epsilons]

        rel_mean = [float(np.mean(eps_sweep[e]["rel_error"])) for e in epsilons]
        # ``1 - cosine`` makes a 4-decade log scale meaningful when most
        # solvers cluster near 1; clamp the noise floor to 1e-9.
        cos_vals = [max(1 - float(eps_sweep[e]["cosine"]), 1e-9) for e in epsilons]

        _label, color, ls, mk = solver_props(solver)
        kw: dict[str, Any] = {
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
            seen["ns"].add(solver)
        if solver in FEM_ORDER:
            seen["fem"].add(solver)


def _style_axes(ax_err, ax_cos, *, title: str, ylabel_left: bool) -> None:
    """Apply consistent axis labels / ticks to one (err, cos) column."""
    ax_err.set_title(title)
    ax_err.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_err.set_ylabel("Relative FD error" if ylabel_left else "")
    ax_err.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_err.yaxis.set_minor_locator(mticker.NullLocator())

    ax_cos.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_cos.set_ylabel(
        r"$1 - \cos(\nabla_\mathrm{AD},\, \nabla_\mathrm{FD})$" if ylabel_left else ""
    )
    ax_cos.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_cos.yaxis.set_minor_locator(mticker.NullLocator())


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "fd_check",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure:
    """Single-experiment fd_check figure for ``(cfg, exp_key)``.

    Registered as the per-experiment plot via ``plot=`` on
    :meth:`Problem.add_experiment`. Reads
    ``<results>/<cfg.name>/gradient/<exp_key>{suffix}/result.json``
    and writes a 1×2 PDF (rel error, cosine) next to it.
    """
    plt.rcParams.update(RCPARAMS)

    out_dir = experiment_dir(results_dir(), cfg.name, "gradient", exp_key + suffix)
    data = load_json(out_dir / "result.json")

    fig, (ax_err, ax_cos) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.3), dpi=300
    )
    fig.subplots_adjust(bottom=0.38, wspace=0.45)

    seen = {"ns": set(), "fem": set()}
    _plot_curves(ax_err, ax_cos, data, seen)
    _style_axes(
        ax_err,
        ax_cos,
        title=f"FD check — {cfg.category_label or cfg.name}",
        ylabel_left=True,
    )

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen["ns"] and s in SOLVER_STYLES]
        + [make_handle(s) for s in FEM_ORDER if s in seen["fem"] and s in SOLVER_STYLES]
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 7),
            fontsize=6.0,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / "fd_check.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def generate_main(out_dir: Path) -> plt.Figure:
    """Main-paper figure: ns-grid fd_check only, 1×2 layout."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    plt.rcParams.update(RCPARAMS)

    data = load_json(
        results_dir() / "ns-grid" / "gradient" / "fd_check" / "result.json"
    )
    fig, (ax_err, ax_cos) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.3), dpi=300
    )
    fig.subplots_adjust(bottom=0.38, wspace=0.45)

    seen = {"ns": set(), "fem": set()}
    _plot_curves(ax_err, ax_cos, data, seen)
    ax_err.set_title("Relative error (2D NS)")
    ax_cos.set_title("Cosine similarity (2D NS)")
    _style_axes(ax_err, ax_cos, title=ax_err.get_title(), ylabel_left=True)
    # Re-stamp titles (the helper overwrites them).
    ax_err.set_title("Relative error (2D NS)")
    ax_cos.set_title("Cosine similarity (2D NS)")

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen["ns"] and s in SOLVER_STYLES]
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
    del cfg
    return fig


def generate(out_dir: Path) -> None:
    """Appendix figure: 2×4 grid over all four domains + main-paper figure."""
    plt.rcParams.update(RCPARAMS)

    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.60), sharex="col")
    fig.subplots_adjust(bottom=0.34, wspace=0.30, hspace=0.45)

    seen = {"ns": set(), "fem": set()}

    for col, (domain_label, subdir, exp_key) in enumerate(_PAPER_DOMAINS):
        path = results_dir() / subdir / "gradient" / exp_key / "result.json"
        if not path.exists():
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        data = load_json(path)
        ax_err, ax_cos = axes[0, col], axes[1, col]
        _plot_curves(ax_err, ax_cos, data, seen)
        _style_axes(ax_err, ax_cos, title=domain_label, ylabel_left=(col == 0))

    all_handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen["ns"]]
        + [make_handle(s) for s in FEM_ORDER if s in seen["fem"]]
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
