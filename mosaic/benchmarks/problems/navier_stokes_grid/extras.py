# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-domain ``_extra/`` aggregator plots for ns-grid.

Each ``_plot_<name>`` function has the signature ``(cfg, **kw) -> None``
expected by :meth:`Problem.add_extra_plot`. It resolves
``<results>/<cfg.name>/_extra/`` itself and writes one figure scoped to
this single domain (2D NS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.cost_overview import (
    plot_cost_overview_for,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    RCPARAMS,
    TEXTWIDTH,
    dedup_handles,
    make_handle,
    rc_context,
    resolve_solver_alias,
    solver_props,
)


def _extra_dir(cfg: Problem) -> Path:
    out_dir = results_dir() / cfg.name / "_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _n_to_elements_2d(N: int) -> int:
    return N**2


# ─────────────────────────────────────────────────────────────────────────────
# cost_overview
# ─────────────────────────────────────────────────────────────────────────────


def _plot_cost_overview(cfg: Problem, **_kw: Any) -> None:
    plot_cost_overview_for(cfg, steady_state=False)


# ─────────────────────────────────────────────────────────────────────────────
# scaling — 2D NS forward / VJP / ratio
# ─────────────────────────────────────────────────────────────────────────────


def _extract_scaling(by_n: dict) -> dict[int, float]:
    out = {}
    for k, v in by_n.items():
        if v is not None and isinstance(v, dict) and v.get("mean") is not None:
            out[int(k)] = float(v["mean"])
    return out


def _load_cost(subdir: str, experiment: str) -> dict[str, dict[int, float]]:
    p = results_dir() / subdir / "cost" / experiment / "result.json"
    if not p.exists():
        return {}
    data = load_json(p)
    return {s: _extract_scaling(nd) for s, nd in data.get("by_N", {}).items()}


def _scaling_impl(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    subdir = "ns-grid"
    fwd_data = _load_cost(subdir, "spatial_cost")
    vjp_data = _load_cost(subdir, "vjp_cost")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.34),
        sharey=False,
        dpi=300,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.88, wspace=0.40)
    ax_fwd, ax_vjp, ax_ratio = axes

    all_els: set[int] = set()
    seen: set[str] = set()

    # ``fwd_data`` / ``vjp_data`` are keyed by spec.name (display form);
    # build an alias→display map so we can iterate NS_ORDER (alias form).
    _display_names = set(fwd_data) | set(vjp_data)
    alias_to_display: dict[str, str] = {}
    for display_name in _display_names:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name

    for alias in NS_ORDER:
        display_name = alias_to_display.get(alias)
        if display_name is None:
            continue
        fwd_pts = fwd_data.get(display_name, {})
        vjp_pts = vjp_data.get(display_name, {})
        if not fwd_pts and not vjp_pts:
            continue

        _label, color, ls, mk = solver_props(alias)
        kw = {
            "color": color,
            "linestyle": ls,
            "marker": mk,
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.5,
        }

        if fwd_pts:
            ns_f = sorted(fwd_pts)
            els_f = [_n_to_elements_2d(n) for n in ns_f]
            ax_fwd.loglog(els_f, [fwd_pts[n] for n in ns_f], **kw)
            all_els.update(els_f)

        if vjp_pts:
            ns_v = sorted(vjp_pts)
            els_v = [_n_to_elements_2d(n) for n in ns_v]
            ax_vjp.loglog(els_v, [vjp_pts[n] for n in ns_v], **kw)
            all_els.update(els_v)

        common_ns = sorted(set(fwd_pts) & set(vjp_pts))
        if len(common_ns) >= 2:
            els_c = [_n_to_elements_2d(n) for n in common_ns]
            ratios = [vjp_pts[n] / fwd_pts[n] for n in common_ns]
            ax_ratio.loglog(els_c, ratios, **kw)
            all_els.update(els_c)

        seen.add(alias)

    ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)

    ax_fwd.set_title("Forward time")
    ax_vjp.set_title("VJP time")
    ax_ratio.set_title("VJP / forward")
    ax_fwd.set_ylabel("2D NS\nTime (s)", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("DOFs", fontsize=7.5)

    for ax in axes:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    tick_els = sorted(all_els)
    if len(tick_els) > 4:
        idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
        tick_els = [tick_els[i] for i in idx]

    fmt = mticker.FuncFormatter(
        lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
    )
    for ax in axes:
        ax.set_xticks(tick_els)
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="x", labelsize=7, rotation=35)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", labelsize=7)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )

    out = out_dir / "scaling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_scaling(cfg: Problem, **_kw: Any) -> None:
    _scaling_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# ucurves — F2 (2D NS) FD U-curves
# ─────────────────────────────────────────────────────────────────────────────


def _plot_ucurve_domain(cfg_dict: dict, out_dir: Path) -> None:
    path: Path = cfg_dict["path"]
    if not path.exists():
        print(f"[ucurves] {path} not found — skipping")
        return

    data = load_json(path)
    by_solver: dict = data["by_solver"]

    all_steps: list[int] = sorted(
        {int(s) for sv in by_solver.values() for s in sv},
        key=int,
    )
    ncols: int = cfg_dict["ncols"]
    nrows: int = int(np.ceil(len(all_steps) / ncols))

    panel_w = TEXTWIDTH / ncols
    panel_h = panel_w * 0.92
    fig_h = nrows * panel_h + 0.55

    fig = plt.figure(figsize=(TEXTWIDTH, fig_h))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        left=0.10,
        right=0.98,
        top=1.0 - 0.12 / fig_h,
        bottom=0.52 / fig_h,
        hspace=0.65,
        wspace=0.40,
    )

    seen: set[str] = set()

    # ``by_solver`` is keyed by spec.name (display form); build alias→display.
    alias_to_display: dict[str, str] = {}
    for display_name in by_solver:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name

    for idx, steps in enumerate(all_steps):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        for alias in NS_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            sv = by_solver.get(display_name)
            if sv is None:
                continue
            entry = sv.get(str(steps))
            if entry is None:
                continue
            eps_sweep: dict = entry.get("eps_sweep", {})
            if not eps_sweep:
                continue

            eps_vals = sorted(eps_sweep.keys(), key=float)
            xs = [float(e) for e in eps_vals]
            ys = [eps_sweep[e]["rel_error_mean"] for e in eps_vals]

            if not all(np.isfinite(y) and y > 0 for y in ys):
                pairs = [
                    (x, y)
                    for x, y in zip(xs, ys, strict=False)
                    if np.isfinite(y) and y > 0
                ]
                if not pairs:
                    continue
                xs, ys = zip(*pairs, strict=False)

            _, color, ls, mk = solver_props(alias)
            ax.loglog(
                xs,
                ys,
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=3.5,
                markeredgewidth=0,
                linewidth=1.4,
            )
            seen.add(alias)

        ax.set_title(f"$T={steps}$", fontsize=8)
        ax.set_xlabel(r"$\varepsilon$", fontsize=7.5)
        if col == 0:
            ax.set_ylabel("Rel. FD error", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    for idx in range(len(all_steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 6) if handles else 1,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.0,
    )

    out = out_dir / cfg_dict["out"]
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_ucurves(cfg: Problem, **_kw: Any) -> None:
    out_dir = _extra_dir(cfg)
    cfg_dict = {
        "path": results_dir()
        / "ns-grid"
        / "gradient"
        / "horizon_sweep"
        / "result.json",
        "out": "ucurves.png",
        "ncols": 4,
    }
    with rc_context():
        _plot_ucurve_domain(cfg_dict, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Registration entry point
# ─────────────────────────────────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Register the per-domain ``_extra/`` plot fns on *problem*."""
    problem.add_extra_plot("_extra/cost_overview", _plot_cost_overview)
    problem.add_extra_plot("_extra/scaling", _plot_scaling)
    problem.add_extra_plot("_extra/ucurves", _plot_ucurves)
