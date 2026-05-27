# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the navier-stokes-3d-grid recovery experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    fig_shared_legend,
    save_fig,
    solver_plot_props,
    solver_styles,
    subplots_grid,
)


def _resolve_recovery_out_dir(base_dir: Path, ic: str | None) -> Path:
    """Resolve the experiment directory: root-level, explicit IC, or auto-detected."""
    root_result = base_dir / "result.json"
    if root_result.exists() and ic is None:
        return base_dir
    if ic is not None:
        return base_dir / ic
    ic_dirs = sorted(
        p.parent for p in base_dir.glob("*/result.json") if p.parent != base_dir
    )
    if not ic_dirs:
        raise FileNotFoundError(
            f"No result.json found in {base_dir} or its subdirectories."
        )
    return ic_dirs[0]


def _sorted_sweep_vals(by_sweep: dict) -> list:
    """Collect ordered sweep values from the first solver's keys."""
    _first = next(iter(by_sweep.values()), {})
    return sorted(
        _first.keys(),
        key=lambda v: float(v) if str(v).replace(".", "").lstrip("-").isdigit() else 0,
    )


def _draw_convergence_panel(
    ax: Any, v: Any, by_sweep: dict, styles: dict, sweep_key: str
) -> None:
    """Draw one convergence-curve panel for sweep value *v*."""
    for name, s_results in by_sweep.items():
        r = s_results.get(v) or s_results.get(str(v))
        if not (r and r.get("errors")):
            continue
        errors = r["errors"]
        sty = styles.get(name, {})
        is_flat = (
            len(errors) > 1
            and (errors[0] - errors[-1]) / (abs(errors[0]) + 1e-30) < 0.01
        )
        label_str = sty.get("label", name)
        if is_flat:
            label_str += " (no grad)"
        line_kw = solver_plot_props(sty, marker=False)
        if is_flat:
            line_kw = {**line_kw, "linestyle": ":", "alpha": 0.55}
        ax.semilogy(errors, label=label_str, **line_kw)
        final_ic = r.get("final_ic_error")
        if final_ic is not None:
            color = sty.get("color", "gray")
            converged = r.get("converged", False)
            ic_label = f"IC={final_ic:.2f}"
            if not converged:
                ic_label += " ✗"
            ax.annotate(
                ic_label,
                xy=(len(errors) - 1, errors[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=6,
                color=color,
                va="center",
            )
    ax.set_xlabel("Iteration")
    ax.set_title(f"{sweep_key}={v}")


def plot_recovery(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    ic: str | None = None,
    exp_key: str = "optimization",
    **_kw: Any,
) -> Any:
    """Recovery convergence-curve plot: one panel per sweep value, all solvers overlaid."""
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    out_dir = _resolve_recovery_out_dir(base_dir, ic)
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)

    by_sweep = data.get("by_sweep") or data.get("by_horizon", {})
    sweep_key = data.get("sweep_key", "steps")
    sweep_vals = _sorted_sweep_vals(by_sweep)

    has_any_errors = any(
        (s_results.get(v) or s_results.get(str(v)) or {}).get("errors")
        for v in sweep_vals
        for s_results in by_sweep.values()
    )
    if not has_any_errors:
        return None

    fig_lc, axes_lc = subplots_grid(len(sweep_vals), panel_w=5, panel_h=4, sharey=True)
    for ax, v in zip(axes_lc, sweep_vals, strict=False):
        _draw_convergence_panel(ax, v, by_sweep, styles, sweep_key)
    axes_lc[0].set_ylabel("Optim loss (MSE)")
    fig_lc.suptitle(
        f"{cfg.name} — R1 convergence curves (all {sweep_key} values)\n"
        "IC=X.XX annotated at curve end = final IC recovery error "
        "(✗ means IC error > threshold; dotted line = no gradient / flat loss)"
    )
    fig_shared_legend(fig_lc, axes_lc)
    if save:
        save_fig(fig_lc, "convergence_curves", out_dir)
    return fig_lc
