# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the thermal-mesh conductivity recovery experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, v1_to_legacy
from mosaic.benchmarks.problems.shared.plots.style import (
    RCPARAMS,
    THERMAL_ORDER,
    paper_row,
    resolve_solver_alias,
    save_fig,
    solver_plot_props,
    solver_styles,
)


def plot_conductivity_recovery(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "conductivity_recovery",
    **_kw: Any,
) -> Any:
    """Identification-error convergence curves for conductivity recovery.

    A single log-scale loss-curve panel: each solver's final identification
    error is its curve's endpoint, so a separate final-error bar chart added
    nothing (and its linear scale buried the well-converged solvers under the
    poorly-converged ones). The curves carry both the convergence behaviour and
    the final ranking.
    """
    out_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    data = v1_to_legacy(load_json(out_dir / "result.json"))
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    plt.rcParams.update(RCPARAMS)

    styles = solver_styles(cfg, differentiable_only=False)
    names = list(by_solver.keys())

    fig_cv, ax_lc = paper_row(1)

    # Collect (alias, handle) per plotted curve so the legend uses the *same*
    # styles as the curves (build the legend from the artists, not from a
    # second style source, which would mis-colour the swatches).
    handles_by_alias: dict[str, Any] = {}
    for name in names:
        res = by_solver[name]
        errors = res.get("errors", [])
        sty = styles.get(name, {})
        if errors:
            (line,) = ax_lc.semilogy(
                errors,
                label=sty.get("label", name),
                **solver_plot_props(sty, marker=False),
            )
            alias = resolve_solver_alias(name, prefer=THERMAL_ORDER)
            if alias is not None:
                handles_by_alias[alias] = line

    ax_lc.set_xlabel("Optimizer iteration")
    ax_lc.set_ylabel(r"Identification error  $\|T(k) - T_\mathrm{obs}\|^2$")
    ax_lc.set_title("Conductivity recovery: error vs iteration (lower is better)")
    ax_lc.grid(True, which="both", alpha=0.3)

    # Legend in canonical thermal order, drawn from the actual curve handles.
    ordered = [handles_by_alias[a] for a in THERMAL_ORDER if a in handles_by_alias]
    ordered += [h for a, h in handles_by_alias.items() if a not in THERMAL_ORDER]
    if ordered:
        loc = (
            {"loc": "outside lower center"}
            if getattr(fig_cv, "get_constrained_layout", lambda: False)()
            else {"loc": "lower center", "bbox_to_anchor": (0.5, 0.01)}
        )
        fig_cv.legend(
            handles=ordered, ncol=min(len(ordered), 6), handlelength=2.0, **loc
        )
    if save:
        save_fig(fig_cv, "conductivity_recovery_convergence", out_dir)

    return fig_cv
