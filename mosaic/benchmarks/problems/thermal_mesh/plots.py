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
    solver_legend,
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
    """Loss curves + final-error bar chart for conductivity recovery."""
    out_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    data = v1_to_legacy(load_json(out_dir / "result.json"))
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    plt.rcParams.update(RCPARAMS)

    styles = solver_styles(cfg, differentiable_only=False)
    names = list(by_solver.keys())

    fig_cv, (ax_lc, ax_bar) = paper_row(2)

    seen: set[str] = set()
    final_errors: list[float] = []
    for name in names:
        res = by_solver[name]
        errors = res.get("errors", [])
        sty = styles.get(name, {})
        if errors:
            ax_lc.semilogy(
                errors,
                label=sty.get("label", name),
                **solver_plot_props(sty, marker=False),
            )
            alias = resolve_solver_alias(name)
            if alias is not None:
                seen.add(alias)
        final_errors.append(float(res.get("final_error", float("nan"))))

    ax_lc.set_xlabel("Iteration")
    ax_lc.set_ylabel("Identification error")
    ax_lc.set_title("Loss curves")
    ax_lc.grid(True, which="both", alpha=0.3)

    colors = [cfg.solver(n).color if n in cfg.solvers else "#888888" for n in names]
    labels = [styles.get(n, {}).get("label", n) for n in names]
    ax_bar.bar(labels, final_errors, color=colors)
    ax_bar.set_ylabel("Final identification error")
    ax_bar.set_title("Final error")
    ax_bar.tick_params(axis="x", rotation=30)
    ax_bar.grid(axis="y", alpha=0.4)

    solver_legend(fig_cv, seen, order=THERMAL_ORDER)
    if save:
        save_fig(fig_cv, "conductivity_recovery_convergence", out_dir)

    return fig_cv
