# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the structural-mesh topology optimisation experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import experiment_dir, load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    RCPARAMS,
    SOLVER_STYLES,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    resolve_solver_alias,
    solver_props,
)


def _solver_order_for(cfg_name: str) -> list[str]:
    """Best-effort solver-ordering pick based on problem name."""
    if "structural" in cfg_name:
        return STRUCTURAL_ORDER
    if "thermal" in cfg_name:
        return THERMAL_ORDER
    return STRUCTURAL_ORDER


def plot_topopt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "topopt",
    **_kw: Any,
) -> Any:
    """Topopt convergence plot — compliance vs iteration per solver."""
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[topopt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)
    data = load_json(result_path)
    by_solver = data.get("by_solver", {})
    solver_order = _solver_order_for(cfg.name)

    fig, ax_c = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.38), dpi=300)
    fig.subplots_adjust(bottom=0.32, top=0.92)

    present: set[str] = set()
    for solver, sdata in by_solver.items():
        alias = resolve_solver_alias(solver)
        _label, color, ls, _mk = solver_props(alias or solver)
        compliances = sdata.get("compliances", [])
        if compliances:
            ax_c.semilogy(
                range(len(compliances)),
                compliances,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            if alias is not None:
                present.add(alias)

    ax_c.set_title(f"Compliance — {cfg.category_label or cfg.name}")
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Compliance")

    handles = dedup_handles(
        [make_handle(s) for s in solver_order if s in present and s in SOLVER_STYLES]
    )
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
