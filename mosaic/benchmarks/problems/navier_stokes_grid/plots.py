# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the navier-stokes-grid drag-optimisation experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    dedup_handles,
    make_handle,
    resolve_solver_alias,
    solver_props,
)


def plot_drag_opt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "drag_opt",
    **_kw: Any,
) -> list:
    """Drag-reduction vs iteration per solver.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    figs: list = []

    def _plot_one(data: Any, out_dir: Any, *, label_key: str) -> None:
        fig = _drag_opt_convergence(cfg, data, out_dir, label_key, save=save)
        if fig is not None:
            figs.append(fig)

    single_path = base_dir / "result.json"
    if single_path.exists():
        _plot_one(load_json(single_path), base_dir, label_key=exp_key)
        return figs

    if base_dir.is_dir():
        for sub in sorted(base_dir.iterdir()):
            sub_data = load_json(sub / "result.json")
            if sub_data is not None:
                _plot_one(sub_data, sub, label_key=f"{exp_key}/{sub.name}")
    return figs


def _drag_opt_convergence(
    cfg: Problem,
    data: dict,
    out_dir: Any,
    label_key: str,
    *,
    save: bool,
) -> plt.Figure | None:
    """Single-panel drag-reduction vs iteration plot."""
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.38), dpi=300)
    fig.subplots_adjust(bottom=0.32, top=0.92)

    present: set[str] = set()
    for solver, sdata in by_solver.items():
        alias = resolve_solver_alias(solver)
        drags = sdata.get("drags", [])
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]
        step = max(1, len(drags) // 50)
        indices = list(range(0, len(drags), step))
        if indices[-1] != len(drags) - 1:
            indices.append(len(drags) - 1)
        reductions = [(drag_0 - drags[i]) / drag_0 * 100 for i in indices]
        _label, color, ls, _mk = solver_props(alias or solver)
        ax.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        if alias is not None:
            present.add(alias)

    ax.set_title(f"Drag reduction — {cfg.category_label or cfg.name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Drag reduction (%)")
    ax.set_ylim(bottom=0)

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in present and s in SOLVER_STYLES]
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
        out = out_dir / f"{label_key.rsplit('/', 1)[-1]}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"Saved {out}")
    return fig
