# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-experiment extra plots for thermal-mesh.

Registered on the :class:`Problem` via :meth:`Problem.add_extra_plot` from
:mod:`thermal_mesh.config`. Each plot is wrapped to take the standard
``(cfg, **kw)`` signature used by the runner and writes its outputs under
``results/<cfg.name>/_extra/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.cost_overview import (
    plot_cost_overview_for,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    PAPER_RCPARAMS,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    resolve_solver_alias,
    solver_props,
)

# See note in recovery_overview.py: optax L-BFGS averages ~3 grad evaluations
# per outer iteration (zoom line search), Adam exactly 1.
_GRAD_EVALS_PER_ITER: dict[str, int] = {"adam": 1, "bfgs": 3}
_GRAD_EVAL_LABEL = "Gradient evaluations"


def _conductivity_methods() -> dict[str, tuple]:
    base = results_dir() / "thermal-mesh" / "optimization"
    return {
        "adam": ("Adam", "-", base / "conductivity_recovery"),
        "bfgs": ("L-BFGS", "--", base / "conductivity_recovery_bfgs"),
    }


def _conductivity_overview_generate(out_dir: Path) -> None:
    """Generate ``conductivity_recovery_overview.pdf`` into *out_dir*.

    Layout:
      Row 0: identification error history — loglog, all solvers × both methods
      Row 1: Adam recovered conductivity profiles, all solvers + truth
      Row 2: L-BFGS recovered conductivity profiles, all solvers + truth
    """
    loaded: dict[str, tuple] = {}
    for key, (*_, path) in _conductivity_methods().items():
        rp = path / "result.json"
        fp = path / "rho_fields.npz"
        if not rp.exists():
            print(f"[conductivity_overview] {rp} not found — skipping {key}")
            continue
        npz = try_load_npz(fp) if fp.exists() else None
        loaded[key] = (load_json(rp), npz)

    if not loaded:
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 1.10))
        outer = gridspec.GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[1.0, 0.65, 0.65],
            left=0.10,
            right=0.98,
            top=0.96,
            bottom=0.10,
            hspace=0.52,
        )
        ax_conv = fig.add_subplot(outer[0])
        ax_adam = fig.add_subplot(outer[1])
        ax_bfgs = fig.add_subplot(outer[2])

        seen_solvers: set[str] = set()

        # ── Convergence — all solvers × both methods ──────────────────────
        for key, (_m_label, m_ls, *_) in _conductivity_methods().items():
            if key not in loaded:
                continue
            result, _ = loaded[key]
            # ``by_solver`` keyed by spec.name; bridge to alias.
            by_solver_d = result["by_solver"]
            alias_to_display: dict[str, str] = {}
            for display_name in by_solver_d:
                a = resolve_solver_alias(display_name)
                if a is not None:
                    alias_to_display[a] = display_name
            for alias in THERMAL_ORDER:
                display_name = alias_to_display.get(alias)
                if display_name is None:
                    continue
                sdata = by_solver_d.get(display_name)
                if sdata is None:
                    continue
                errors = sdata.get("errors", [])
                if not errors:
                    continue
                _, s_color, _, _ = solver_props(alias)
                f = _GRAD_EVALS_PER_ITER.get(key, 1)
                xs = [(i + 1) * f for i in range(len(errors))]
                ax_conv.loglog(
                    xs,
                    errors,
                    color=s_color,
                    linestyle=m_ls,
                    linewidth=1.3,
                    alpha=0.9,
                )
                seen_solvers.add(alias)

        ax_conv.set_title("Thermal conductivity recovery")
        ax_conv.set_xlabel(_GRAD_EVAL_LABEL)
        ax_conv.set_ylabel("Identification error")

        # ── Profile panels ────────────────────────────────────────────────
        for ax, key, title in [
            (ax_adam, "adam", "Adam — recovered profiles"),
            (ax_bfgs, "bfgs", "L-BFGS — recovered profiles"),
        ]:
            if key not in loaded:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_title(title)
                continue

            _, npz = loaded[key]
            if npz is None:
                ax.set_title(title)
                continue

            xs = np.arange(npz["rho_truth"].shape[0])
            ax.plot(
                xs,
                npz["rho_truth"],
                color="0.2",
                linestyle="--",
                linewidth=1.4,
                label="Truth",
                zorder=3,
            )

            # ``rho_final_<name>`` keys use spec.name (display form); build
            # alias→display from the npz keys.
            npz_keys = npz.files if hasattr(npz, "files") else list(npz.keys())
            _display_names = [
                k[len("rho_final_") :] for k in npz_keys if k.startswith("rho_final_")
            ]
            alias_to_display: dict[str, str] = {}
            for display_name in _display_names:
                a = resolve_solver_alias(display_name)
                if a is not None:
                    alias_to_display[a] = display_name
            for alias in THERMAL_ORDER:
                display_name = alias_to_display.get(alias)
                if display_name is None:
                    continue
                rho_key = f"rho_final_{display_name}"
                if rho_key not in npz_keys:
                    continue
                _, s_color, _, _ = solver_props(alias)
                ax.plot(xs, npz[rho_key], color=s_color, linewidth=1.1, alpha=0.85)
                seen_solvers.add(alias)

            ax.set_title(title)
            ax.set_xlabel("Node index")
            ax.set_ylabel("Conductivity")

        # ── Legend ────────────────────────────────────────────────────────
        truth_handle = mlines.Line2D(
            [], [], color="0.2", linestyle="--", linewidth=1.4, label="Truth"
        )
        solver_handles = dedup_handles(
            [make_handle(s) for s in THERMAL_ORDER if s in seen_solvers]
        )
        fig.legend(
            handles=[truth_handle, *solver_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            fontsize=6.5,
            framealpha=0.8,
            edgecolor="0.8",
            handlelength=1.8,
        )

        out = out_dir / "conductivity_recovery_overview.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


# ── Adapter + registration ────────────────────────────────────────────────────


def _conductivity_overview_plot(cfg: Problem, **_kw: Any) -> None:
    """Runner-facing adapter: writes ``conductivity_recovery_overview.pdf``."""
    out_dir = results_dir() / cfg.name / "_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    _conductivity_overview_generate(out_dir)


def _plot_cost_overview(cfg: Problem, **_kw: Any) -> None:
    plot_cost_overview_for(cfg, steady_state=True)


def register(problem: Problem) -> None:
    """Attach cross-experiment extras to *problem*."""
    problem.add_extra_plot("_extra/conductivity_overview", _conductivity_overview_plot)
    problem.add_extra_plot("_extra/cost_overview", _plot_cost_overview)
