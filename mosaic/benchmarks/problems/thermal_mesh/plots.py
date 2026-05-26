# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the thermal-mesh conductivity recovery experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
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
    """Two outputs: loss curves + final-error bar; recovered conductivity field comparison.

    Files written to ``results/<problem>/optimization/conductivity_recovery<suffix>/``:
    - ``conductivity_recovery_convergence.{png,pdf}`` — semilogy loss + final-error bar
    - ``conductivity_recovery_fields.{png,pdf}``      — rho_init / rho_truth / rho_final per solver
    """
    out_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    styles = solver_styles(cfg, differentiable_only=False)
    names = list(by_solver.keys())

    # ── 1. Convergence + final-error figure ───────────────────────────────────
    fig_cv, (ax_lc, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

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
        final_errors.append(float(res.get("final_error", float("nan"))))

    ax_lc.set_xlabel("Iteration")
    ax_lc.set_ylabel("Identification error")
    ax_lc.set_title("Conductivity recovery — loss curves")
    ax_lc.legend(fontsize=8)
    ax_lc.grid(True, which="both", alpha=0.3)

    colors = [cfg.solver(n).color if n in cfg.solvers else "#888888" for n in names]
    labels = [styles.get(n, {}).get("label", n) for n in names]
    ax_bar.bar(labels, final_errors, color=colors)
    ax_bar.set_ylabel("Final identification error")
    ax_bar.set_title("Conductivity recovery — final error")
    ax_bar.tick_params(axis="x", rotation=30)
    ax_bar.grid(axis="y", alpha=0.4)

    fig_cv.suptitle(f"{cfg.name} — conductivity recovery")
    fig_cv.tight_layout()
    if save:
        save_fig(fig_cv, "conductivity_recovery_convergence", out_dir)

    # ── 2. Conductivity field comparison + evolution GIFs ─────────────────────
    fields_path = out_dir / "rho_fields.npz"
    if fields_path.exists():
        _plot_conductivity_recovery_fields(
            cfg, fields_path, out_dir, names, styles, save=save
        )
        if save:
            _render_conductivity_recovery_evolution_gifs(
                fields_path, out_dir, names, styles
            )

    return fig_cv


def _plot_conductivity_recovery_fields(
    cfg: Problem,
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
    *,
    save: bool,
) -> None:
    """Render ``conductivity_recovery_fields.{png,pdf}``.

    Shows rho_init, rho_truth, and rho_final per solver as 1-D line plots
    (conductivity field is a 1-D vector over mesh faces/cells).
    """
    npz = try_load_npz(fields_path)

    rho_init = np.asarray(npz["rho_init"]) if "rho_init" in npz.files else None
    rho_truth = np.asarray(npz["rho_truth"]) if "rho_truth" in npz.files else None

    n_active = sum(1 for n in solver_names if f"rho_final_{n}" in npz.files)
    if n_active == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    xs = np.arange(rho_init.shape[0]) if rho_init is not None else None

    if rho_truth is not None:
        ax.plot(
            np.arange(rho_truth.shape[0]),
            rho_truth,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="truth",
            zorder=10,
        )
    if rho_init is not None:
        ax.plot(
            xs,
            rho_init,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="init",
            zorder=9,
        )

    for name in solver_names:
        key = f"rho_final_{name}"
        if key not in npz.files:
            continue
        rho_f = np.asarray(npz[key])
        sty = styles.get(name, {})
        ax.plot(
            np.arange(rho_f.shape[0]),
            rho_f,
            label=sty.get("label", name),
            **solver_plot_props(sty, marker=False),
        )

    ax.set_xlabel("Cell index")
    ax.set_ylabel("Conductivity ρ")
    ax.set_title(f"{cfg.name} — recovered conductivity field")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        save_fig(fig, "conductivity_recovery_fields", out_dir)


def _render_conductivity_recovery_evolution_gifs(
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
) -> None:
    """Write ``conductivity_recovery_evolution_<solver>.gif`` per solver.

    Each frame is a 1-D line plot of ``rho_history_<name>[frame, :]`` with
    dashed truth and dotted init references.  Y-range is fixed across frames.
    Silently skips solvers without a recorded history.
    """
    npz = try_load_npz(fields_path)
    rho_truth = np.asarray(npz["rho_truth"]) if "rho_truth" in npz.files else None
    rho_init = np.asarray(npz["rho_init"]) if "rho_init" in npz.files else None

    for name in solver_names:
        hkey = f"rho_history_{name}"
        if hkey not in npz.files:
            continue
        hist = np.asarray(npz[hkey])  # (n_frames, n_cells)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        n_frames, n_cells = hist.shape
        xs = np.arange(n_cells)

        extrema = [float(hist.min()), float(hist.max())]
        if rho_truth is not None:
            extrema.extend([float(rho_truth.min()), float(rho_truth.max())])
        if rho_init is not None:
            extrema.extend([float(rho_init.min()), float(rho_init.max())])
        ylo, yhi = min(extrema), max(extrema)
        pad = 0.05 * (yhi - ylo + 1e-12)
        ylo -= pad
        yhi += pad

        sty = styles.get(name, {})
        color = sty.get("color", "#3366CC")
        label = sty.get("label", name)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        if rho_truth is not None:
            ax.plot(
                xs,
                rho_truth,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="truth",
            )
        if rho_init is not None:
            ax.plot(
                xs, rho_init, color="gray", linestyle=":", linewidth=1.0, label="init"
            )
        (line,) = ax.plot(xs, hist[0], color=color, linewidth=1.8, label=label)
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Conductivity ρ")
        ax.set_ylim(ylo, yhi)
        title = ax.set_title(f"{label} — snapshot 1 / {n_frames}", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        def _update(
            idx: Any,
            _line: Any = line,
            _title: Any = title,
            _hist: Any = hist,
            _label: Any = label,
            _n: Any = n_frames,
        ) -> Any:
            _line.set_ydata(_hist[idx])
            _title.set_text(f"{_label} — snapshot {idx + 1} / {_n}")
            return _line, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"conductivity_recovery_evolution_{name}", out_dir, fps=4)
