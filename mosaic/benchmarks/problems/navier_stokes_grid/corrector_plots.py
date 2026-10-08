"""Plots for the bounded, three-arm solver-in-the-loop benchmark."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
    v1_to_legacy,
)
from mosaic.benchmarks.problems.shared.plots.style import save_fig, solver_props

_ARMS = (
    ("uncorrected", "Solver only", "0.5", ":"),
    ("supervised", "Supervised", "tab:orange", "-."),
    ("stop_gradient", "Stopped gradients", "tab:blue", "--"),
    ("corrected", "Full gradients", "tab:green", "-"),
)


def plot_solver_in_loop(
    cfg: Problem, *, save: bool = True, suffix: str = "", **_kwargs: Any
) -> list:
    """Compare methods within each solver; never rank different reference targets."""
    directory = experiment_dir(
        results_dir(), cfg.name, "optimization", f"solver_in_loop{suffix}"
    )
    result = directory / "result.json"
    fields = directory / "corrector_fields.npz"
    if not result.exists() or not fields.exists():
        return []
    metrics = v1_to_legacy(load_json(result)).get("by_solver", {})
    arrays = try_load_npz(fields)
    names = [str(name) for name in arrays.get("solver_names", np.array([])).tolist()]
    if not names:
        return []
    times = np.asarray(arrays.get("evaluation_times", []))
    figures = []
    rollout, axes = plt.subplots(
        len(names),
        1,
        figsize=(7, 2.7 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for index, name in enumerate(names):
        ax = axes[index, 0]
        for key, label, color, style in _ARMS:
            values = np.asarray(arrays.get(f"error_{key}_{index}", []))
            if values.size:
                x = times[: len(values)] if times.size else np.arange(len(values))
                ax.plot(x[1:], values[1:], label=label, color=color, linestyle=style)
        ax.set(
            title=solver_props(name)[0],
            xlabel="Physical time",
            ylabel="Relative L2 velocity error",
        )
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(ncol=2, fontsize="small")
    rollout.suptitle("Held-out rollouts — each solver uses its own refined reference")
    figures.append(rollout)

    field_figure, axes = plt.subplots(
        len(names),
        5,
        figsize=(13, 2.4 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for index, name in enumerate(names):
        keys = [f"reference_rollout_{index}"] + [
            f"rollout_{arm}_{index}" for arm, _, _, _ in _ARMS
        ]
        snapshots = []
        for key in keys:
            values = np.asarray(arrays.get(key, []))
            snapshots.append(
                values[-1, :, :, 0, 0] if values.ndim == 5 and len(values) else None
            )
        valid = [
            np.abs(value[np.isfinite(value)])
            for value in snapshots
            if value is not None
        ]
        limit = (
            max((float(value.max()) for value in valid if value.size), default=1.0)
            or 1.0
        )
        image = None
        for column, (value, title) in enumerate(
            zip(snapshots, ["Reference"] + [arm[1] for arm in _ARMS], strict=True)
        ):
            ax = axes[index, column]
            if value is None:
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                image = ax.imshow(
                    np.ma.masked_invalid(value.T),
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
            if index == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[index, 0].set_ylabel(solver_props(name)[0])
        if image is not None:
            field_figure.colorbar(
                image, ax=axes[index, :].tolist(), label="x velocity", shrink=0.8
            )
    field_figure.suptitle(
        "Final held-out fields — first model/IC; shared unclipped scale within each row"
    )
    figures.append(field_figure)

    diagnostics, axes = plt.subplots(
        len(names),
        2,
        figsize=(10, 2.8 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    traces = [
        ("loss_supervised", "Supervised", "tab:orange"),
        ("loss_stop_gradient", "Stopped", "tab:blue"),
        ("loss", "Full", "tab:green"),
    ]
    cost_keys = [
        "supervised_training_wall_time_s",
        "stop_gradient_training_wall_time_s",
        "training_wall_time_s",
    ]
    for index, name in enumerate(names):
        row = metrics.get(name, {})
        for prefix, label, color in traces:
            values = np.asarray(arrays.get(f"{prefix}_{index}", []))
            if values.size:
                axes[index, 0].plot(
                    np.arange(1, len(values) + 1), values, label=label, color=color
                )
        axes[index, 0].set(
            title=solver_props(name)[0],
            xlabel="Optimizer update",
            ylabel="Training loss",
        )
        if axes[index, 0].lines:
            axes[index, 0].legend(fontsize="small")
        costs = [row.get(key) for key in cost_keys]
        for position, (cost, (_, _label, color)) in enumerate(
            zip(costs, traces, strict=True)
        ):
            if isinstance(cost, int | float) and np.isfinite(cost):
                axes[index, 1].bar(position, cost, color=color)
            else:
                axes[index, 1].text(position, 0, "Missing", ha="center", va="bottom")
        axes[index, 1].set(
            xticks=range(3),
            xticklabels=[row[1] for row in traces],
            ylabel="Fitting time (s)",
        )
        fd = row.get("end_to_end_fd_rel_error")
        text = (
            f"Model gradient FD relative error: {fd:.3g}"
            if isinstance(fd, int | float)
            else "Model gradient FD: unavailable"
        )
        axes[index, 1].set_title(text)
    diagnostics.suptitle(
        "Training diagnostics — fitting time excludes reference and fixed-pair preparation"
    )
    figures.append(diagnostics)
    if save:
        for figure, name in zip(
            figures,
            ["solver_in_loop", "solver_in_loop_fields", "solver_in_loop_training"],
            strict=True,
        ):
            save_fig(figure, name, directory)
    return figures
