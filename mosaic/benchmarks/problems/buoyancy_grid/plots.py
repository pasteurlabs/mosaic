# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Held-out field and error plots with shared scales across training methods."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_experiment_result, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.style import save_fig

_METHODS = {
    "uncorrected": ("Uncorrected", "#777777"),
    "solver_in_loop": ("Solver in the loop", "#177c71"),
}


def _speed(state: np.ndarray) -> np.ndarray:
    # Include the missing low wall face before averaging to cell centers.
    ux = np.pad(state[:-1, :, 0], ((1, 1), (0, 0)))
    uy = np.pad(state[:, :-1, 1], ((0, 0), (1, 1)))
    return np.hypot((ux[:-1] + ux[1:]) / 2, (uy[:, :-1] + uy[:, 1:]) / 2)


def plot_solver_in_loop(
    cfg: Problem,
    *,
    suffix: str = "",
    exp_key: str = "solver_in_loop",
    save: bool = True,
    **_: Any,
) -> list:
    """Plot full scalar/velocity fields, rollout errors and measured training cost."""
    out = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    result = load_experiment_result(out)
    if result is None:
        return []
    snapshots = try_load_npz(out / "corrector_fields.npz")
    names = [str(name) for name in snapshots.get("solver_names", [])]
    figures = []
    for record in result.get("results", []):
        name, metrics = record["solver"], record.get("metrics", {})
        if metrics.get("status") != "ok" or name not in names:
            continue
        index = names.index(name)
        selected_update = metrics["solver_in_loop"]["selected_update"]
        selected_label = (
            "Selected: no correction"
            if selected_update == 0
            else f"Solver in the loop (update {selected_update})"
        )
        keys = ["reference", *_METHODS]
        if any(f"{key}_{index}" not in snapshots for key in keys):
            continue
        states = {key: snapshots[f"{key}_{index}"] for key in keys}
        physics = result.get("params", {}).get("physics", {})
        interval = physics.get("dt", 0.04) * physics.get("steps", 2)
        times = np.arange(len(states["reference"])) * interval
        final = {key: value[-1] for key, value in states.items()}
        scalar_max = max(float(s[..., 2].max()) for s in final.values())
        scalar_min = min(0.0, min(float(s[..., 2].min()) for s in final.values()))
        speeds = {key: _speed(state) for key, state in final.items()}
        speed_max = max(float(s.max()) for s in speeds.values())
        errors = {
            key: state[..., 2] - final["reference"][..., 2]
            for key, state in final.items()
        }
        error_max = max(max(float(np.abs(e).max()) for e in errors.values()), 1e-8)
        fig, axes = plt.subplots(3, 3, figsize=(9, 8), constrained_layout=True)
        for column, key in enumerate(keys):
            label = (
                "Restricted reference"
                if key == "reference"
                else (selected_label if key == "solver_in_loop" else _METHODS[key][0])
            )
            axes[0, column].set_title(label, fontsize=10)
            fields = (final[key][..., 2], speeds[key], errors[key])
            limits = ((scalar_min, scalar_max), (0, speed_max), (-error_max, error_max))
            for row, (field, (low, high)) in enumerate(
                zip(fields, limits, strict=True)
            ):
                im = axes[row, column].imshow(
                    field.T,
                    origin="lower",
                    extent=(0, 1, 0, 1),
                    cmap="RdBu_r" if row == 2 else "inferno",
                    vmin=low,
                    vmax=high,
                )
                axes[row, column].set_xticks([0, 1])
                axes[row, column].set_yticks([0, 1])
                if column == 0:
                    axes[row, column].set_ylabel(
                        ("Buoyancy scalar", "Speed", "Scalar error")[row]
                    )
                if column == len(keys) - 1:
                    fig.colorbar(im, ax=axes[row, :], shrink=0.8)
        fig.suptitle(
            f"{name}: held-out plume at t={times[-1]:.2f} — shared scales per row"
        )
        figures.append(fig)
        if save:
            save_fig(fig, f"fields_{name}", out)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for key, (label, color) in _METHODS.items():
            if key == "solver_in_loop":
                label = selected_label
            for ax, metric in zip(axes, ("velocity_mse", "scalar_mse"), strict=True):
                ax.plot(
                    times[1:],
                    np.sqrt(metrics[key][metric][1:]),
                    label=label,
                    color=color,
                )
                ax.set_xlabel("Physical time")
                ax.set_ylabel("Normalized RMS error")
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
        axes[0].set_title("Velocity")
        axes[1].set_title("Buoyancy scalar")
        axes[1].legend(fontsize=8)
        fig.suptitle(
            f"{name}: held-out rollout; normalization fixed from first training case"
        )
        figures.append(fig)
        if save:
            save_fig(fig, f"rollout_error_{name}", out)

        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        trained = metrics["solver_in_loop"]
        for candidate in trained["candidates"]:
            checkpoints = candidate["validation_checkpoints"]
            ax.plot(
                [point["update"] for point in checkpoints],
                [point["loss"] for point in checkpoints],
                marker="o",
                label=f"Learning rate {candidate['learning_rate']:g}",
            )
        selected = trained["candidates"][trained["selected_candidate"]]
        ax.scatter(
            [selected_update],
            [selected["validation_loss"]],
            marker="*",
            s=150,
            color="black",
            zorder=3,
            label="Selected checkpoint",
        )
        ax.set(
            xlabel="Completed optimizer updates",
            ylabel="Full-rollout validation loss",
            title=f"{name}: selection includes the uncorrected model at update 0",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        figures.append(fig)
        if save:
            save_fig(fig, f"validation_checkpoints_{name}", out)

        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        trained = metrics["solver_in_loop"]
        cost = trained["total_seconds"]
        ax.barh(
            ["Solver in the loop"],
            [trained["total_training_seconds"]],
            color=_METHODS["solver_in_loop"][1],
            label="Training",
        )
        bars = ax.barh(
            ["Solver in the loop"],
            [trained["total_validation_seconds"]],
            left=[trained["total_training_seconds"]],
            color="#aaa",
            label="Validation checkpoints",
        )
        ax.bar_label(bars, labels=[f"{cost:.1f} s total"], padding=4)
        ax.set_xlim(0, max(cost, 1e-6) * 1.25)
        ax.set_xlabel("Measured training and validation time across all candidates (s)")
        ax.legend(fontsize=8)
        ax.set_title(
            f"{name}: {metrics['updates']} completed updates per candidate; selected update {selected_update}"
        )
        figures.append(fig)
        if save:
            save_fig(fig, f"training_cost_{name}", out)
    return figures
