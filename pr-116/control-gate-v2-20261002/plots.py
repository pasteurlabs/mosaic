"""Full-field plots for development gates; render only inside a Slurm job."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gate(
    arrays: dict,
    metrics: dict,
    destination: Path,
    *,
    learned_method: str | None = None,
) -> None:
    """Show physical outcomes and optimization history without implying learning."""
    destination.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(metrics["tasks"]):
        goal = arrays["fine_goal"][index, :, :, 0, :]
        zero = arrays["fine_zero_rollouts"][index, -1, :, :, 0, :]
        controlled = arrays["fine_shooting_rollouts"][index, -1, :, :, 0, :]
        fields = [goal, zero]
        titles = ["Target", "No control"]
        if "fine_linear_rollouts" in arrays:
            fields.append(arrays["fine_linear_rollouts"][index, -1, :, :, 0, :])
            titles.append("Linear controller")
        fields.append(controlled)
        titles.append("Learned control" if learned_method else "Optimized control")
        columns = len(fields)
        dx = 2 * np.pi / goal.shape[0]
        curls = [
            (
                np.roll(v[..., 1], -1, axis=0)
                - np.roll(v[..., 1], 1, axis=0)
                - np.roll(v[..., 0], -1, axis=1)
                + np.roll(v[..., 0], 1, axis=1)
            )
            / (2 * dx)
            for v in fields
        ]
        vmax = max(float(np.max(np.abs(c))) for c in curls) or 1
        errors = [np.linalg.norm(v - goal, axis=-1) for v in fields[1:]]
        emax = max(float(e.max()) for e in errors) or 1
        fig, axes = plt.subplots(
            2, columns, figsize=(4 * columns + 1, 8), layout="constrained"
        )
        for ax, curl, title in zip(
            axes[0],
            curls,
            titles,
            strict=True,
        ):
            im = ax.imshow(
                curl.T,
                origin="lower",
                extent=(0, 2 * np.pi, 0, 2 * np.pi),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=axes[0].tolist(), label="Terminal vorticity", shrink=0.8)
        for ax, error, title in zip(
            axes[1, :-1],
            errors,
            [f"{title}: error" for title in titles[1:]],
            strict=True,
        ):
            im = ax.imshow(
                error.T,
                origin="lower",
                extent=(0, 2 * np.pi, 0, 2 * np.pi),
                cmap="magma",
                vmin=0,
                vmax=emax,
            )
            ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(
            im, ax=axes[1, :-1].tolist(), label="Velocity error magnitude", shrink=0.8
        )
        trace = row["shooting"]["trace"]
        axes[1, -1].plot(np.arange(len(trace)), trace, color="#0072B2")
        axes[1, -1].set(
            title="Policy training at 64²"
            if learned_method
            else "Direct optimization at 64²",
            xlabel="Gradient update",
            ylabel="Mean perturbed objective"
            if learned_method == "spsa"
            else "Method training loss"
            if learned_method
            else "Terminal error + control effort",
        )
        axes[1, -1].grid(alpha=0.2)
        status = "passed" if row["passed"] else "FAILED"
        heading = f"Control gate · task {row['seed']} · 192² evaluation · {status}"
        detail = "Individual optimized controls; no neural controller trained yet"
        if learned_method:
            heading = (
                f"{learned_method} · validation task {row['seed']} · 192² · {status}"
            )
            detail = "Development pilot; validation data, not held-out confirmation"
        fig.suptitle(f"{heading}\n{detail}", fontsize=13)
        fig.savefig(destination / f"task-{row['seed']}.png", dpi=160)
        plt.close(fig)


def plot_pilot(arrays: dict, metrics: dict, destination: Path) -> None:
    """Plot the first two preselected validation tasks at the final checkpoint."""
    if not metrics.get("completed") or "final_validation" not in metrics:
        return
    update = metrics["final_validation"]["update"]
    rows = metrics["final_validation"]["tasks"][:2]
    trace = [
        row.get("loss", row.get("paired_loss_mean", np.nan))
        for row in metrics["training_trace"]
    ]
    mapped = {
        "fine_goal": arrays["validation_fine_goal"][:2],
        "fine_zero_rollouts": arrays["validation_fine_zero_rollouts"][:2],
        "fine_shooting_rollouts": arrays[f"validation_update{update}_fine_rollouts"][
            :2
        ],
    }
    if "validation_fine_linear_rollouts" in arrays:
        mapped["fine_linear_rollouts"] = arrays["validation_fine_linear_rollouts"][:2]
    outcome = {
        "tasks": [
            {
                "seed": row["task_seed"],
                "passed": row["admitted"],
                "shooting": {"trace": trace},
            }
            for row in rows
        ]
    }
    plot_gate(mapped, outcome, destination, learned_method=metrics["method"])
