"""Summarize packed cluster results with paired seed/IC uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def _plot_cell(
    archive_path: Path, destination: Path, payload: dict, metrics: dict
) -> None:
    """Publish deterministic fields, held-out curves, and measured training cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tarfile.open(archive_path) as archive:
        member = archive.extractfile(
            "./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz"
        )
        with np.load(io.BytesIO(member.read())) as data:
            arrays = dict(data)
    destination.mkdir(parents=True, exist_ok=True)
    arms = (
        ("uncorrected", "Solver only", "0.5", ":"),
        ("supervised", "Supervised", "tab:orange", "-."),
        ("stop_gradient", "Recurrent, stopped", "tab:blue", "--"),
        ("corrected", "Recurrent, full", "tab:green", "-"),
    )
    times = arrays["evaluation_times"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for arm, label, color, style in arms:
        ax.plot(
            times[1:], arrays[f"error_{arm}_0"][1:], label=label, color=color, ls=style
        )
    ax.set(
        yscale="log", xlabel="Physical time", ylabel="Held-out mean relative L2 error"
    )
    ax.axvline(
        payload["run"]["dataset"]["train_frames"] * (times[1] - times[0]),
        color="0.8",
        label="Training trajectory horizon",
    )
    ax.legend(fontsize=8)
    ax.set_title(archive_path.parent.name)
    fig.savefig(destination / "rollout.png", dpi=170)
    plt.close(fig)

    reference = arrays.get("reference_rollout_0", arrays.get("reference_rollout"))
    trajectories = [reference] + [arrays[f"rollout_{arm}_0"] for arm, *_ in arms]
    frames = [(len(reference) - 1) // 2, len(reference) - 1]

    def vorticity(velocity: np.ndarray) -> np.ndarray:
        # Stored grid velocities have shape (Nx, Ny, 1, 2).
        if velocity.ndim == 4:
            velocity = velocity[:, :, 0, :]
        ux, uy = velocity[..., 0], velocity[..., 1]
        dx = 2 * np.pi / ux.shape[0]
        return (
            np.roll(uy, -1, 0)
            - np.roll(uy, 1, 0)
            - np.roll(ux, -1, 1)
            + np.roll(ux, 1, 1)
        ) / (2 * dx)

    fields = [[vorticity(value[frame]) for value in trajectories] for frame in frames]
    vmax = float(np.nanpercentile(np.abs([row[0] for row in fields]), 99)) or 1
    fig, axes = plt.subplots(2, 5, figsize=(13, 5), layout="constrained")
    for row, frame in enumerate(frames):
        for col, field in enumerate(fields[row]):
            im = axes[row, col].imshow(
                np.ma.masked_invalid(field.T),
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title((["Reference"] + [a[1] for a in arms])[col])
        axes[row, 0].set_ylabel(f"t={times[frame]:.2f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Vorticity", shrink=0.8)
    fig.suptitle(
        "First model seed and first held-out IC; common scale clipped at reference p99"
    )
    fig.savefig(destination / "fields.png", dpi=170)
    plt.close(fig)

    seconds = [
        metrics[k]
        for k in (
            "supervised_total_wall_time_s",
            "stop_gradient_training_wall_time_s",
            "training_wall_time_s",
        )
    ]
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    ax.bar(
        ["Supervised + pairs", "Recurrent, stopped", "Recurrent, full"],
        np.array(seconds) / 60,
        color=["tab:orange", "tab:blue", "tab:green"],
    )
    ax.set(
        ylabel="Training wall time (minutes)",
        title="Equal updates; measured compute cost",
    )
    fig.savefig(destination / "cost.png", dpi=170)
    plt.close(fig)


def _read_result(path: Path) -> tuple[dict, dict]:
    with tarfile.open(path) as archive:
        outcome = archive.extractfile("./outcome.json")
        if outcome is None:
            raise ValueError("outcome.json is missing")
        metrics = json.load(outcome)
        fields = archive.extractfile(
            "./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz"
        )
        if fields is None:
            raise ValueError("corrector_fields.npz is missing")
        with np.load(io.BytesIO(fields.read())) as arrays:
            errors = {
                arm: arrays[f"error_{arm}_samples_0"]
                for arm in ("corrected", "stop_gradient", "supervised", "uncorrected")
                if f"error_{arm}_samples_0" in arrays
            }
    return metrics, errors


def _ratio(baseline: np.ndarray, corrected: np.ndarray) -> dict:
    """Bootstrap paired model seeds and shared held-out ICs, never frames."""
    # Integrate over time first so tiny early errors cannot outweigh a poor
    # late rollout. The harness also preserves geometric gains as diagnostics.
    base = np.mean(baseline[..., 1:], axis=-1)
    full = np.mean(corrected[..., 1:], axis=-1)
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(full)):
        return {"ratio": None, "ci95": None, "reason": "nonfinite rollout"}
    point = float(np.mean(base) / (np.mean(full) + 1e-12))
    if base.shape[0] < 3:
        return {"ratio": point, "ci95": None}
    rng = np.random.default_rng(20261001)
    seeds = rng.integers(base.shape[0], size=(10000, base.shape[0]))
    ics = rng.integers(base.shape[1], size=(10000, base.shape[1]))
    indices = (seeds[:, :, None], ics[:, None, :])
    draws = np.mean(base[indices], axis=(1, 2)) / (
        np.mean(full[indices], axis=(1, 2)) + 1e-12
    )
    return {"ratio": point, "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def _plot_comparisons(rows: list[dict], destination: Path) -> None:
    """Keep every completed configuration visible, including admission failures."""
    import matplotlib.pyplot as plt

    if not rows:
        return
    fig, axes = plt.subplots(
        1, 3, figsize=(12, max(4, len(rows) * 0.3)), sharey=True, layout="constrained"
    )
    labels = [
        f"{r['phase']} {r['solver']} N={r['N']} k={r['k0']:g} "
        f"H={r['unroll']} U={r['updates']} seeds={r['n_model_seeds']}"
        + (" [admission failed]" if not r["admitted"] else "")
        for r in rows
    ]
    for ax, comparison, title in zip(
        axes,
        ("vs_supervised", "vs_stopped", "vs_uncorrected"),
        ("Supervised / full error", "Stopped / full error", "Solver only / full error"),
        strict=True,
    ):
        for index, row in enumerate(rows):
            result = row[comparison]
            value, interval = result["ratio"], result["ci95"]
            if value is None or not np.isfinite(value):
                ax.text(
                    0.98,
                    index,
                    "nonfinite rollout",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    fontsize=7,
                )
                continue
            color = "tab:green" if row["admitted"] else "0.6"
            ax.plot(value, index, "o", color=color, ms=4)
            if interval:
                ax.plot(interval, [index, index], color=color)
        ax.axvline(1, color="black", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel=title)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(rows)), labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Mean rollout error ratios: >1 favours full gradients; paired seed/IC 95% intervals when ≥3 seeds"
    )
    fig.savefig(destination / "comparisons.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Keep missing and failed cells visible alongside successful comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plots", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    pending, failures = [], []
    for config in sorted((args.campaign / "configs").glob("*.json")):
        payload = json.loads(config.read_text())
        run = payload["run"]
        normalized = json.loads(json.dumps(run))
        normalized["training"].pop("model_seeds", None)
        protocol_hash = hashlib.sha256(
            json.dumps(normalized, sort_keys=True).encode()
        ).hexdigest()
        key = (
            config.stem.split("-", 1)[0],
            payload["solver"],
            run["physics"]["N"],
            run["dataset"]["k0"],
            run["training"]["unroll"],
            run["training"]["max_updates"],
            payload["source_sha256"],
            protocol_hash,
        )
        archive = args.campaign / "results" / config.stem / "results.tar"
        if not archive.exists():
            pending.append(config.stem)
            continue
        try:
            metrics, errors = _read_result(archive)
        except (KeyError, ValueError) as exc:
            failures.append({"cell": config.stem, "reason": str(exc)})
            continue
        if not metrics.get("completed") or len(errors) != 4:
            failures.append(
                {
                    "cell": config.stem,
                    "reason": "incomplete or reference admission failed",
                    "metrics": metrics,
                }
            )
            continue
        if args.plots:
            _plot_cell(archive, args.plots / config.stem, payload, metrics)
        groups[key].append((metrics, errors))
    rows = []
    for key, cells in groups.items():
        phase, solver, n, k0, unroll, updates, source, protocol_hash = key
        errors = {
            arm: np.concatenate([e[arm] for _, e in cells], axis=0)
            for arm in cells[0][1]
        }
        # Native trajectories do not depend on model seed; repeat for paired broadcasting.
        native = np.concatenate(
            [np.broadcast_to(e["uncorrected"], e["corrected"].shape) for _, e in cells]
        )
        ratios = {
            "vs_supervised": _ratio(errors["supervised"], errors["corrected"]),
            "vs_stopped": _ratio(errors["stop_gradient"], errors["corrected"]),
            "vs_uncorrected": _ratio(native, errors["corrected"]),
        }
        admitted = all(m.get("valid_for_vjp_ranking", False) for m, _ in cells)
        fd = [m.get("end_to_end_fd_rel_error") for m, _ in cells]
        gradient_checked = all(
            value is not None and np.isfinite(value) and value < 0.05 for value in fd
        )
        arm_errors = {**errors, "uncorrected": native}
        summaries = {}
        for arm, samples in arm_errors.items():
            final = samples[..., -1]
            mean = samples[..., 1:].mean()
            summaries[arm] = {
                "mean_rollout_error": float(mean) if np.isfinite(mean) else None,
                "mean_final_error": float(final.mean())
                if np.all(np.isfinite(final))
                else None,
                "finite_rollout_fraction": float(
                    np.isfinite(samples).all(axis=-1).mean()
                ),
                "stable_rollout_fraction": float(
                    (
                        np.isfinite(samples).all(axis=-1) & (samples <= 1).all(axis=-1)
                    ).mean()
                ),
            }
        rows.append(
            {
                "phase": phase,
                "solver": solver,
                "N": n,
                "k0": k0,
                "unroll": unroll,
                "updates": updates,
                "n_model_seeds": errors["corrected"].shape[0],
                "source_sha256": source,
                "protocol_sha256": protocol_hash,
                "admitted": admitted,
                "fd_relative_errors": fd,
                **ratios,
                "arm_summaries": summaries,
                "resolved_joint_benefit": admitted
                and gradient_checked
                and all(
                    r["ci95"] is not None and r["ci95"][0] > 1 for r in ratios.values()
                ),
                "full_training_seconds": sum(
                    m["training_wall_time_s"] for m, _ in cells
                ),
                "supervised_total_seconds": sum(
                    m["supervised_total_wall_time_s"] for m, _ in cells
                ),
                "stopped_training_seconds": sum(
                    m["stop_gradient_training_wall_time_s"] for m, _ in cells
                ),
            }
        )
    result = {"comparisons": rows, "pending": pending, "failures": failures}
    if args.plots:
        _plot_comparisons(rows, args.plots)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
