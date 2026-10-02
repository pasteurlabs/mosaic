"""Summarize packed cluster results with paired seed/IC uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
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
    if "error_pretrained_0" in arrays:
        arms = (
            arms[0],
            ("pretrained", "Pretrained start", "tab:purple", ":"),
            *arms[1:],
        )
    times = arrays["evaluation_times"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for arm, label, color, style in arms:
        ax.plot(
            times[1:],
            arrays[f"error_{arm}_0"][1:],
            label=label,
            color=color,
            ls=style,
            lw=2,
        )
    ax.set(
        yscale="log", xlabel="Physical time", ylabel="Held-out mean relative L2 error"
    )
    ax.axvline(
        payload["run"]["dataset"]["train_frames"] * (times[1] - times[0]),
        color="0.7",
        ls=":",
        lw=1,
    )
    ax.text(
        0.51,
        0.96,
        "Beyond training horizon →",
        transform=ax.transAxes,
        fontsize=9,
        color="0.4",
        va="top",
    )
    ax.legend(
        fontsize=10, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2, frameon=False
    )
    ax.set_title(
        "How does rollout error grow?", loc="left", fontsize=15, weight="bold", pad=14
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
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
    reference_label = (
        "Fine reference\n(restricted)"
        if payload["run"]["dataset"]["reference_kind"] == "solver_self_refined"
        else "Reference"
    )
    vmax = float(np.nanpercentile(np.abs([row[0] for row in fields]), 99)) or 1
    fig, axes = plt.subplots(
        2, len(trajectories), figsize=(2.6 * len(trajectories), 5), layout="constrained"
    )
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
                axes[row, col].set_title(
                    ([reference_label] + [a[1] for a in arms])[col]
                )
        axes[row, 0].set_ylabel(f"t={times[frame]:.2f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Vorticity", shrink=0.8)
    fig.suptitle("Vorticity across the full domain", fontsize=16, weight="bold")
    fig.savefig(destination / "fields.png", dpi=170)
    plt.close(fig)

    # Absolute fields can hide small but meaningful correction errors. Show
    # velocity error separately, using one scale across methods and times.
    errors = [
        [
            np.linalg.norm(value[frame] - reference[frame], axis=-1).squeeze()
            for value in trajectories[1:]
        ]
        for frame in frames
    ]
    finite_errors = np.asarray(errors)
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    error_max = (float(finite_errors.max()) or 1) if finite_errors.size else 1
    fig, axes = plt.subplots(
        2, len(arms), figsize=(2.6 * len(arms), 5), layout="constrained"
    )
    for row, frame in enumerate(frames):
        for col, field in enumerate(errors[row]):
            im = axes[row, col].imshow(
                np.ma.masked_invalid(field.T),
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=error_max,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(arms[col][1])
        axes[row, 0].set_ylabel(f"t={times[frame]:.2f}")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), label="Velocity error magnitude", shrink=0.8
    )
    fig.suptitle("Where does the rollout differ from the reference?", fontsize=16)
    fig.savefig(destination / "field_errors.png", dpi=170)
    plt.close(fig)

    common_cost = metrics.get("pretrain_common_wall_time_s", 0.0)
    seconds = [
        common_cost + metrics[k]
        for k in (
            "supervised_total_wall_time_s",
            "stop_gradient_training_wall_time_s",
            "training_wall_time_s",
        )
    ]
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    bars = ax.barh(
        ["Supervised (including pairs)", "Recurrent, stopped", "Recurrent, full"],
        np.array(seconds) / 60,
        color=["tab:orange", "tab:blue", "tab:green"],
        height=0.5,
    )
    ax.bar_label(bars, fmt="%.1f min", padding=6, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(seconds) / 60 * 1.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set(
        xlabel="Training wall time (minutes)",
    )
    ax.set_title(
        "Training cost including shared pretraining"
        if common_cost
        else "Training cost at equal updates",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.savefig(destination / "cost.png", dpi=170)
    plt.close(fig)


def _plot_reference(archive_path: Path, destination: Path, metrics: dict) -> None:
    """Inspect post-burn-in flow without implying a learned or stationary result."""
    import matplotlib.pyplot as plt

    with tarfile.open(archive_path) as archive:
        member = archive.extractfile(
            "./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz"
        )
        with np.load(io.BytesIO(member.read())) as data:
            reference = data["reference_rollout_0"]
    destination.mkdir(parents=True, exist_ok=True)
    velocity = reference[:, :, :, 0, :]
    dx = 2 * np.pi / velocity.shape[1]
    curl = (
        np.roll(velocity[..., 1], -1, axis=1)
        - np.roll(velocity[..., 1], 1, axis=1)
        - np.roll(velocity[..., 0], -1, axis=2)
        + np.roll(velocity[..., 0], 1, axis=2)
    ) / (2 * dx)
    times = np.linspace(0, metrics["rollout_final_time"], len(reference))
    frames = [0, (len(reference) - 1) // 2, len(reference) - 1]
    finite_curl = np.abs(curl[frames])
    finite_curl = finite_curl[np.isfinite(finite_curl)]
    vmax = (float(finite_curl.max()) or 1) if finite_curl.size else 1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
    for ax, frame in zip(axes, frames, strict=True):
        im = ax.imshow(
            np.ma.masked_invalid(curl[frame].T),
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"After burn-in: t={times[frame]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="Vorticity", shrink=0.75)
    fig.suptitle(
        "Restricted fine reference · first held-out IC · no learned correction"
    )
    fig.savefig(destination / "reference_fields.png", dpi=170)
    plt.close(fig)


def _read_result(path: Path) -> tuple[dict, dict]:
    with tarfile.open(path) as archive:
        try:
            outcome = archive.extractfile("./outcome.json")
        except KeyError:
            result = archive.extractfile(
                "./ns-grid/optimization/solver_in_loop_supervised/result.json"
            )
            raw = json.load(result) if result is not None else {}
            return {
                "completed": False,
                "solver_failures": raw.get("_solver_failures", raw),
            }, {}
        if outcome is None:
            raise ValueError("outcome.json is missing")
        metrics = json.load(outcome)
        if not metrics.get("eligible_for_corrector_training", True):
            return metrics, {}
        try:
            fields = archive.extractfile(
                "./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz"
            )
        except KeyError:
            return metrics, {}
        if fields is None:
            raise ValueError("corrector_fields.npz is missing")
        with np.load(io.BytesIO(fields.read())) as arrays:
            errors = {
                arm: arrays[f"error_{arm}_samples_0"]
                for arm in ("corrected", "stop_gradient", "supervised", "uncorrected")
                if f"error_{arm}_samples_0" in arrays
            }
    return metrics, errors


def _job_states(campaign: Path) -> dict[str, str]:
    """Read accounting once; numerical results remain independent of Slurm."""
    manifest = campaign / "job_ids.json"
    if not manifest.exists():
        return {}
    jobs = json.loads(manifest.read_text())
    if not jobs:
        return {}
    try:
        result = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--jobs",
                ",".join(str(int(job)) for job in jobs.values()),
                "--format",
                "JobIDRaw,State%30",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Slurm accounting unavailable; missing results remain pending: {exc}")
        return {}
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[1].strip():
            states[fields[0]] = fields[1].split()[0].rstrip("+")
    return {cell: states[str(job)] for cell, job in jobs.items() if str(job) in states}


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


def _plot_outcomes(rows: list[dict], destination: Path) -> None:
    """Show absolute error and small paired effects without a broad ratio axis."""
    import matplotlib.pyplot as plt

    arms = ["uncorrected", "supervised", "stop_gradient", "corrected"]
    labels = ["Solver only", "Supervised", "Recurrent, stopped", "Recurrent, full"]
    colors = ["0.5", "tab:orange", "tab:blue", "tab:green"]
    for row in rows:
        means = [row["arm_summaries"][a]["mean_rollout_error"] for a in arms]
        if any(v is None or not np.isfinite(v) or v <= 0 for v in means):
            continue
        fig, (absolute, effect) = plt.subplots(1, 2, figsize=(11, 4.7))
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.25, top=0.72, wspace=0.8)
        for i, (arm, color, mean) in enumerate(zip(arms, colors, means, strict=True)):
            values = np.asarray(row["arm_summaries"][arm]["model_seed_means"])
            absolute.scatter(
                100 * values, np.full(len(values), i), color=color, alpha=0.5, s=25
            )
            absolute.plot(100 * mean, i, "D", color=color, ms=7)
            absolute.annotate(
                f"{100 * mean:.3f}%",
                (100 * mean, i),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        absolute.set_yticks(range(4), labels)
        absolute.set_ylim(3.55, -0.65)
        absolute.set_xlim(
            left=0,
            right=125
            * max(max(row["arm_summaries"][a]["model_seed_means"]) for a in arms),
        )
        absolute.set_xlabel("Mean relative L2 error (%)")
        absolute.set_title("Absolute rollout error", loc="left", weight="bold")
        full = np.asarray(row["arm_summaries"]["corrected"]["model_seed_means"])
        bounds = [0.0]
        for i, (key, arm) in enumerate(
            [("vs_supervised", "supervised"), ("vs_stopped", "stop_gradient")]
        ):
            result = row[key]
            value = 100 * (1 - 1 / result["ratio"])
            interval = result["ci95"]
            interval = [100 * (1 - 1 / x) for x in interval] if interval else None
            seed_base = np.asarray(row["arm_summaries"][arm]["model_seed_means"])
            seed_gain = 100 * (1 - full / seed_base)
            bounds.extend(seed_gain.tolist() + [value] + (interval or []))
            checked = row["admitted"] and row["gradient_checks_passed"]
            resolved = interval is not None and not interval[0] <= 0 <= interval[1]
            color = (
                ("#147d64" if value > 0 else "#b55b35")
                if checked and resolved
                else "0.5"
            )
            effect.scatter(
                seed_gain, np.full(len(seed_gain), i), color=color, alpha=0.4, s=25
            )
            if interval:
                effect.plot(interval, [i, i], color=color, lw=2)
            effect.plot(value, i, "D" if checked else "x", color=color, ms=7)
            label = f"{value:+.2f}%"
            if interval:
                label += f" [{interval[0]:+.2f}, {interval[1]:+.2f}]"
            effect.annotate(
                label,
                (value, i),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        pad = max(1.0, (max(bounds) - min(bounds)) * 0.3)
        effect.set_xlim(min(bounds) - pad, max(bounds) + pad)
        effect.axvline(0, color="0.5", ls="--", lw=1)
        effect.set_yticks([0, 1], ["vs supervised", "vs stopped"])
        effect.set_ylim(1.65, -0.65)
        effect.set_xlabel("Error reduction with full gradients (%)")
        effect.set_title(
            "Paired benefit · positive is better", loc="left", weight="bold"
        )
        for ax in (absolute, effect):
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", alpha=0.1)
        curriculum = row.get("curriculum", [])
        rates = " → ".join(f"{s['lr']:g}" for s in curriculum)
        subtitle = (
            f"{row['phase'].capitalize()} · {row['solver']} · "
            f"{row['n_model_seeds']} model seeds × {row['n_test_ics']} held-out ICs · "
            f"{row['updates']:,} updates · horizon {row['unroll']}"
        )
        if rates:
            subtitle += f" · LR {rates}"
        fig.text(
            0.05, 0.94, "How much do solver gradients help?", fontsize=18, weight="bold"
        )
        fig.text(0.05, 0.86, subtitle, fontsize=10, color="0.3")
        note = "Diamonds: pooled means/effects. Dots: model-seed means/effects."
        note += "\nIntervals: paired model-seed/IC bootstrap, 95%. Error averages all noninitial rollout frames."
        if not (row["admitted"] and row["gradient_checks_passed"]):
            note += "\n×: reference or gradient check failed; not an admitted gradient-benefit result."
        fig.text(0.05, 0.06, note, fontsize=9, color="0.35")
        fig.savefig(
            destination
            / f"outcome-{row['phase']}-{row['solver']}-{row['protocol_sha256'][:8]}.png",
            dpi=180,
            facecolor="white",
        )
        plt.close(fig)


def _plot_comparisons(rows: list[dict], destination: Path) -> None:
    """Compare solvers within a fixed protocol on identical, readable axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    if not rows:
        return
    names = {
        "jax-cfd": "JAX-CFD",
        "phiflow": "PhiFlow",
        "pict": "PICT",
        "warp-ns": "Warp",
        "xlb": "XLB",
        "ins-jl": "INS.jl",
    }
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["protocol_sha256"], row["source_sha256"])].append(row)

    def render(group: list[dict], path: Path, overview: bool = False) -> None:
        exemplar = group[0]
        by_solver = {r["solver"]: r for r in group}
        solvers = list(names) if overview else [s for s in names if s in by_solver]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.72, wspace=0.13)
        finite = [
            r[k]["ratio"]
            for r in group
            for k in ("vs_supervised", "vs_stopped", "vs_uncorrected")
            if r[k]["ratio"] is not None and r[k]["ratio"] > 0
        ]
        bounds = finite + [
            v
            for r in group
            for k in ("vs_supervised", "vs_stopped", "vs_uncorrected")
            for v in (r[k]["ci95"] or [])
            if np.isfinite(v) and v > 0
        ]
        lo = min(0.25, 2 ** np.floor(np.log2(min(bounds)))) if bounds else 0.25
        hi = max(2, 2 ** np.ceil(np.log2(max(bounds)))) if bounds else 2
        ticks = [lo, 0.5, 1, hi] if lo < 0.5 else [lo, 1, hi]
        for ax, key, title in zip(
            axes,
            ("vs_supervised", "vs_stopped", "vs_uncorrected"),
            ("vs supervised", "vs stopped gradients", "vs solver only"),
            strict=True,
        ):
            ax.axvspan(lo * 0.8, 1, color="#faf4ef", zorder=0)
            ax.axvspan(1, hi * 1.3, color="#edf6f3", zorder=0)
            ax.axvline(1, color="#555555", lw=1, ls="--")
            for i, solver in enumerate(solvers):
                row = by_solver.get(solver)
                if row is None:
                    ax.text(
                        0.04,
                        i,
                        "No trained result",
                        transform=ax.get_yaxis_transform(),
                        color="0.5",
                        fontsize=9,
                    )
                    continue
                value = row[key]["ratio"]
                if value is None or not np.isfinite(value) or value <= 0:
                    ax.text(
                        0.04,
                        i,
                        "Nonfinite rollout",
                        transform=ax.get_yaxis_transform(),
                        color="0.5",
                        fontsize=9,
                    )
                    continue
                rankable = row["admitted"] and row["gradient_checks_passed"]
                interval = row[key]["ci95"]
                unresolved = interval is not None and interval[0] <= 1 <= interval[1]
                color = (
                    ("#147d64" if value > 1 else "#b55b35")
                    if rankable and not unresolved
                    else "#888888"
                )
                ax.plot([1, value], [i, i], color=color, alpha=0.4, lw=2)
                ax.plot(value, i, "o" if rankable else "x", color=color, ms=6)
                if interval:
                    ax.plot(interval, [i, i], color=color, lw=2)
                label = f"{value:.2f}×"
                if interval:
                    label += f"\n[{interval[0]:.2f}, {interval[1]:.2f}]"
                ax.annotate(
                    label,
                    (value, i),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=color,
                )
            ax.set_xscale("log", base=2)
            ax.set_xlim(lo * 0.8, hi * 1.3)
            ax.set_ylim(len(solvers) - 0.5, -0.65)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_title(title, fontsize=12, pad=15, weight="bold")
            ax.tick_params(axis="both", length=0, labelsize=10, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_yticks(range(len(solvers)), [names[s] for s in solvers])
        fig.text(
            0.13,
            0.94,
            "Does differentiating through the solver help?",
            fontsize=17,
            weight="bold",
        )
        curriculum = exemplar.get("curriculum", [])
        horizon_label = "→".join(str(stage["unroll"]) for stage in curriculum) or str(
            exemplar["unroll"]
        )
        if curriculum and len({stage["unroll"] for stage in curriculum}) == 1:
            horizon_label = str(curriculum[0]["unroll"])
        rate_label = ""
        if len({stage["lr"] for stage in curriculum}) > 1:
            rate_label = " · lr " + "→".join(f"{stage['lr']:g}" for stage in curriculum)
        physical_label = (
            f"forcing k={exemplar['forcing_wavenumber']} a={exemplar['forcing_amplitude']:g} · "
            f"burn-in {exemplar['burn_in_time']:g}"
            if exemplar.get("forcing_amplitude", 0)
            else f"vortex scale {exemplar['k0']:g} · amplitude {exemplar['amplitude']:g}"
        )
        fig.text(
            0.13,
            0.87,
            f"{exemplar['phase'].capitalize()} · {exemplar['N']}² · {physical_label} · "
            f"unroll {horizon_label} · {exemplar['updates']:,} updates{rate_label}",
            fontsize=10,
            color="0.35",
        )
        if exemplar.get("reference_kind") == "solver_self_refined":
            fig.text(
                0.13,
                0.80,
                "Each solver uses its own refined reference; compare training methods within each row.",
                fontsize=9,
                color="0.35",
            )
        fig.text(
            0.55,
            0.13,
            "Baseline error / full-gradient error    ·    Below 1: worse    |    Above 1: better",
            ha="center",
            fontsize=10,
        )
        seeds = sorted({r["n_model_seeds"] for r in group})
        ic_counts = sorted({r["n_test_ics"] for r in group})
        note = (
            "Single model seed; exploratory estimates without confidence intervals."
            if seeds == [1]
            else f"Model seeds per row: {','.join(map(str, seeds))}; held-out ICs: {','.join(map(str, ic_counts))}. "
            "Bars: paired bootstrap 95% intervals."
        )
        if any(not (r["admitted"] and r["gradient_checks_passed"]) for r in group):
            note += "  ×: reference or gradient check failed."
        fig.text(0.13, 0.035, note, fontsize=9, color="0.4")
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)

    for group in groups.values():
        row = group[0]
        render(
            group,
            destination / f"comparison-{row['phase']}-{row['protocol_sha256'][:8]}.png",
        )
    # The headline uses the common, original pilot protocol, never the best result.
    primary = [
        r
        for r in rows
        if r["phase"] == "explore"
        and r["N"] == 32
        and r["k0"] == 4
        and r["amplitude"] == 0.5
        and r["unroll"] == 8
        and r["updates"] == 300
    ]
    if primary:
        render(primary, destination / "comparisons.png", overview=True)
    elif groups:
        # Select the most widely evaluated protocol, never the best outcome.
        render(
            max(groups.values(), key=len),
            destination / "comparisons.png",
            overview=True,
        )


def main() -> None:
    """Keep missing and failed cells visible alongside successful comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plots", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    pending, failures, reference_checks = [], [], []
    job_states = _job_states(args.campaign)
    terminal_states = {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
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
            run["dataset"]["amplitude"],
            run["training"]["unroll"],
            run["training"]["max_updates"],
            payload["source_sha256"],
            protocol_hash,
        )
        archive = args.campaign / "results" / config.stem / "results.tar"
        if not archive.exists():
            state = job_states.get(config.stem)
            if state in terminal_states:
                failures.append(
                    {
                        "cell": config.stem,
                        "reason": f"Slurm {state}: no result archive",
                    }
                )
            else:
                pending.append(config.stem)
            continue
        try:
            metrics, errors = _read_result(archive)
        except (KeyError, ValueError) as exc:
            failures.append({"cell": config.stem, "reason": str(exc)})
            continue
        if metrics.get("reference_only"):
            if args.plots:
                _plot_reference(archive, args.plots / config.stem, metrics)
            reference_checks.append(
                {"cell": config.stem, "solver": payload["solver"], "metrics": metrics}
            )
            continue
        if not metrics.get("completed") or len(errors) != 4:
            failures.append(
                {
                    "cell": config.stem,
                    "reason": "reference admission failed"
                    if not metrics.get("eligible_for_corrector_training", True)
                    else "incomplete run",
                    "metrics": metrics,
                }
            )
            continue
        if args.plots:
            _plot_cell(archive, args.plots / config.stem, payload, metrics)
        groups[key].append((metrics, errors))
    rows = []
    for key, cells in groups.items():
        phase, solver, n, k0, amplitude, unroll, updates, source, protocol_hash = key
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
                "model_seed_means": [
                    float(v) if np.isfinite(v) else None
                    for v in samples[..., 1:].mean(axis=(1, 2))
                ],
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
                "reference_kind": cells[0][0].get("reference_kind"),
                "solver": solver,
                "N": n,
                "k0": k0,
                "amplitude": amplitude,
                "unroll": unroll,
                "updates": updates,
                "pretrain_updates": cells[0][0].get("pretrain_updates_per_seed", 0),
                "curriculum": cells[0][0].get("training_curriculum", []),
                "training_examples_per_seed": cells[0][0].get(
                    "training_examples_per_seed"
                ),
                "pretrained_mean_rollout_error": (
                    float(
                        np.mean([m["pretrained_mean_rollout_error"] for m, _ in cells])
                    )
                    if "pretrained_mean_rollout_error" in cells[0][0]
                    else None
                ),
                "pretrain_common_seconds": sum(
                    m.get("pretrain_common_wall_time_s", 0.0) for m, _ in cells
                ),
                "n_model_seeds": errors["corrected"].shape[0],
                "n_test_ics": errors["corrected"].shape[1],
                "forcing_amplitude": cells[0][0].get("forcing_amplitude", 0.0),
                "forcing_wavenumber": cells[0][0].get("forcing_wavenumber"),
                "burn_in_time": cells[0][0].get("burn_in_time", 0.0),
                "source_sha256": source,
                "protocol_sha256": protocol_hash,
                "admitted": admitted,
                "gradient_checks_passed": gradient_checked,
                "fd_relative_errors": fd,
                **ratios,
                "arm_summaries": summaries,
                "resolved_joint_benefit": admitted
                and gradient_checked
                and all(
                    r["ci95"] is not None and r["ci95"][0] > 1 for r in ratios.values()
                ),
                "full_training_seconds": sum(
                    m.get("pretrain_common_wall_time_s", 0.0)
                    + m["training_wall_time_s"]
                    for m, _ in cells
                ),
                "supervised_total_seconds": sum(
                    m.get("pretrain_common_wall_time_s", 0.0)
                    + m["supervised_total_wall_time_s"]
                    for m, _ in cells
                ),
                "stopped_training_seconds": sum(
                    m.get("pretrain_common_wall_time_s", 0.0)
                    + m["stop_gradient_training_wall_time_s"]
                    for m, _ in cells
                ),
            }
        )
    result = {
        "comparisons": rows,
        "pending": pending,
        "failures": failures,
        "reference_checks": reference_checks,
        "job_states": job_states,
    }
    if args.plots:
        _plot_comparisons(rows, args.plots)
        _plot_outcomes(rows, args.plots)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
