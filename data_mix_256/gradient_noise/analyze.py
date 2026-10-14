"""Analyze all paired noise arms. Run numerical work/rendering on Slurm only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("clean", "large", "medium", "small")
SEEDS = (0, 1, 2)
COLORS = {
    "clean": "#333333",
    "large": "#0072B2",
    "medium": "#E69F00",
    "small": "#009E73",
}


def read(path: Path):
    return json.loads(path.read_text())


def array_hash(array):
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_kwargs(kwargs):
    """Remove seed and process addresses, retaining callable names and IC metadata."""
    result = json.loads(json.dumps(kwargs))
    # IcSpec repr contains process-specific callable addresses. Normalize only
    # this known field/pattern; names, descriptions, plot parameters and every
    # substantive experiment setting remain compared. Source hashes and exact
    # generated-data hashes independently bind the implementation and inputs.
    for name, spec in result.get("make_ic", {}).items():
        require(isinstance(spec, str), f"Unexpected make_ic representation: {name}")
        normalized, count = re.subn(
            r"(<function [A-Za-z_][A-Za-z0-9_.<>]* at )0x[0-9a-fA-F]+(>)",
            r"\1<process-address>\2",
            spec,
        )
        require(count == 1, f"Unknown make_ic callable representation: {name}")
        result["make_ic"][name] = normalized
    for run in result["runs"]:
        del run["training"]["model_seed"]
    return result


def collect(campaign: Path):
    """Never silently drop failed cells or compare differently prepared data."""
    rows = []
    shared = None
    paired_initializations = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            cell = campaign / f"{condition}-seed{seed}"
            outcome = read(cell / "outcome.json")
            require(
                outcome.get("exit_code") == 0
                and outcome.get("script_completed") is True,
                f"Incomplete cell {cell}",
            )
            receipt_paths = [
                p
                for p in (
                    cell / "noise_receipt.json",
                    cell / "results/noise_receipt.json",
                )
                if p.exists()
            ]
            require(len(receipt_paths) == 1, f"Missing or ambiguous receipt {cell}")
            receipt_path = receipt_paths[0]
            receipt = read(receipt_path)
            require(
                receipt["complete"] is True and receipt["checks_only"] is False,
                f"Incomplete receipt {cell}",
            )
            require(
                receipt["arm"] == condition and receipt["model_seed"] == seed,
                f"Wrong receipt identity {cell}",
            )
            probe = receipt["initial_gradient_probe"]
            require(
                all(
                    probe[key] is True
                    for key in (
                        "forward_states_bitwise_equal",
                        "forward_loss_bitwise_equal",
                        "reproducible_bitwise",
                    )
                ),
                f"Forward/replay failure {cell}",
            )
            require(
                receipt["noise_seed"] == 17000 + seed
                and receipt["relative_strength"]
                == (0 if condition == "clean" else 0.1),
                f"Noise protocol mismatch {cell}",
            )
            require(
                set(receipt["data_hashes"]) == {"training_jax", "scales", "validation"},
                f"Missing dataset provenance {cell}",
            )
            source_identity = {
                Path(path).name: sha for path, sha in receipt["source_sha256"].items()
            }
            require(
                len(source_identity) == len(receipt["source_sha256"]),
                f"Ambiguous source basenames {cell}",
            )
            records = receipt["training_vjp_records"]
            counts = [
                sum(r["update"] == update for r in records) for update in range(256)
            ]
            expected_count = len(probe["state_vjp_records"])
            require(
                expected_count > 0
                and counts == [expected_count] * 256
                and sum(counts) == len(records),
                f"Missing training interventions {cell}",
            )
            folder = cell / "results/buoyancy-grid/optimization/solver_in_loop"
            result = read(folder / "result.json")
            require(
                len(result["results"]) == 1, f"Expected one registered run in {cell}"
            )
            entry = result["results"][0]
            require(entry["solver"] == "JAX-CFD", f"Unexpected solver in {cell}")
            metrics = entry["metrics"]
            require(
                metrics["status"] == "ok" and metrics["updates"] == 256,
                f"Failed or changed training budget {cell}",
            )
            method = metrics["solver_in_loop"]
            candidates = method["candidates"]
            require(len(candidates) == 1, f"Pilot expects one fixed LR in {cell}")
            candidate = candidates[0]
            require(
                candidate["completed_updates"] == 256
                and len(candidate["training_loss"]) == 256,
                f"Incomplete training {cell}",
            )
            checkpoints = candidate["validation_checkpoints"]
            require(
                [p["update"] for p in checkpoints] == [0, 64, 128, 192, 256],
                f"Changed checkpoint protocol {cell}",
            )
            require(
                all(np.isfinite(p["loss"]) and p["loss"] >= 0 for p in checkpoints),
                f"Invalid validation losses {cell}",
            )
            winner = min(checkpoints, key=lambda p: (p["loss"], p["update"]))
            require(
                method["selected_candidate"] == 0
                and method["selected_update"]
                == winner["update"]
                == candidate["selected_update"]
                and candidate["validation_loss"] == winner["loss"],
                f"Invalid validation selection {cell}",
            )
            with np.load(folder / "corrector_fields.npz", allow_pickle=False) as data:
                index = list(map(str, data["solver_names"])).index("JAX-CFD")
                arrays = {
                    key: np.asarray(data[f"{key}_{index}"])
                    for key in ("reference", "uncorrected", "solver_in_loop")
                }
            require(
                all(
                    a.shape == (101, 64, 64, 3) and np.isfinite(a).all()
                    for a in arrays.values()
                ),
                f"Malformed rollout arrays {cell}",
            )
            invariants = {
                "source_identity": source_identity,
                "data_hashes": receipt["data_hashes"],
                "experiment_kwargs": common_kwargs(
                    receipt["resolved_experiment_kwargs"]
                ),
                "reference_array": array_hash(arrays["reference"]),
                "uncorrected_array": array_hash(arrays["uncorrected"]),
                **{
                    key: metrics[key]
                    for key in (
                        "scales",
                        "training_cases",
                        "seeds",
                        "window_seed",
                        "window_starts",
                        "window_cases",
                        "unroll",
                        "eval_frames",
                        "train_frames",
                        "learning_rates",
                        "validation_interval",
                    )
                },
            }
            if shared is None:
                shared = invariants
            require(
                invariants == shared,
                f"Source, references, baseline, normalization, or schedule differ in {cell}",
            )
            initial_hash = receipt["initial_model_sha256"]
            paired_initializations.setdefault(seed, initial_hash)
            require(
                paired_initializations[seed] == initial_hash,
                f"Unpaired initialization {cell}",
            )
            losses = np.asarray(method["loss"], dtype=float)
            baseline = np.asarray(metrics["uncorrected"]["loss"], dtype=float)
            require(
                losses.shape == baseline.shape == (101,),
                f"Incorrect rollout metric length {cell}",
            )
            require(
                np.isfinite(losses).all()
                and np.isfinite(baseline).all()
                and np.all(losses >= 0)
                and np.all(baseline >= 0),
                f"Invalid test errors {cell}",
            )
            scales = np.asarray(metrics["scales"])
            require(
                scales.shape == (3,)
                and np.isfinite(scales).all()
                and np.all(scales > 0),
                f"Invalid scales {cell}",
            )
            for arm, recorded in (
                ("solver_in_loop", losses),
                ("uncorrected", baseline),
            ):
                error = ((arrays[arm] - arrays["reference"]) / scales) ** 2
                reconstructed = 0.5 * (
                    error[..., :2].mean(axis=(1, 2, 3))
                    + error[..., 2].mean(axis=(1, 2))
                )
                require(
                    np.allclose(reconstructed, recorded, rtol=2e-6, atol=1e-10),
                    f"Saved fields and loss metrics disagree: {cell}/{arm}",
                )
            test_mean, baseline_mean = (
                float(losses[1:].mean()),
                float(baseline[1:].mean()),
            )
            require(baseline_mean > 0, f"Cannot normalize zero baseline error {cell}")
            rows.append(
                {
                    "condition": condition,
                    "model_seed": seed,
                    "selected_update": winner["update"],
                    "validation_checkpoints": checkpoints,
                    "mean_test_loss": test_mean,
                    "uncorrected_mean_test_loss": baseline_mean,
                    "test_loss_over_uncorrected": test_mean / baseline_mean,
                    "total_training_seconds": method["total_training_seconds"],
                    "total_validation_seconds": method["total_validation_seconds"],
                    "initial_gradient_perturbation": {
                        "relative_l2": probe["parameter_gradient_relative_l2"],
                        "cosine": probe["parameter_gradient_cosine"],
                    },
                    "result_sha256": hashlib.sha256(
                        (folder / "result.json").read_bytes()
                    ).hexdigest(),
                    "receipt_sha256": hashlib.sha256(
                        receipt_path.read_bytes()
                    ).hexdigest(),
                }
            )
    for row in rows:
        clean = next(
            r
            for r in rows
            if r["condition"] == "clean" and r["model_seed"] == row["model_seed"]
        )
        require(
            clean["mean_test_loss"] > 0,
            "Cannot form paired relative change against zero clean loss",
        )
        row["relative_change_vs_matched_clean_percent"] = 100 * (
            row["mean_test_loss"] / clean["mean_test_loss"] - 1
        )
    return rows, shared


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160, facecolor="white")
    plt.close(fig)


def render(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, key, ylabel, reference in (
        (
            axes[0],
            "test_loss_over_uncorrected",
            "Selected held-out loss / uncorrected loss",
            1,
        ),
        (
            axes[1],
            "relative_change_vs_matched_clean_percent",
            "Selected held-out loss change vs matched clean (%)",
            0,
        ),
    ):
        for seed, color in zip(SEEDS, ("#0072B2", "#D55E00", "#009E73"), strict=True):
            values = [
                next(
                    r[key]
                    for r in rows
                    if r["condition"] == condition and r["model_seed"] == seed
                )
                for condition in CONDITIONS
            ]
            ax.plot(
                np.arange(4) + (seed - 1) * 0.055,
                values,
                marker="o",
                color=color,
                alpha=0.8,
                lw=1,
                label=f"Network seed {seed}",
            )
        means = [
            np.mean([r[key] for r in rows if r["condition"] == condition])
            for condition in CONDITIONS
        ]
        ax.scatter(
            np.arange(4),
            means,
            color="black",
            marker="D",
            s=35,
            zorder=5,
            label="Mean of 3 paired runs",
        )
        ax.axhline(reference, color="gray", linestyle="--", lw=1)
        ax.set(
            xticks=range(4),
            xticklabels=CONDITIONS,
            ylabel=ylabel,
            xlabel="Training adjoint intervention",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Scale-targeted gradient-noise pilot: all three matched seeds\nLower is better; validation-selected checkpoints; descriptive results, no confidence intervals",
        fontsize=12,
    )
    save_figure(fig, output, "paired_test_errors")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS, strict=True):
        for condition in CONDITIONS:
            row = next(
                r
                for r in rows
                if r["condition"] == condition and r["model_seed"] == seed
            )
            points = row["validation_checkpoints"]
            ax.plot(
                [p["update"] for p in points],
                [p["loss"] for p in points],
                marker="o",
                color=COLORS[condition],
                label=condition,
            )
            selected = next(p for p in points if p["update"] == row["selected_update"])
            ax.scatter(
                selected["update"],
                selected["loss"],
                marker="*",
                s=145,
                color=COLORS[condition],
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
        ax.set(
            title=f"Network seed {seed}",
            xlabel="Completed optimizer updates",
            xticks=[0, 64, 128, 192, 256],
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Full-trajectory validation loss")
    axes[-1].legend(title="Noise condition", fontsize=8)
    fig.suptitle(
        "Validation trajectories and selected checkpoints (stars)\nEvery arm completes 256 updates, including when update 0 is selected",
        fontsize=12,
    )
    save_figure(fig, output, "validation_checkpoints")

    present = [r for r in rows if r["initial_gradient_perturbation"] is not None]
    if present:
        require(
            len(present) == len(rows),
            "Partial gradient receipts: refusing selective gradient plot",
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for ax, key, label in (
            (
                axes[0],
                "relative_l2",
                "Initial neural-gradient perturbation / clean norm",
            ),
            (axes[1], "cosine", "Initial noisy/clean neural-gradient cosine"),
        ):
            for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True):
                values = [
                    next(
                        r["initial_gradient_perturbation"][key]
                        for r in rows
                        if r["condition"] == condition and r["model_seed"] == seed
                    )
                    for condition in CONDITIONS
                ]
                require(np.isfinite(values).all(), "Nonfinite initial-gradient receipt")
                ax.plot(range(4), values, marker=marker, label=f"Network seed {seed}")
            ax.set(xticks=range(4), xticklabels=CONDITIONS, ylabel=label)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Measured initial parameter-gradient effect\nState-VJP perturbation amplitude is fixed; resulting neural-gradient change need not be 10%",
            fontsize=12,
        )
        save_figure(fig, output, "initial_gradient_perturbation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    rows, invariants = collect(args.campaign)
    report = {
        "rows": rows,
        "verified_invariants": invariants,
        "conditions": CONDITIONS,
        "model_seeds": SEEDS,
        "definition": "Mean normalized held-out loss over predicted frames1..100; selection uses full validation only. Relative changes are paired within network seed.",
        "limitations": [
            "Three paired replicates, descriptive only; no significance or confidence-interval claims.",
            "Scale-targeted input-state adjoint corruption does not directly manipulate the earlier output-restricted conditioning diagnostic.",
            "Reference generation and selected-model test evaluation are excluded from reported training/validation timings.",
        ],
        "all_expected_cells_present": True,
    }
    (output / "pilot_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    render(rows, output)
    receipt = {
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file() and p.name != "analysis_receipt.json"
        },
    }
    (output / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
