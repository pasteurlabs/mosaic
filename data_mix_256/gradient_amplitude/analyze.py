"""Amplitude sweep analysis. Numerical work and rendering must run on Slurm."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pilot_checks import array_hash, common_kwargs, read, require

BANDS = ("large", "medium", "small")
STRENGTHS = (0.03, 0.1, 0.3, 1.0, 3.0)
SEEDS = tuple(range(5))


def expected_cells(manifest):
    cells = manifest["cells"]
    keys = [(c["arm"], float(c["strength"]), int(c["model_seed"])) for c in cells]
    expected = {("clean", 0.0, seed) for seed in SEEDS} | {
        (band, strength, seed)
        for band in BANDS
        for strength in STRENGTHS
        for seed in SEEDS
    }
    require(
        len(keys) == 80 and set(keys) == expected,
        "Manifest must contain exactly the frozen80 cells, no duplicates",
    )
    return cells


def failure_classification(receipt):
    """Recognize only the evidenced native advective-CFL rejection during fitting."""
    raw = receipt.get("failure_kind", "check_or_execution_failure")
    training_failure = receipt.get("training_failure") or {}
    message = "Initial advective CFL must be <= 0.5; reduce dt."
    if (
        raw == "check_or_execution_failure"
        and receipt.get("phase") == "training"
        and training_failure.get("type") == "ValidationError"
        and message in training_failure.get("message", "")
        and message in (receipt.get("failure") or {}).get("message", "")
    ):
        return "solver_admissibility_failure"
    return raw


def collect(campaign, manifest):
    rows, shared, initial = [], None, {}
    for cell in expected_cells(manifest):
        folder = campaign / cell["path"]
        require(
            folder.resolve().is_relative_to(campaign.resolve()), "Cell escapes campaign"
        )
        receipt_paths = [
            p
            for p in (
                folder / "results/noise_receipt.json",
                folder / "noise_receipt.json",
            )
            if p.exists()
        ]
        row = {key: cell[key] for key in ("arm", "strength", "model_seed", "path")}
        row["classification"] = "missing_receipt"
        if not receipt_paths:
            rows.append(row)
            continue
        require(len(receipt_paths) == 1, f"Ambiguous receipt: {folder}")
        receipt = read(receipt_paths[0])
        row["receipt_sha256"] = hashlib.sha256(
            receipt_paths[0].read_bytes()
        ).hexdigest()
        require(
            receipt["arm"] == cell["arm"]
            and receipt["model_seed"] == cell["model_seed"]
            and receipt["relative_strength"] == cell["strength"],
            f"Cell identity mismatch {folder}",
        )
        # Missing/check/infra failures remain visible; never infer divergence from a nonzero shell exit.
        if (
            receipt.get("outcome") == "passed"
            and receipt.get("complete")
            and not receipt.get("checks_only")
        ):
            row["classification"] = "completed"
        elif receipt.get("outcome") == "failed":
            row["classification"] = failure_classification(receipt)
        else:
            row["classification"] = "incomplete_or_check_only"
        row["raw_failure_kind"] = receipt.get("failure_kind")
        row["classification_evidence"] = (
            "Native ValidationError explicitly rejects Initial advective CFL > 0.5 during fit; corroborated in wrapped failure. This does not establish nonfinite divergence."
            if row["classification"] == "solver_admissibility_failure"
            else None
        )
        row["failure"] = receipt.get("failure")
        row["phase"] = receipt.get("phase")
        row["training_failure"] = receipt.get("training_failure")
        row["attempted_update"] = receipt.get("attempted_update")
        row["completed_updates_lower_bound"] = receipt.get(
            "completed_updates_lower_bound"
        )
        if "data_hashes" in receipt:
            require(
                set(receipt["data_hashes"]) == {"training_jax", "validation", "scales"},
                f"Missing data hashes {folder}",
            )
            source = {
                Path(p).name: value for p, value in receipt["source_sha256"].items()
            }
            require(
                len(source) == len(receipt["source_sha256"]),
                "Ambiguous source basenames",
            )
            invariant = {
                "source": source,
                "data": receipt["data_hashes"],
                "kwargs": common_kwargs(receipt["resolved_experiment_kwargs"]),
            }
            if shared is None:
                shared = invariant
            require(invariant == shared, f"Unmatched source/data/protocol {folder}")
            seed = cell["model_seed"]
            initial.setdefault(seed, receipt["initial_model_sha256"])
            require(
                initial[seed] == receipt["initial_model_sha256"],
                f"Unpaired initial model {folder}",
            )
        probe = receipt.get("initial_gradient_probe")
        if probe:
            require(
                all(
                    probe[k]
                    for k in (
                        "forward_states_bitwise_equal",
                        "forward_loss_bitwise_equal",
                        "reproducible_bitwise",
                    )
                ),
                f"Failed intervention checks {folder}",
            )
            row["gradient_relative_l2"] = probe["parameter_gradient_relative_l2"]
            row["gradient_cosine"] = probe["parameter_gradient_cosine"]
        if row["classification"] != "completed":
            row["partial_validation_checkpoints"] = receipt.get(
                "observed_validation_checkpoints", []
            )
            rows.append(row)
            continue
        require(
            "data_hashes" in receipt and probe is not None,
            f"Completed run missing provenance/checks {folder}",
        )
        records = receipt.get("training_vjp_records", [])
        expected_count = len(probe["state_vjp_records"])
        counts = [sum(r["update"] == update for r in records) for update in range(256)]
        require(
            expected_count > 0
            and counts == [expected_count] * 256
            and sum(counts) == len(records),
            f"Incomplete training intervention ledger {folder}",
        )
        require(
            receipt["noise_seed"] == 17000 + cell["model_seed"],
            f"Noise RNG mismatch {folder}",
        )
        result_folder = folder / "results/buoyancy-grid/optimization/solver_in_loop"
        result = read(result_folder / "result.json")
        require(len(result["results"]) == 1, "Expected one solver result")
        entry = result["results"][0]
        require(entry["solver"] == "JAX-CFD", "Wrong solver")
        metrics = entry["metrics"]
        method = metrics["solver_in_loop"]
        require(
            metrics["status"] == "ok"
            and metrics["updates"] == 256
            and len(method["candidates"]) == 1,
            "Incomplete or changed recipe",
        )
        candidate = method["candidates"][0]
        checkpoints = candidate["validation_checkpoints"]
        require(
            candidate["completed_updates"] == 256
            and len(candidate["training_loss"]) == 256,
            "Incomplete training",
        )
        require(
            [p["update"] for p in checkpoints] == [0, 64, 128, 192, 256],
            "Changed checkpoint schedule",
        )
        require(
            all(np.isfinite(p["loss"]) and p["loss"] >= 0 for p in checkpoints),
            "Invalid validation metrics",
        )
        winner = min(checkpoints, key=lambda p: (p["loss"], p["update"]))
        require(
            method["selected_update"]
            == candidate["selected_update"]
            == winner["update"]
            and candidate["validation_loss"] == winner["loss"],
            "Wrong selected checkpoint",
        )
        with np.load(
            result_folder / "corrector_fields.npz", allow_pickle=False
        ) as data:
            index = list(map(str, data["solver_names"])).index("JAX-CFD")
            arrays = {
                k: np.asarray(data[f"{k}_{index}"])
                for k in ("reference", "uncorrected", "solver_in_loop")
            }
        require(
            all(
                a.shape == (101, 64, 64, 3) and np.isfinite(a).all()
                for a in arrays.values()
            ),
            "Malformed saved fields",
        )
        field_invariants = {
            k: array_hash(arrays[k]) for k in ("reference", "uncorrected")
        }
        field_invariants.update(
            {
                k: metrics[k]
                for k in (
                    "scales",
                    "window_cases",
                    "window_starts",
                    "window_seed",
                    "training_cases",
                    "seeds",
                    "unroll",
                    "learning_rates",
                )
            }
        )
        row["_field_invariants"] = field_invariants
        losses = np.asarray(method["loss"])
        baseline = np.asarray(metrics["uncorrected"]["loss"])
        require(
            losses.shape == baseline.shape == (101,)
            and np.isfinite(losses).all()
            and np.isfinite(baseline).all(),
            "Invalid test metrics",
        )
        for arm, recorded in (("solver_in_loop", losses), ("uncorrected", baseline)):
            error = (
                (arrays[arm] - arrays["reference"]) / np.asarray(metrics["scales"])
            ) ** 2
            recomputed = 0.5 * (
                error[..., :2].mean(axis=(1, 2, 3)) + error[..., 2].mean(axis=(1, 2))
            )
            require(
                np.allclose(recomputed, recorded, rtol=2e-6, atol=1e-10),
                "Saved fields disagree with metrics",
            )
        row.update(
            selected_update=winner["update"],
            validation_checkpoints=checkpoints,
            selected_test_loss=float(losses[1:].mean()),
            uncorrected_test_loss=float(baseline[1:].mean()),
            final_validation_loss=checkpoints[-1]["loss"],
            selected_validation_loss=winner["loss"],
            final_validation_over_selected=(
                checkpoints[-1]["loss"] / winner["loss"] if winner["loss"] > 0 else None
            ),
        )
        rows.append(row)
    completed = [r for r in rows if r["classification"] == "completed"]
    if completed:
        fields = completed[0]["_field_invariants"]
        require(
            all(r["_field_invariants"] == fields for r in completed),
            "Baseline/reference/scales/schedules differ",
        )
        for row in completed:
            del row["_field_invariants"]
    for row in completed:
        clean = next(
            (
                r
                for r in completed
                if r["arm"] == "clean" and r["model_seed"] == row["model_seed"]
            ),
            None,
        )
        require(
            clean is not None,
            "Cannot report paired effect without completed matched clean run",
        )
        for key in ("selected_test_loss", "final_validation_loss"):
            require(clean[key] > 0, "Zero clean denominator")
            row[key + "_change_percent"] = 100 * (row[key] / clean[key] - 1)
        require(row["uncorrected_test_loss"] > 0, "Zero uncorrected error")
        row["selected_test_over_uncorrected"] = (
            row["selected_test_loss"] / row["uncorrected_test_loss"]
        )
    return rows, shared


def render(rows, output):
    colors = plt.cm.tab10(np.arange(5))
    for key, title, ylabel, name in (
        (
            "selected_test_loss_change_percent",
            "Validation-selected checkpoint: held-out test error",
            "Paired change vs clean selected test (%)",
            "selected_test",
        ),
        (
            "final_validation_loss_change_percent",
            "Final update 256 checkpoint: validation error",
            "Paired change vs clean final validation (%)",
            "final_validation",
        ),
        (
            "gradient_relative_l2",
            "Initial parameter-gradient effect",
            "Gradient perturbation / clean gradient norm",
            "gradient_effect",
        ),
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4.7), sharey=True, layout="constrained"
        )
        for ax, band in zip(axes, BANDS, strict=True):
            for seed, color in zip(SEEDS, colors, strict=True):
                selected = [
                    next(
                        r
                        for r in rows
                        if r["arm"] == ("clean" if a == 0 else band)
                        and r["strength"] == a
                        and r["model_seed"] == seed
                    )
                    for a in (0.0, *STRENGTHS)
                ]
                values = [r.get(key, np.nan) for r in selected]
                ax.plot(
                    (0.0, *STRENGTHS),
                    values,
                    marker="o",
                    color=color,
                    label=f"Seed {seed}",
                )
            means = []
            for a in (0.0, *STRENGTHS):
                cells = [
                    r
                    for r in rows
                    if r["arm"] == ("clean" if a == 0 else band) and r["strength"] == a
                ]
                # Never replace a missing/diverged seed by a survivor-only mean.
                means.append(
                    float(np.mean([r[key] for r in cells]))
                    if len(cells) == 5 and all(key in r for r in cells)
                    else np.nan
                )
            ax.plot(
                (0.0, *STRENGTHS),
                means,
                color="black",
                marker="D",
                lw=2,
                label="Mean: all 5 required",
            )
            ax.set_xscale("symlog", linthresh=0.03)
            ax.set_yscale(
                "symlog", linthresh=0.01 if key == "gradient_relative_l2" else 10
            )
            ax.set(
                xticks=(0.0, *STRENGTHS),
                xticklabels=["0", ".03", ".1", ".3", "1", "3"],
                xlabel="Relative native state-VJP noise",
                title=band,
            )
            ax.axhline(0, color="gray", ls="--", lw=0.7)
            ax.grid(alpha=0.2)
        transform_label = (
            "symlog; linear below 0.01"
            if key == "gradient_relative_l2"
            else "symlog; linear within ±10%"
        )
        axes[0].set_ylabel(ylabel + "\n(" + transform_label + ")")
        axes[-1].legend(fontsize=7)
        fig.suptitle(
            title
            + "\nFive matched seeds; descriptive only; gaps retain unavailable outcomes"
        )
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, band in zip(axes, BANDS, strict=True):
        matrix = np.full((5, 5), np.nan)
        for i, seed in enumerate(SEEDS):
            for j, a in enumerate(STRENGTHS):
                row = next(
                    r
                    for r in rows
                    if r["arm"] == band
                    and r["strength"] == a
                    and r["model_seed"] == seed
                )
                if row["classification"] == "completed":
                    matrix[i, j] = row["selected_update"]
                else:
                    ax.text(
                        j,
                        i,
                        row["classification"].replace("_", "\n"),
                        ha="center",
                        va="center",
                        fontsize=5,
                    )
        im = ax.imshow(matrix, vmin=0, vmax=256, cmap="viridis", aspect="auto")
        ax.set(
            title=band,
            xticks=range(5),
            xticklabels=STRENGTHS,
            yticks=range(5),
            yticklabels=SEEDS,
            xlabel="Noise amplitude",
            ylabel="Network seed",
        )
        for i, j in zip(*np.where(np.isfinite(matrix)), strict=True):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="black" if matrix[i, j] >= 192 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=axes, label="Selected optimizer update")
    fig.suptitle("Validation selection and explicit unavailable outcomes")
    for extension in ("png", "svg"):
        fig.savefig(output / f"checkpoint_selection.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).parent / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    rows, invariants = collect(
        args.campaign, read(args.manifest or args.campaign / "analysis_manifest.json")
    )
    paired_summaries = []
    for band in BANDS:
        for strength in STRENGTHS:
            group = [r for r in rows if r["arm"] == band and r["strength"] == strength]
            for metric in (
                "selected_test_loss_change_percent",
                "final_validation_loss_change_percent",
            ):
                values = [r[metric] for r in group if metric in r]
                paired_summaries.append(
                    {
                        "band": band,
                        "strength": strength,
                        "metric": metric,
                        "available_pairs": len(values),
                        "mean_change_percent": float(np.mean(values))
                        if len(values) == 5
                        else None,
                        "all_five_worsen": all(v > 0 for v in values)
                        if len(values) == 5
                        else None,
                        "seed_changes": {
                            str(r["model_seed"]): r.get(metric) for r in group
                        },
                    }
                )
    report = {
        "paired_summaries": paired_summaries,
        "rows": rows,
        "verified_invariants": invariants,
        "limitations": [
            "Five paired replicates, descriptive only; no significance claims.",
            "Selected heldout test and final-checkpoint validation are distinct outcomes.",
            "Divergence and unavailable results remain explicit; no survivor-only summary means.",
        ],
        "classification_counts": {
            status: sum(r["classification"] == status for r in rows)
            for status in sorted({r["classification"] for r in rows})
        },
    }
    (output / "amplitude_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    render(rows, output)
    (output / "analysis_receipt.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        Path(__file__),
                        Path(__file__).with_name("pilot_checks.py"),
                    )
                }
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
