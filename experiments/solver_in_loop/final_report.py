"""Report the registered correction confirmation on a Slurm CPU allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "full": "Full solver gradients",
    "stopped": "Tuned stopped gradients",
    "supervised": "Tuned supervision",
    "matched_stopped": "Matched stopped gradients",
    "native": "Solver only",
}


def paired_ratio(a: np.ndarray, b: np.ndarray) -> dict:
    """Resample paired model seeds and shared ICs, never individual time frames."""
    if a.shape != b.shape or a.ndim != 2 or not np.all(np.isfinite([a, b])):
        raise ValueError("paired ratio requires finite matching model-by-IC arrays")
    rng = np.random.default_rng(11620261003)
    ratios = []
    for _ in range(10000):
        models = rng.integers(a.shape[0], size=a.shape[0])
        ics = rng.integers(a.shape[1], size=a.shape[1])
        indices = np.ix_(models, ics)
        ratios.append(float(a[indices].mean() / b[indices].mean()))
    return {
        "ratio": float(a.mean() / b.mean()),
        "ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "mean_paired_difference": float((a - b).mean()),
        "model_seed_win_count": int(np.sum(a.mean(axis=1) < b.mean(axis=1))),
        "ic_win_count": int(np.sum(a.mean(axis=0) < b.mean(axis=0))),
    }


def collect(inputs: dict) -> tuple[dict, dict, list[str]]:
    """Require the entire registered paired matrix; retain every missing cell."""
    models, ics = inputs["expected_model_seeds"], inputs["expected_ic_seeds"]
    arms = list(inputs["selected"])
    values, examples, failures = {}, {}, list(inputs.get("failures", []))
    seen: set[tuple] = set()
    dataset_hashes: set[str] = set()
    checkpoint_hashes: dict[tuple, str] = {}
    for row in inputs["test"]:
        arm, model = row["arm"], row["model_seed"]
        path = Path(row["path"])
        try:
            metrics = json.loads((path / "outcome.json").read_text())
            if not metrics.get("completed") or not metrics.get("admitted"):
                raise ValueError("incomplete or failed admission")
            dataset_hashes.add(metrics["evaluation_dataset_sha256"])
            checkpoint_key = arm, model
            previous_hash = checkpoint_hashes.setdefault(
                checkpoint_key, metrics["model_sha256"]
            )
            if previous_hash != metrics["model_sha256"]:
                raise ValueError("model checkpoint differs across IC batches")
            if (
                metrics["model_seed"] != model
                or metrics["evaluation_seeds"] != row["ic_seeds"]
            ):
                raise ValueError("evaluation identities differ from registered row")
            with np.load(path / "fields.npz", allow_pickle=False) as data:
                errors = np.asarray(data["error_corrected"], dtype=float)
                native = np.asarray(data["error_uncorrected"], dtype=float)
                if (
                    errors.shape != (len(row["ic_seeds"]), 49)
                    or native.shape != errors.shape
                ):
                    raise ValueError("unexpected error matrix shape")
                if not np.all(np.isfinite([errors, native])):
                    raise ValueError("nonfinite held-out errors")
                for index, ic in enumerate(row["ic_seeds"]):
                    key = arm, model, ic
                    if (
                        key in seen
                        or arm not in arms
                        or model not in models
                        or ic not in ics
                    ):
                        raise ValueError(f"duplicate or unregistered cell {key}")
                    seen.add(key)
                    values[key] = errors[index]
                    native_key = "native", model, ic
                    if native_key in values and not np.array_equal(
                        values[native_key], native[index]
                    ):
                        raise ValueError("native reference errors differ between arms")
                    values[native_key] = native[index]
                if model == models[0] and row["ic_seeds"][0] == ics[0]:
                    examples[arm] = {key: np.asarray(data[key]) for key in data.files}
        except (OSError, KeyError, ValueError) as error:
            failures.append(f"{arm}/model{model}/{row['ic_seeds']}: {error}")
    if len(dataset_hashes) != 1:
        failures.append(
            "evaluation dataset hashes must be identical across all arms and seeds"
        )
    training_seen = set()
    for row in inputs.get("training", []):
        key = row["arm"], row["model_seed"]
        try:
            trained = json.loads((Path(row["path"]) / "outcome.json").read_text())
            if key in training_seen:
                raise ValueError("duplicate training row")
            training_seen.add(key)
            if not trained.get("completed") or not trained.get("admitted"):
                raise ValueError("training did not pass completion/admission")
            if trained["model_sha256"] != checkpoint_hashes.get(key):
                raise ValueError(
                    "evaluation differs from registered training checkpoint"
                )
        except (OSError, KeyError, ValueError) as error:
            failures.append(f"training {key}: {error}")
    for arm in arms:
        for model in models:
            if (arm, model) not in training_seen:
                failures.append(f"missing training record for {arm}/model{model}")
    matrices = {}
    for arm in [*arms, "native"]:
        missing = [
            (model, ic)
            for model in models
            for ic in ics
            if (arm, model, ic) not in values
        ]
        if missing:
            failures.append(f"{arm}: {len(missing)} missing model/IC pairs")
        else:
            matrices[arm] = np.asarray(
                [[values[arm, model, ic] for ic in ics] for model in models]
            )
    return matrices, examples, failures


def plots(matrices: dict, examples: dict, out: Path, ics: list[int]) -> None:
    """Show all test errors and prospectively fixed full-domain fields."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for arm, errors in matrices.items():
        axes[0].plot(
            np.arange(1, 49) * 0.04,
            errors[..., 1:].mean(axis=(0, 1)) * 100,
            label=LABELS[arm],
        )
        if arm != "native":
            axes[1].scatter(
                matrices["supervised"][..., 1:].mean(axis=(0, 2)) * 100,
                errors[..., 1:].mean(axis=(0, 2)) * 100,
                label=LABELS[arm],
                alpha=0.7,
            )
    axes[0].set(
        xlabel="Physical time",
        ylabel="Mean relative velocity L2 error (%)",
        title="All 32 test flows · 8 model seeds",
    )
    axes[0].legend(fontsize=8)
    lim = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, lim], [0, lim], color="0.6", ls="--")
    axes[1].set(
        xlabel="Tuned supervision error (%)",
        ylabel="Method error (%)",
        title="Each dot is one test flow",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(out / "rollout-and-paired.png", dpi=170)
    plt.close(fig)
    first = examples["full"]
    trajectories = {
        "Fine reference (restricted)": first["reference_rollout"],
        "Solver only": first["rollout_uncorrected"],
    }
    for arm, example in examples.items():
        if not np.array_equal(example["reference_rollout"], first["reference_rollout"]):
            raise ValueError("fixed field reference differs across methods")
        trajectories[LABELS[arm]] = example["rollout_corrected"]
    fields, errors = [], []
    for frame in (24, 48):
        curls, differences = [], []
        for velocity in trajectories.values():
            v = velocity[frame].squeeze()
            curl = (
                np.roll(v[..., 1], -1, 0)
                - np.roll(v[..., 1], 1, 0)
                - np.roll(v[..., 0], -1, 1)
                + np.roll(v[..., 0], 1, 1)
            ) / (4 * np.pi / 64)
            curls.append(curl)
            differences.append(
                np.linalg.norm(
                    velocity[frame] - first["reference_rollout"][frame], axis=-1
                ).squeeze()
            )
        fields.append(curls)
        errors.append(differences)
    for name, rows, cmap, label in (
        ("fields.png", fields, "RdBu_r", "Vorticity"),
        ("field-errors.png", errors, "magma", "Velocity error magnitude"),
    ):
        maximum = float(np.max(np.abs(rows))) or 1
        fig, axes = plt.subplots(
            2,
            len(trajectories),
            figsize=(3.3 * len(trajectories), 6.8),
            layout="constrained",
        )
        for i, row in enumerate(rows):
            for j, field in enumerate(row):
                im = axes[i, j].imshow(
                    field.T,
                    origin="lower",
                    extent=(0, 2 * np.pi, 0, 2 * np.pi),
                    cmap=cmap,
                    vmin=-maximum if cmap == "RdBu_r" else 0,
                    vmax=maximum,
                )
                axes[i, j].set(xticks=[], yticks=[])
                if i == 0:
                    axes[i, j].set_title(list(trajectories)[j], fontsize=10)
            axes[i, 0].set_ylabel(f"t={(24, 48)[i] * 0.04:.2f}")
        fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.75)
        fig.suptitle(
            f"Fixed test flow {ics[0]} · model seed 8 · shared unclipped scale"
        )
        fig.savefig(out / name, dpi=170)
        plt.close(fig)


def main() -> None:
    """Publish a positive, negative, or explicitly inconclusive final report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    inputs = json.loads((args.campaign / "report-input.json").read_text())
    out = args.campaign / "report"
    out.mkdir(exist_ok=True)
    matrices, examples, failures = collect(inputs)
    report = {
        "report_complete": True,
        "failures": failures,
        "selected": inputs["selected"],
        "model_seeds": inputs["expected_model_seeds"],
        "ic_seeds": inputs["expected_ic_seeds"],
        "research_direction_closed": True,
        "scope": "Registered INS correction setting and search only",
    }
    lines = ["### Final bounded neural-corrector comparison", ""]
    if failures:
        report["conclusion"] = (
            "Inconclusive: incomplete results or failed admission; no superiority claim."
        )
        lines += [
            f"**{report['conclusion']}**",
            "",
            "Research stops here; failure is not evidence of equivalence.",
            "",
        ]
        lines.extend(f"- {failure}" for failure in failures)
    else:
        averages = {
            arm: array[..., 1:].mean(axis=-1) for arm, array in matrices.items()
        }
        comparisons = {
            arm: paired_ratio(averages["full"], averages[arm])
            for arm in averages
            if arm != "full"
        }
        report["comparisons"] = comparisons
        report["mean_errors"] = {
            arm: float(value.mean()) for arm, value in averages.items()
        }
        report["errors_by_model_ic"] = {
            arm: value.tolist() for arm, value in averages.items()
        }
        report["credible_supervised_advantage"] = bool(
            comparisons["supervised"]["ratio"] <= 0.95
            and comparisons["supervised"]["ci95"][1] < 1
        )
        matched = comparisons.get("matched_stopped", comparisons["stopped"])
        report["matched_solver_gradient_benefit"] = bool(matched["ci95"][1] < 1)
        report["conclusion"] = (
            "Solver-in-the-loop beats tuned supervision in this registered setting."
            if report["credible_supervised_advantage"]
            else "No convincing advantage over tuned supervision within the registered search; "
            "correction research closed."
        )
        lines += [
            f"**{report['conclusion']}**",
            "",
            "| Method | Mean rollout error ↓ |",
            "|---|---:|",
        ]
        lines.extend(
            f"| {LABELS[arm]} | {value * 100:.3f}% |"
            for arm, value in report["mean_errors"].items()
        )
        ratio = comparisons["supervised"]
        lines += [
            "",
            (
                f"Full/supervised error ratio: **{ratio['ratio']:.3f}**, "
                f"paired 95% interval [{ratio['ci95'][0]:.3f}, {ratio['ci95'][1]:.3f}]."
            ),
            "",
            (
                f"Matched stopped-gradient ratio: **{matched['ratio']:.3f}**, "
                f"paired 95% interval [{matched['ci95'][0]:.3f}, {matched['ci95'][1]:.3f}]."
            ),
            "",
            (
                "Selection used 16 separate validation flows and three model seeds. Confirmation uses "
                "eight fresh model seeds and 32 untouched flows. Intervals resample paired model seeds "
                "and shared flows, not time frames. The positive threshold was a ≥5% mean reduction and a "
                "95% interval excluding no improvement."
            ),
            "",
            (
                "The same INS solver supplies the 192² reference restricted to 64². The physical task, "
                "reference checks and model architecture are unchanged. Search covers learning rates, "
                "unroll lengths, update budgets and supervised warm starts; it is not an exhaustive "
                "impossibility test."
            ),
        ]
        plots(matrices, examples, out, inputs["expected_ic_seeds"])
        for name, label in (
            ("rollout-and-paired.png", "Rollout and paired results"),
            ("fields.png", "Full-domain fields"),
            ("field-errors.png", "Spatial velocity errors"),
        ):
            lines += ["", f"![{label}](ARTIFACT_URL/report/{name})"]
    costs = []
    for row in inputs.get("training", []):
        path = Path(row["path"]) / "outcome.json"
        if path.exists():
            costs.append(
                {
                    "arm": row["arm"],
                    "model_seed": row["model_seed"],
                    "metrics": json.loads(path.read_text()),
                }
            )
    report["training_cost_records"] = costs
    if not failures:
        report["mean_training_seconds_including_pretrain_and_pairs"] = {
            arm: float(
                np.mean(
                    [
                        row["metrics"]["method_training_including_pretrain_and_pairs_s"]
                        for row in costs
                        if row["arm"] == arm
                    ]
                )
            )
            for arm in inputs["selected"]
        }
        lines += [
            "",
            "Mean training cost per model (including warm starts and supervised-pair generation):",
        ]
        for arm, seconds in report[
            "mean_training_seconds_including_pretrain_and_pairs"
        ].items():
            lines += ["", f"- {LABELS[arm]}: {seconds / 60:.1f} minutes"]
        lines += [
            "",
            (
                "Shared reference generation, evaluation and hyperparameter search are separate costs; "
                "equal updates do not imply equal compute."
            ),
        ]
    lines += [
        "",
        (
            "[Frozen protocol, every candidate, failures, costs and numerical "
            "results](ARTIFACT_TREE). Earlier negative correction and neural-control results remain "
            "retained in this PR; direct control optimization is a separate non-neural result."
        ),
        "",
    ]
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "PRsection.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
