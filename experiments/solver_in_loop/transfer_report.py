"""Strict, solver-specific frozen-recipe transfer report; run in a CPU allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.solver_in_loop import final_report as shared


def validate_transfer(inputs: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Check transfer identities directly; no fictitious solver-specific selection."""
    failures = []
    base = plan["base_payload"]
    for phase in ("training", "test"):
        for row in inputs.get(phase, []):
            try:
                result = json.loads((Path(row["path"]) / "outcome.json").read_text())
                identity = result["identity"]
                for key in ("solver", "source_sha256", "image", "image_sha256"):
                    if identity[key] != base[key]:
                        raise ValueError(f"{key} differs from frozen transfer")
                if identity["physics"] != base["run"]["physics"]:
                    raise ValueError("physics differs from frozen INS recipe")
                if (
                    result["arm"] != row["arm"]
                    or result["model_seed"] != row["model_seed"]
                ):
                    raise ValueError("arm/model identity mismatch")
                if phase == "training":
                    recipe = plan["confirmation_recipes"][row["arm"]]
                    for setting, key in (
                        ("lr", "lr"),
                        ("unroll", "unroll"),
                        ("max_updates", "updates"),
                    ):
                        if result["training"][setting] != recipe[key]:
                            raise ValueError("training recipe changed")
                    if result.get("optimizer_updates") != recipe["updates"]:
                        raise ValueError("fixed optimizer budget not completed")
                    if not result.get("evaluation_deferred") or result.get(
                        "evaluation_seeds"
                    ):
                        raise ValueError(
                            "confirmation inputs were evaluated during fitting"
                        )
            except (OSError, KeyError, ValueError) as error:
                failures.append(f"{phase}/{row['arm']}/{row['model_seed']}: {error}")
    return failures


def main() -> None:
    """Report complete matrices or explicit admission/infrastructure failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    plan = json.loads((campaign / "plan.json").read_text())
    inputs = json.loads((campaign / "report-input.json").read_text())
    inputs.pop("base_payload", None)
    inputs["selected"] = plan["confirmation_recipes"]
    inputs.setdefault("failures", []).extend(validate_transfer(inputs, plan))
    matrices, examples, failures = shared.collect(inputs)
    out = campaign / "report"
    out.mkdir(exist_ok=True)
    solver = plan["base_payload"]["solver"]
    result = {
        "report_complete": True,
        "solver": solver,
        "scope": "Frozen INS recipe transfer; no transfer-specific model tuning",
        "resources": plan["resources"],
        "recipes": plan["confirmation_recipes"],
        "failures": failures,
        "model_seeds": inputs["expected_model_seeds"],
        "ic_seeds": inputs["expected_ic_seeds"],
        "source_sha256": plan["source_sha256"],
        "image_sha256": plan["base_payload"]["image_sha256"],
        "reference_grid": 192,
        "coarse_grid": 64,
        "reference_temporal_factor": plan["base_payload"]["run"]["dataset"][
            "reference_temporal_factor"
        ],
        "audit_temporal_factor": plan["base_payload"]["run"]["dataset"][
            "reference_audit_temporal_factor"
        ],
        "reference_tolerance": 0.005,
    }
    lines = [
        f"### Frozen corrector transfer: {solver}",
        "",
        (
            "The INS-selected model settings are reused unchanged. All three methods use B200 GPUs, "
            "the same 64² solver and solver-specific 192² references. Temporal refinement is solver-specific "
            "to meet the common 0.5% reference-admission criterion. Every training and test IC must pass admission."
        ),
        "",
    ]
    if failures:
        result["conclusion"] = (
            "Inconclusive: incomplete or inadmissible transfer; no superiority claim."
        )
        lines += [result["conclusion"], "", *[f"- {failure}" for failure in failures]]
    else:
        averages = {
            arm: array[..., 1:].mean(axis=-1) for arm, array in matrices.items()
        }
        comparisons = {
            arm: shared.paired_ratio(averages["full"], averages[arm])
            for arm in ("supervised", "stopped", "native")
        }
        result.update(
            mean_errors={arm: float(value.mean()) for arm, value in averages.items()},
            comparisons=comparisons,
            errors_by_model_ic={arm: value.tolist() for arm, value in averages.items()},
        )
        result["joint_transfer_benefit"] = bool(
            comparisons["supervised"]["ratio"] <= 0.95
            and comparisons["supervised"]["ci95"][1] < 1
            and comparisons["stopped"]["ci95"][1] < 1
        )
        result["conclusion"] = (
            "Both supervision and matched stopped-gradient comparisons favor full solver gradients."
            if result["joint_transfer_benefit"]
            else "The registered joint solver-gradient advantage did not replicate."
        )
        lines += [
            result["conclusion"],
            "",
            "| Method | Mean held-out relative L2 error |",
            "|---|---:|",
        ]
        for arm, value in result["mean_errors"].items():
            lines.append(f"| {arm} | {value:.6f} |")
        lines += [
            "",
            "Paired model/IC bootstrap ratios below 1 favor full gradients:",
            "",
        ]
        for arm, comparison in comparisons.items():
            lo, hi = comparison["ci95"]
            lines.append(
                f"- Full/{arm}: {comparison['ratio']:.4f} (95% CI {lo:.4f}–{hi:.4f})."
            )
        costs = {arm: 0.0 for arm in plan["confirmation_recipes"]}
        for row in inputs["training"]:
            trained = json.loads((Path(row["path"]) / "outcome.json").read_text())
            costs[row["arm"]] += trained["training_wall_time_s"]
        dataset = json.loads(
            (campaign / "results" / "assembled" / "dataset.json").read_text()
        )
        pair_cost = dataset["supervised_dataset_wall_time_s"]
        common_cost = dataset["dataset_preparation_wall_time_s"] - pair_cost
        result["costs"] = {
            "training_wall_s_all_eight_models": costs,
            "supervised_pair_generation_wall_s": pair_cost,
            "common_reference_preparation_wall_s": common_cost,
            "rule": (
                "Standalone method cost adds common reference cost; supervision also adds pair cost. "
                "Training includes all continuation allocations. Fine evaluation separate."
            ),
        }
        shared.LABELS.update(
            stopped="Stopped gradients (INS recipe)",
            supervised="Supervision (INS recipe)",
        )
        shared.plots(matrices, examples, out, inputs["expected_ic_seeds"])
        fig, ax = shared.plt.subplots(figsize=(8, 4.5), layout="constrained")
        arms = list(costs)
        ax.bar(
            arms,
            [costs[arm] / 8 for arm in arms],
            label="Model fitting including continuation",
        )
        ax.bar(
            arms,
            [pair_cost / 8 if arm == "supervised" else 0 for arm in arms],
            bottom=[costs[arm] / 8 for arm in arms],
            label="Amortized supervised pairs",
        )
        ax.set(
            ylabel="Seconds per model (eight models)",
            title=f"{solver}: all arms on B200",
        )
        ax.legend()
        fig.savefig(out / "training-cost.png", dpi=170)
        shared.plt.close(fig)
        lines += [
            "",
            "![Paired rollout errors](ARTIFACT_URL/report/rollout-and-paired.png)",
            "![Full vorticity fields](ARTIFACT_URL/report/fields.png)",
            "![Full velocity errors](ARTIFACT_URL/report/field-errors.png)",
            "![Same-hardware training cost](ARTIFACT_URL/report/training-cost.png)",
            "",
            (
                "These are within-solver, same-hardware comparisons. They do not establish equal DNS accuracy "
                "across solvers "
                "or compare B200 runtimes with the earlier RTX5090 INS result."
            ),
        ]
    (out / "report.json").write_text(
        json.dumps(shared.json_safe(result), indent=2, allow_nan=False)
    )
    (out / "PRsection.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
