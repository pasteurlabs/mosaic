"""Aggregate every developmental control result without exact-pairing claims."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    "full": "Full gradients",
    "spsa": "Action SPSA",
    "demonstration_imitation": "Demonstration imitation",
    "improved_imitation": "Expert-attempt imitation",
}
COLORS = dict(zip(NAMES, ["#2166ac", "#d97706", "#278353", "#8051a5"], strict=True))


def collect(campaign: Path) -> list[dict[str, Any]]:
    """Retain missing/failed cells rather than selecting successful runs."""
    cells = []
    for config_path in sorted((campaign / "configs").glob("*.json")):
        cell = config_path.stem
        row = {
            "cell": cell,
            "config": json.loads(config_path.read_text()),
            "status": "missing_result",
        }
        archive = campaign / "results" / cell / "results.tar"
        if archive.exists():
            with tarfile.open(archive) as bundle:
                matches = [
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith("outcome.json")
                ]
                if matches:
                    metrics = json.load(bundle.extractfile(matches[0]))
                    row["metrics"] = metrics
                    row["status"] = (
                        "complete"
                        if metrics.get("completed", True)
                        and metrics.get("admitted", False)
                        else "failed_or_unadmitted"
                    )
        cells.append(row)
    return cells


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Use descriptive seed summaries; mismatched datasets disable exact pairing."""
    pilots = [
        row
        for row in cells
        if row.get("metrics", {}).get("phase") == "developmental_pilot"
    ]
    checkpoints = defaultdict(list)
    hashes = defaultdict(list)
    initializations = defaultdict(set)
    cost_rows = []
    for row in pilots:
        data = row["metrics"]
        hashes[data.get("dataset_sha256", "MISSING")].append(row["cell"])
        initializations[data["model_seed"]].add(
            data.get("initial_model_sha256", "MISSING")
        )
        phases = data.get("costs", {}).get("phases", {})
        cost_rows.append(
            {
                "cell": row["cell"],
                "method": data["method"],
                "model_seed": data["model_seed"],
                "status": row["status"],
                "query_counts_complete": data.get("query_counts_complete", False),
                "phase_seconds": {
                    phase: values["wall_time_s"] for phase, values in phases.items()
                },
                "additional_expert_labels_used": sum(
                    label["label_source"] == "shooting"
                    for label in data.get("expert_labels", [])
                ),
                "expert_labels_attempted": len(data.get("expert_labels", [])),
            }
        )
        for checkpoint in data.get("validation_checkpoints", []):
            checkpoints[data["method"], checkpoint["update"]].append(
                {
                    "cell": row["cell"],
                    "model_seed": data["model_seed"],
                    "objective": checkpoint["mean_objective"],
                    "admitted": checkpoint["admitted"],
                    "zero_objective": checkpoint["mean_zero_objective"],
                    "linear_objective": data.get("linear_validation", {}).get(
                        "mean_objective"
                    ),
                }
            )
    groups = []
    for (method, update), rows in sorted(checkpoints.items()):
        values = [row["objective"] for row in rows]
        groups.append(
            {
                "method": method,
                "update": update,
                "models": len(rows),
                "mean_objective": statistics.mean(values),
                "minimum": min(values),
                "maximum": max(values),
                "all_admitted": all(row["admitted"] for row in rows),
                "rows": rows,
            }
        )
    warm = defaultdict(list)
    for row in cells:
        data = row.get("metrics", {})
        if data.get("phase") != "linear_warm_start_diagnostic":
            continue
        physics = data["control"]
        duration = physics["dt"] * physics["steps_per_slot"] * physics["control_slots"]
        for task in data["tasks"]:
            warm[duration].append({"cell": row["cell"], **task})
    warm_groups = []
    for duration, rows in sorted(warm.items()):
        linear = statistics.mean(row["linear_fine_objective"] for row in rows)
        optimized = statistics.mean(row["fine_objective"] for row in rows)
        warm_groups.append(
            {
                "duration": duration,
                "tasks": len(rows),
                "mean_linear": linear,
                "mean_optimized": optimized,
                "relative_reduction": 1 - optimized / linear,
                "wins": sum(
                    row["fine_objective"] < row["linear_fine_objective"] for row in rows
                ),
                "mean_optimizer_seconds": statistics.mean(
                    row["shooting"]["wall_time_s"] for row in rows
                ),
                "all_admitted": all(row["admitted"] for row in rows),
                "rows": rows,
            }
        )
    return {
        "scope": (
            "Development validation only; unequal, untuned budgets. "
            "Descriptive points, no superiority or confidence-interval claim."
        ),
        "exact_paired_dataset": bool(pilots)
        and len(hashes) == 1
        and "MISSING" not in hashes,
        "dataset_hash_groups": dict(hashes),
        "initial_model_hashes_match_within_seed": all(
            len(value) == 1 and "MISSING" not in value
            for value in initializations.values()
        ),
        "checkpoints": groups,
        "costs": cost_rows,
        "warm_start": warm_groups,
        "cell_statuses": {row["cell"]: row["status"] for row in cells},
    }


def plot_summary(summary: dict[str, Any], out: Path) -> None:
    """Show every checkpoint and seed, plus the cheap controller prominently."""
    groups = sorted(
        summary["checkpoints"],
        key=lambda row: (list(NAMES).index(row["method"]), row["update"]),
    )
    if groups:
        fig, ax = plt.subplots(figsize=(13, 5.7), constrained_layout=True)
        for x, group in enumerate(groups):
            values = np.asarray([row["objective"] for row in group["rows"]])
            ax.bar(
                x, group["mean_objective"], color=COLORS[group["method"]], alpha=0.25
            )
            ax.scatter(
                x + np.linspace(-0.12, 0.12, len(values)),
                values,
                color=COLORS[group["method"]],
                s=35,
                zorder=3,
            )
        all_rows = [row for group in groups for row in group["rows"]]
        for label, key, color in [
            ("Cheap linear control", "linear_objective", "#111111"),
            ("Zero control", "zero_objective", "#777777"),
        ]:
            values = [row[key] for row in all_rows if row.get(key) is not None]
            if values:
                ax.axhline(
                    statistics.mean(values), color=color, linestyle="--", label=label
                )
        ax.set_xticks(
            range(len(groups)),
            [
                f"{NAMES[group['method']]}\n{group['update']} updates"
                for group in groups
            ],
            rotation=20,
            ha="right",
        )
        ax.set_yscale("log")
        ax.set_ylabel("192² validation objective · lower is better")
        ax.set_title("The cheap controller beats every learned-policy checkpoint")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        flag = (
            "Dataset hashes differ: these are descriptive, not exactly paired results."
            if not summary["exact_paired_dataset"]
            else "Dataset hashes match. Descriptive developmental results only."
        )
        fig.supxlabel("Dots are model seeds; bars are means. " + flag, fontsize=9)
        fig.savefig(out / "pilot_objectives.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
        for method, label in NAMES.items():
            candidates = [group for group in groups if group["method"] == method]
            if not candidates:
                continue
            last = max(candidates, key=lambda group: group["update"])
            for index, row in enumerate(last["rows"]):
                cost = next(
                    cost for cost in summary["costs"] if cost["cell"] == row["cell"]
                )
                seconds = sum(
                    cost["phase_seconds"].get(phase, 0)
                    for phase in ["policy_training", "label_generation"]
                )
                ax.scatter(
                    seconds / 60,
                    row["objective"],
                    color=COLORS[method],
                    s=65,
                    label=label if index == 0 else None,
                )
                ax.annotate(
                    str(row["model_seed"]),
                    (seconds / 60, row["objective"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        cheap = [
            row["linear_objective"]
            for group in groups
            for row in group["rows"]
            if row.get("linear_objective") is not None
        ]
        if cheap:
            ax.axhline(
                statistics.mean(cheap),
                color="black",
                linestyle="--",
                label="Cheap linear control",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Policy fitting + extra expert labels · measured minutes")
        ax.set_ylabel("Final-checkpoint validation objective")
        ax.set_title("Extra zero-start teacher optimization produced no new labels")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        fig.supxlabel(
            "Common data preparation and validation are reported separately.\n"
            "The linear controller requires a coarse rollout; its line does not imply zero inference cost.",
            fontsize=9,
        )
        fig.savefig(out / "pilot_costs.png", dpi=180)
        plt.close(fig)

    if summary["warm_start"]:
        fig, axes = plt.subplots(
            1,
            len(summary["warm_start"]),
            figsize=(12, 5),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, group in zip(axes.flat, summary["warm_start"], strict=True):
            for index, row in enumerate(
                sorted(group["rows"], key=lambda value: value["task_seed"])
            ):
                ax.plot(
                    [index, index],
                    [row["linear_fine_objective"], row["fine_objective"]],
                    color="#999999",
                    linewidth=1,
                )
                ax.scatter(
                    index,
                    row["linear_fine_objective"],
                    color="black",
                    marker="o",
                    label="Linear baseline" if index == 0 else None,
                )
                ax.scatter(
                    index,
                    row["fine_objective"],
                    color="#2166ac",
                    marker="v",
                    label="25 AD updates from linear" if index == 0 else None,
                )
            ax.set_yscale("log")
            ax.set_xlabel("Development task seed")
            ax.set_ylabel("192² objective · lower is better")
            ax.set_title(
                f"T = {group['duration']:g}: {100 * group['relative_reduction']:.1f}% lower mean objective\n"
                f"{group['wins']}/{group['tasks']} tasks improve; "
                f"{group['mean_optimizer_seconds']:.1f} s/task optimization"
            )
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=9)
        fig.supxlabel(
            "All tasks retained. Paired within each diagnostic; "
            "this is per-instance optimization, not a learned policy.",
            fontsize=9,
        )
        fig.savefig(out / "warm_start.png", dpi=180)
        plt.close(fig)


def markdown(summary: dict[str, Any]) -> str:
    """Keep the numeric table and failed/missing cells alongside the figures."""
    lines = [
        "# Developmental flow-control results",
        "",
        summary["scope"],
        "",
        (
            f"Exact dataset pairing: **{summary['exact_paired_dataset']}**. "
            "All dataset hash groups are retained in report.json."
        ),
        "",
        "## Learned-policy checkpoints",
        "",
        "| Method | Updates | Model seeds | Mean objective | Seed minimum–maximum | All admitted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for group in summary["checkpoints"]:
        lines.append(
            f"| {NAMES[group['method']]} | {group['update']} | {group['models']} | "
            f"{group['mean_objective']:.8g} | {group['minimum']:.8g}–{group['maximum']:.8g} | "
            f"{group['all_admitted']} |"
        )
    lines += [
        "",
        "![All checkpoints](pilot_objectives.png)",
        "",
        "![Measured final-checkpoint training costs](pilot_costs.png)",
        "",
        "## Per-instance warm starts",
        "",
        (
            "| Horizon | Tasks | Linear mean | Warm AD mean | Mean reduction | "
            "AD optimization seconds/task | All admitted |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for group in summary["warm_start"]:
        lines.append(
            f"| {group['duration']:g} | {group['tasks']} | {group['mean_linear']:.8g} | "
            f"{group['mean_optimized']:.8g} | {100 * group['relative_reduction']:.2f}% | "
            f"{group['mean_optimizer_seconds']:.2f} | {group['all_admitted']} |"
        )
    lines += [
        "",
        "Optimization timings exclude construction and fine-grid validation of the initial linear controller.",
        "",
        "![Warm starts](warm_start.png)",
        "",
        "## Complete cost ledger",
        "",
        (
            "| Cell | Common data s | Policy fit s | Extra labels s | Validation s | "
            "Expert labels selected/attempted | Status |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["costs"]:
        costs = row["phase_seconds"]
        lines.append(
            f"| {row['cell']} | {costs.get('common_data_preparation', 0):.1f} | "
            f"{costs.get('policy_training', 0):.1f} | {costs.get('label_generation', 0):.1f} | "
            f"{costs.get('validation', 0):.1f} | "
            f"{row['additional_expert_labels_used']}/{row['expert_labels_attempted']} | {row['status']} |"
        )
    lines += [
        "",
        "All cell statuses: " + json.dumps(summary["cell_statuses"], sort_keys=True),
        "",
        "## Fixed-example full fields",
        "",
        "![Long-horizon warm start, task 0](warm-long-0-task-0.png)",
        "",
        "![Full-gradient policy seed 0, validation task 2000](full-0-task-2000.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run only in a Slurm allocation; preserve raw outcomes in the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = collect(args.campaign)
    summary = summarize(cells)
    (args.out / "report.json").write_text(
        json.dumps({"summary": summary, "cells": cells}, indent=2)
    )
    plot_summary(summary, args.out)
    for cell, filename in [("warm-long-0", "task-0.png"), ("full-0", "task-2000.png")]:
        archive = args.campaign / "results" / cell / "results.tar"
        if archive.exists():
            with tarfile.open(archive) as bundle:
                matches = [
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith("plots/" + filename)
                ]
                if matches:
                    with (args.out / f"{cell}-{filename}").open("wb") as destination:
                        shutil.copyfileobj(bundle.extractfile(matches[0]), destination)
    (args.out / "README.md").write_text(markdown(summary))
    print(
        json.dumps(
            {key: summary[key] for key in ["exact_paired_dataset", "cell_statuses"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
