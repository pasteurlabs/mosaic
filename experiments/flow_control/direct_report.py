"""CPU-only reports for registered direct-control development and confirmation."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(campaign: Path, cell: str, *, fields: bool = False) -> tuple[dict, dict]:
    """Preserve missing outputs; load numerical fields only for fixed examples."""
    try:
        with tarfile.open(campaign / "results" / cell / "results.tar") as archive:
            metrics = json.load(archive.extractfile("./outcome.json"))
            arrays = {}
            if fields:
                with np.load(
                    io.BytesIO(archive.extractfile("./fields.npz").read()),
                    allow_pickle=False,
                ) as data:
                    arrays = {key: data[key] for key in data.files}
            return metrics, arrays
    except (OSError, KeyError, ValueError) as error:
        return {"completed": False, "admitted": False, "failure": repr(error)}, {}


def field_plot(metrics: dict, arrays: dict, selected: dict, path: Path) -> None:
    """Use the prospectively fixed first task, independent of which method wins."""
    if not all(
        key in arrays for key in ["fine_initial", "fine_goal", "linear_fine_rollout"]
    ):
        return
    fields = [
        arrays["fine_initial"],
        arrays["fine_goal"],
        arrays["linear_fine_rollout"][-1],
    ]
    labels = ["Initial", "Target", "Linear control"]
    for method, setting in selected.items():
        runs = [
            row
            for row in metrics.get("settings", [])
            if row["setting_id"] == setting["setting_id"]
        ]
        if not runs:
            continue
        for index, snapshot in enumerate(runs[0]["snapshots"]):
            key = f"snapshots_{setting['setting_id']}_{index}_fine_rollout"
            if (
                snapshot["budget_kind"] == "wall_time_s"
                and snapshot["budget"] == 120
                and key in arrays
            ):
                fields.append(arrays[key][-1])
                labels.append(method.upper() + " · 120 s")
    extent = metrics["control"]["domain_extent"]
    dx = extent / fields[0].shape[0]
    velocities = [field[:, :, 0, :] for field in fields]
    curls = [
        (
            np.roll(v[..., 1], -1, 0)
            - np.roll(v[..., 1], 1, 0)
            - np.roll(v[..., 0], -1, 1)
            + np.roll(v[..., 0], 1, 1)
        )
        / (2 * dx)
        for v in velocities
    ]
    errors = [np.linalg.norm(v - velocities[1], axis=-1) for v in velocities]
    vmax = max(float(np.max(np.abs(v))) for v in curls) or 1
    emax = max(float(np.max(v)) for v in errors[2:]) or 1
    fig, axes = plt.subplots(
        2, len(fields), figsize=(3.6 * len(fields), 7), layout="constrained"
    )
    for i, (label, curl, error) in enumerate(zip(labels, curls, errors, strict=True)):
        im = axes[0, i].imshow(
            curl.T,
            origin="lower",
            extent=(0, extent, 0, extent),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[0, i].set(title=label, xlabel="x", ylabel="y")
        if i < 2:
            axes[1, i].axis("off")
        else:
            err = axes[1, i].imshow(
                error.T,
                origin="lower",
                extent=(0, extent, 0, extent),
                cmap="magma",
                vmin=0,
                vmax=emax,
            )
            axes[1, i].set(title="Velocity error", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=axes[0].tolist(), label="Vorticity", shrink=0.75)
    fig.colorbar(
        err, ax=axes[1, 2:].tolist(), label="Velocity error magnitude", shrink=0.75
    )
    fig.suptitle(
        f"Task {metrics['task_seed']} · 192² direct-control evaluation · no neural training"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    """Show every configuration/failure; require complete paired tasks for summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.campaign / "direct-plan.json").read_text())
    selected = (
        json.loads((args.campaign / "selections.json").read_text()).get("selected", {})
        if (args.campaign / "selections.json").exists()
        else {}
    )
    stages = {}
    for stage, seeds, settings in [
        ("development", plan["development_task_seeds"], plan["settings"]),
        ("test", plan["test_task_seeds"], list(selected.values())),
    ]:
        prefix = "dev" if stage == "development" else "test"
        results = [read(args.campaign, f"{prefix}-{seed}")[0] for seed in seeds]
        if not any(row.get("settings") for row in results):
            continue
        groups = []
        for setting in settings:
            for budget in [30, 60, 120]:
                rows = []
                for seed, result in zip(seeds, results, strict=True):
                    run = next(
                        (
                            r
                            for r in result.get("settings", [])
                            if r["setting_id"] == setting["setting_id"]
                        ),
                        {},
                    )
                    snapshot = next(
                        (
                            r
                            for r in run.get("snapshots", [])
                            if r["budget_kind"] == "wall_time_s"
                            and r["budget"] == budget
                        ),
                        {},
                    )
                    valid = bool(
                        result.get("common_admitted")
                        and not run.get("failure")
                        and snapshot.get("available")
                        and snapshot.get("admitted")
                    )
                    rows.append(
                        {
                            "task_seed": seed,
                            "admitted": valid,
                            "objective": snapshot.get("fine_objective"),
                            "linear_objective": result.get("linear_baseline", {}).get(
                                "fine_objective"
                            ),
                            "optimization_accounting": run.get("accounting", {}),
                            "failure": run.get("failure", result.get("failure")),
                        }
                    )
                complete = all(row["admitted"] for row in rows)
                groups.append(
                    {
                        "setting_id": setting["setting_id"],
                        "method": setting["method"],
                        "budget_s": budget,
                        "all_admitted": complete,
                        "failed_or_missing_tasks": sum(
                            not row["admitted"] for row in rows
                        ),
                        "mean_objective": float(
                            np.mean([row["objective"] for row in rows])
                        )
                        if complete
                        else None,
                        "rows": rows,
                    }
                )
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        for setting in settings:
            points = [
                r
                for r in groups
                if r["setting_id"] == setting["setting_id"] and r["all_admitted"]
            ]
            if points:
                ax.plot(
                    [r["budget_s"] for r in points],
                    [r["mean_objective"] for r in points],
                    marker="o",
                    label=setting["setting_id"],
                )
        linear = [r.get("linear_baseline", {}).get("fine_objective") for r in results]
        if all(v is not None for v in linear):
            ax.axhline(
                float(np.mean(linear)), color="black", ls="--", label="Linear control"
            )
        zero = [r.get("zero_baseline", {}).get("fine_objective") for r in results]
        if all(v is not None for v in zero):
            ax.axhline(float(np.mean(zero)), color="gray", ls=":", label="No control")
        ax.set(
            xlabel="Charged optimization budget (seconds)",
            ylabel="Mean 192² objective",
            yscale="log",
            title=f"{stage.title()} · objective at strict completed-query deadlines",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.savefig(args.out / f"{stage}-time.png", dpi=170)
        plt.close(fig)
        query_groups = []
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        for setting in settings:
            points = []
            for budget in [32, 64, 128]:
                values, reverse_calls = [], []
                for result in results:
                    run = next(
                        (
                            r
                            for r in result.get("settings", [])
                            if r["setting_id"] == setting["setting_id"]
                        ),
                        {},
                    )
                    snapshot = next(
                        (
                            r
                            for r in run.get("snapshots", [])
                            if r["budget_kind"] == "queries" and r["budget"] == budget
                        ),
                        {},
                    )
                    if (
                        result.get("common_admitted")
                        and not run.get("failure")
                        and snapshot.get("available")
                        and snapshot.get("admitted")
                    ):
                        values.append(snapshot["fine_objective"])
                        reverse_calls.append(
                            sum(
                                e["vjp_count"]
                                for e in run["query_events"]
                                if e["query"] <= budget
                            )
                        )
                complete = len(values) == len(seeds)
                query_groups.append(
                    {
                        "setting_id": setting["setting_id"],
                        "forward_queries": budget,
                        "failed_or_unavailable_tasks": len(seeds) - len(values),
                        "mean_objective": float(np.mean(values)) if complete else None,
                        "mean_additional_vjp_calls": float(np.mean(reverse_calls))
                        if complete
                        else None,
                    }
                )
                if complete:
                    points.append((budget, float(np.mean(values))))
            if points:
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    label=setting["setting_id"],
                )
        ax.set(
            xlabel="Forward objective queries (AD additionally uses VJPs)",
            ylabel="Mean 192² objective",
            yscale="log",
            title=f"{stage.title()} · query frontiers within the 120-second run",
        )
        if ax.lines:
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(alpha=0.2)
        fig.savefig(args.out / f"{stage}-queries.png", dpi=170)
        plt.close(fig)
        paired = []
        if stage == "test":
            by_method = {
                g["method"]: g
                for g in groups
                if g["budget_s"] == 120 and g["all_admitted"]
            }
            for baseline in ["spsa", "powell"]:
                if "adam" not in by_method or baseline not in by_method:
                    continue
                a = np.asarray([r["objective"] for r in by_method["adam"]["rows"]])
                b = np.asarray([r["objective"] for r in by_method[baseline]["rows"]])
                difference = a - b
                rng = np.random.default_rng(116)
                boot = difference[rng.integers(0, len(a), size=(5000, len(a)))].mean(
                    axis=1
                )
                paired.append(
                    {
                        "comparison": f"adam minus {baseline}",
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(np.median(difference)),
                        "adam_wins": int((a < b).sum()),
                        "tasks": len(a),
                        "paired_task_bootstrap95_mean_difference": np.quantile(
                            boot, [0.025, 0.975]
                        ).tolist(),
                        "note": (
                            "Task-level bootstrap; one fixed SPSA seed per task, "
                            "no optimizer-seed variability estimate."
                        ),
                    }
                )
                fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
                ax.axhline(0, color="black", lw=1)
                ax.bar(
                    seeds,
                    difference,
                    color=np.where(difference < 0, "#0072b2", "#d55e00"),
                )
                ax.set(
                    xlabel="Untouched task seed",
                    ylabel=f"AD objective minus {baseline}",
                    title="120-second paired differences · negative favors AD",
                )
                fig.savefig(args.out / f"test-adam-vs-{baseline}.png", dpi=170)
                plt.close(fig)
        stages[stage] = {
            "groups": groups,
            "query_groups": query_groups,
            "paired": paired,
            "task_seeds": seeds,
            "raw_metrics": results,
        }
        examples = [3000] if stage == "development" else [5000, 5001]
        for seed in examples:
            metrics, arrays = read(args.campaign, f"{prefix}-{seed}", fields=True)
            field_plot(metrics, arrays, selected, args.out / f"{stage}-task-{seed}.png")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Direct per-instance optimization. No neural training claim. "
                    "All configured tasks and failures retained."
                ),
                "stages": stages,
            },
            indent=2,
        )
    )
    lines = [
        "# Direct-control comparison",
        "",
        (
            "Budget:120seconds; checkpoints30/60/120. Fine evaluation and audits occur "
            "after optimization. Forward and VJP counts are separate; see raw metrics."
        ),
        "",
    ]
    for stage, data in stages.items():
        lines.extend(
            [
                f"## {stage.title()}",
                "",
                "|Setting|Seconds|Mean objective|Failed/missing tasks|",
                "|---|---:|---:|---:|",
            ]
        )
        for g in data["groups"]:
            lines.append(
                f"|{g['setting_id']}|{g['budget_s']}|{g['mean_objective']}|{g['failed_or_missing_tasks']}|"
            )
        lines.extend(["", f"![Objective versus time]({stage}-time.png)", ""])
    (args.out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
