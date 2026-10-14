"""External plots for the isolated FD diagnostic evaluation. Run on Slurm."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=160, facecolor="white")
    plt.close(fig)


def controls(report, out):
    cases = report["analytic_controls"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    concrete = []
    for ax, case in zip(axes.flat, cases, strict=True):
        eps = [r["epsilon"] for r in case["epsilons"]]
        ax.plot(
            eps,
            [r["local_ad_fd"]["relative"] for r in case["epsilons"]],
            marker="o",
            label="Center vs FD",
            color="black",
        )
        for order in (8, 16, 32):
            ax.plot(
                eps,
                [
                    next(o for o in r["orders"] if o["order"] == order)["path_ad_fd"][
                        "relative"
                    ]
                    for r in case["epsilons"]
                ],
                marker=".",
                label=f"Average {order} vs FD",
            )
        ax.set(
            xscale="log",
            xlabel="Epsilon",
            ylabel="Relative discrepancy",
            title=case["name"].replace("_", " "),
        )
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1e-7))
        ax.grid(alpha=0.2)
        if case["true_center_derivative"] is None:
            truth = "Local derivative undefined"
        else:
            truth = (
                "Center derivative correct"
                if case["center_backward_error"]["illustrative_agreement"]
                else "Center derivative WRONG"
            )
        ax.text(
            0.03,
            0.04,
            truth,
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        for row in case["epsilons"]:
            for order in row["orders"]:
                concrete.append(
                    {
                        "case": case["name"],
                        "epsilon": row["epsilon"],
                        "order": order["order"],
                        "local_derivative_exists": case["local_derivative_exists"],
                        "true_center_derivative": case["true_center_derivative"],
                        "supplied_center_derivative": case[
                            "supplied_center_derivative"
                        ],
                        "center_correct_under_display_tolerance": None
                        if case["center_backward_error"] is None
                        else case["center_backward_error"]["illustrative_agreement"],
                        "local_vs_fd_agrees": row["local_ad_fd"][
                            "illustrative_agreement"
                        ],
                        "average_vs_fd_agrees": order["path_ad_fd"][
                            "illustrative_agreement"
                        ],
                        "sampled_backward_error_max": order[
                            "sampled_backward_error_max"
                        ],
                    }
                )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Analytic controls: local and finite-interval comparisons answer different questions\nAll declared epsilons and orders; illustrative tolerance, not a proposed CI gate; y symlog (linear below 1e-8)"
    )
    save(fig, out, "analytic_controls")
    return concrete


def native(report, out):
    cases = report.get("native_cases", [])
    if not cases:
        return []
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(13, 4.5 * len(cases)),
        squeeze=False,
        layout="constrained",
    )
    cost = []
    for i, case in enumerate(cases):
        for j, direction in enumerate(case["directions"]):
            ax = axes[i, j]
            rows = direction["epsilons"]
            eps = [r["epsilon"] for r in rows]
            ax.plot(
                eps,
                [r.get("local_ad_fd", {}).get("relative", np.nan) for r in rows],
                color="black",
                marker="o",
                label="Center vs FD",
            )
            for order in (8, 16, 32):
                values = []
                for row in rows:
                    entry = next((o for o in row["orders"] if o["order"] == order), {})
                    values.append(entry.get("path_ad_fd", {}).get("relative", np.nan))
                    if "seconds" in entry:
                        cost.append(
                            {
                                "steps": case["steps"],
                                "direction": direction["name"],
                                "epsilon": row["epsilon"],
                                "order": order,
                                "fd_seconds": row.get("fd_seconds"),
                                "path_average_seconds": entry["seconds"],
                                "path_calls": entry["api_calls"],
                                "unavailable": entry.get("unavailable"),
                            }
                        )
                ax.plot(eps, values, marker=".", label=f"Average {order} vs FD")
            ax.set(
                xscale="log",
                xlabel="Relative epsilon (absolute step = epsilon × IC RMS)",
                ylabel="Relative discrepancy",
                title=f"{case['steps']} native steps / {direction['name']}",
            )
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Unchanged main JAX-CFD: exact shared float32 objective and FD arithmetic\nNo independent derivative oracle; all orders retained; unavailable evaluations remain gaps"
    )
    save(fig, out, "native_comparison")
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(6 * len(cases), 4.8),
        squeeze=False,
        layout="constrained",
    )
    for ax, case in zip(axes.flat, cases, strict=True):
        local = [
            r["fd_seconds"]
            for d in case["directions"]
            for r in d["epsilons"]
            if "fd_seconds" in r
        ]
        groups = [local] + [
            [
                r["path_average_seconds"]
                for r in cost
                if r["steps"] == case["steps"]
                and r["order"] == o
                and not r["unavailable"]
            ]
            for o in (8, 16, 32)
        ]
        for x, group in enumerate(groups):
            ax.scatter([x] * len(group), group, alpha=0.6, s=15)
            if group:
                ax.scatter([x], [np.median(group)], color="black", marker="D", s=45)
        ax.set(
            xticks=range(4),
            xticklabels=["FD endpoints", "Average 8", "Average 16", "Average 32"],
            ylabel="Measured seconds",
            title=f"{case['steps']} native steps",
        )
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Warm evaluation costs: all measured direction/epsilon combinations\nDiamonds are medians; center gradient reusable; each average adds one objective-gradient call per node"
    )
    save(fig, out, "native_costs")
    return cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["complete"]
    args.output.mkdir(parents=True, exist_ok=True)
    table = controls(report, args.output)
    cost = native(report, args.output)
    summary = {
        "source_commit": report["source_commit"],
        "input_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analytic_comparisons": table,
        "native_cost_records": cost,
        "native_case_receipts": [
            {k: v for k, v in r.items() if k != "directions"}
            for r in report.get("native_cases", [])
        ],
        "interpretation": "Curated controls establish concrete counterexamples, not population false-pass rates. Local derivative undefined cases excluded from true/false correctness labels. Native results are consistency diagnostics, not correctness certification.",
    }
    (args.output / "analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
