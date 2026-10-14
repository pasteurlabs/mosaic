"""Plot fixed native-gradient audit. Run numerical analysis/rendering on Slurm only."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, folder, name):
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=160, facecolor="white")
    plt.close(fig)


def sweep(record, key):
    return [r["epsilon"] for r in record["epsilon_sweep"]], [
        r.get(key, np.nan) for r in record["epsilon_sweep"]
    ]


def render(report, output):
    samples = report["samples"]
    for direction in ("projected_gradient", "fixed_random"):
        fig, axes = plt.subplots(
            2, 3, figsize=(14, 8), sharex=True, layout="constrained"
        )
        for row in samples:
            ax = axes[
                ("scalar", "velocity").index(row["field"]),
                ("large", "medium", "small").index(row["band"]),
            ]
            ax.plot(
                *sweep(row["directions"][direction], "central_relative_error"),
                marker="o",
                ms=3,
                label=f"Frame {row['frame']}",
            )
            ax.set(
                title=f"{row['field']} / {row['band']}",
                xscale="log",
                yscale="symlog",
                xlabel="FD epsilon",
                ylabel="Relative central-FD vs AD error",
            )
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.grid(alpha=0.2)
        for ax in axes.flat:
            if ax.lines:
                ax.legend(fontsize=7)
        fig.suptitle(
            f"Unchanged native forward map: {direction.replace('_', ' ')}\nAll declared epsilons retained; y is symlog (linear below 1e-5); gaps denote unavailable evaluations"
        )
        save(fig, output, f"state_fd_{direction}")
    worst = next(
        (
            r
            for r in samples
            if (r["frame"], r["field"], r["band"]) == (72, "scalar", "small")
        ),
        None,
    )
    if worst:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for column, direction in enumerate(("projected_gradient", "fixed_random")):
            record = worst["directions"][direction]
            for key, label, style in [
                ("central", "Central FD", "-"),
                ("right", "Right-sided FD", "--"),
                ("left", "Left-sided FD", ":"),
            ]:
                axes[0, column].plot(
                    *sweep(record, key), marker="o", ms=3, ls=style, label=label
                )
            axes[0, column].axhline(
                record["autodiff_directional"], color="black", lw=1, label="Native AD"
            )
            axes[0, column].set(
                xscale="log",
                xlabel="FD epsilon",
                ylabel="Directional slope",
                title=direction.replace("_", " "),
            )
            axes[0, column].legend(fontsize=8)
            for key, label in [
                ("realized_plus_direction_norm", "Realized + step norm / epsilon"),
                ("realized_minus_direction_norm", "Realized − step norm / epsilon"),
            ]:
                axes[1, column].plot(*sweep(record, key), marker="o", ms=3, label=label)
            axes[1, column].axhline(
                1, color="black", lw=1, ls="--", label="Requested unit direction"
            )
            axes[1, column].set(
                xscale="log",
                xlabel="FD epsilon",
                ylabel="Realized input-direction norm",
            )
            axes[1, column].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        fig.suptitle(
            "Prespecified worst historical probe: frame 72, scalar / small\nOne-sided asymmetry alone does not prove nonsmoothness; public states and outputs remain float32"
        )
        save(fig, output, "worst_historical_probe")
    neural = report.get("neural_audit", {})
    if neural.get("available"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for row in neural["rows"]:
            for ax, direction in zip(
                axes, ("native_gradient", "fixed_random"), strict=True
            ):
                ax.plot(
                    *sweep(row["directions"][direction], "central_relative_error"),
                    marker="o",
                    ms=3,
                    label=f"Update {row['update']}, case {row['case']}, start {row['start']}",
                )
                ax.set(
                    xscale="log",
                    xlabel="FD epsilon",
                    ylabel="Relative central-FD vs AD error",
                    title=direction.replace("_", " "),
                )
                ax.set_yscale("symlog", linthresh=1e-5)
                ax.grid(alpha=0.2)
        axes[-1].legend(fontsize=6)
        fig.suptitle(
            "Actual neural window-loss directional derivatives\nAll windows and checkpoints; y symlog (linear below 1e-5)"
        )
        save(fig, output, "neural_loss_fd")


def summarize(report):
    samples = report["samples"]
    if not report.get("complete"):
        raise ValueError("Audit is incomplete; refusing a final plot/report")
    if not all(r["forward_bitwise_equal"] for r in samples):
        raise ValueError("Native/wrapper forward mismatch")
    records = []
    for row in samples:
        for direction, record in row["directions"].items():
            records.append(
                {
                    "identity": {k: row[k] for k in ("frame", "field", "band")},
                    "direction": direction,
                    "autodiff_directional": record["autodiff_directional"],
                    "same_ad_consistency": record["jvp_vjp_same_ad_check"],
                    "epsilon_sweep": record["epsilon_sweep"],
                }
            )
    return {
        "complete": True,
        "sample_count": len(samples),
        "all_native_wrapper_forwards_bitwise_equal": True,
        "max_native_wrapper_state_vjp_relative_l2": max(
            r["native_wrapper_vjp_relative_l2"] for r in samples
        ),
        "records": records,
        "neural_audit": report.get("neural_audit"),
        "interpretation_limits": [
            "JVP and VJP use the same AD system: agreement is not an independent gradient reference.",
            "Central-FD error depends on finite-step truncation, branch changes, and public float32 rounding.",
            "One-sided asymmetry can arise from curvature or branch crossings; it alone cannot establish a derivative defect.",
            "No epsilon is selected to declare correctness; complete fixed ladders and unavailable evaluations are retained.",
            "This audit does not alter the forward map or replace gradients with a surrogate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text())
    summary = summarize(report)
    summary["input_sha256"] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    render(report, args.output)


if __name__ == "__main__":
    main()
