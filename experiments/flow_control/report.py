"""Render all development gate outcomes, including the cheap control baseline."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.flow_control.plots import plot_gate


def main() -> None:
    """Read frozen cluster archives and retain every task, including failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for path in sorted((args.campaign / "results").glob("*/results.tar")):
        with tarfile.open(path) as archive:
            try:
                metrics = json.load(archive.extractfile("./outcome.json"))
                with np.load(
                    io.BytesIO(archive.extractfile("./fields.npz").read())
                ) as data:
                    arrays = {name: data[name] for name in data.files}
                plot_gate(arrays, metrics, args.out / path.parent.name)
                rows.extend(metrics["tasks"])
            except (KeyError, ValueError) as error:
                failures.append({"cell": path.parent.name, "error": repr(error)})
    rows.sort(key=lambda row: row["seed"])
    if not rows:
        raise RuntimeError("No completed task metrics to report")
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for key, label, color, marker in [
        ("fine_zero_objective", "No control", "#777777", "o"),
        ("linear_fine_objective", "Linear controller", "#009E73", "s"),
        ("fine_objective", "25 gradient updates from zero", "#0072B2", "^"),
    ]:
        ax.plot(x, [row[key] for row in rows], marker=marker, color=color, label=label)
    ax.set(
        xticks=x,
        xticklabels=[row["seed"] for row in rows],
        xlabel="Development task",
        ylabel="Terminal error + control effort (192²)",
        yscale="log",
        title="Direct control gate · no neural controller trained",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.out / "objectives.png", dpi=170)
    plt.close(fig)
    summary = {
        "tasks": rows,
        "failures": failures,
        "passed_tasks": sum(row["passed"] for row in rows),
        "linear_beats_shooting_tasks": sum(
            row["linear_fine_objective"] < row["fine_objective"] for row in rows
        ),
        "mean_objectives": {
            key: float(np.mean([row[key] for row in rows]))
            for key in [
                "fine_zero_objective",
                "linear_fine_objective",
                "fine_objective",
            ]
        },
        "scope": "developmental direct-control gate; not neural training or held-out confirmation",
    }
    (args.out / "report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
