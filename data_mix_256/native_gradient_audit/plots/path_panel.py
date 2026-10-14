"""Render all path-average orders without treating same-AD quadrature as ground truth."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["complete"]
    assert [r["epsilon"] for r in report["samples"]] == [0.001, 0.03, 0.1]
    assert all(
        [o["order"] for o in r["orders"]] == [8, 16, 32] for r in report["samples"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    summaries = []
    for column, row in enumerate(report["samples"]):
        ax = axes[0, column]
        for order in row["orders"]:
            ax.plot(
                [n["nominal_distance"] / row["epsilon"] for n in order["nodes"]],
                [n["directional_ad"] for n in order["nodes"]],
                marker="o",
                ms=3,
                label=f"Order {order['order']}",
            )
        ax.scatter(
            [0],
            [row["center_ad"]],
            color="black",
            marker="*",
            s=90,
            label="Center AD",
            zorder=5,
        )
        ax.axhline(row["central_fd"], color="gray", ls="--", label="Central FD")
        ax.set(
            title=f"Epsilon {row['epsilon']:g}",
            xlabel="Position along symmetric path / epsilon",
            ylabel="Directional AD slope",
        )
        ax = axes[1, column]
        orders = [o["order"] for o in row["orders"]]
        ax.plot(
            orders,
            [o["average_directional_ad"] for o in row["orders"]],
            marker="o",
            label="Path-average AD",
        )
        ax.axhline(row["central_fd"], color="gray", ls="--", label="Central FD")
        ax.axhline(row["center_ad"], color="black", ls=":", label="Center AD")
        ax.set(xticks=orders, xlabel="Gauss–Legendre order", ylabel="Directional slope")
        summaries.append(
            {
                "epsilon": row["epsilon"],
                "center_ad": row["center_ad"],
                "central_fd": row["central_fd"],
                "orders": [
                    {k: v for k, v in o.items() if k != "nodes"} for o in row["orders"]
                ],
            }
        )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Native frame 72 scalar / small: finite-path derivative variation\nSame-AD consistency diagnostic; float32 casts preclude an exact integral-theorem claim; no branch-causal attribution"
    )
    save(fig, args.output, "path_average")
    receipt = {
        "input_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": summaries,
        "interpretation": report["interpretation"],
        "all_orders_preserved": True,
    }
    (args.output / "path_panel_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
