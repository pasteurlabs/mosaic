"""Separate-map precision diagnostic panel; render on Slurm only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import save, sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native = json.loads(args.native.read_text())
    companion = json.loads(args.companion.read_text())
    assert native["complete"] and companion["complete"]
    row = next(
        r
        for r in native["samples"]
        if (r["frame"], r["field"], r["band"]) == (72, "scalar", "small")
    )
    assert (
        row["input_sha256"] == companion["native_input_sha256"]
        and row["probe_sha256"] == companion["probe_sha256"]
    )
    original = row["directions"]["projected_gradient"]
    paired = companion["directions"]["original_native_projected_gradient"]
    own = companion["directions"]["companion_projected_gradient"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positive_errors = []
    exact_zero_count = 0
    for record, label, color in (
        (original, "Native float32 public map, original direction", "#D55E00"),
        (paired, "Float64 companion map, same direction", "#0072B2"),
        (own, "Float64 companion map, its gradient direction", "#009E73"),
    ):
        epsilons, errors = sweep(record, "central_relative_error")
        exact_zero_count += sum(error == 0 for error in errors)
        positive_errors.extend(
            error for error in errors if math.isfinite(error) and error > 0
        )
        axes[0].plot(
            epsilons,
            [error if error > 0 else math.nan for error in errors],
            marker="o",
            ms=3,
            label=label,
            color=color,
        )
    axes[0].set(
        xscale="log",
        xlabel="FD epsilon",
        ylabel="Central-FD / AD relative discrepancy\n(each map compared with its own AD)",
    )
    if not positive_errors:
        raise ValueError(
            "No positive discrepancies available for the logarithmic panel"
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(min(positive_errors) / 1.3, max(positive_errors) * 1.3)
    if exact_zero_count:
        axes[0].text(
            0.02,
            0.02,
            f"{exact_zero_count} exact-zero discrepancies omitted on log axis; retained in JSON",
            transform=axes[0].transAxes,
            fontsize=7,
        )
    for record, prefix, color in (
        (original, "Native", "#D55E00"),
        (paired, "Companion", "#0072B2"),
    ):
        for key, style in (("right", "--"), ("left", ":")):
            axes[1].plot(
                *sweep(record, key),
                ls=style,
                color=color,
                label=f"{prefix} {key}-sided FD",
            )
        axes[1].axhline(
            record["autodiff_directional"], color=color, lw=1, label=f"{prefix} AD"
        )
    axes[1].set(
        xscale="log",
        xlabel="FD epsilon",
        ylabel="Directional slope (same original direction)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Frame 72 scalar / small: numerical-precision diagnostic\nCompanion changes interval-output and normalization precision; it is a different map, not a native gradient reference"
    )
    save(fig, args.output, "precision_companion")
    receipt = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.native, args.companion)
        },
        "map_differences": companion["changes"],
        "exact_zero_discrepancies_omitted_on_log_axis": exact_zero_count,
        "comparisons": {
            k: v for k, v in companion.items() if k.startswith("native_companion_")
        },
        "interpretation": "Compare each map with its own AD. Better FD agreement for the companion alone does not prove native AD wrong or establish a branch-causal explanation.",
    }
    (args.output / "precision_panel_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
