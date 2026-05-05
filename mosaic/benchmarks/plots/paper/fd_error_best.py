"""Generate Figure: Best (minimum) FD relative error per solver across all domains.

For each solver the minimum relative error over the ε sweep is shown, i.e. the
error at the optimal perturbation size.  This summarises the V-shaped sweep
curve from fd_check into a single per-solver scalar.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import (
    RCPARAMS, SOLVER_STYLES, NS_ORDER, STRUCTURAL_ORDER, THERMAL_ORDER,
    solver_props,
)

RESULTS = Path(__file__).parent.parent.parent / "results"

DOMAINS = [
    ("2D NS",      "ns-grid",         NS_ORDER),
    ("3D NS",      "ns-3d-grid",      NS_ORDER),
    ("Structural", "structural-mesh", STRUCTURAL_ORDER),
    ("Thermal",    "thermal-mesh",    THERMAL_ORDER),
]


def _best_err(eps_sweep: dict) -> float:
    best = None
    for vals in eps_sweep.values():
        err = vals.get("rel_error")
        mean_err = float(np.mean(err)) if isinstance(err, list) else float(err)
        if best is None or mean_err < best:
            best = mean_err
    return best if best is not None else float("nan")


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42))
    fig.subplots_adjust(wspace=0.12, left=0.02, right=0.98)

    for ax, (domain_label, subdir, order) in zip(axes, DOMAINS):
        path = RESULTS / subdir / "gradient" / "fd_check" / "result.json"
        data = json.loads(path.read_text())
        by_solver = data.get("by_solver", {})

        # Collect (label, color, err) in canonical order, bottom-to-top
        rows: list[tuple[str, str, float]] = []
        for key in order:
            if key not in by_solver:
                continue
            eps_sweep = by_solver[key].get("eps_sweep", {})
            err = _best_err(eps_sweep)
            label, color, *_ = solver_props(key)
            rows.append((label, color, err))

        if not rows:
            ax.set_visible(False)
            continue

        labels  = [r[0] for r in rows]
        colors  = [r[1] for r in rows]
        errors  = [r[2] for r in rows]
        y       = np.arange(len(rows))

        ax.barh(y, errors, color=colors, height=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(domain_label)
        ax.set_xlabel("Min. relative FD error", fontsize=7)
        ax.invert_yaxis()  # best (lowest error) at top
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.88", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Hide y-tick labels on all panels except the first
        if ax is not axes[0]:
            ax.set_yticklabels([])

    out = out_dir / "fd_error_best.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
