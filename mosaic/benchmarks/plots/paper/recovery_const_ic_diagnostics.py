"""Figure: Optimisation diagnostics for recovery_constant_ic (zero-init, cold start).

5-panel, 2-row figure:
  Row 1: loss | IC error | grad norm
  Row 2: grad divergence | IC divergence
One line per solver.

Output: recovery_const_ic_diagnostics.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

RESULTS = Path(__file__).parent.parent.parent / "results"
BASE_DIR = RESULTS / "ns-3d-grid" / "optimization" / "recovery_constant_ic"

_SOLVER_ORDER = ["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict"]

_PANELS = [
    # (data_key,        x_mode,    label,                                  title)
    ("errors", "iter", r"$\mathcal{L}$", "Loss"),
    ("ic_error_history", "snap", r"IC error $\|\hat{u}_0 - u_0\|$", "IC error"),
    ("grad_norms", "iter", r"$\|\nabla\mathcal{L}\|$", "Gradient norm"),
    (
        "grad_divs",
        "iter",
        r"$\|\nabla\cdot\nabla\mathcal{L}\|_\infty$",
        "Grad divergence",
    ),
    ("ic_divs", "iter", r"$\|\nabla\cdot\hat{u}_0\|_\infty$", "IC divergence"),
]


def _resolve_json(base_dir: Path) -> Path | None:
    for name in ("result_partial.json", "result.json"):
        p = base_dir / name
        if p.exists():
            return p
    return None


def generate(out_dir: Path) -> None:
    json_path = _resolve_json(BASE_DIR)
    if json_path is None:
        print("[recovery_const_ic_diagnostics] no result JSON found, skipping")
        return

    data = json.loads(json_path.read_text())
    by_sweep = data.get("by_sweep", {})

    # Collect all sweep values present
    sweep_vals = sorted(
        {sv for sd in by_sweep.values() for sv in sd},
        key=int,
    )
    if not sweep_vals:
        print("[recovery_const_ic_diagnostics] by_sweep is empty, skipping")
        return

    # Use the first (only) sweep value
    steps = sweep_vals[0]

    plt.rcParams.update(RCPARAMS)
    fig = plt.figure(figsize=(TEXTWIDTH * 1.5, TEXTWIDTH * 0.60))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.07,
        right=0.98,
        bottom=0.20,
        top=0.87,
        hspace=0.55,
        wspace=0.42,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    # Hide unused 6th cell
    fig.add_subplot(gs[1, 2]).set_visible(False)

    seen: set[str] = set()

    for ax, (key, x_mode, ylabel, title) in zip(axes, _PANELS):
        for solver in _SOLVER_ORDER:
            entry = by_sweep.get(solver, {}).get(str(steps))
            if not entry:
                continue
            vals = entry.get(key)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)

            if x_mode == "snap":
                # ic_error_history sampled every snap_interval iterations
                n_full = len(entry.get("errors") or []) or 500
                si = max(1, n_full // len(arr))
                iters = np.arange(1, len(arr) + 1) * si
            else:
                iters = np.arange(1, len(arr) + 1)

            lbl, color, ls, mk = solver_props(solver)
            ax.plot(iters, arr, color=color, linestyle=ls, lw=1.4, label=lbl)
            seen.add(solver)

        ax.set_yscale("log")
        ax.set_xlabel("Iteration", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_title(title, fontweight="bold", fontsize=8.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"recovery_constant_ic  (steps={steps}, zero-init, rand_div_free IC)",
        fontsize=8.5,
        fontweight="bold",
        y=0.96,
    )

    handles = dedup_handles([make_handle(s) for s in _SOLVER_ORDER if s in seen])
    fig.legend(
        handles=handles,
        fontsize=6.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.4,
        labelspacing=0.25,
        columnspacing=1.0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"recovery_const_ic_diagnostics.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate(BASE_DIR)
