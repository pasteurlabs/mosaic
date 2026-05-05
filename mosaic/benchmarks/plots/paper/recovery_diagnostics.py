"""Figure: Optimisation diagnostics for IC recovery (v3, σ=0.25).

Three-panel figure per representative step value:
  Row 1: gradient norm vs iteration
  Row 2: gradient divergence vs iteration
  Row 3: IC divergence vs iteration
One line per solver.  Separate figure for each step value.

Output: recovery_diagnostics_steps{N}.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, solver_props, make_handle, dedup_handles

RESULTS  = Path(__file__).parent.parent.parent / "results"
BASE_DIR = RESULTS / "ns-3d-grid" / "optimization" / "recovery_long_steps_v3"

_SOLVER_ORDER = ["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict"]
_DIAG_KEYS    = ["errors", "grad_norms", "grad_divs", "ic_divs"]
_DIAG_LABELS  = [
    r"$\mathcal{L}$",
    r"$\|\nabla\mathcal{L}\|$",
    r"$\|\nabla\cdot\nabla\mathcal{L}\|_\infty \;/\; \|\nabla\mathcal{L}\|$",
    r"$\|\nabla\cdot u_0\|_\infty$",
]
_DIAG_TITLES  = ["Loss", "Gradient norm", "Gradient divergence (normalised)", "IC divergence"]


def _resolve_json(base_dir: Path) -> Path | None:
    for name in ("result.json", "result_partial.json"):
        p = base_dir / name
        if p.exists():
            return p
    return None


def _plot_step(by_sweep: dict, steps: str, out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        1, 4,
        figsize=(TEXTWIDTH * 1.45, TEXTWIDTH * 0.38),
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.82, wspace=0.38)

    seen: set[str] = set()
    for ax, key, label, title in zip(axes, _DIAG_KEYS, _DIAG_LABELS, _DIAG_TITLES):
        for solver in _SOLVER_ORDER:
            entry = by_sweep.get(solver, {}).get(steps)
            if not entry:
                continue
            vals = entry.get(key)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            if key == "grad_divs":
                norms = np.array(entry.get("grad_norms") or [], dtype=float)
                if len(norms) == len(arr):
                    arr = arr / np.where(norms > 0, norms, np.nan)
            lbl, color, ls, mk = solver_props(solver)
            iters = np.arange(1, len(arr) + 1)
            ax.plot(iters, arr, color=color, linestyle=ls, lw=1.4)
            seen.add(solver)

        ax.set_yscale("log")
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Recovery diagnostics  (steps={steps}, σ=0.25)", fontsize=9, fontweight="bold", y=0.96)

    handles = dedup_handles([make_handle(s) for s in _SOLVER_ORDER if s in seen])
    fig.legend(handles=handles, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.53, 0.01),
               ncol=len(handles), framealpha=0.8, handlelength=2.0,
               borderpad=0.4, labelspacing=0.25, columnspacing=1.0)

    for ext in ("pdf", "png"):
        out = out_dir / f"recovery_diagnostics_steps{steps}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def generate(out_dir: Path) -> None:
    json_path = _resolve_json(BASE_DIR)
    if json_path is None:
        print("[recovery_diagnostics] no result JSON found, skipping")
        return

    data     = json.loads(json_path.read_text())
    by_sweep = data["by_sweep"]
    steps_vals = sorted(
        {s for solver_data in by_sweep.values() for s in solver_data},
        key=int,
    )

    with plt.rc_context(RCPARAMS):
        for steps in steps_vals:
            _plot_step(by_sweep, steps, out_dir)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
