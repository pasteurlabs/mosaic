"""Figure: IC recovery error and final loss vs. rollout horizon (3D NS).

Two-panel figure:
  Left:  IC reconstruction error vs. rollout steps (linear y)
  Right: Final optimisation loss vs. rollout steps (log y)

One line per solver, shaded band = min/max across seeds.
Data from result.json (reconstructed from /tmp/ns3d_recovery_v2.log).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, solver_props, make_handle, dedup_handles

RESULTS   = Path(__file__).parent.parent.parent / "results"
_BASE_DIR = RESULTS / "ns-3d-grid" / "optimization" / "recovery_long_steps_v2"

STEPS_VALS    = [5, 10, 20, 40, 80, 160]
_SOLVER_ORDER = ["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict"]


def _resolve_json(base_dir: Path) -> Path | None:
    for name in ("result.json", "result_partial.json"):
        p = base_dir / name
        if p.exists():
            return p
    return None


def _load(json_path: Path):
    """Return (ic_err, final_loss) dicts: {(solver, steps): [values]}."""
    if not json_path.exists():
        return {}, {}
    data = json.loads(json_path.read_text())["by_sweep"]
    ic_err, loss = {}, {}
    for solver, by_steps in data.items():
        for steps_str, v in by_steps.items():
            key = (solver, int(steps_str))
            ic_err[key] = v.get("final_ic_error_trials", [v["final_ic_error"]])
            loss[key]   = v.get("final_loss_trials",     [v["final_loss"]])
    return ic_err, loss


def _plot_panel(ax, data, solver_order, steps_vals, seen, log_y=False):
    for solver in solver_order:
        label, color, ls, mk = solver_props(solver)
        xs, means = [], []
        for steps in steps_vals:
            vals = data.get((solver, steps), [])
            if not vals:
                continue
            xs.append(steps)
            means.append(float(np.median(vals)))
        if not xs:
            continue
        ax.plot(xs, means, color=color, linestyle=ls, marker=mk,
                markersize=4, markeredgewidth=0, linewidth=1.6)
        seen.add(solver)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(steps_vals)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Rollout steps", fontsize=8)
    ax.tick_params(labelsize=7)


def generate(out_dir: Path) -> None:
    json_path = _resolve_json(_BASE_DIR)
    if json_path is None:
        print("[recovery_long_steps] no result.json or result_partial.json found, skipping")
        return
    ic_err, loss = _load(json_path)
    if not ic_err:
        print("[recovery_long_steps] no data loaded, skipping")
        return

    plt.rcParams.update(RCPARAMS)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.52),
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.91, wspace=0.38)

    seen: set[str] = set()
    _plot_panel(ax_l, ic_err, _SOLVER_ORDER, STEPS_VALS, seen, log_y=False)
    _plot_panel(ax_r, loss,   _SOLVER_ORDER, STEPS_VALS, seen, log_y=True)

    ax_l.set_ylabel("IC recovery error", fontsize=8)
    ax_l.set_title("IC recovery error", fontweight="bold")
    ax_l.set_ylim(bottom=-0.003)

    ax_r.set_ylabel("Final optimisation loss", fontsize=8)
    ax_r.set_title("Final optimisation loss", fontweight="bold")

    handles = dedup_handles([make_handle(s) for s in _SOLVER_ORDER if s in seen])
    fig.legend(handles=handles, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.54, 0.01),
               ncol=3, framealpha=0.8, handlelength=2.0,
               borderpad=0.4, labelspacing=0.25, columnspacing=1.0)

    for ext in ("pdf", "png"):
        out = out_dir / f"recovery_long_steps.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
