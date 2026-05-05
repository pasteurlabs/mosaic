"""Figure: Jacobian properties vs. IC recovery error (3D NS).

Two-panel scatter — every point is one (solver, rollout-steps, seed):

  Left:  log10(κ) [condition number from jacobian_svd] vs. IC error
  Right: gradient norm [from horizon_sweep]           vs. IC error

Marker size encodes rollout steps (larger = more steps).
No connecting lines — raw scatter only.
"""

from __future__ import annotations

import json
import re
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, solver_props, make_handle, dedup_handles

RESULTS  = Path(__file__).parent.parent.parent / "results"
SVD_BASE = RESULTS / "ns-3d-grid" / "gradient"
LOG_PATH = Path("/tmp/ns3d_recovery_v2.log")

_SVD_EXPS  = {10: "jacobian_svd", 20: "jacobian_svd_steps20", 40: "jacobian_svd_steps40"}
_HS_PATH   = SVD_BASE / "horizon_sweep" / "result.json"

_STEPS_VALS = [10, 20, 40, 80, 160]
_STEP_SIZES = {10: 20, 20: 35, 40: 55, 80: 75, 160: 100}


def _load_recovery_seeds(log_path: Path) -> dict[tuple[str, int], list[float]]:
    rec: dict[tuple, list] = collections.defaultdict(list)
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r"(\w+) steps=(\d+).*ic_err=([0-9.e+\-]+)", line)
            if m and "done" in line:
                rec[(m.group(1), int(m.group(2)))].append(float(m.group(3)))
    return dict(rec)


def generate(out_dir: Path) -> None:
    rec = _load_recovery_seeds(LOG_PATH)
    if not rec:
        print("[jacobian_recovery_correlation] recovery log not found, skipping")
        return

    # ── condition number data (steps 10, 20, 40) ─────────────────────────────
    kappa: dict[tuple[str, int], float] = {}
    for steps, exp in _SVD_EXPS.items():
        p = SVD_BASE / exp / "result.json"
        if not p.exists():
            continue
        conds = json.loads(p.read_text())["per_solver_cond"]
        for solver, k in conds.items():
            kappa[(solver, steps)] = float(k)

    # ── gradient norm data (steps 10–160) ────────────────────────────────────
    grad_norm: dict[tuple[str, int], float] = {}
    if _HS_PATH.exists():
        hs = json.loads(_HS_PATH.read_text())["by_solver"]
        for solver, sd in hs.items():
            for t, v in sd.items():
                g = v.get("grad_norm")
                if g and not (g != g):   # skip NaN
                    grad_norm[(solver, int(t))] = float(g)

    plt.rcParams.update(RCPARAMS)
    fig, (ax_k, ax_g) = plt.subplots(
        1, 2,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.52),
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.93, wspace=0.38)

    solver_order = ["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict"]
    seen: set[str] = set()

    for solver in solver_order:
        label, color, ls, mk = solver_props(solver)

        # ── left panel: κ ────────────────────────────────────────────────────
        for steps in [10, 20, 40]:
            k = kappa.get((solver, steps))
            seeds = rec.get((solver, steps), [])
            if k is None or not seeds:
                continue
            ax_k.scatter(
                [np.log10(k)] * len(seeds), seeds,
                color=color, marker=mk,
                s=_STEP_SIZES[steps], alpha=0.75,
                linewidths=0, zorder=3,
            )
            seen.add(solver)

        # ── right panel: grad norm ────────────────────────────────────────────
        for steps in _STEPS_VALS:
            g = grad_norm.get((solver, steps))
            seeds = rec.get((solver, steps), [])
            if g is None or not seeds:
                continue
            ax_g.scatter(
                [g] * len(seeds), seeds,
                color=color, marker=mk,
                s=_STEP_SIZES[steps], alpha=0.75,
                linewidths=0, zorder=3,
            )
            seen.add(solver)

    # ── axes labels / limits ─────────────────────────────────────────────────
    ax_k.set_xlabel(r"$\log_{10}(\kappa)$", fontsize=8)
    ax_k.set_ylabel("IC recovery error", fontsize=8)
    ax_k.set_title("Condition number vs. recovery error")
    ax_k.set_xlim(-0.3, 13)
    ax_k.set_ylim(-0.005, 0.098)

    ax_g.set_xlabel("Gradient norm", fontsize=8)
    ax_g.set_title("Gradient norm vs. recovery error")
    ax_g.set_xlim(40, 270)
    ax_g.set_ylim(-0.005, 0.098)
    ax_g.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax_g.set_yticklabels([])   # shared y-axis visually

    for ax in (ax_k, ax_g):
        ax.tick_params(labelsize=7)

    # ── legends ──────────────────────────────────────────────────────────────
    solver_handles = dedup_handles([
        make_handle(s) for s in solver_order if s in seen
    ])
    step_handles = [
        mlines.Line2D([], [], marker="o", color="0.4", linestyle="none",
                      markersize=np.sqrt(_STEP_SIZES[s]), alpha=0.75,
                      label=f"steps={s}", markeredgewidth=0)
        for s in _STEPS_VALS
    ]

    leg1 = ax_k.legend(handles=solver_handles, fontsize=6.5, loc="upper left",
                       framealpha=0.8, handlelength=1.5,
                       borderpad=0.4, labelspacing=0.25)
    ax_k.add_artist(leg1)

    ax_g.legend(handles=step_handles, fontsize=6.5, loc="upper right",
                framealpha=0.8, handlelength=0.8,
                borderpad=0.4, labelspacing=0.25)

    for ext in ("pdf", "png"):
        out = out_dir / f"jacobian_recovery_correlation.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
