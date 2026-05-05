"""Generate Figure: Drag optimisation convergence for Re=20 and Re=100.

Per Re: 1×2 figure with drag convergence lines and inlet profiles.
Outputs: drag_opt_re20.pdf, drag_opt_re100.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

RESULTS = Path(__file__).parent.parent.parent / "results"

SOLVER_ORDER = ["xlb", "phiflow", "pict", "jax_cfd"]


def _plot_re(re_tag: str, out_dir: Path) -> None:
    base = RESULTS / "ns-grid" / "optimization" / "drag_opt" / re_tag
    result_path = base / "result.json"
    profiles_path = base / "profiles.npz"

    data = json.loads(result_path.read_text())
    profiles = np.load(profiles_path)

    fig, (ax_drag, ax_prof) = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.45))
    fig.subplots_adjust(bottom=0.30, wspace=0.35)

    present: set[str] = set()

    for solver, sdata in data["by_solver"].items():
        if solver not in SOLVER_ORDER:
            continue
        label, color, ls, mk = SOLVER_STYLES.get(
            solver, (solver, "#888888", "-", "o"))
        drags = sdata["drags"]
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]
        step = 5
        indices = list(range(0, len(drags), step))
        if (len(drags) - 1) % step != 0:
            indices.append(len(drags) - 1)
        kw = dict(color=color, linestyle=ls, marker="", linewidth=1.6, label=label)
        ax_drag.plot(indices, [drags[i] / drag_0 for i in indices], **kw)
        present.add(solver)

    ax_drag.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_drag.set_title(f"Drag reduction (Re={re_tag[2:]})")
    ax_drag.set_xlabel("Iteration")
    ax_drag.set_ylabel("drag / drag$_0$")

    # Inlet profiles
    y_arr = np.linspace(0, 1, profiles["initial"].shape[0])

    # initial profile
    ax_prof.plot(profiles["initial"], y_arr,
                 color="#999999", linestyle="--", linewidth=1.4, label="Initial")

    for solver in SOLVER_ORDER:
        final_key = f"final_{solver}"
        if final_key not in profiles:
            continue
        label, color, ls, mk = SOLVER_STYLES.get(
            solver, (solver, "#888888", "-", "o"))
        ax_prof.plot(profiles[final_key], y_arr,
                     color=color, linestyle=ls, linewidth=1.6, label=label)
        present.add(solver)

    ax_prof.set_title(f"Optimised inlet profile (Re={re_tag[2:]})")
    ax_prof.set_xlabel(r"$u_x$")
    ax_prof.set_ylabel("$y$")

    handles = [
        mlines.Line2D([], [], color="#999999", linestyle="--", linewidth=1.4,
                      label="Initial")
    ] + [
        mlines.Line2D([], [],
                      color=SOLVER_STYLES[s][1],
                      linestyle=SOLVER_STYLES[s][2],
                      linewidth=1.6,
                      label=SOLVER_STYLES[s][0])
        for s in SOLVER_ORDER if s in present
    ]

    fig.legend(handles=handles,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=4,
               fontsize=7.5,
               framealpha=0.7,
               handlelength=2.0)

    out = out_dir / f"drag_opt_{re_tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _plot_re("re20", out_dir)
        _plot_re("re100", out_dir)
