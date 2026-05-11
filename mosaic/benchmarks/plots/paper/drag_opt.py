"""Generate Figure: Drag optimisation at Re=20.

Layout: 3-column figure
  col 0  — drag reduction (%) vs iteration  [spans full height]
  col 1  — final + initial inlet profiles   [spans full height]
  col 2  — profile evolution imshow         [3 rows, one per solver]

Output: drag_opt_re20.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES, solver_props

SOLVER_ORDER = ["xlb", "phiflow", "pict"]


def _plot_re(re_tag: str, out_dir: Path) -> None:
    base = results_dir() / "ns-grid" / "optimization" / "drag_opt" / re_tag
    result_path = base / "result.json"
    profiles_path = base / "profiles.npz"

    if not result_path.exists():
        print(f"[drag_opt] {result_path} not found — skipping")
        return

    data = json.loads(result_path.read_text())
    profiles = np.load(profiles_path) if profiles_path.exists() else None

    hist_solvers = [
        s
        for s in SOLVER_ORDER
        if profiles is not None and f"profile_history_{s}" in profiles
    ]
    n_rows = max(len(hist_solvers), 1)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * (0.14 + 0.13 * n_rows)))
    gs = gridspec.GridSpec(
        n_rows,
        3,
        figure=fig,
        width_ratios=[1.4, 0.9, 1.1],
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.22,
        hspace=0.12,
        wspace=0.45,
    )

    ax_drag = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[:, 1])
    imshow_axes = [fig.add_subplot(gs[r, 2]) for r in range(n_rows)]

    present: set[str] = set()

    # ── Drag reduction panel ─────────────────────────────────────────────────
    for solver in SOLVER_ORDER:
        sdata = data["by_solver"].get(solver)
        if sdata is None:
            continue
        drags = sdata["drags"]
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]

        step = max(1, len(drags) // 50)
        indices = list(range(0, len(drags), step))
        if indices[-1] != len(drags) - 1:
            indices.append(len(drags) - 1)

        reductions = [(drag_0 - drags[i]) / drag_0 * 100 for i in indices]

        label, color, ls, _ = solver_props(solver)
        ax_drag.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        present.add(solver)

    ax_drag.set_title("Drag reduction")
    ax_drag.set_xlabel("Iteration")
    ax_drag.set_ylabel("Drag reduction (%)")
    ax_drag.set_ylim(bottom=0)

    # ── Final inlet profiles panel ───────────────────────────────────────────
    if profiles is not None:
        y_arr = np.linspace(0, 1, profiles["initial"].shape[0])
        ax_prof.plot(
            profiles["initial"], y_arr, color="#999999", linestyle="--", linewidth=1.4
        )
        for solver in SOLVER_ORDER:
            if f"final_{solver}" not in profiles:
                continue
            label, color, ls, _ = solver_props(solver)
            ax_prof.plot(
                profiles[f"final_{solver}"],
                y_arr,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            present.add(solver)

    ax_prof.set_title("Optimised profile")
    ax_prof.set_xlabel(r"$u_x$")
    ax_prof.set_ylabel("$y$")

    # ── Profile history imshow panels ─────────────────────────────────────────
    for idx, (ax_im, solver) in enumerate(zip(imshow_axes, hist_solvers, strict=False)):
        hist = profiles[f"profile_history_{solver}"]  # (n_snaps, ny)
        label, color, _, _ = solver_props(solver)

        ax_im.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )

        n_snaps = hist.shape[0]
        n_iters = len(data["by_solver"].get(solver, {}).get("drags", [1]))
        snap_step = n_iters / max(n_snaps - 1, 1)
        tick_pos = [0, n_snaps // 2, n_snaps - 1]
        ax_im.set_xticks(tick_pos)
        ax_im.set_xticklabels([f"{int(t * snap_step)}" for t in tick_pos], fontsize=6.5)
        ax_im.tick_params(labelsize=6.5)
        ax_im.set_yticks([])

        ax_im.text(
            0.03,
            0.95,
            label,
            transform=ax_im.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=color,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0},
        )

        if idx == 0:
            ax_im.set_title("Profile history")
        if idx < len(hist_solvers) - 1:
            ax_im.tick_params(labelbottom=False)
        else:
            ax_im.set_xlabel("Iteration", fontsize=7.0)

    # hide unused imshow rows
    for ax_im in imshow_axes[len(hist_solvers) :]:
        ax_im.set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D(
            [], [], color="#999999", linestyle="--", linewidth=1.4, label="Initial"
        )
    ] + [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_STYLES[s][1],
            linestyle=SOLVER_STYLES[s][2],
            linewidth=1.6,
            label=SOLVER_STYLES[s][0],
        )
        for s in SOLVER_ORDER
        if s in present and s in SOLVER_STYLES
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(handles), 5),
        fontsize=7.5,
        framealpha=0.7,
        edgecolor="0.8",
        handlelength=2.0,
    )

    out = out_dir / f"drag_opt_{re_tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _plot_re("re20", out_dir)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
