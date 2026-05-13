"""Drag-optimisation single-experiment + paper-figure generator.

Two public entry points:

  * :func:`plot_experiment(cfg, *, exp_key, suffix, save)` — the canonical
    single-experiment 3-column figure (drag reduction, optimised inlet
    profile, profile history) produced in paper styling. Reads
    ``<results>/<cfg.name>/optimization/<exp_key><suffix>/result.json``
    and writes a paper-quality PDF in the same experiment directory.
    Used both as the per-experiment plot delegate (called from
    :func:`mosaic.benchmarks.problems.navier_stokes_grid.plots.plot_drag_opt`)
    and as the source figure for the paper-output pipeline.
  * :func:`generate(out_dir)` — thin wrapper that resolves the ns-grid
    cfg and dispatches to ``plot_experiment``, then copies the PDF to
    *out_dir* under the canonical paper filename for the build pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
)
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    dedup_handles,
    make_handle,
    solver_props,
)

# Solvers shown in the drag_opt panel, in display order.
_SOLVER_ORDER = ["xlb", "phiflow", "pict"]


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "drag_opt",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure | None:
    """Single-experiment paper-styled drag-optimisation figure.

    Layout (3-column GridSpec):
      * col 0 — drag reduction (%) vs iteration
      * col 1 — final + initial inlet profiles
      * col 2 — profile-history imshow, one row per solver

    Reads ``result.json`` (+ optional ``profiles.npz``) from the
    experiment directory and writes ``<exp_key>.pdf`` next to them when
    ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    profiles_path = out_dir / "profiles.npz"

    if not result_path.exists():
        print(f"[drag_opt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)

    data = load_json(result_path)
    profiles = try_load_npz(profiles_path) if profiles_path.exists() else None

    hist_solvers = [
        s
        for s in _SOLVER_ORDER
        if profiles is not None and f"profile_history_{s}" in profiles
    ]
    n_rows = max(len(hist_solvers), 1)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * (0.14 + 0.13 * n_rows)), dpi=300)
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
    for solver in _SOLVER_ORDER:
        sdata = data["by_solver"].get(solver)
        if sdata is None:
            continue
        drags = sdata.get("drags", [])
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]

        step = max(1, len(drags) // 50)
        indices = list(range(0, len(drags), step))
        if indices[-1] != len(drags) - 1:
            indices.append(len(drags) - 1)

        reductions = [(drag_0 - drags[i]) / drag_0 * 100 for i in indices]

        _label, color, ls, _mk = solver_props(solver)
        ax_drag.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        present.add(solver)

    ax_drag.set_title("Drag reduction")
    ax_drag.set_xlabel("Iteration")
    ax_drag.set_ylabel("Drag reduction (%)")
    ax_drag.set_ylim(bottom=0)

    # ── Final inlet profiles panel ───────────────────────────────────────────
    if profiles is not None and "initial" in profiles:
        y_arr = np.linspace(0, 1, profiles["initial"].shape[0])
        ax_prof.plot(
            profiles["initial"], y_arr, color="#999999", linestyle="--", linewidth=1.4
        )
        for solver in _SOLVER_ORDER:
            if f"final_{solver}" not in profiles:
                continue
            _label, color, ls, _mk = solver_props(solver)
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

    # ── Profile history imshow panels ────────────────────────────────────────
    for idx, (ax_im, solver) in enumerate(zip(imshow_axes, hist_solvers, strict=False)):
        hist = profiles[f"profile_history_{solver}"]  # (n_snaps, ny)
        label, color, _ls, _mk = solver_props(solver)

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

    # Hide unused imshow rows
    for ax_im in imshow_axes[len(hist_solvers) :]:
        ax_im.set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D(
            [], [], color="#999999", linestyle="--", linewidth=1.4, label="Initial"
        ),
        *dedup_handles(
            [make_handle(s) for s in NS_ORDER if s in present and s in SOLVER_STYLES]
        ),
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

    if save:
        out = out_dir / f"{exp_key}.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def generate(out_dir: Path) -> None:
    """Paper-output entry point: ns-grid drag_opt figure → ``drag_opt_re20.pdf``."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    with plt.rc_context(RCPARAMS):
        fig = plot_experiment(cfg, exp_key="drag_opt", suffix="", save=False)
        if fig is None:
            return
        out = out_dir / "drag_opt_re20.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
