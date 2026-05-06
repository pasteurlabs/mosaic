"""Generate Figure: FD U-curves (relative error vs ε) for F2 and F3 NS domains.

For each rollout length (horizon), plots relative FD error vs perturbation size ε
across solvers.  Produces two PDFs: one for F2 (ns-grid) and one for F3 (ns-3d-grid).

Output: appendix_ucurves_f2.pdf, appendix_ucurves_f3.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

RESULTS = Path(__file__).parent.parent.parent / "results"

_CONFIGS = {
    "f2": {
        "path": RESULTS / "ns-grid" / "gradient" / "horizon_sweep" / "result.json",
        "out": "appendix_ucurves_f2.pdf",
        "title": "F2 — FD U-curves (2D NS)",
        "ncols": 4,
    },
    "f3": {
        "path": RESULTS / "ns-3d-grid" / "gradient" / "horizon_sweep" / "result.json",
        "out": "appendix_ucurves_f3.pdf",
        "title": "F3 — FD U-curves (3D NS)",
        "ncols": 5,
    },
}


def _plot_domain(cfg: dict, out_dir: Path) -> None:
    path: Path = cfg["path"]
    if not path.exists():
        print(f"[ucurves] {path} not found — skipping")
        return

    data = json.loads(path.read_text())
    by_solver: dict = data["by_solver"]

    # Collect all step values across solvers, sorted numerically
    all_steps: list[int] = sorted(
        {int(s) for sv in by_solver.values() for s in sv},
        key=int,
    )
    ncols: int = cfg["ncols"]
    nrows: int = int(np.ceil(len(all_steps) / ncols))

    panel_w = TEXTWIDTH / ncols
    panel_h = panel_w * 0.92
    fig_h = nrows * panel_h + 0.55  # extra for legend

    fig = plt.figure(figsize=(TEXTWIDTH, fig_h))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        left=0.10,
        right=0.98,
        top=1.0 - 0.12 / fig_h,
        bottom=0.52 / fig_h,
        hspace=0.65,
        wspace=0.40,
    )

    seen: set[str] = set()

    for idx, steps in enumerate(all_steps):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        for solver in NS_ORDER:
            sv = by_solver.get(solver)
            if sv is None:
                continue
            entry = sv.get(str(steps))
            if entry is None:
                continue
            eps_sweep: dict = entry.get("eps_sweep", {})
            if not eps_sweep:
                continue

            eps_vals = sorted(eps_sweep.keys(), key=float)
            xs = [float(e) for e in eps_vals]
            ys = [eps_sweep[e]["rel_error_mean"] for e in eps_vals]

            if not all(np.isfinite(y) and y > 0 for y in ys):
                # drop non-finite points
                pairs = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y) and y > 0]
                if not pairs:
                    continue
                xs, ys = zip(*pairs)

            _, color, ls, mk = solver_props(solver)
            ax.loglog(
                xs,
                ys,
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=3.5,
                markeredgewidth=0,
                linewidth=1.4,
            )
            seen.add(solver)

        ax.set_title(f"$T={steps}$", fontsize=8)
        ax.set_xlabel(r"$\varepsilon$", fontsize=7.5)
        if col == 0:
            ax.set_ylabel("Rel. FD error", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    # hide unused panels
    for idx in range(len(all_steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 6),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.0,
    )

    out = out_dir / cfg["out"]
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        for cfg in _CONFIGS.values():
            _plot_domain(cfg, out_dir)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
