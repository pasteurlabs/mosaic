# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Figure: F2 cylinder-flow forward accuracy vs viscosity + flow fields.

Left: consensus error vs ν for each valid solver.
Right: 2×2 vorticity fields at ν=0.01 for phiflow, openfoam, pict, xlb.

Output: appendix_cylinder.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

# ν index to show for field panels (index 2 = ν=0.01)
_NU_IDX = 2
# Solvers to show as field panels, in display order (2×2)
_FIELD_SOLVERS = ["phiflow", "openfoam", "pict", "xlb"]


def _vorticity(field: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Compute vorticity ω = dv/dx − du/dy from velocity field (nx,ny,1,2)."""
    u = field[:, :, 0, 0]
    v = field[:, :, 0, 1]
    nx, ny = u.shape
    dv_dx = np.gradient(v, L / nx, axis=0)
    du_dy = np.gradient(u, L / ny, axis=1)
    return dv_dx - du_dy


def generate(out_dir: Path) -> None:
    """Generate cylinder-flow forward accuracy figure."""
    _base = results_dir() / "ns-grid" / "forward" / "cylinder"
    _path = _base / "result.json"
    _fields = _base / "fields.npz"

    if not _path.exists():
        print(f"[cylinder] {_path} not found — skipping")
        return

    with plt.rc_context(RCPARAMS):
        data = json.loads(_path.read_text())
        by_param = data["by_param"]
        params = sorted(by_param.keys(), key=float)
        nu_vals = [float(p) for p in params]

        fields_data = np.load(_fields, allow_pickle=True) if _fields.exists() else None

        # Layout: left column = line plot, right 2×2 = vorticity fields
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.50))
        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[1.7, 1, 1],
            height_ratios=[1, 1],
            left=0.10,
            right=0.98,
            top=0.90,
            bottom=0.22,
            hspace=0.05,
            wspace=0.10,
        )

        ax_line = fig.add_subplot(gs[:, 0])  # spans both rows
        field_axes = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]

        # ── Error line plot ──────────────────────────────────────────────────
        seen: set[str] = set()
        for solver in NS_ORDER:
            _, color, ls, mk = solver_props(solver)
            xs, ys = [], []
            for p in params:
                entry = by_param[p].get(solver)
                if isinstance(entry, dict):
                    err = entry.get("error")
                    if isinstance(err, float) and np.isfinite(err) and err > 0:
                        xs.append(float(p))
                        ys.append(err)
            if xs:
                ax_line.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=4,
                    markeredgewidth=0,
                    linewidth=1.6,
                )
                seen.add(solver)

        ax_line.set_xscale("log")
        ax_line.set_xlabel(r"$\nu$")
        ax_line.set_ylabel("Consensus error")
        ax_line.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_line.xaxis.set_major_locator(mticker.FixedLocator(nu_vals))
        ax_line.set_xlim(min(nu_vals) * 0.65, max(nu_vals) * 1.5)
        ax_line.tick_params(axis="x", labelsize=7.5, rotation=30)
        ax_line.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

        # legend inside the line plot (upper left, away from x-axis ticks)
        handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
        ax_line.legend(
            handles=handles,
            loc="upper left",
            ncol=1,
            fontsize=7.0,
            framealpha=0.9,
            edgecolor="0.8",
            handlelength=2.0,
        )

        # ── Vorticity field panels ───────────────────────────────────────────
        nu_show = (
            fields_data["sweep_values"][_NU_IDX] if fields_data is not None else None
        )
        nu_label = f"$\\nu$ = {nu_show:.3g}" if nu_show is not None else ""

        for ax, solver in zip(field_axes, _FIELD_SOLVERS, strict=False):
            label, color, _, _ = solver_props(solver)
            if fields_data is not None:
                key = f"{solver}_{_NU_IDX}"
                if key in fields_data:
                    omega = _vorticity(fields_data[key])
                    vmax = float(np.abs(omega).max()) or 1.0
                    ax.imshow(
                        omega.T,
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=-vmax,
                        vmax=vmax,
                        aspect="equal",
                        interpolation="bilinear",
                    )
            # label inside the panel, top-left corner
            ax.text(
                0.04,
                0.96,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color=color,
                bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1.0},
            )
            ax.axis("off")

        # shared title above the field panel block, placed relative to fig.transFigure
        fig.text(
            0.735,
            0.93,
            f"Final vorticity ({nu_label})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        out = out_dir / "appendix_cylinder.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
