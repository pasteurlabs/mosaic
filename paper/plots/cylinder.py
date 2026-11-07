"""Generate Figure: F2 DFG 2D-1 cylinder benchmark (Re = 20, steady).

Left: signed drag-coefficient error against the literature value vs cells per
diameter, one line per solver. Right: steady velocity magnitude of the most
accurate solver at the finest resolution.

Output: appendix_cylinder.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from . import TEXTWIDTH, by_solver, is_ok, load_result, results_dir
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

_EXPERIMENT = "forward/cylinder"
# DFG 2D-1 geometry (Schäfer & Turek 1996).
_LX, _LY = 2.2, 0.41
_CYL_CENTER, _CYL_RADIUS = (0.2, 0.2), 0.05


def generate(out_dir: Path) -> None:
    result = load_result("ns-grid", _EXPERIMENT)
    cd_ref = result["extras"]["reference"]["cd"]
    data = by_solver(result)
    resolutions = sorted({int(v) for runs in data.values() for v in runs})
    solvers = [
        s for s in NS_ORDER if s in data and all(is_ok(m) for m in data[s].values())
    ]
    fields = np.load(results_dir() / "ns-grid" / _EXPERIMENT / "fields.npz")

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.36))
        gs = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1, 1.75],
            left=0.09,
            right=0.99,
            top=0.90,
            bottom=0.34,
            wspace=0.18,
        )
        ax = fig.add_subplot(gs[0])
        ax_field = fig.add_subplot(gs[1])

        # ── Signed Cd error vs resolution ────────────────────────────────────
        ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
        for solver in solvers:
            _, color, ls, mk = solver_props(solver)
            errs = [
                (data[solver][str(r)]["cd"] - cd_ref) / cd_ref * 100
                for r in resolutions
            ]
            ax.plot(
                resolutions,
                errs,
                color=color,
                linestyle=ls,
                marker=mk,
                markeredgewidth=0,
            )
        ax.set_xticks(resolutions)
        ax.set_xlim(resolutions[0] - 2, resolutions[-1] + 2)
        ax.set_xlabel("Cells per diameter")
        ax.set_ylabel(r"$(C_D - C_D^{\mathrm{ref}})\,/\,C_D^{\mathrm{ref}}$ (%)")
        ax.set_title(f"Drag error ($C_D^{{\\mathrm{{ref}}}} = {cd_ref}$)")
        fig.legend(
            handles=[make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(solvers),
            handlelength=2.0,
        )

        # ── Steady velocity magnitude of the most accurate solver ───────────
        finest = str(resolutions[-1])
        best = min(solvers, key=lambda s: abs(data[s][finest]["cd"] - cd_ref))
        u = fields[f"{best}_{resolutions.index(resolutions[-1])}"][:, :, 0, :]
        speed = np.hypot(u[..., 0], u[..., 1])
        im = ax_field.imshow(
            speed.T,
            origin="lower",
            extent=(0, _LX, 0, _LY),
            cmap="viridis",
            interpolation="bilinear",
        )
        ax_field.add_patch(
            plt.Circle(_CYL_CENTER, _CYL_RADIUS, fc="0.85", ec="0.3", lw=0.6)
        )
        ax_field.set_xticks([0, _LX / 2, _LX])
        ax_field.set_yticks([0, _LY])
        ax_field.set_xlabel("$x$")
        ax_field.set_ylabel("$y$", labelpad=-6)
        ax_field.grid(False)
        ax_field.set_title(
            f"{solver_props(best)[0]}, {finest} cells per diameter: $|\\mathbf{{u}}|$"
        )
        fig.colorbar(im, ax=ax_field, fraction=0.03, pad=0.02, shrink=0.35)

        out = out_dir / "appendix_cylinder.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
