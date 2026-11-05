"""Generate Figure: H thermal-conductivity recovery with L-BFGS-B.

Top: temperature misfit and relative error of the recovered conductivity vs
iteration per solver. Bottom: ground-truth and recovered conductivity fields.

Output: conductivity_recovery_overview.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from . import TEXTWIDTH, by_solver, is_ok, load_result, results_dir
from .style import RCPARAMS, THERMAL_ORDER, make_handle, solver_props

_EXPERIMENT = "optimization/conductivity_recovery_bfgs"


def _rel_err(rho: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Relative l2 error of *rho* (..., n_cells) against *truth*."""
    return np.linalg.norm(rho - truth, axis=-1) / np.linalg.norm(truth)


def generate(out_dir: Path) -> None:
    result = load_result("thermal-mesh", _EXPERIMENT)
    phys = result["params"]["physics"]
    runs = {s: m["None"] for s, m in by_solver(result).items()}
    solvers = [s for s in THERMAL_ORDER if s in runs and is_ok(runs[s])]
    rho = np.load(results_dir() / "thermal-mesh" / _EXPERIMENT / "rho_fields.npz")
    truth = rho["rho_truth"]
    # Cells are stored with x innermost, so (ny, nx) maps rows to y.
    shape = (phys["ny"], phys["nx"])
    extent = (0, phys["Lx"], 0, phys["Ly"])

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.55))
        outer = gridspec.GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[1.6, 1],
            left=0.09,
            right=0.99,
            top=0.94,
            bottom=0.08,
            hspace=0.45,
        )
        top = outer[0].subgridspec(1, 2, wspace=0.3)
        ax_mis = fig.add_subplot(top[0])
        ax_rho = fig.add_subplot(top[1])
        maps = outer[1].subgridspec(1, len(solvers) + 1, wspace=0.08)
        map_axes = [fig.add_subplot(maps[i]) for i in range(len(solvers) + 1)]

        # ── Convergence ──────────────────────────────────────────────────────
        snap = phys["snap_interval"]
        for solver in solvers:
            _, color, ls, _ = solver_props(solver)
            kw = {"color": color, "linestyle": ls, "linewidth": 1.4}
            misfit = runs[solver]["errors"]
            ax_mis.semilogy(np.arange(len(misfit)), misfit, **kw)
            # History snapshots follow every snap_interval-th iterate.
            hist = rho[f"rho_history_{solver}"]
            xs = np.arange(len(hist) + 1) * snap
            errs = np.concatenate(
                [[_rel_err(rho["rho_init"], truth)], _rel_err(hist, truth)]
            )
            ax_rho.semilogy(xs, errs * 100, **kw)
        for ax in (ax_mis, ax_rho):
            ax.set_xlabel("L-BFGS-B iteration")
            ax.set_xlim(0, len(misfit) - 1)
        ax_mis.set_title("Temperature misfit")
        ax_rho.set_title(
            r"Conductivity error $\|\rho - \rho^\ast\| / \|\rho^\ast\|$ (%)"
        )

        # ── Fields ───────────────────────────────────────────────────────────
        fields = [("Truth", truth)] + [
            (
                f"{solver_props(s)[0]}: {_rel_err(rho[f'rho_final_{s}'], truth):.1%}",
                rho[f"rho_final_{s}"],
            )
            for s in solvers
        ]
        vmax = max(float(f.max()) for _, f in fields)
        for ax, (title, field) in zip(map_axes, fields, strict=True):
            im = ax.imshow(
                field.reshape(shape),
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
            ax.set_title(title, fontsize=7, fontweight="normal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        cbar = fig.colorbar(im, ax=map_axes, fraction=0.015, pad=0.01, shrink=0.6)
        cbar.set_label(r"$\rho$")

        fig.legend(
            handles=[make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(solvers),
            handlelength=2.0,
        )

        out = out_dir / "conductivity_recovery_overview.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
