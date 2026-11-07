"""Generate Figure: F2 drag minimization over the DFG 2D-1 inflow profile.

Top left: Cd / Cd0 vs Adam iteration per solver, with a dot where the loss
levels off. Top right: initial and optimized inflow profiles. Bottom: velocity
magnitude near the cylinder before and after, for the solver with the largest
drag reduction.

Output: drag_opt_re20.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from . import TEXTWIDTH, by_solver, is_ok, load_result, results_dir
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

_EXPERIMENT = "optimization/drag_opt"
# DFG 2D-1 geometry (Schäfer & Turek 1996).
_LX, _LY = 2.2, 0.41
_CYL_CENTER, _CYL_RADIUS = (0.2, 0.2), 0.05
# The field panels show the upstream half of the channel.
_X_SHOW = _LX / 2


def _speed(field: np.ndarray) -> np.ndarray:
    """Velocity magnitude of a (nx, ny, 1, 2) field, cropped to x < _X_SHOW."""
    u = field[:, :, 0, :]
    nx_show = round(u.shape[0] * _X_SHOW / _LX)
    return np.hypot(u[:nx_show, :, 0], u[:nx_show, :, 1])


def generate(out_dir: Path) -> None:
    data = {
        s: m["None"] for s, m in by_solver(load_result("ns-grid", _EXPERIMENT)).items()
    }
    solvers = [s for s in NS_ORDER if s in data and is_ok(data[s])]
    base = results_dir() / "ns-grid" / _EXPERIMENT
    profiles = np.load(base / "profiles.npz")
    flows = np.load(base / "flow_fields.npz")
    best = max(solvers, key=lambda s: data[s]["drag_reduction_pct"])

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.62))
        outer = gridspec.GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[1.6, 1],
            left=0.09,
            right=0.99,
            top=0.95,
            bottom=0.19,
            hspace=0.55,
        )
        top = outer[0].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3)
        bottom = outer[1].subgridspec(1, 2, wspace=0.12)
        ax_drag = fig.add_subplot(top[0])
        ax_prof = fig.add_subplot(top[1])
        field_axes = [fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])]

        # ── Cd / Cd0 vs iteration ────────────────────────────────────────────
        for solver in solvers:
            _, color, ls, mk = solver_props(solver)
            drags = np.asarray(data[solver]["drags"])
            ratio = drags / drags[0]
            k = data[solver]["iters_to_plateau"]
            ax_drag.plot(np.arange(len(ratio)), ratio, color=color, linestyle=ls)
            ax_drag.plot(
                k, ratio[k], marker=mk, color=color, markeredgewidth=0, markersize=5
            )
        ax_drag.set_ylim(0.4, 1.02)
        ax_drag.set_xlim(left=0)
        ax_drag.set_xlabel("Adam iteration")
        ax_drag.set_ylabel(r"$C_D / C_{D,0}$")
        ax_drag.set_title("Drag")

        # ── Inflow profiles ──────────────────────────────────────────────────
        y = np.linspace(0, _LY, profiles["initial"].shape[0])
        ax_prof.plot(profiles["initial"], y, color="0.5", linestyle="--", linewidth=1.4)
        for solver in solvers:
            _, color, ls, _ = solver_props(solver)
            ax_prof.plot(profiles[f"final_{solver}"], y, color=color, linestyle=ls)
        ax_prof.set_ylim(0, _LY)
        ax_prof.set_yticks([0, _LY])
        ax_prof.set_xlabel(r"Inflow $u_x$")
        ax_prof.set_ylabel("$y$", labelpad=-6)
        ax_prof.set_title("Inflow profile")

        # ── Flow fields before / after ───────────────────────────────────────
        speeds = [
            _speed(flows[f"flow_initial_{best}"]),
            _speed(flows[f"flow_final_{best}"]),
        ]
        vmax = max(float(s.max()) for s in speeds)
        label = solver_props(best)[0]
        titles = [f"Initial ({label})", f"Optimized ({label})"]
        for ax, speed, title in zip(field_axes, speeds, titles, strict=True):
            im = ax.imshow(
                speed.T,
                origin="lower",
                extent=(0, _X_SHOW, 0, _LY),
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                interpolation="bilinear",
            )
            ax.add_patch(
                plt.Circle(_CYL_CENTER, _CYL_RADIUS, fc="0.85", ec="0.3", lw=0.6)
            )
            ax.grid(False)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, _LY])
            ax.set_xlabel("$x$", labelpad=1)
            ax.set_title(title, fontweight="normal")
        field_axes[0].set_ylabel("$y$", labelpad=-6)
        field_axes[1].tick_params(labelleft=False)
        cbar = fig.colorbar(im, ax=field_axes, fraction=0.02, pad=0.01)
        cbar.set_label(r"$|\mathbf{u}|$")

        # ── Legend ───────────────────────────────────────────────────────────
        initial = mlines.Line2D(
            [], [], color="0.5", linestyle="--", linewidth=1.4, label="Initial"
        )
        fig.legend(
            handles=[initial] + [make_handle(s) for s in solvers],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(solvers) + 1,
            handlelength=2.0,
        )

        out = out_dir / "drag_opt_re20.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")
