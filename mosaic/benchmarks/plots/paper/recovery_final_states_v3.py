"""Figure: recovered vs. true final flow states at steps=160 (v3, σ=0.25).

Layout: 3 rows × 6 columns
  Row 0: Ground truth final state  (each solver's own forward from ic_true)
  Row 1: Recovered final state     (forward from optimised IC)
  Row 2: Absolute point-wise error

Central z-slice of ∇·u (divergence) shown for each solver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, solver_props

RESULTS  = Path(__file__).parent.parent.parent / "results"
BASE_DIR = RESULTS / "ns-3d-grid" / "optimization" / "recovery_long_steps_v3"

_STEPS_IDX    = 5          # index of steps=160 in sweep_values=[5,10,20,40,80,160]
_DOMAIN_EXTENT = 2 * np.pi


def _vorticity_z_slice(u: np.ndarray, dx: float, z_mid: int) -> np.ndarray:
    """Return central z-slice of ω_z = du_y/dx - du_x/dy for field u of shape (N,N,N,3)."""
    omega_z = (
        np.gradient(u[..., 1], dx, axis=0)   # du_y/dx
      - np.gradient(u[..., 0], dx, axis=1)   # du_x/dy
    )
    return omega_z[:, :, z_mid]   # (N, N)


def generate(out_dir: Path) -> None:
    npz_path = BASE_DIR / "recovery_fields.npz"
    if not npz_path.exists():
        print("[recovery_final_states_v3] recovery_fields.npz not found, skipping")
        return

    f       = np.load(npz_path)
    solvers = list(f["solver_names"])
    n       = len(solvers)
    N       = f["ic_true"].shape[0]
    z_mid   = N // 2
    dx      = _DOMAIN_EXTENT / N

    # IC fields
    ic_gt   = _vorticity_z_slice(f["ic_true"], dx, z_mid)
    ic_init = _vorticity_z_slice(f["ic_init"], dx, z_mid)
    ic_err  = np.abs(ic_gt - ic_init)

    # Per-solver vorticity (ω_z) slices at steps=160
    gt_fields  = [_vorticity_z_slice(f[f"final_gt_{i}"],                    dx, z_mid) for i in range(n)]
    rec_fields = [_vorticity_z_slice(f[f"final_rec_all_{i}"][_STEPS_IDX],   dx, z_mid) for i in range(n)]
    errors     = [np.abs(g - r) for g, r in zip(gt_fields, rec_fields)]

    # Shared vorticity limits (IC + all solver panels)
    vel_all = np.concatenate(
        [ic_gt.ravel(), ic_init.ravel()]
        + [g.ravel() for g in gt_fields]
        + [r.ravel() for r in rec_fields]
    )
    vabs       = max(abs(vel_all.min()), abs(vel_all.max()))
    vmin, vmax = -vabs, vabs
    err_max    = max(ic_err.max(), max(e.max() for e in errors))

    plt.rcParams.update(RCPARAMS)

    # 3 rows × (1 IC col + gap + n solver cols) — use gridspec for spacing
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(TEXTWIDTH * 1.6, TEXTWIDTH * 0.75))
    gs  = GridSpec(
        3, n + 1,
        figure=fig,
        width_ratios=[1] + [1] * n,
        left=0.06, right=0.88, bottom=0.08, top=0.91,
        hspace=0.10, wspace=0.08,
    )
    # Extra gap between IC column and solver columns via manual rect shift
    # (achieved by wspace; the IC col will be visually separated by its title)

    row_labels = ["GT", "Init / Rec.", r"$|$Error$|$"]
    row_cmaps  = ["RdBu_r", "RdBu_r", "Reds"]
    row_vmins  = [vmin,  vmin,  0]
    row_vmaxs  = [vmax,  vmax,  err_max]

    ic_data     = [ic_gt,   ic_init, ic_err]
    solver_data = [gt_fields, rec_fields, errors]

    ims_vel = []
    ims_err = []

    def _show(ax, field, cmap, vm_lo, vm_hi):
        return ax.imshow(
            field.T, origin="lower", cmap=cmap,
            vmin=vm_lo, vmax=vm_hi,
            aspect="equal", interpolation="nearest",
        )

    def _clean(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    for row in range(3):
        cmap, vm_lo, vm_hi = row_cmaps[row], row_vmins[row], row_vmaxs[row]

        # IC column (col 0)
        ax = fig.add_subplot(gs[row, 0])
        im = _show(ax, ic_data[row], cmap, vm_lo, vm_hi)
        _clean(ax)
        if row == 0:
            ax.set_title("IC (true)", fontsize=7.5, fontweight="bold", pad=3)
        elif row == 1:
            ax.set_title("IC (init)", fontsize=7, pad=3)
        ax.set_ylabel(row_labels[row], fontsize=7, labelpad=4)
        if row < 2:
            ims_vel.append(im)
        else:
            ims_err.append(im)

        # Solver columns (cols 1..n)
        for col, solver in enumerate(solvers):
            ax = fig.add_subplot(gs[row, col + 1])
            im = _show(ax, solver_data[row][col], cmap, vm_lo, vm_hi)
            _clean(ax)
            if row == 0:
                _, color, _, _ = solver_props(solver)
                ax.set_title(solver_props(solver)[0], fontsize=7.5,
                             fontweight="bold", color=color, pad=3)
            if row < 2:
                ims_vel.append(im)
            else:
                ims_err.append(im)

    # Colorbars
    cax_vel = fig.add_axes([0.895, 0.38, 0.012, 0.52])
    cb_vel  = fig.colorbar(ims_vel[0], cax=cax_vel)
    cb_vel.set_label(r"$\omega_z$", fontsize=7, labelpad=2)
    cb_vel.ax.tick_params(labelsize=6, length=0)

    cax_err = fig.add_axes([0.895, 0.08, 0.012, 0.24])
    cb_err  = fig.colorbar(ims_err[0], cax=cax_err)
    cb_err.set_label(r"$|error|$", fontsize=7, labelpad=2)
    cb_err.ax.tick_params(labelsize=6, length=0)

    fig.suptitle(r"Vorticity $\omega_z$ — IC and final states at steps=160  (σ=0.25, central z-slice)",
                 fontsize=8.5, fontweight="bold", y=0.97)

    for ext in ("pdf", "png"):
        out = out_dir / f"recovery_final_states_v3.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
