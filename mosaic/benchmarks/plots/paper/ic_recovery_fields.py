"""IC recovery field visualization: best-vs-worst solver + horizon degradation.

Panel A (top 2 rows × 4 cols): field comparison at representative step count (40)
  Row 0 — Exponax (best):  true IC | perturbed | recovered | error
  Row 1 — XLB     (worst): true IC | perturbed | recovered | error

Panel B (bottom row × 6 cols): Exponax recovered IC across all step counts
  Columns: steps = 5, 10, 20, 40, 80, 160

Fields shown as z-midplane vorticity (ω_z = ∂u_y/∂x − ∂u_x/∂y, spectral).
Error panels show pointwise vorticity difference (rec − true).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, solver_props

RESULTS = Path(__file__).parent.parent.parent / "results"
_NPZ  = RESULTS / "ns-3d-grid" / "optimization" / "recovery_long_steps" / "recovery_fields.npz"
_JSON = RESULTS / "ns-3d-grid" / "optimization" / "recovery_long_steps" / "result.json"

_BEST  = "exponax"
_WORST = "xlb"


def _vorticity_z(field: np.ndarray) -> np.ndarray:
    """Spectral ω_z at z-midplane.  field: (N,N,N,3) → (N,N)"""
    N = field.shape[0]
    sl = field[N // 2]          # (N, N, 3)  — x,y,z velocity components
    ux = sl[..., 0].astype(np.float64)
    uy = sl[..., 1].astype(np.float64)
    kn = np.fft.fftfreq(N) * N  # wavenumber grid (units: 1/cell)
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    wy = np.fft.ifftn(1j * KX * np.fft.fftn(uy)).real   # ∂u_y/∂x
    wx = np.fft.ifftn(1j * KY * np.fft.fftn(ux)).real   # ∂u_x/∂y
    return (wy - wx).astype(np.float32)


def _ic_err_str(by_sweep: dict, solver: str, steps: int) -> str:
    v = by_sweep.get(solver, {}).get(str(steps), {})
    e = v.get("final_ic_error") if isinstance(v, dict) else None
    return f"err={e:.3f}" if e is not None and np.isfinite(e) else "err=NaN"


def generate(out_dir: Path) -> None:
    if not _NPZ.exists():
        print(f"[ic_recovery_fields] missing {_NPZ}, skipping")
        return

    data     = np.load(_NPZ)
    by_sweep = json.loads(_JSON.read_text())["by_sweep"]

    solvers     = list(data["solver_names"])
    sweep_vals  = [int(v) for v in data["sweep_values"]]   # [5,10,20,40,80,160]
    rep_idx     = sweep_vals.index(40)

    best_i  = solvers.index(_BEST)
    worst_i = solvers.index(_WORST)

    ic_true = data["ic_true"]
    ic_init = data["ic_init"]

    vort_true = _vorticity_z(ic_true)
    vort_init = _vorticity_z(ic_init)
    vlim = float(np.abs(vort_true).max()) * 1.05

    # ── layout ────────────────────────────────────────────────────────────────
    plt.rcParams.update(RCPARAMS)

    fig_w = TEXTWIDTH
    fig_h = fig_w * 1.35
    fig = plt.figure(figsize=(fig_w, fig_h))

    # outer: row 0 = Part A (2 solver rows), row 1 = Part B (horizon)
    outer = mgridspec.GridSpec(
        2, 1,
        height_ratios=[2.2, 1.0],
        hspace=0.55,
        top=0.95, bottom=0.04, left=0.08, right=0.95,
    )

    # Part A: 2 rows × 5 cols (4 panels + narrow colorbar)
    gs_a = mgridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=outer[0],
        wspace=0.06, hspace=0.45,
        width_ratios=[1, 1, 1, 1, 0.07],
    )

    # Part B: 1 row × 7 cols (6 panels + narrow colorbar)
    gs_b = mgridspec.GridSpecFromSubplotSpec(
        1, 7, subplot_spec=outer[1],
        wspace=0.06,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.07],
    )

    col_titles_a = ["True IC", "Perturbed", "Recovered\n(steps=40)", "Error\n(rec − true)"]

    def _panel(ax, arr, cmap, vmin, vmax, title=None, ylabel=None, titlecolor="black"):
        im = ax.imshow(arr, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        if title:
            ax.set_title(title, fontsize=7, pad=2, color=titlecolor)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7.5, labelpad=4)
        return im

    # ── Part A ────────────────────────────────────────────────────────────────
    for row, (si, sname) in enumerate([(best_i, _BEST), (worst_i, _WORST)]):
        label, color, _, _ = solver_props(sname)
        quality = "best" if row == 0 else "worst"
        row_label = f"{label}\n({quality})"

        rec   = data[f"ic_rec_all_{si}"][rep_idx]      # (N,N,N,3)
        vort_rec = _vorticity_z(rec)
        vort_err = vort_rec - vort_true                # signed difference

        err_lim = float(np.abs(vort_err).max()) * 1.05

        ax0 = fig.add_subplot(gs_a[row, 0])
        ax1 = fig.add_subplot(gs_a[row, 1])
        ax2 = fig.add_subplot(gs_a[row, 2])
        ax3 = fig.add_subplot(gs_a[row, 3])
        ax_cb = fig.add_subplot(gs_a[row, 4])

        err_str = _ic_err_str(by_sweep, sname, 40)
        titles  = [col_titles_a[0], col_titles_a[1],
                   f"Recovered\n(steps=40, {err_str})",
                   "Error (rec−true)"]

        _panel(ax0, vort_true, "RdBu_r", -vlim, vlim,
               title=titles[0], ylabel=row_label)
        _panel(ax1, vort_init, "RdBu_r", -vlim, vlim, title=titles[1])
        im_rec = _panel(ax2, vort_rec, "RdBu_r", -vlim, vlim, title=titles[2])
        _panel(ax3, vort_err, "RdBu_r", -err_lim, err_lim, title=titles[3])

        cb = fig.colorbar(im_rec, cax=ax_cb)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label("ω_z", fontsize=6, labelpad=1)
        cb.set_ticks([round(-vlim * 0.8, 1), 0, round(vlim * 0.8, 1)])

    # ── Part B ────────────────────────────────────────────────────────────────
    best_label, best_color, _, _ = solver_props(_BEST)
    ic_rec_all = data[f"ic_rec_all_{best_i}"]   # (6, N,N,N,3)

    last_im = None
    for si, steps in enumerate(sweep_vals):
        ax = fig.add_subplot(gs_b[0, si])
        vort_rec = _vorticity_z(ic_rec_all[si])
        err_str  = _ic_err_str(by_sweep, _BEST, steps)
        title    = f"steps={steps}\n{err_str}"
        ylabel   = f"{best_label}\n(horizon)" if si == 0 else None
        last_im  = _panel(ax, vort_rec, "RdBu_r", -vlim, vlim,
                          title=title, ylabel=ylabel)

    ax_cb_b = fig.add_subplot(gs_b[0, 6])
    cb_b = fig.colorbar(last_im, cax=ax_cb_b)
    cb_b.ax.tick_params(labelsize=5.5)
    cb_b.set_label("ω_z", fontsize=6, labelpad=1)
    cb_b.set_ticks([round(-vlim * 0.8, 1), 0, round(vlim * 0.8, 1)])

    # ── section labels ────────────────────────────────────────────────────────
    pos_a = outer[0].get_position(fig)
    pos_b = outer[1].get_position(fig)
    fig.text(0.01, pos_a.y1 + 0.005, "(a) Solver comparison — steps=40",
             fontsize=8, fontweight="bold", va="bottom")
    fig.text(0.01, pos_b.y1 + 0.005, "(b) Observability horizon — Exponax",
             fontsize=8, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        out = out_dir / f"ic_recovery_fields.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
