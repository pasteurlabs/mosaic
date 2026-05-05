"""Generate Figure: Topology optimisation convergence and density fields.

Structural and thermal domains.
Outputs:
  structural_topopt_convergence.pdf
  thermal_topopt_convergence.pdf
  structural_topopt_fields.pdf     (only when topopt_fields.npz present)
  thermal_topopt_fields.pdf        (only when topopt_fields.npz present)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES, STRUCTURAL_ORDER, THERMAL_ORDER

RESULTS = Path(__file__).parent.parent.parent / "results"


def _plot_combined_convergence(out_path: Path) -> None:
    domains = [
        ("Structural", RESULTS / "structural-mesh" / "optimization" / "topopt" / "result.json", STRUCTURAL_ORDER),
        ("Thermal",    RESULTS / "thermal-mesh"    / "optimization" / "topopt" / "result.json", THERMAL_ORDER),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
    fig.subplots_adjust(bottom=0.42, wspace=0.35)

    all_present: set[str] = set()
    all_order: list[str] = []

    for ax, (domain_label, result_path, solver_order) in zip(axes, domains):
        data = json.loads(result_path.read_text())
        by_solver = data["by_solver"]

        for solver, sdata in by_solver.items():
            label, color, ls, mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
            compliances = sdata.get("compliances", [])
            if compliances:
                ax.semilogy(range(len(compliances)), compliances,
                            color=color, linestyle=ls, marker="", linewidth=1.6)
            all_present.add(solver)

        for s in solver_order:
            if s not in all_order:
                all_order.append(s)

        ax.set_title(domain_label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Compliance" if ax is axes[0] else "")

    handles = [
        mlines.Line2D([], [], color=SOLVER_STYLES[s][1], linestyle=SOLVER_STYLES[s][2],
                      linewidth=1.6, label=SOLVER_STYLES[s][0])
        for s in all_order if s in all_present and s in SOLVER_STYLES
    ]
    ncols = max(1, len(handles) // 2)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=ncols, fontsize=7.5, framealpha=0.7, handlelength=2.0)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_fields(
    domain_label: str,
    fields_path: Path,
    params_path: Path,
    out_path: Path,
) -> None:
    if not fields_path.exists():
        print(f"Skipping fields figure — {fields_path} not found")
        return

    npz = np.load(fields_path)
    params = json.loads(params_path.read_text())
    ph = params["physics"]
    nx = ph["nx"]
    ny = ph["ny"]
    nz = ph["nz"]

    npz_solvers = list(npz["solver_names"])
    result_path = fields_path.parent / "result.json"
    all_solvers = (list(json.loads(result_path.read_text())["by_solver"].keys())
                   if result_path.exists() else npz_solvers)

    def _reshape(rho_flat: np.ndarray) -> np.ndarray:
        """Reshape flat density to 2D by summing along the thinnest axis."""
        rho_3d = rho_flat.reshape(nz, ny, nx)
        if ny <= nz and ny <= nx:
            return rho_3d.sum(axis=1)   # sum over y → (nz, nx)
        return rho_3d.sum(axis=0)       # sum over z → (ny, nx)

    CMAP = "gray_r"
    VMIN, VMAX = 0.0, 1.0

    n_panels = len(all_solvers)
    ncols = math.ceil(n_panels / 2)
    nrows = 2
    fig, axes_2d = plt.subplots(nrows, ncols,
                                figsize=(TEXTWIDTH * 0.92, TEXTWIDTH * 0.55))
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.25)
    axes = axes_2d.flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    im = None
    for panel_i, sname in enumerate(all_solvers):
        ax = axes[panel_i]
        label = SOLVER_STYLES.get(sname, (sname,))[0]
        if sname in npz_solvers:
            npz_i = npz_solvers.index(sname)
            rho_final_2d = _reshape(npz[f"rho_final_{npz_i}"])
            rho_norm = rho_final_2d / (rho_final_2d.max() or 1.0)
            im = ax.imshow(rho_norm, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                           aspect="auto", interpolation="bilinear")
        else:
            img_path = fields_path.parent / f"topopt_3d_{sname}.png"
            if img_path.exists():
                img = plt.imread(str(img_path))
                im = ax.imshow(img[:, :, 0] if img.ndim == 3 else img,
                               cmap=CMAP, vmin=VMIN, vmax=VMAX,
                               aspect="auto", interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
        ax.set_title(label, fontsize=7.5)
        ax.axis("off")

    if im is not None:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(["0", "0.5", "1"])
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Density", fontsize=7.5)

    fig.suptitle(f"{domain_label}", fontsize=8, y=1.01)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_conductivity_recovery(out_path: Path) -> None:
    result_path = RESULTS / "thermal-mesh" / "optimization" / "conductivity_recovery" / "result.json"
    fields_png   = RESULTS / "thermal-mesh" / "optimization" / "conductivity_recovery" / "conductivity_recovery_fields.png"

    data = json.loads(result_path.read_text())
    by_solver = data["by_solver"]

    has_fields = fields_png.exists()
    ncols = 2 if has_fields else 1
    fig, axes = plt.subplots(1, ncols,
                             figsize=(TEXTWIDTH, TEXTWIDTH * 0.38),
                             gridspec_kw={"width_ratios": [1, 1.6]} if has_fields else {})
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    ax_conv = axes[0] if has_fields else axes

    present: set[str] = set()
    for solver, sdata in by_solver.items():
        errors = sdata.get("errors", [])
        if not errors:
            continue
        label, color, ls, mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
        ax_conv.semilogy(range(len(errors)), errors,
                         color=color, linestyle=ls, linewidth=1.6, label=label)
        present.add(solver)

    ax_conv.set_title("Thermal — conductivity recovery")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Identification error")

    handles = [
        mlines.Line2D([], [], color=SOLVER_STYLES[s][1], linestyle=SOLVER_STYLES[s][2],
                      linewidth=1.6, label=SOLVER_STYLES[s][0])
        for s in THERMAL_ORDER if s in present and s in SOLVER_STYLES
    ]
    ax_conv.legend(handles=handles, fontsize=7.5, framealpha=0.7,
                   handlelength=2.0, loc="upper right")

    if has_fields:
        ax_fields = axes[1]
        img = plt.imread(str(fields_png))
        ax_fields.imshow(img, aspect="auto", interpolation="bilinear")
        ax_fields.set_title("Recovered conductivity fields")
        ax_fields.axis("off")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _plot_combined_convergence(out_path=out_dir / "topopt_convergence.pdf")
        _plot_conductivity_recovery(out_path=out_dir / "conductivity_recovery.pdf")
        _plot_fields(
            domain_label="Structural",
            fields_path=RESULTS / "structural-mesh" / "optimization" / "topopt" / "topopt_fields.npz",
            params_path=RESULTS / "structural-mesh" / "optimization" / "topopt" / "params.json",
            out_path=out_dir / "structural_topopt_fields.pdf",
        )
        _plot_fields(
            domain_label="Thermal",
            fields_path=RESULTS / "thermal-mesh" / "optimization" / "topopt" / "topopt_fields.npz",
            params_path=RESULTS / "thermal-mesh" / "optimization" / "topopt" / "params.json",
            out_path=out_dir / "thermal_topopt_fields.pdf",
        )

