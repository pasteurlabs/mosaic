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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    RCPARAMS,
    SOLVER_STYLES,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
)


def _plot_combined_convergence(out_path: Path) -> None:
    domains = [
        (
            "Structural",
            results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "result.json",
            STRUCTURAL_ORDER,
        ),
        (
            "Thermal",
            results_dir() / "thermal-mesh" / "optimization" / "topopt" / "result.json",
            THERMAL_ORDER,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
    fig.subplots_adjust(bottom=0.42, wspace=0.35)

    all_present: set[str] = set()
    all_order: list[str] = []

    for ax, (domain_label, result_path, solver_order) in zip(
        axes, domains, strict=False
    ):
        if not result_path.exists():
            print(f"[topopt] {result_path} not found — skipping {domain_label}")
            ax.set_title(f"{domain_label} (no data)")
            ax.set_visible(False)
            continue
        data = json.loads(result_path.read_text())
        by_solver = data["by_solver"]

        for solver, sdata in by_solver.items():
            _label, color, ls, _mk = SOLVER_STYLES.get(
                solver, (solver, "#888888", "-", "o")
            )
            compliances = sdata.get("compliances", [])
            if compliances:
                ax.semilogy(
                    range(len(compliances)),
                    compliances,
                    color=color,
                    linestyle=ls,
                    marker="",
                    linewidth=1.6,
                )
            all_present.add(solver)

        for s in solver_order:
            if s not in all_order:
                all_order.append(s)

        ax.set_title(domain_label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Compliance" if ax is axes[0] else "")

    handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_STYLES[s][1],
            linestyle=SOLVER_STYLES[s][2],
            linewidth=1.6,
            label=SOLVER_STYLES[s][0],
        )
        for s in all_order
        if s in all_present and s in SOLVER_STYLES
    ]
    ncols = max(1, len(handles) // 2)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncols,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


_THRESH = 0.35
_ELEV = 22
_AZIM = 35
_CLR_FIXED = "#888888"
_CLR_LOAD = "#FF1744"


def _add_bcs(ax, nx: int, ny: int, nz: int, ph: dict) -> None:
    """Overlay fixed-face patch and load arrow on a voxel Axes3D.

    Voxel axes are (x=length, y=width, z=height) after transpose.
    Fixed: x=0 face.  Load: right-face corner, direction from params.
    """
    # Fixed face — semi-transparent gray wall at x=0
    wall = Poly3DCollection(
        [[(0, 0, 0), (0, ny, 0), (0, ny, nz), (0, 0, nz)]],
        alpha=0.55,
        facecolor=_CLR_FIXED,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.add_collection3d(wall)

    # Load point in voxel coords (first cell of the relevant corner)
    corner_z_high = ph.get("corner_z_high", False)
    corner_y_high = ph.get("corner_y_high", False)
    load_axis = ph.get("load_axis", "z")
    ly = (ny - 0.5) if corner_y_high else 0.5
    lz = (nz - 0.5) if corner_z_high else 0.5

    arrow_len = nz * 0.65
    if load_axis == "y":
        y_sign = -1 if corner_y_high else 1
        dx, dy_a, dz_a = 0, y_sign * arrow_len, 0
    else:
        z_sign = -1 if corner_z_high else 1
        dx, dy_a, dz_a = 0, 0, z_sign * arrow_len

    # Arrow originates at the load surface and points in the force direction
    ox = nx + 0.6 if load_axis != "y" else nx
    ax.quiver(
        ox,
        ly,
        lz,
        dx,
        dy_a,
        dz_a,
        color=_CLR_LOAD,
        linewidth=2.5,
        arrow_length_ratio=0.28,
    )


def _voxel_facecolors(
    rho_xyz: np.ndarray,
    filled: np.ndarray,
    base_color: str,
) -> np.ndarray:
    """Return RGBA facecolor array; empty voxels are fully transparent."""
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(base_color)
    fc = np.zeros((*rho_xyz.shape, 4))
    norm = np.where(filled, (rho_xyz - _THRESH) / (1.0 - _THRESH), 0.0)
    # Lighten low-density voxels slightly toward white
    fc[..., 0] = r + (1 - r) * (1 - norm) * 0.45
    fc[..., 1] = g + (1 - g) * (1 - norm) * 0.45
    fc[..., 2] = b + (1 - b) * (1 - norm) * 0.45
    fc[..., 3] = filled.astype(float)
    return fc


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
    nx, ny, nz = ph["nx"], ph["ny"], ph["nz"]

    npz_solvers = list(npz["solver_names"])

    n = len(npz_solvers)
    nrows, ncols = 2, math.ceil(n / 2)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.58))

    for i, sname in enumerate(npz_solvers):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        rho_flat = npz[f"rho_final_{i}"]
        # reshape to (nz, ny, nx) then transpose to (nx, ny, nz) = (x, y, z)
        rho_xyz = rho_flat.reshape(nz, ny, nx).transpose(2, 1, 0)
        filled = rho_xyz > _THRESH

        _, color, _, _ = SOLVER_STYLES.get(sname, (sname, "#555555", "-", "o"))
        fc = _voxel_facecolors(rho_xyz, filled, color)
        # edgecolors matching face keeps grid lines invisible while shade=True works
        ax.voxels(filled, facecolors=fc, edgecolors=fc, shade=True)
        _add_bcs(ax, nx, ny, nz, params["physics"])

        label = SOLVER_STYLES.get(sname, (sname,))[0]
        ax.set_title(label, fontsize=7.5, pad=-4)
        ax.view_init(elev=_ELEV, azim=_AZIM)
        ax.set_axis_off()

    fig.subplots_adjust(wspace=0.0, hspace=-0.08, top=0.88, bottom=0.0)
    fig.suptitle(domain_label, fontsize=8, y=0.97)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_conductivity_recovery(out_path: Path) -> None:
    result_path = (
        results_dir()
        / "thermal-mesh"
        / "optimization"
        / "conductivity_recovery"
        / "result.json"
    )
    fields_png = (
        results_dir()
        / "thermal-mesh"
        / "optimization"
        / "conductivity_recovery"
        / "conductivity_recovery_fields.png"
    )

    data = json.loads(result_path.read_text())
    by_solver = data["by_solver"]

    has_fields = fields_png.exists()
    ncols = 2 if has_fields else 1
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.38),
        gridspec_kw={"width_ratios": [1, 1.6]} if has_fields else {},
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    ax_conv = axes[0] if has_fields else axes

    present: set[str] = set()
    for solver, sdata in by_solver.items():
        errors = sdata.get("errors", [])
        if not errors:
            continue
        label, color, ls, _mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
        ax_conv.semilogy(
            range(len(errors)),
            errors,
            color=color,
            linestyle=ls,
            linewidth=1.6,
            label=label,
        )
        present.add(solver)

    ax_conv.set_title("Thermal — conductivity recovery")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Identification error")

    handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_STYLES[s][1],
            linestyle=SOLVER_STYLES[s][2],
            linewidth=1.6,
            label=SOLVER_STYLES[s][0],
        )
        for s in THERMAL_ORDER
        if s in present and s in SOLVER_STYLES
    ]
    ax_conv.legend(
        handles=handles,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
        loc="upper right",
    )

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
            fields_path=results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "topopt_fields.npz",
            params_path=results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "params.json",
            out_path=out_dir / "structural_topopt_fields.pdf",
        )
        _plot_fields(
            domain_label="Thermal",
            fields_path=results_dir()
            / "thermal-mesh"
            / "optimization"
            / "topopt"
            / "topopt_fields.npz",
            params_path=results_dir()
            / "thermal-mesh"
            / "optimization"
            / "topopt"
            / "params.json",
            out_path=out_dir / "thermal_topopt_fields.pdf",
        )
