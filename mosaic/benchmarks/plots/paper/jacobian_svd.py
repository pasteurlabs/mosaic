"""Generate Figure: Jacobian SVD spectra for 2D and 3D NS.

One panel per solver, one line per (nu, T) configuration.
Produces:
  appendix_jacobian_svd_2d.pdf
  appendix_jacobian_svd_3d.pdf
  jacobian_svd_comparison.pdf / .png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

NS_ORDER_2D = [
    "jax_cfd",
    "phiflow",
    "ins_jl",
    "xlb",
    "pict",
    "warp_ns",
    "openfoam",
]
NS_ORDER_3D = ["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict", "openfoam"]

VARIANT_STYLES = [
    {"color": "#0077BB", "linestyle": "-"},
    {"color": "#CC3311", "linestyle": "--"},
    {"color": "#009988", "linestyle": "-."},
    {"color": "#EE7733", "linestyle": ":"},
]


def _variant_label(phys: dict) -> str:
    nu = phys["nu"]
    steps = phys["steps"]
    dt = phys["dt"]
    t = steps * dt
    # Human-readable: low/high viscosity, short/long rollout
    visc = "low ν" if nu <= 0.001 else "high ν"
    horizon = f"T={t:.2g}s"
    return f"{visc}, {horizon}"


def _solver_color_label(solver: str) -> tuple[str, str]:
    entry = SOLVER_STYLES.get(solver)
    if entry:
        return entry[1], entry[0]
    return "#888888", solver


def _svd_panels(fig, axes, variants, solvers, n_show) -> list:
    """Fill axes panels; return legend handles for the variant lines."""
    legend_handles: list[mlines.Line2D] = []
    legend_built = False

    ncols = axes.shape[1]

    for idx, solver in enumerate(solvers):
        ax = axes[idx // ncols][idx % ncols]
        color, label = _solver_color_label(solver)
        y_min_data = np.inf

        for vi, (_, data) in enumerate(variants):
            spectra = data["per_solver_spectra"]
            phys = data["params"]["physics"]

            if solver not in spectra:
                continue

            sv = np.array(spectra[solver], dtype=float)
            sv_norm = sv / sv[0] if sv[0] > 0 else sv
            n = min(n_show, len(sv_norm))
            modes = np.arange(1, n + 1)

            vstyle = VARIANT_STYLES[vi % len(VARIANT_STYLES)]
            vlabel = _variant_label(phys)

            mk = "o" if n <= 32 else ""
            (line,) = ax.semilogy(
                modes,
                sv_norm[:n],
                f"{mk}{vstyle['linestyle']}",
                color=vstyle["color"],
                markersize=3 if mk else 0,
                linewidth=1.5,
                label=vlabel,
            )

            if not legend_built:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=vstyle["color"],
                        linestyle=vstyle["linestyle"],
                        linewidth=1.5,
                        label=vlabel,
                    )
                )

            pos = sv_norm[:n][sv_norm[:n] > 0]
            if len(pos):
                y_min_data = min(y_min_data, float(pos.min()))

        legend_built = True  # only collect handles from the first solver panel

        if np.isfinite(y_min_data) and y_min_data > 0:
            y_floor = 10 ** (np.floor(np.log10(y_min_data)) - 0.5)
            ax.set_ylim(bottom=y_floor, top=2.0)

        ax.set_title(label)
        ax.set_xlabel("Mode index $i$")  # hidden on non-bottom rows after subplots call
        if idx % ncols == 0:
            ax.set_ylabel(r"$\sigma_i\,/\,\sigma_1$")
        else:
            ax.set_ylabel("")

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    return legend_handles


def _plot_svd_figure(
    subdir: str,
    experiments: list[str],
    n_show: int,
    solver_order: list[str],
    out_path: Path,
) -> None:
    variants: list[tuple[str, dict]] = []
    for exp_key in experiments:
        path = results_dir() / subdir / "gradient" / exp_key / "result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data.get("per_solver_spectra"):
            variants.append((exp_key, data))

    if not variants:
        print(f"No data found for {subdir}")
        return

    _NS_EXCLUDED = {"fenics_ns", "su2"}
    all_solver_sets = [set(d["per_solver_spectra"].keys()) for _, d in variants]
    solvers = [s for s in solver_order if any(s in ss for ss in all_solver_sets)]
    for ss in all_solver_sets:
        for s in sorted(ss):
            if s not in solvers and s not in _NS_EXCLUDED:
                solvers.append(s)

    n_solvers = len(solvers)
    ncols = min(3, n_solvers)
    nrows = math.ceil(n_solvers / ncols)

    panel_h = TEXTWIDTH / ncols * 0.85
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(TEXTWIDTH, panel_h * nrows + 0.3),
        squeeze=False,
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.18)

    handles = _svd_panels(fig, axes, variants, solvers, n_show)

    for row in range(nrows - 1):
        for col in range(ncols):
            axes[row][col].set_xlabel("")
            axes[row][col].tick_params(labelbottom=False)

    for idx in range(n_solvers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    ncols_leg = min(len(handles), 4)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncols_leg,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.5,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _svd_comparison(
    subdir: str,
    experiments: list[str],
    solvers: list[str],
    n_show: int,
    out_stem: str,
    out_dir: Path,
    fig_w_scale: float = 1.15,
    extra_outputs: list[str] | None = None,
) -> None:
    """Shared implementation for SVD comparison figures (2D and 3D)."""
    _EXCLUDED = {"fenics_ns", "openfoam", "su2"}

    variants: list[tuple[str, dict]] = []
    for exp_key in experiments:
        path = results_dir() / subdir / "gradient" / exp_key / "result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data.get("per_solver_spectra"):
            variants.append((exp_key, data))

    if not variants:
        print(f"No data found for {subdir} jacobian_svd")
        return

    all_solver_sets = [set(d["per_solver_spectra"].keys()) for _, d in variants]
    present_solvers = [
        s
        for s in solvers
        if any(s in ss for ss in all_solver_sets) and s not in _EXCLUDED
    ]

    n_solvers = len(present_solvers)
    ncols = 3
    nrows = math.ceil(n_solvers / ncols)

    fig_w = TEXTWIDTH * fig_w_scale
    panel_h = fig_w / ncols * 0.4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, panel_h * nrows + 0.4),
        squeeze=False,
        sharex=True,
        dpi=300,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.18)

    handles = _svd_panels(fig, axes, variants, present_solvers, n_show=n_show)

    for row in range(nrows - 1):
        for col in range(ncols):
            axes[row][col].set_xlabel("")
            axes[row][col].tick_params(labelbottom=False)

    for idx in range(n_solvers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    axes[0][0].annotate(
        "steeper $\\downarrow$ = larger $\\kappa$\n(worse conditioning)",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        fontsize=6.5,
        ha="right",
        va="bottom",
        color="0.35",
        fontstyle="italic",
    )

    ncols_leg = min(len(handles), 4)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=ncols_leg,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.5,
    )

    stems = [out_stem] + (extra_outputs or [])
    for stem in stems:
        for ext in ("pdf", "png"):
            out = out_dir / f"{stem}.{ext}"
            fig.savefig(out, dpi=200 if ext == "png" else None)
            print(f"Saved {out}")
    plt.close(fig)


_EXPERIMENTS = [
    "jacobian_svd",
    "jacobian_svd_nu01",
    "jacobian_svd_steps20",
    "jacobian_svd_steps40",
]


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        # 3D NS — main paper figure
        _svd_comparison(
            subdir="ns-3d-grid",
            experiments=_EXPERIMENTS,
            solvers=["exponax", "phiflow", "xlb", "ins_jl", "warp_ns", "pict"],
            n_show=1536,
            out_stem="jacobian_svd_comparison",
            out_dir=out_dir,
            fig_w_scale=1.15,
        )
        # 2D NS — appendix figure, same style, fenics_ns excluded
        _svd_comparison(
            subdir="ns-grid",
            experiments=_EXPERIMENTS,
            solvers=["jax_cfd", "phiflow", "ins_jl", "xlb", "pict", "warp_ns"],
            n_show=128,
            out_stem="appendix_jacobian_svd_2d",
            out_dir=out_dir,
            fig_w_scale=1.0,
        )


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
