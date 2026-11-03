"""Generate Figure: Physical accuracy of the Navier-Stokes solvers.

Produces:
  appendix_pa_ns2d.pdf       — 3×3 grid (2D NS)
  appendix_pa_ns3d.pdf       — 3×3 grid (3D NS)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, results_dir
from .style import (
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

RESULTS = results_dir()

SWEEPS = [
    ("vs_N", True, True),
    ("vs_nu", True, False),
    ("vs_steps", False, False),
]
METRICS = [
    ("analytic_error", "Analytic error", True),
    ("divergence_rms", "Divergence RMS", True),
    ("kinetic_energy", "Kinetic energy", False),
]
SWEEP_XLABELS = {
    "vs_N": "Elements",
    "vs_nu": r"$\nu$",
    "vs_steps": "Steps",
}
ROW_LABELS = {
    "vs_N": "vs N",
    "vs_nu": r"vs $\nu$",
    "vs_steps": "vs steps",
}

KE0 = 0.25


def _n_to_elements(N: int, subdir: str) -> int:
    if subdir == "ns-grid":
        return N**2
    if subdir == "ns-3d-grid":
        return N**3
    return N


def _set_axis_ticks(
    ax, vals: list, is_log_x: bool, is_log_y: bool, is_elements: bool = False
):
    tick_x = sorted(set(vals))
    if len(tick_x) > 4:
        idx = np.round(np.linspace(0, len(tick_x) - 1, 4)).astype(int)
        tick_x = [tick_x[i] for i in idx]
    ax.set_xticks(tick_x)
    if is_elements:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
            )
        )
    else:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis="x", labelsize=7.5, rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    if is_log_y:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=7.5)


def _ke_analytic(sweep_key: str, params: list[str], phys: dict, subdir: str):
    if subdir != "ns-grid":
        return None
    dt = float(phys.get("dt", 0.01))
    nu_fixed = float(phys.get("nu", 0.05))
    steps_fixed = int(phys.get("steps", 20))
    xs, ys = [], []
    for p in params:
        raw = float(p)
        if sweep_key == "vs_nu":
            t = steps_fixed * dt
            nu = raw
        elif sweep_key == "vs_steps":
            t = raw * dt
            nu = nu_fixed
        else:
            t = steps_fixed * dt
            nu = nu_fixed
        ke = KE0 * np.exp(-4.0 * nu * t)
        x = _n_to_elements(int(p), subdir) if sweep_key == "vs_N" else raw
        xs.append(x)
        ys.append(ke)
    return xs, ys


def _plot_ns_domain(subdir: str, domain_title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.85))
    fig.suptitle(domain_title, fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22, wspace=0.35, hspace=0.55)

    ns_seen: set[str] = set()

    for row, (sweep_key, log_x, use_elements) in enumerate(SWEEPS):
        path = RESULTS / subdir / "forward/physical_laws" / sweep_key / "result.json"
        data = json.loads(path.read_text())
        by_param = data["by_param"]
        params = sorted(by_param.keys(), key=float)
        phys = data.get("params", {}).get("physics", {})

        _EXCLUDED = {"fenics_ns", "su2"}
        all_solvers: list[str] = []
        for pdata in by_param.values():
            for s in pdata:
                if s not in all_solvers and s not in _EXCLUDED:
                    all_solvers.append(s)

        for col, (metric_key, metric_label, log_y) in enumerate(METRICS):
            ax = axes[row, col]
            x_all: list[float] = []

            for solver in all_solvers:
                _label, color, ls, mk = solver_props(solver)
                kw = {
                    "color": color,
                    "linestyle": ls,
                    "marker": mk,
                    "markersize": 4,
                    "markeredgewidth": 0,
                    "linewidth": 1.6,
                }
                xs, ys = [], []
                for p in params:
                    entry = by_param[p].get(solver)
                    val = entry.get(metric_key) if isinstance(entry, dict) else None
                    if val is not None:
                        raw_x = float(p)
                        x = _n_to_elements(int(p), subdir) if use_elements else raw_x
                        xs.append(x)
                        ys.append(float(val))
                if xs:
                    plot_fn = (
                        ax.loglog
                        if (log_x and log_y)
                        else (
                            ax.semilogy
                            if log_y
                            else (ax.semilogx if log_x else ax.plot)
                        )
                    )
                    plot_fn(xs, ys, **kw)
                    x_all.extend(xs)
                    if solver in NS_ORDER:
                        ns_seen.add(solver)

            if metric_key == "kinetic_energy":
                ke_ref = _ke_analytic(sweep_key, params, phys, subdir)
                if ke_ref is not None:
                    plot_fn = ax.semilogx if log_x else ax.plot
                    plot_fn(
                        ke_ref[0],
                        ke_ref[1],
                        color="#aaaaaa",
                        linestyle="--",
                        linewidth=1.2,
                        zorder=0,
                        label="analytic",
                    )

            if row == 0:
                ax.set_title(metric_label)
            if col == 0:
                ax.set_ylabel(ROW_LABELS[sweep_key], fontsize=9)
            if row == len(SWEEPS) - 1:
                ax.set_xlabel(SWEEP_XLABELS[sweep_key])

            _set_axis_ticks(ax, x_all, log_x, log_y, is_elements=use_elements)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in ns_seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    _plot_ns_domain(
        "ns-grid", "2D NS — physical accuracy", out_dir / "appendix_pa_ns2d.pdf"
    )
    _plot_ns_domain(
        "ns-3d-grid", "3D NS — physical accuracy", out_dir / "appendix_pa_ns3d.pdf"
    )
