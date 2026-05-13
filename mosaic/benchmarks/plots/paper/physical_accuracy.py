"""Physical accuracy figures — single-experiment + cross-domain.

Two public entry points:

  * :func:`plot_experiment(cfg, exp_key, suffix, save)` — single-experiment
    figure for ``forward/physical_laws`` (or its sweep variants). Used
    both as the canonical mosaic experiment plot (delegated from
    :func:`mosaic.benchmarks.problems.shared.plots.forward.plot_physical_laws`)
    and per-problem from the cross-domain generator.
  * :func:`generate(out_dir)` — appendix figures: 3×3 grids for each NS
    domain, plus one-panel ``Compliance`` / ``Thermal compliance`` figures
    for the FEM domains.

Layout detection lives in :func:`plot_experiment`. If the experiment dir
contains a top-level ``result.json`` it renders the FEM single-panel
figure. Otherwise it scans for sub-dirs (one per sweep variant: ``vs_N``,
``vs_steps``, ``vs_nu``) and renders the 3×3 NS grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import experiment_dir, load_json, results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

# ── shared sweep / metric constants for the NS 3×3 grid ──────────────────────

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

# Per-problem FEM specs: which metric + xlabel + ylabel + title to render.
_FEM_SPECS: dict[str, dict] = {
    "structural-mesh": {
        "metric": "compliance",
        "xlabel": r"$F_\mathrm{total}$",
        "ylabel": "Compliance",
        "title": "Structural",
    },
    "thermal-mesh": {
        "metric": "thermal_compliance",
        "xlabel": r"$Q_\mathrm{total}$",
        "ylabel": "Thermal compliance",
        "title": "Thermal",
    },
}

# Per-problem NS specs: subdir (== problem name) and figure title.
_NS_DOMAINS: list[tuple[str, str]] = [
    ("ns-grid", "2D NS — physical accuracy"),
    ("ns-3d-grid", "3D NS — physical accuracy"),
]

# Mapping from problem name -> appendix paper-figure filename.
_PAPER_FILENAMES = {
    "ns-grid": "appendix_pa_ns2d.pdf",
    "ns-3d-grid": "appendix_pa_ns3d.pdf",
    "structural-mesh": "appendix_pa_structural.pdf",
    "thermal-mesh": "appendix_pa_thermal.pdf",
}


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


# ── NS rendering helpers ──────────────────────────────────────────────────────


def _plot_ns_row(
    axes_row,
    sweep_key: str,
    log_x: bool,
    use_elements: bool,
    data: dict,
    subdir: str,
    ns_seen: set[str],
    row_is_top: bool,
) -> None:
    """Render one row of the 3×3 NS grid for a single sweep (3 metrics)."""
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
        ax = axes_row[col]
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
                    else (ax.semilogy if log_y else (ax.semilogx if log_x else ax.plot))
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

        if row_is_top:
            ax.set_title(metric_label)
        if col == 0:
            ax.set_ylabel(ROW_LABELS[sweep_key], fontsize=9)
        ax.set_xlabel(SWEEP_XLABELS[sweep_key])

        _set_axis_ticks(ax, x_all, log_x, log_y, is_elements=use_elements)


def _plot_ns_grid(
    cfg_name: str,
    sweeps_data: dict[str, dict],
    domain_title: str,
    out_path: Path | None,
) -> plt.Figure:
    """3×3 NS grid: rows=sweep (vs_N, vs_nu, vs_steps), cols=metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.85))
    fig.suptitle(domain_title, fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22, wspace=0.35, hspace=0.55)

    ns_seen: set[str] = set()

    for row, (sweep_key, log_x, use_elements) in enumerate(SWEEPS):
        data = sweeps_data.get(sweep_key)
        if data is None:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue
        _plot_ns_row(
            axes[row],
            sweep_key,
            log_x,
            use_elements,
            data,
            cfg_name,
            ns_seen,
            row_is_top=(row == 0),
        )

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

    if out_path is not None:
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    return fig


# ── FEM rendering helper ──────────────────────────────────────────────────────


def _plot_fem_single(
    data: dict,
    spec: dict,
    out_path: Path | None,
) -> plt.Figure:
    """Single-panel FEM physical-laws figure (metric vs total load)."""
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH * 0.47, TEXTWIDTH * 0.47 * 1.1))
    fig.subplots_adjust(bottom=0.22)

    metric = spec["metric"]
    by_param = data["by_param"]
    params = sorted(by_param.keys(), key=float)
    x_vals = np.array([float(p) for p in params])

    all_solvers: list[str] = []
    for pdata in by_param.values():
        for s in pdata:
            if s not in all_solvers:
                all_solvers.append(s)

    fem_seen: set[str] = set()

    # Reference slope-2 curve calibrated on the first parameter point.
    valid_first = [
        by_param[params[0]][s][metric]
        for s in all_solvers
        if metric in by_param[params[0]].get(s, {})
    ]
    if valid_first and len(x_vals) > 0:
        c0 = np.mean(valid_first)
        ref_y = float(c0) * (x_vals / x_vals[0]) ** 2
        ax.loglog(
            x_vals,
            ref_y,
            color="#aaaaaa",
            linestyle="--",
            linewidth=1.0,
            zorder=0,
            label="slope 2",
        )

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
        for px, p in zip(x_vals, params, strict=False):
            val = by_param[p].get(solver, {}).get(metric)
            if val is not None:
                xs.append(px)
                ys.append(float(val))
        if xs:
            ax.loglog(xs, ys, **kw)
            if solver in FEM_ORDER:
                fem_seen.add(solver)

    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    _set_axis_ticks(ax, list(x_vals), True, True)

    handles = dedup_handles([make_handle(s) for s in FEM_ORDER if s in fem_seen])
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    if out_path is not None:
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    return fig


# ── public API ────────────────────────────────────────────────────────────────


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "physical_laws",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure | None:
    """Single-experiment physical-accuracy figure for ``(cfg, exp_key)``.

    Reads ``<results>/<cfg.name>/forward/<exp_key>{suffix}/result.json`` (or
    its sub-dirs when the experiment is a multi-run sweep), and writes a
    paper-styled PDF named ``physical_accuracy.pdf`` next to it.

    Detection:
      * If the experiment dir contains ``result.json``: single FEM panel
        (compliance / thermal_compliance vs total load).
      * If it contains sub-dirs ``vs_N`` / ``vs_steps`` / ``vs_nu``: 3×3 NS
        grid covering both axes (sweep × metric).
      * If neither, returns ``None`` (no data available).
    """
    plt.rcParams.update(RCPARAMS)

    out_dir = experiment_dir(results_dir(), cfg.name, "forward", exp_key + suffix)

    # ── single-result layout (FEM or per-sub-key NS call) ────────────────────
    single_path = out_dir / "result.json"
    if single_path.exists():
        data = load_json(single_path)
        spec = _FEM_SPECS.get(cfg.name)
        if spec is not None:
            return _plot_fem_single(
                data, spec, out_dir / "physical_accuracy.pdf" if save else None
            )
        # NS sub-experiment (one sweep variant): render a 1×3 row.
        sweep_key = (exp_key + suffix).rsplit("/", 1)[-1] or "vs_param"
        sweep_meta = next(
            ((sk, log_x, use_el) for sk, log_x, use_el in SWEEPS if sk == sweep_key),
            (sweep_key, True, sweep_key == "vs_N"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.32))
        fig.subplots_adjust(bottom=0.32, wspace=0.45)
        ns_seen: set[str] = set()
        _plot_ns_row(
            axes,
            sweep_meta[0],
            sweep_meta[1],
            sweep_meta[2],
            data,
            cfg.name,
            ns_seen,
            row_is_top=True,
        )
        handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in ns_seen])
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(handles), 6),
                fontsize=7.0,
                framealpha=0.7,
                handlelength=2.0,
            )
        if save:
            out = out_dir / "physical_accuracy.pdf"
            fig.savefig(out)
            print(f"Saved {out}")
        return fig

    # ── multi-run sweep layout: 3×3 NS grid over sub-dirs ────────────────────
    sweeps_data: dict[str, dict] = {}
    if out_dir.is_dir():
        for sub in sorted(out_dir.iterdir()):
            if not sub.is_dir():
                continue
            sub_result = sub / "result.json"
            if sub_result.exists():
                sweeps_data[sub.name] = load_json(sub_result)
    if not sweeps_data:
        return None

    title = next(
        (t for n, t in _NS_DOMAINS if n == cfg.name),
        f"{cfg.category_label or cfg.name} — physical accuracy",
    )
    return _plot_ns_grid(
        cfg.name,
        sweeps_data,
        title,
        out_dir / "physical_accuracy.pdf" if save else None,
    )


def generate(out_dir: Path) -> None:
    """Appendix figures: per-domain physical-accuracy panel grids."""
    from mosaic.benchmarks.problems import get_config

    plt.rcParams.update(RCPARAMS)

    for problem_name in _PAPER_FILENAMES:
        try:
            cfg = get_config(problem_name)
        except Exception:
            continue
        fig = plot_experiment(cfg, exp_key="physical_laws", save=False)
        if fig is None:
            continue
        out = out_dir / _PAPER_FILENAMES[problem_name]
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
