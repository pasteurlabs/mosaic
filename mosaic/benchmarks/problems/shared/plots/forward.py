# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plots for the forward suite (agreement, physical_laws)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    apply_style,
    dedup_handles,
    field_grid,
    fig_shared_legend,
    make_handle,
    save_fig,
    solver_plot_props,
    solver_props,
    solver_styles,
    subplots_grid,
    unit_label,
    vorticity_2d,
)

apply_style()


def _smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smooth a 1-D array (for noisy per-snapshot observables like RDF)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(arr, k, mode="same")


def _resolve_field_to_2d(field_to_2d: Any) -> Any:
    """Return the field→2D callable, falling back to vorticity_2d."""
    return field_to_2d if field_to_2d is not None else vorticity_2d


def _field_grid_kw(field_cmap: str, field_symmetric: bool) -> dict:
    """Return field_grid keyword overrides for the problem's colormap / symmetry."""
    return {"cmap": field_cmap, "symmetric": field_symmetric}


def _is_field(arr: np.ndarray) -> bool:
    """True if *arr* is a plottable field (at least 2-D and not a scalar stub)."""
    return arr.ndim >= 2


_SUITE = "forward"


# ── agreement ─────────────────────────────────────────────────────────────────


def _agreement_plot_scalar(
    cfg: Any,
    npz: Any,
    solver_names: Any,
    sweep_vals: Any,
    sweep_key: Any,
    styles: Any,
    out_dir: Any,
    save: Any,
    *,
    output_key: str,
    units: dict | None,
) -> Any:
    """Scalar-output agreement: plot the scalar value vs sweep parameter."""
    n_vals = len(sweep_vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    all_y: list[float] = []
    solver_series: list[tuple] = []
    for name in solver_names:
        y_vals = [
            float(npz[f"{name}_{i}"]) if f"{name}_{i}" in npz else np.nan
            for i in range(n_vals)
        ]
        style = styles.get(name, {})
        props = solver_plot_props(style)
        lbl = style.get("label", name)
        solver_series.append((y_vals, props, lbl))
        all_y.extend(v for v in y_vals if np.isfinite(v) and v > 0)
    # Use log-y when the output spans more than one decade (e.g. SIMP compliance)
    if all_y and max(all_y) / min(all_y) > 10:
        ax.set_yscale("log")
    for y_vals, props, lbl in solver_series:
        ax.plot(sweep_vals, y_vals, label=lbl, **props)
    ax.set_xlabel(unit_label(sweep_key, units))
    ylabel = output_key.replace("_", " ")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{cfg.name} — {ylabel} vs {sweep_key}")
    ax.grid(True, alpha=0.3)
    fig_shared_legend(fig, [ax])
    if save:
        save_fig(fig, "curves", out_dir)
    return fig


def _agreement_curve_panel(
    ax_top: Any, ax_bot: Any, x: Any, npz: Any, solver_names: Any, i: Any, styles: Any
) -> None:
    """Render one column (one sweep value) of the curve-mode agreement grid."""
    cons_key = f"consensus_{i}"
    consensus = _smooth(npz[cons_key]) if cons_key in npz else None
    for name in solver_names:
        key_s = f"{name}_{i}"
        if key_s not in npz:
            continue
        curve = _smooth(npz[key_s])
        style = styles.get(name, {})
        props = solver_plot_props(style, marker=False)
        lbl = style.get("label", name)
        ax_top.plot(x, curve, label=lbl, **props)
        if consensus is not None:
            ax_bot.plot(x, curve - consensus, label=lbl, **props)
    if consensus is not None:
        ax_bot.axhline(0, color="0.55", lw=0.8, ls="--")


def _agreement_plot_curves(
    cfg: Any,
    npz: Any,
    sample_consensus: Any,
    solver_names: Any,
    sweep_vals: Any,
    sweep_key: Any,
    styles: Any,
    out_dir: Any,
    save: Any,
    *,
    agreement_xlabel: str,
    agreement_ylabel: str,
) -> Any:
    """1-D observable agreement (RDF g(r), P(k), etc.) with residual row."""
    n_vals = len(sweep_vals)
    x = npz["x_axis"] if "x_axis" in npz else np.arange(len(sample_consensus))
    fig_agr, ax_grid = plt.subplots(
        2,
        n_vals,
        figsize=(4 * n_vals, 7),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    for i, val in enumerate(sweep_vals):
        ax_top = ax_grid[0, i]
        ax_bot = ax_grid[1, i]
        _agreement_curve_panel(ax_top, ax_bot, x, npz, solver_names, i, styles)
        ax_top.set_title(f"{sweep_key}={val:.3g}")
        ax_bot.set_xlabel(agreement_xlabel)
        if i == 0:
            ax_top.set_ylabel(agreement_ylabel)
            ax_bot.set_ylabel(f"Δ {agreement_ylabel}")
    fig_agr.suptitle(f"{cfg.name} — agreement ({agreement_ylabel})")
    fig_shared_legend(fig_agr, list(ax_grid.flat))
    if save:
        save_fig(fig_agr, "curves", out_dir)
    return fig_agr


def _agreement_raw_fields(
    cfg: Any,
    npz: Any,
    solver_names: Any,
    sweep_vals: Any,
    sweep_key: Any,
    styles: Any,
    f2d: Any,
    out_dir: Any,
    save: Any,
    *,
    field_cmap: str,
    field_symmetric: bool,
) -> None:
    """Raw field grid: rows=solvers, cols=sweep values."""
    raw_panels = []
    for name in solver_names:
        for i, val in enumerate(sweep_vals):
            key_s = f"{name}_{i}"
            if key_s not in npz or not _is_field(npz[key_s]):
                continue
            label = f"{styles.get(name, {}).get('label', name)}\n{sweep_key}={val:.3g}"
            raw_panels.append((label, f2d(npz[key_s])))
    if not raw_panels:
        return
    fig_raw = field_grid(
        raw_panels,
        f"{cfg.name} — solver fields",
        ncols=len(sweep_vals),
        **_field_grid_kw(field_cmap, field_symmetric),
    )
    if save:
        save_fig(fig_raw, "fields_raw", out_dir)


def _agreement_error_fields(
    cfg: Any,
    npz: Any,
    solver_names: Any,
    sweep_vals: Any,
    sweep_key: Any,
    styles: Any,
    reference_label: Any,
    f2d: Any,
    out_dir: Any,
    save: Any,
    *,
    field_cmap: str,
    field_symmetric: bool,
) -> Any:
    """Field error grid: rows=solvers, cols=sweep values."""
    panels = []
    for name in solver_names:
        for i, val in enumerate(sweep_vals):
            key_s = f"{name}_{i}"
            key_c = f"consensus_{i}"
            if key_s not in npz or key_c not in npz:
                continue
            if not _is_field(npz[key_s]) or not _is_field(npz[key_c]):
                continue
            err = f2d(npz[key_s]) - f2d(npz[key_c])
            label = f"{styles.get(name, {}).get('label', name)}\n{sweep_key}={val:.3g}"
            panels.append((label, err))
    if not panels:
        return None
    _ref_desc = "analytic solution" if reference_label == "analytic" else "consensus"
    fig_err = field_grid(
        panels,
        f"{cfg.name} — field error vs {_ref_desc}",
        ncols=len(sweep_vals),
        **_field_grid_kw(field_cmap, field_symmetric),
    )
    if save:
        save_fig(fig_err, "fields", out_dir)
    return fig_err


# Greek-letter sweep keys → LaTeX math; everything else is wrapped in $…$.
_AGREEMENT_MATH_LABELS = {
    "nu": r"$\nu$",
    "mu": r"$\mu$",
    "rho": r"$\rho$",
    "sigma": r"$\sigma$",
    "alpha": r"$\alpha$",
}


def _agreement_math_label(sweep_key: str) -> str:
    """Render *sweep_key* as a math-mode axis label."""
    if not sweep_key:
        return ""
    return _AGREEMENT_MATH_LABELS.get(sweep_key, f"${sweep_key}$")


def _alias_to_display_name(cfg: Any) -> dict[str, str]:
    """Map ``SOLVER_STYLES`` alias keys → ``cfg.solvers[i].name`` display names.

    ``result.json`` keys per-solver entries by the display name (``spec.name``),
    but ``NS_ORDER`` / ``FEM_ORDER`` are alias-keyed (``spec.dir.replace("-","_")``).
    Bridge the two so the paper plots can iterate the canonical order while
    looking up by the on-disk key.
    """
    return {s.dir.replace("-", "_"): s.name for s in cfg.solvers}


def _agreement_paper_plot_curves(ax: Any, data: dict, seen: set[str], cfg: Any) -> None:
    """Plot per-solver error-vs-sweep curves onto ``ax``.

    Walks :data:`NS_ORDER`, drawing every solver that produced at least
    one valid finite positive error. Updates ``seen`` so the caller can
    build a legend covering only solvers that actually appear.
    """
    by_param = data.get("by_param", {})
    if not by_param:
        return
    params = sorted(by_param.keys(), key=float)
    alias_to_name = _alias_to_display_name(cfg)

    for solver in NS_ORDER:
        display_name = alias_to_name.get(solver)
        if display_name is None:
            continue
        _label, color, ls, mk = solver_props(solver)
        xs, ys = [], []
        for p in params:
            entry = by_param[p].get(display_name)
            if isinstance(entry, dict):
                err = entry.get("error")
                if (
                    err is not None
                    and isinstance(err, float)
                    and np.isfinite(err)
                    and err > 0
                ):
                    xs.append(float(p))
                    ys.append(err)
        if not xs:
            continue
        ax.semilogy(
            xs,
            ys,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4,
            markeredgewidth=0,
            linewidth=1.6,
        )
        seen.add(solver)


def _agreement_paper_style_axis(
    ax: Any,
    *,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool,
    has_data: bool = True,
) -> None:
    """Apply consistent axis labels / ticks to one panel."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if not has_data:
        ax.text(
            0.5,
            0.5,
            "no positive data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
            fontsize=7,
        )
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        return
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if log_x:
        ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def _agreement_paper_figure(
    cfg: Problem, *, exp_key: str, suffix: str, save: bool, out_dir: Path
) -> plt.Figure:
    """Paper-styled single-experiment agreement figure.

    Writes ``agreement.pdf`` next to ``result.json``. Single axis plots
    per-solver error vs the sweep parameter (sweep_key inferred from the
    result file), styled with the paper palette + rcParams.
    """
    plt.rcParams.update(RCPARAMS)

    data = load_json(out_dir / "result.json")
    sweep_key = data.get("sweep_key", "param")
    reference_label = data.get("reference_label", "consensus")
    ref_desc = "analytic" if reference_label == "analytic" else "consensus"

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, TEXTWIDTH * 0.4), dpi=300)
    fig.subplots_adjust(bottom=0.30, left=0.18, right=0.95, top=0.88)

    seen: set[str] = set()
    _agreement_paper_plot_curves(ax, data, seen, cfg)

    x_label = _agreement_math_label(sweep_key)
    _agreement_paper_style_axis(
        ax,
        title=f"{cfg.category_label or cfg.name} — vs {sweep_key}",
        x_label=x_label,
        y_label=f"Error vs {ref_desc}",
        log_x=True,
        has_data=bool(seen),
    )

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen and s in SOLVER_STYLES]
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=6.5,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / "agreement.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def _agreement_convergence(
    cfg: Any, exp_key: Any, suffix: Any, save: Any, out_dir: Any
) -> None:
    """Error vs sweep param line chart (paper styling).

    Renders the canonical single-experiment paper-styled figure
    (``agreement.pdf`` next to ``result.json``). The previous
    ``convergence.png`` / ``errors.png`` shared-style files are
    superseded by this paper-styled output.
    """
    fig = _agreement_paper_figure(
        cfg, exp_key=exp_key, suffix=suffix, save=save, out_dir=out_dir
    )
    plt.close(fig)


def _agreement_power_spectra(
    cfg: Any,
    npz: Any,
    solver_names: Any,
    sweep_vals: Any,
    sweep_key: Any,
    styles: Any,
    out_dir: Any,
    save: Any,
    *,
    power_spectrum_fn: Any,
    domain_extent: float,
) -> None:
    """Power spectra (one subplot per sweep value, all solvers overlaid)."""
    if power_spectrum_fn is None:
        return
    n_vals = len(sweep_vals)
    fig_ps, axes = subplots_grid(n_vals, panel_w=4, panel_h=4, sharey=True)
    for i, (val, ax) in enumerate(zip(sweep_vals, axes, strict=False)):
        for name in solver_names:
            key_s = f"{name}_{i}"
            if key_s not in npz or not _is_field(npz[key_s]):
                continue
            k, Pk = power_spectrum_fn(npz[key_s], domain_extent=domain_extent)
            style = styles.get(name, {})
            ax.loglog(
                k,
                Pk,
                label=style.get("label", name),
                **solver_plot_props(style, marker=False),
            )
        ax.set_title(f"{sweep_key}={val:.3g}")
        ax.set_xlabel("k  [h/Mpc]")
        if i == 0:
            ax.set_ylabel("P(k)  [(Mpc/h)³]")
    fig_ps.suptitle(f"{cfg.name} — matter power spectrum")
    fig_shared_legend(fig_ps, axes)
    if save:
        save_fig(fig_ps, "power_spectra", out_dir)


def plot_agreement(
    cfg: Problem,
    *,
    field_to_2d: Any = None,
    output_key: str = "output",
    domain_extent: float = 2 * np.pi,
    resolution_key: str = "N",
    units: dict | None = None,
    agreement_xlabel: str = "x",
    agreement_ylabel: str = "value",
    pairwise_xlabel: str = "k",
    pairwise_ylabels: dict | None = None,
    field_cmap: str = "RdBu_r",
    field_symmetric: bool = True,
    diagnostic_fields: bool = True,
    power_spectrum_fn: Any = None,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "agreement",
    **_kw: Any,
) -> Any:
    """Field-error grid (rows=solvers × cols=sweep values) + optional power spectra."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    fields_path = out_dir / "fields.npz"
    data = load_json(out_dir / "result.json")
    sweep_key = data.get("sweep_key", "param")
    reference_label = data.get("reference_label", "consensus")
    styles = solver_styles(cfg)
    f2d = _resolve_field_to_2d(field_to_2d)

    npz = try_load_npz(fields_path)
    sweep_vals = npz["sweep_values"].tolist()
    solver_names = npz["solver_names"].tolist()

    # ── detect comparison type from consensus shape ───────────────────────────
    sample_consensus = npz.get("consensus_0", None)

    # Scalar outputs (ndim == 0): plot the scalar value vs sweep parameter.
    if sample_consensus is not None and sample_consensus.ndim == 0:
        return _agreement_plot_scalar(
            cfg,
            npz,
            solver_names,
            sweep_vals,
            sweep_key,
            styles,
            out_dir,
            save,
            output_key=output_key,
            units=units,
        )

    if sample_consensus is not None and sample_consensus.ndim == 1:
        # 1-D observable agreement (e.g. RDF g(r) vs r, or P(k) vs k)
        # Layout: row 0 = absolute curves, row 1 = residual (solver − consensus).
        # The residual row makes %-level differences visible — mandatory in
        # code-comparison papers (Euclid, HACC, etc.) where the absolute panel
        # shows everything agreeing.
        return _agreement_plot_curves(
            cfg,
            npz,
            sample_consensus,
            solver_names,
            sweep_vals,
            sweep_key,
            styles,
            out_dir,
            save,
            agreement_xlabel=agreement_xlabel,
            agreement_ylabel=agreement_ylabel,
        )

    _agreement_raw_fields(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        f2d,
        out_dir,
        save,
        field_cmap=field_cmap,
        field_symmetric=field_symmetric,
    )
    fig_err = _agreement_error_fields(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        reference_label,
        f2d,
        out_dir,
        save,
        field_cmap=field_cmap,
        field_symmetric=field_symmetric,
    )
    _agreement_convergence(cfg, exp_key, suffix, save, out_dir)
    _agreement_power_spectra(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        out_dir,
        save,
        power_spectrum_fn=power_spectrum_fn,
        domain_extent=domain_extent,
    )
    return fig_err


def plot_forward_fields(
    cfg: Problem,
    *,
    field_to_2d: Any = None,
    domain_extent: float = 2 * np.pi,
    field_cmap: str = "RdBu_r",
    field_symmetric: bool = True,
    power_spectrum_fn: Any = None,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "cylinder",
    **_kw: Any,
) -> Any:
    """Field grids (rows=solvers × cols=sweep values) + optional power spectra.

    Like :func:`plot_agreement` but without the convergence-vs-sweep
    figure — for experiments where the sweep is a per-physics
    visualization knob (e.g. cylinder wake at several viscosities)
    rather than a quantitative convergence axis.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    fields_path = out_dir / "fields.npz"
    data = load_json(out_dir / "result.json")
    sweep_key = data.get("sweep_key", "param")
    reference_label = data.get("reference_label", "consensus")
    styles = solver_styles(cfg)
    f2d = _resolve_field_to_2d(field_to_2d)

    npz = try_load_npz(fields_path)
    sweep_vals = npz["sweep_values"].tolist()
    solver_names = npz["solver_names"].tolist()

    _agreement_raw_fields(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        f2d,
        out_dir,
        save,
        field_cmap=field_cmap,
        field_symmetric=field_symmetric,
    )
    fig_err = _agreement_error_fields(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        reference_label,
        f2d,
        out_dir,
        save,
        field_cmap=field_cmap,
        field_symmetric=field_symmetric,
    )
    _agreement_power_spectra(
        cfg,
        npz,
        solver_names,
        sweep_vals,
        sweep_key,
        styles,
        out_dir,
        save,
        power_spectrum_fn=power_spectrum_fn,
        domain_extent=domain_extent,
    )
    return fig_err


# ── physical_laws ──────────────────────────────────────────────────────────────

# Shared sweep / metric constants for the NS 3×3 grid.
_PA_SWEEPS = [
    ("vs_N", True, True),
    ("vs_nu", True, False),
    ("vs_steps", False, False),
]
_PA_METRICS = [
    ("analytic_error", "Analytic error", True),
    ("divergence_rms", "Divergence RMS", True),
    ("kinetic_energy", "Kinetic energy", False),
]
_PA_SWEEP_XLABELS = {
    "vs_N": "Elements",
    "vs_nu": r"$\nu$",
    "vs_steps": "Steps",
}
_PA_ROW_LABELS = {
    "vs_N": "vs N",
    "vs_nu": r"vs $\nu$",
    "vs_steps": "vs steps",
}

_PA_KE0 = 0.25

# Per-problem FEM specs: which metric + xlabel + ylabel + title to render.
_PA_FEM_SPECS: dict[str, dict] = {
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
_PA_NS_DOMAINS: list[tuple[str, str]] = [
    ("ns-grid", "2D NS — physical accuracy"),
    ("ns-3d-grid", "3D NS — physical accuracy"),
]


def _pa_n_to_elements(N: int, subdir: str) -> int:
    if subdir == "ns-grid":
        return N**2
    if subdir == "ns-3d-grid":
        return N**3
    return N


def _pa_set_axis_ticks(
    ax: Any, vals: list, is_log_x: bool, is_log_y: bool, is_elements: bool = False
) -> None:
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


def _pa_ke_analytic(sweep_key: str, params: list[str], phys: dict, subdir: str) -> Any:
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
        ke = _PA_KE0 * np.exp(-4.0 * nu * t)
        x = _pa_n_to_elements(int(p), subdir) if sweep_key == "vs_N" else raw
        xs.append(x)
        ys.append(ke)
    return xs, ys


def _pa_plot_ns_row(
    axes_row: Any,
    sweep_key: str,
    log_x: bool,
    use_elements: bool,
    data: dict,
    cfg: Any,
    ns_seen: set[str],
    row_is_top: bool,
) -> None:
    """Render one row of the 3×3 NS grid for a single sweep (3 metrics)."""
    by_param = data["by_param"]
    params = sorted(by_param.keys(), key=float)
    phys = data.get("params", {}).get("physics", {})
    subdir = cfg.name

    alias_to_name = _alias_to_display_name(cfg)
    name_to_alias = {v: k for k, v in alias_to_name.items()}
    _EXCLUDED = {"fenics_ns", "su2"}
    all_solvers: list[str] = []
    for pdata in by_param.values():
        for s in pdata:
            if s not in all_solvers and name_to_alias.get(s) not in _EXCLUDED:
                all_solvers.append(s)

    for col, (metric_key, metric_label, log_y) in enumerate(_PA_METRICS):
        ax = axes_row[col]
        x_all: list[float] = []

        for solver in all_solvers:
            alias = name_to_alias.get(solver, solver)
            _label, color, ls, mk = solver_props(alias)
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
                    x = _pa_n_to_elements(int(p), subdir) if use_elements else raw_x
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
                if alias in NS_ORDER:
                    ns_seen.add(alias)

        if metric_key == "kinetic_energy":
            ke_ref = _pa_ke_analytic(sweep_key, params, phys, subdir)
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
            ax.set_ylabel(_PA_ROW_LABELS[sweep_key], fontsize=9)
        ax.set_xlabel(_PA_SWEEP_XLABELS[sweep_key])

        _pa_set_axis_ticks(ax, x_all, log_x, log_y, is_elements=use_elements)


def _pa_plot_ns_grid(
    cfg: Any,
    sweeps_data: dict[str, dict],
    domain_title: str,
    out_path: Path | None,
) -> plt.Figure:
    """3×3 NS grid: rows=sweep (vs_N, vs_nu, vs_steps), cols=metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.85))
    fig.suptitle(domain_title, fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.22, wspace=0.35, hspace=0.55)

    ns_seen: set[str] = set()

    for row, (sweep_key, log_x, use_elements) in enumerate(_PA_SWEEPS):
        data = sweeps_data.get(sweep_key)
        if data is None:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue
        _pa_plot_ns_row(
            axes[row],
            sweep_key,
            log_x,
            use_elements,
            data,
            cfg,
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


def _pa_plot_fem_single(
    cfg: Any,
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

    alias_to_name = _alias_to_display_name(cfg)
    name_to_alias = {v: k for k, v in alias_to_name.items()}

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
        alias = name_to_alias.get(solver, solver)
        _label, color, ls, mk = solver_props(alias)
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
            if alias in FEM_ORDER:
                fem_seen.add(alias)

    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    _pa_set_axis_ticks(ax, list(x_vals), True, True)

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


def plot_physical_laws(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "physical_laws",
    **_kw: Any,
) -> Any:
    """Per-experiment physical-laws figure (paper styling).

    Reads ``<results>/<cfg.name>/forward/<exp_key>{suffix}/result.json``
    (or its sub-dirs when the experiment is a multi-run sweep) and writes
    a paper-styled PDF named ``physical_accuracy.pdf`` next to it.

    Detection:
      * If the experiment dir contains ``result.json``: single FEM panel
        (compliance / thermal_compliance vs total load), or 1×3 NS row
        when the experiment is a single sweep variant.
      * If it contains sub-dirs ``vs_N`` / ``vs_steps`` / ``vs_nu``: 3×3
        NS grid covering both axes (sweep × metric).
      * If neither, returns ``None`` (no data available).
    """
    plt.rcParams.update(RCPARAMS)

    out_dir = results_dir() / cfg.name / "forward" / f"{exp_key}{suffix}"

    # ── single-result layout (FEM or per-sub-key NS call) ────────────────────
    single_path = out_dir / "result.json"
    if single_path.exists():
        data = load_json(single_path)
        spec = _PA_FEM_SPECS.get(cfg.name)
        if spec is not None:
            return _pa_plot_fem_single(
                cfg, data, spec, out_dir / "physical_accuracy.pdf" if save else None
            )
        # NS sub-experiment (one sweep variant): render a 1×3 row.
        sweep_key = (exp_key + suffix).rsplit("/", 1)[-1] or "vs_param"
        sweep_meta = next(
            (
                (sk, log_x, use_el)
                for sk, log_x, use_el in _PA_SWEEPS
                if sk == sweep_key
            ),
            (sweep_key, True, sweep_key == "vs_N"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.32))
        fig.subplots_adjust(bottom=0.32, wspace=0.45)
        ns_seen: set[str] = set()
        _pa_plot_ns_row(
            axes,
            sweep_meta[0],
            sweep_meta[1],
            sweep_meta[2],
            data,
            cfg,
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
        (t for n, t in _PA_NS_DOMAINS if n == cfg.name),
        f"{cfg.category_label or cfg.name} — physical accuracy",
    )
    return _pa_plot_ns_grid(
        cfg,
        sweeps_data,
        title,
        out_dir / "physical_accuracy.pdf" if save else None,
    )
