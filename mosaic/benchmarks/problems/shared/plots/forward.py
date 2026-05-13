"""Plots for the forward suite (agreement, physical_laws)."""

from __future__ import annotations

import contextlib

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.style import (
    apply_style,
    field_grid,
    fig_shared_legend,
    save_fig,
    solver_plot_props,
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


def _resolve_field_to_2d(field_to_2d):
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
    cfg,
    npz,
    solver_names,
    sweep_vals,
    sweep_key,
    styles,
    out_dir,
    save,
    *,
    output_key: str,
    units: dict | None,
):
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


def _agreement_curve_panel(ax_top, ax_bot, x, npz, solver_names, i, styles):
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
    cfg,
    npz,
    sample_consensus,
    solver_names,
    sweep_vals,
    sweep_key,
    styles,
    out_dir,
    save,
    *,
    agreement_xlabel: str,
    agreement_ylabel: str,
):
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
    cfg,
    npz,
    solver_names,
    sweep_vals,
    sweep_key,
    styles,
    f2d,
    out_dir,
    save,
    *,
    field_cmap: str,
    field_symmetric: bool,
):
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
    *,
    field_cmap: str,
    field_symmetric: bool,
):
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


def _agreement_convergence(
    cfg,
    data,
    solver_names,
    sweep_key,
    styles,
    reference_label,
    exp_key,
    out_dir,
    save,
    *,
    units: dict | None,
):
    """Error vs sweep param line chart (baseline: convergence; agreement: error vs ν)."""
    by_param = data.get("by_param", {})
    if not by_param:
        return
    fig_conv, ax_conv = plt.subplots(figsize=(6, 5))
    param_vals = sorted(by_param.keys(), key=lambda v: float(v))
    all_errs: list[float] = []
    for name in solver_names:
        style = styles.get(name, {})
        xs, ys = [], []
        for pv in param_vals:
            entry = by_param[pv].get(name, {})
            if entry.get("valid") and entry.get("error") is not None:
                xs.append(float(pv))
                ys.append(float(entry["error"]))
                all_errs.append(float(entry["error"]))
        if xs:
            ax_conv.plot(
                xs, ys, label=style.get("label", name), **solver_plot_props(style)
            )
    if all_errs:
        ax_conv.set_yscale("log")
        if exp_key == "baseline":
            with contextlib.suppress(Exception):
                ax_conv.set_xscale("log")
    ref_desc = (
        "analytic solution" if reference_label == "analytic" else "solver consensus"
    )
    ax_conv.set_xlabel(unit_label(sweep_key, units))
    ax_conv.set_ylabel(f"Relative L₂ error vs {ref_desc}")
    title = (
        f"{cfg.name} — spatial convergence (steps=1)"
        if exp_key == "baseline"
        else f"{cfg.name} — inter-solver agreement vs {sweep_key} (ref: {ref_desc})"
    )
    ax_conv.set_title(title)
    ax_conv.grid(True, alpha=0.3, which="both")
    fig_shared_legend(fig_conv, [ax_conv])
    if save:
        save_fig(
            fig_conv, "convergence" if exp_key == "baseline" else "errors", out_dir
        )


def _agreement_power_spectra(
    cfg,
    npz,
    solver_names,
    sweep_vals,
    sweep_key,
    styles,
    out_dir,
    save,
    *,
    power_spectrum_fn,
    domain_extent: float,
):
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


def plot_agreement(  # noqa: PLR0913 — explicit-deps signature
    cfg: Problem,
    *,
    field_to_2d=None,
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
    power_spectrum_fn=None,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "agreement",
    **_kw,
):
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
    _agreement_convergence(
        cfg,
        data,
        solver_names,
        sweep_key,
        styles,
        reference_label,
        exp_key,
        out_dir,
        save,
        units=units,
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


def _plot_physical_laws_single(cfg, data, out_dir, styles, save, *, units: dict | None):
    """Render one physical-laws sweep result (one sweep key)."""
    by_param = data.get("by_param", {})
    sweep_key = data.get("sweep_key", "param")
    vals = sorted(by_param.keys(), key=float)
    if not vals:
        return None

    diag_names: list[str] = []
    for val in vals:
        for solver_data in by_param[val].values():
            if isinstance(solver_data, dict):
                diag_names = list(solver_data.keys())
                break
        if diag_names:
            break

    if not diag_names:
        return None

    x = np.array([float(v) for v in vals])
    n_diags = len(diag_names)
    fig, axes = subplots_grid(n_diags, panel_w=5, panel_h=4)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, dname in zip(axes, diag_names, strict=False):
        for spec in cfg.solvers:
            name = spec.name
            y = []
            for val in vals:
                sd = by_param[val].get(name)
                y.append(sd.get(dname, np.nan) if isinstance(sd, dict) else np.nan)
            y = np.array(y, dtype=float)
            style = styles.get(name, {})
            if not np.all(np.isnan(y)):
                ax.plot(
                    x,
                    y,
                    label=style.get("label", name),
                    **solver_plot_props(style),
                )
        ax.set_xlabel(unit_label(sweep_key, units))
        ax.set_ylabel(dname)
        ax.set_title(f"{dname} vs {sweep_key}")
        if np.any(x > 0):
            ax.set_xscale("log")

    fig.suptitle(f"{cfg.name} — physical laws ({sweep_key} sweep)")
    fig_shared_legend(fig, axes)
    if save:
        save_fig(fig, "physical_laws", out_dir)
    return fig


def plot_physical_laws(
    cfg: Problem,
    *,
    units: dict | None = None,
    save: bool = True,
    suffix: str = "",
    **_kw,
):
    """One subplot per diagnostic: value vs sweep parameter for each solver.

    Supports both single-run layout (result.json at top level) and multi-run
    layout (one named subdir per sweep, each with its own result.json).
    Also plots analytic_error vs sweep parameter when available.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"physical_laws{suffix}"
    styles = solver_styles(cfg)

    # Single-run layout: result.json directly in out_dir
    try:
        data = load_json(out_dir / "result.json")
    except FileNotFoundError:
        data = None
    if data is not None:
        return _plot_physical_laws_single(cfg, data, out_dir, styles, save, units=units)

    # Multi-run layout: one named subdir per sweep
    figs = []
    if out_dir.is_dir():
        for sub in sorted(out_dir.iterdir()):
            if not sub.is_dir():
                continue
            try:
                sub_data = load_json(sub / "result.json")
            except FileNotFoundError:
                continue
            if sub_data is not None:
                fig = _plot_physical_laws_single(
                    cfg, sub_data, sub, styles, save, units=units
                )
                if fig is not None:
                    figs.append(fig)
    return figs if figs else None
