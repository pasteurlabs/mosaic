"""Plots for the forward suite (agreement, physical_laws)."""

from __future__ import annotations

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


def _agreement_convergence(cfg, exp_key, suffix, save):
    """Error vs sweep param line chart — delegated to the paper module.

    Renders the canonical single-experiment paper-styled figure
    (``agreement.pdf`` next to ``result.json``) so the per-experiment plot
    and the paper figure share one implementation. The previous
    ``convergence.png`` / ``errors.png`` shared-style files are now
    superseded by this paper-styled output.
    """
    from mosaic.benchmarks.plots.paper import agreement as paper_agreement

    fig = paper_agreement.plot_experiment(
        cfg, exp_key=exp_key, suffix=suffix, save=save
    )
    plt.close(fig)


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
    _agreement_convergence(cfg, exp_key, suffix, save)
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


def plot_physical_laws(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "physical_laws",
    **_kw,
):
    """Per-experiment physical-laws figure (paper styling).

    Delegates to
    :func:`mosaic.benchmarks.plots.paper.physical_accuracy.plot_experiment`
    so the on-disk per-experiment PDF and the paper appendix figure stay
    byte-identical in layout.
    """
    from mosaic.benchmarks.plots.paper import (
        physical_accuracy as paper_physical_accuracy,
    )

    return paper_physical_accuracy.plot_experiment(
        cfg, exp_key=exp_key, suffix=suffix, save=save
    )
