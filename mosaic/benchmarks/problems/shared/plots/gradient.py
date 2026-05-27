# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plots for the gradient evaluation suite (fd_check, param_sweep, jacobian_svd)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    apply_style,
    dedup_handles,
    fig_shared_legend,
    make_handle,
    resolve_solver_alias,
    save_fig,
    solver_plot_props,
    solver_props,
    solver_styles,
)

apply_style()

_SUITE = "gradient"


# ── fd_check helpers ──────────────────────────────────────────────────────────

# Solvers blacklisted from FD-check figures (e.g. variants that don't
# implement VJP at all and would just clutter the legend).
_FD_CHECK_BLACKLIST = {"fenics_ns", "su2"}


def _fd_check_plot_curves(
    ax_err: Any, ax_cos: Any, data: dict, seen: dict[str, set]
) -> None:
    """Plot per-solver rel-error + cosine curves into ``ax_err``/``ax_cos``."""
    for solver, sdata in data["by_solver"].items():
        alias = resolve_solver_alias(solver)
        # Blacklist accepts both display-name and alias forms.
        if solver in _FD_CHECK_BLACKLIST or (
            alias is not None and alias in _FD_CHECK_BLACKLIST
        ):
            continue
        eps_sweep = sdata.get("eps_sweep") or {}
        if not eps_sweep:
            continue
        epsilons = sorted(eps_sweep.keys(), key=float)
        eps_f = [float(e) for e in epsilons]

        rel_mean = [float(np.mean(eps_sweep[e]["rel_error"])) for e in epsilons]
        # ``1 - cosine`` makes a 4-decade log scale meaningful when most
        # solvers cluster near 1; clamp the noise floor to 1e-9.
        cos_vals = [max(1 - float(eps_sweep[e]["cosine"]), 1e-9) for e in epsilons]

        _label, color, ls, mk = solver_props(alias or solver)
        kw: dict[str, Any] = {
            "color": color,
            "linestyle": ls,
            "marker": mk,
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.6,
        }
        ax_err.loglog(eps_f, rel_mean, **kw)
        ax_cos.loglog(eps_f, cos_vals, **kw)

        if alias is not None and alias in NS_ORDER:
            seen["ns"].add(alias)
        if alias is not None and alias in FEM_ORDER:
            seen["fem"].add(alias)


def _fd_check_style_axes(
    ax_err: Any, ax_cos: Any, *, title: str, ylabel_left: bool
) -> None:
    """Apply consistent axis labels / ticks to one (err, cos) column."""
    ax_err.set_title(title)
    ax_err.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_err.set_ylabel("Relative FD error" if ylabel_left else "")
    ax_err.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_err.yaxis.set_minor_locator(mticker.NullLocator())

    ax_cos.set_xlabel(r"Perturbation size $\varepsilon$")
    ax_cos.set_ylabel(
        r"$1 - \cos(\nabla_\mathrm{AD},\, \nabla_\mathrm{FD})$" if ylabel_left else ""
    )
    ax_cos.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax_cos.yaxis.set_minor_locator(mticker.NullLocator())


def _fd_check_paper_figure(
    cfg: Problem, *, exp_key: str, suffix: str, save: bool, out_dir: Any
) -> plt.Figure:
    """Paper-styled 1×2 rel-error + cosine figure for a single fd_check run."""
    plt.rcParams.update(RCPARAMS)

    data = load_json(out_dir / "result.json")

    fig, (ax_err, ax_cos) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.3), dpi=300
    )
    fig.subplots_adjust(bottom=0.38, wspace=0.45)

    seen = {"ns": set(), "fem": set()}
    _fd_check_plot_curves(ax_err, ax_cos, data, seen)
    _fd_check_style_axes(
        ax_err,
        ax_cos,
        title=f"FD check — {cfg.category_label or cfg.name}",
        ylabel_left=True,
    )

    handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen["ns"] and s in SOLVER_STYLES]
        + [make_handle(s) for s in FEM_ORDER if s in seen["fem"] and s in SOLVER_STYLES]
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 7),
            fontsize=6.0,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / "fd_check.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


# ── G0: finite-difference check ───────────────────────────────────────────────


def plot_fd_check(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "fd_check",
    **_kw: Any,
) -> Any:
    """FD-check experiment plot: rel-error + cosine curves vs ε."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    return _fd_check_paper_figure(
        cfg, exp_key=exp_key, suffix=suffix, save=save, out_dir=out_dir
    )


# ── shared U-curve helper ─────────────────────────────────────────────────────


def _plot_ucurve_overlay(
    by_solver: dict,
    sweep_keys: Any,
    sweep_label: str,
    styles: dict,
    title_prefix: str,
    ncols: int = 4,
) -> plt.Figure:
    """Overlay U-curves: one panel per sweep value, all solvers overlaid.

    Each panel shows rel_error_mean ± std (shaded) vs ε for every solver,
    with a shared legend below the figure.
    """
    n_panels = len(sweep_keys)
    n_cols = min(ncols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False
    )

    solver_names = list(by_solver.keys())

    for idx, key in enumerate(sweep_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for name in solver_names:
            props = solver_plot_props(styles[name])

            sweep = by_solver[name][key]["eps_sweep"]
            eps_f = sorted(sweep.keys(), key=float)
            eps_fl = [float(e) for e in eps_f]
            re_m = [sweep[e]["rel_error_mean"] for e in eps_f]

            ax.loglog(eps_fl, re_m, label=styles[name]["label"], **props)

        ax.set_xlabel("ε")
        ax.set_ylabel("Relative FD error")
        ax.set_title(f"{sweep_label} = {key}", fontsize=9)

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title_prefix, fontsize=10)
    fig_shared_legend(fig, axes)
    return fig


# ── G2a: parameter sweep ─────────────────────────────────────────────────────


def plot_param_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "param_sweep",
    **_kw: Any,
) -> Any:
    """Param-sweep experiment plot: per-sweep-value ε U-curve grid."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)
    sweep_key = data.get("sweep_key", "param")

    param_vals = sorted(next(iter(data["by_solver"].values())).keys(), key=float)
    fig_u = _plot_ucurve_overlay(
        data["by_solver"],
        param_vals,
        sweep_key,
        styles,
        f"{cfg.name} — G2a ε U-curves ({sweep_key} sweep)",
        ncols=len(param_vals),
    )
    if save:
        save_fig(fig_u, "ucurves", out_dir)
    return fig_u


# ── G3: Jacobian SVD ──────────────────────────────────────────────────────────


# ── G2c: horizon sweep ────────────────────────────────────────────────────────


def plot_horizon_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "horizon_sweep",
    **_kw: Any,
) -> Any:
    """Horizon-sweep experiment plot: per-horizon ε U-curve grid.

    Writes only ``ucurves.pdf`` — one panel per horizon T, all solvers
    overlaid on the FD-error vs ε U-curve. Summary curves, per-solver
    error panels, and gradient-field renderings were removed; the
    U-curves carry the full convergence story.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    step_keys = sorted(next(iter(data["by_solver"].values())).keys(), key=int)
    fig_u = _plot_ucurve_overlay(
        data["by_solver"],
        step_keys,
        "steps",
        styles,
        f"{cfg.name} — G2c ε U-curves (horizon sweep)",
        ncols=4,
    )
    if save:
        save_fig(fig_u, "ucurves", out_dir)
    return fig_u


def plot_jacobian_svd_comparison(
    cfg: Problem,
    exp_keys: list[str] | None = None,
    *,
    save: bool = True,
    out_name: str = "jacobian_svd_comparison",
    **_kw: Any,
) -> Any:
    """Overlay per-solver singular value spectra across multiple jacobian_svd variants.

    One subplot per solver; one line per variant (e.g. different nu or steps).
    Variants are auto-labelled from their physics config when available, falling
    back to the exp_key name.
    """
    if exp_keys is None:
        exp_keys = [
            "jacobian_svd",
            "jacobian_svd_nu01",
            "jacobian_svd_steps20",
            "jacobian_svd_steps40",
        ]

    # Variant label overrides derived from known naming conventions
    _VARIANT_LABELS = {
        "jacobian_svd": "ν=0.001  T=0.5s",
        "jacobian_svd_nu01": "ν=0.01   T=0.5s",
        "jacobian_svd_steps20": "ν=0.001  T=1.0s",
        "jacobian_svd_steps40": "ν=0.001  T=2.0s",
    }
    _VARIANT_STYLES = [
        {"color": "#1f77b4", "linestyle": "-"},
        {"color": "#ff7f0e", "linestyle": "--"},
        {"color": "#2ca02c", "linestyle": "-."},
        {"color": "#d62728", "linestyle": ":"},
    ]

    # Load available variants
    variants: list[tuple[str, dict]] = []
    for exp_key in exp_keys:
        result_path = results_dir() / cfg.name / _SUITE / exp_key / "result.json"
        if not result_path.exists():
            continue
        data = load_json(result_path)
        if not data.get("per_solver_spectra"):
            continue
        variants.append((exp_key, data))

    if not variants:
        return None

    # Collect all solver names across variants
    all_solvers: list[str] = []
    for _, data in variants:
        for s in data.get("solver_names", []):
            if s not in all_solvers:
                all_solvers.append(s)

    styles = solver_styles(cfg)
    n_solvers = len(all_solvers)
    ncols = min(3, n_solvers)
    nrows = math.ceil(n_solvers / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
    )

    for idx, solver in enumerate(all_solvers):
        ax = axes[idx // ncols][idx % ncols]
        solver_style = styles.get(solver, {})
        color = solver_style.get("color", "#888888")
        label_base = solver_style.get("label", solver)

        for vi, (exp_key, data) in enumerate(variants):
            spectra = data.get("per_solver_spectra", {})
            if solver not in spectra:
                continue
            spec = spectra[solver]
            er = data.get("per_solver_eff_rank", {}).get(solver, float("nan"))
            cond = data.get("per_solver_cond", {}).get(solver, float("nan"))
            var_label = _VARIANT_LABELS.get(exp_key, exp_key)
            vstyle = _VARIANT_STYLES[vi % len(_VARIANT_STYLES)]
            n_modes = len(spec)
            modes = list(range(1, n_modes + 1))
            marker = "o" if n_modes <= 32 else ""
            ax.semilogy(
                modes,
                spec,
                f"{marker}{vstyle['linestyle']}",
                color=vstyle["color"],
                markersize=4 if marker else 0,
                linewidth=1.5,
                label=f"{var_label}  r={er:.0f}  κ={cond:.1e}",
            )

        ax.axhline(1e-1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(1e-4, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{label_base}", color=color, fontsize=10)
        ax.set_xlabel("Mode index i")
        ax.set_ylabel("σᵢ / σ₁")
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_solvers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"{cfg.name} — Jacobian SVD: per-solver spectra comparison", fontsize=11
    )
    fig.tight_layout()

    if save:
        out_dir = results_dir() / cfg.name / _SUITE
        save_fig(fig, out_name, out_dir)
    return fig
