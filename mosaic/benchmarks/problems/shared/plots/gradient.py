# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plots for the gradient evaluation suite (fd_check, param_sweep, jacobian_svd)."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    legacy_by_solver,
    load_json,
    results_dir,
    try_load_npz,
    v1_to_legacy,
)
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
    grad_magnitude_2d,
    make_handle,
    resolve_solver_alias,
    save_fig,
    solver_plot_props,
    solver_props,
    solver_styles,
    unit_label,
    vorticity_2d,
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


def _fd_check_figure(
    cfg: Problem, *, exp_key: str, suffix: str, save: bool, out_dir: Any
) -> plt.Figure:
    """Styled 1×2 rel-error + cosine figure for a single fd_check run."""
    plt.rcParams.update(RCPARAMS)

    data = v1_to_legacy(load_json(out_dir / "result.json"))

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
        out = out_dir / "fd_check.png"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


# ── jacobian_svd helpers ──────────────────────────────────────────────────────

_SVD_VARIANT_STYLES = [
    {"color": "#0077BB", "linestyle": "-"},
    {"color": "#CC3311", "linestyle": "--"},
    {"color": "#009988", "linestyle": "-."},
    {"color": "#EE7733", "linestyle": ":"},
]

# Piecewise-log y-scale: normal log above this threshold, compressed log below.
_SVD_YSCALE_THRESHOLD = 1e-3
_SVD_YSCALE_COMPRESS = 0.18


def _svd_piecewise_log_forward(y: Any) -> Any:
    y = np.asarray(y, dtype=float)
    log_thresh = np.log10(_SVD_YSCALE_THRESHOLD)
    safe = np.where(y > 0, y, 1e-30)
    log_y = np.log10(safe)
    return np.where(
        log_y >= log_thresh,
        log_y,
        log_thresh + _SVD_YSCALE_COMPRESS * (log_y - log_thresh),
    )


def _svd_piecewise_log_inverse(t: Any) -> Any:
    t = np.asarray(t, dtype=float)
    log_thresh = np.log10(_SVD_YSCALE_THRESHOLD)
    return np.where(
        t >= log_thresh,
        np.power(10.0, t),
        np.power(10.0, log_thresh + (t - log_thresh) / _SVD_YSCALE_COMPRESS),
    )


def _svd_variant_label(phys: dict) -> str:
    nu = phys.get("nu")
    steps = phys.get("steps")
    dt = phys.get("dt")
    if nu is None or steps is None or dt is None:
        return ""
    t = steps * dt
    visc = "low ν" if nu <= 0.001 else "high ν"
    horizon = f"T={t:.2g}s"
    return f"{visc}, {horizon}"


def _svd_solver_color_label(solver: str) -> tuple[str, str]:
    entry = SOLVER_STYLES.get(solver)
    if entry:
        return entry[1], entry[0]
    return "#888888", solver


def _svd_panels(fig: Any, axes: Any, variants: Any, solvers: Any, n_show: Any) -> list:
    """Fill axes panels; return legend handles for the variant lines."""
    legend_handles: list[mlines.Line2D] = []
    legend_built = False

    ncols = axes.shape[1]

    for idx, solver in enumerate(solvers):
        ax = axes[idx // ncols][idx % ncols]
        _color, label = _svd_solver_color_label(solver)
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

            vstyle = _SVD_VARIANT_STYLES[vi % len(_SVD_VARIANT_STYLES)]
            vlabel = _svd_variant_label(phys)

            mk = "o" if n <= 32 else ""
            (_line,) = ax.plot(
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

        ax.set_yscale(
            "function",
            functions=(_svd_piecewise_log_forward, _svd_piecewise_log_inverse),
        )
        if np.isfinite(y_min_data) and y_min_data > 0:
            y_floor = 10 ** (np.floor(np.log10(y_min_data)) - 0.5)
            ax.set_ylim(bottom=y_floor, top=2.0)
        # Faint divider at the scale-break to flag the change.
        ax.axhline(
            _SVD_YSCALE_THRESHOLD,
            color="0.7",
            linestyle=":",
            linewidth=0.6,
            zorder=0,
        )

        ax.set_title(label)
        ax.set_xlabel("Mode index $i$")
        if idx % ncols == 0:
            ax.set_ylabel(r"$\sigma_i\,/\,\sigma_1$")
        else:
            ax.set_ylabel("")

        ax.yaxis.set_major_locator(mticker.FixedLocator([1.0, 1e-3, 1e-6, 1e-9]))
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    return legend_handles


def _jacobian_svd_figure(
    cfg: Problem,
    *,
    exp_key: str,
    suffix: str,
    save: bool,
    out_dir: Any,
    n_show: int | None = None,
) -> plt.Figure | None:
    """Styled per-solver SVD spectrum grid for a single jacobian_svd run.

    Returns ``None`` when ``per_solver_spectra`` is missing (e.g. scalar
    outputs) so callers can fall back to alternate diagnostics.
    """
    plt.rcParams.update(RCPARAMS)

    data = v1_to_legacy(load_json(out_dir / "result.json"))

    per_solver_spectra = data.get("per_solver_spectra") or {}
    if not per_solver_spectra:
        return None

    variants = [(exp_key + suffix, data)]
    solvers = list(per_solver_spectra.keys())

    if n_show is None:
        n_show = max(len(v) for v in per_solver_spectra.values())

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
        dpi=300,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.18)

    _handles = _svd_panels(fig, axes, variants, solvers, n_show)

    for row in range(nrows - 1):
        for col in range(ncols):
            axes[row][col].set_xlabel("")
            axes[row][col].tick_params(labelbottom=False)

    for idx in range(n_solvers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if save:
        out = out_dir / "jacobian_svd.png"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


# ── horizon_sweep helpers ─────────────────────────────────────────────────────

_HORIZON_FAILURE_MARKER = "X"
_HORIZON_FAILURE_LABEL = "NaN gradient"
_HORIZON_JITTER_LOG = 0.04

# Solvers blacklisted from horizon-sweep figures.
_HORIZON_EXCLUDED = {"fenics_ns", "su2", "openfoam"}


def _horizon_plot_curves(axes: Any, data: dict, seen: set[str]) -> bool:
    """Plot per-solver grad-norm / best-ε FD error / cosine into ``axes``."""
    from collections import defaultdict

    ax_gn, ax_err, ax_cos = axes
    # A ``steps`` sweep lands under ``by_steps`` (not ``by_solver``) post-v1;
    # collapse whichever group is present.
    by_solver = legacy_by_solver(data)
    if not by_solver:
        return False
    # ``by_solver`` is keyed by spec.name (display form); build an
    # alias→display map and iterate NS_ORDER (alias-keyed) against it.
    alias_to_display: dict[str, str] = {}
    for display_name in by_solver:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name
    ordered = [
        a for a in NS_ORDER if a in alias_to_display and a not in _HORIZON_EXCLUDED
    ]

    fail_at_step: dict[int, list[str]] = defaultdict(list)
    for alias in ordered:
        sv = by_solver[alias_to_display[alias]]
        for k, v in sv.items():
            gn = v.get("grad_norm", 1.0)
            if not np.isfinite(gn) or gn <= 0:
                fail_at_step[int(k)].append(alias)

    jitter_x: dict[tuple[str, int], float] = {}
    for step, solvers_here in fail_at_step.items():
        n = len(solvers_here)
        for i, sv in enumerate(solvers_here):
            if n == 1:
                jitter_x[(sv, step)] = float(step)
            else:
                log_off = (2 * i / (n - 1) - 1) * _HORIZON_JITTER_LOG
                jitter_x[(sv, step)] = step * 10**log_off

    failure_seen = False

    for alias in ordered:
        sv = by_solver[alias_to_display[alias]]
        _label, color, ls, _mk = solver_props(alias)

        step_keys = sorted(sv.keys(), key=int)
        ok_steps, ok_gn, ok_err, ok_cos = [], [], [], []
        fail_steps = []

        for k in step_keys:
            v = sv[k]
            gn = v.get("grad_norm", float("nan"))
            eps_sweep = v.get("eps_sweep", {})
            if eps_sweep:
                best_err = min(float(e["rel_error_mean"]) for e in eps_sweep.values())
                best_cos = max(float(e["cosine_mean"]) for e in eps_sweep.values())
            else:
                best_err = float("nan")
                best_cos = float("nan")

            if np.isfinite(gn) and gn > 0 and np.isfinite(best_err) and best_err > 0:
                ok_steps.append(int(k))
                ok_gn.append(gn)
                ok_err.append(best_err)
                ok_cos.append(best_cos)
            else:
                fail_steps.append(int(k))

        kw = {
            "color": color,
            "linestyle": ls,
            "marker": "o",
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.6,
            "zorder": 3,
        }

        ok_cos_defect = [max(1.0 - c, 1e-12) for c in ok_cos]

        if ok_steps:
            ax_gn.loglog(ok_steps, ok_gn, **kw)
            ax_err.loglog(ok_steps, ok_err, **kw)
            ax_cos.loglog(ok_steps, ok_cos_defect, **kw)
            seen.add(alias)

        for fs in fail_steps:
            jx = jitter_x.get((alias, fs), float(fs))
            mk_kw = {
                "marker": _HORIZON_FAILURE_MARKER,
                "color": color,
                "markersize": 9,
                "markeredgewidth": 1.2,
                "markeredgecolor": "white",
                "linestyle": "none",
                "zorder": 6,
            }
            if ok_gn:
                ax_gn.loglog([jx], [ok_gn[-1]], **mk_kw)
                ax_err.loglog([jx], [ok_err[-1]], **mk_kw)
                ax_cos.loglog([jx], [ok_cos_defect[-1]], **mk_kw)
            failure_seen = True

    return failure_seen


def _horizon_style_axes(axes: Any) -> None:
    """Apply consistent titles / labels to (ax_gn, ax_err, ax_cos)."""
    ax_gn, ax_err, ax_cos = axes
    ax_gn.set_title("Gradient norm")
    ax_gn.set_xlabel("Rollout steps $T$")
    ax_gn.set_ylabel(r"$\|\nabla\mathcal{L}\|$")

    ax_err.set_title("FD relative error (best $\\varepsilon$)")
    ax_err.set_xlabel("Rollout steps $T$")
    ax_err.set_ylabel("Relative FD error")

    ax_cos.set_title("Cosine similarity (best $\\varepsilon$)")
    ax_cos.set_xlabel("Rollout steps $T$")
    ax_cos.set_ylabel("$1 -$ cosine")


def _horizon_attach_legend(fig: Any, seen: set[str], failure_seen: bool) -> None:
    """Build the solver legend (plus an optional × failure handle)."""
    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    if failure_seen:
        handles.append(
            mlines.Line2D(
                [],
                [],
                marker=_HORIZON_FAILURE_MARKER,
                color="0.4",
                linestyle="none",
                markersize=7,
                markeredgewidth=1.0,
                markeredgecolor="white",
                label=_HORIZON_FAILURE_LABEL,
            )
        )
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(handles), 6),
        fontsize=7.5,
        framealpha=0.7,
        edgecolor="0.8",
        handlelength=2.0,
    )


def _horizon_sweep_figure(
    cfg: Problem, *, exp_key: str, suffix: str, save: bool, out_dir: Any
) -> plt.Figure:
    """Styled 1×3 grad-norm / FD-error / cosine figure."""
    plt.rcParams.update(RCPARAMS)

    data = v1_to_legacy(load_json(out_dir / "result.json"))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42), dpi=300)
    fig.subplots_adjust(bottom=0.32, wspace=0.58, left=0.09, right=0.98, top=0.91)

    seen: set[str] = set()
    failure_seen = _horizon_plot_curves(axes, data, seen)
    _horizon_style_axes(axes)
    _horizon_attach_legend(fig, seen, failure_seen)

    if save:
        out = out_dir / "horizon_sweep.png"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


# ── G0: finite-difference check ───────────────────────────────────────────────


def plot_fd_check(
    cfg: Problem,
    *,
    ic_to_2d: Any = None,
    ic_key: str = "ic",
    diagnostic_fields: bool = True,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "fd_check",
    **_kw: Any,
) -> Any:
    """FD-check experiment plot: curves (styled) + gradient-magnitude fields.

    The rel-error / cosine curves use the inlined styling helpers
    (:func:`_fd_check_figure`) so the per-experiment ``fd_check.pdf``
    and the figure renderer stay in lockstep. The
    gradient-magnitude field panels are this problem's extra: they need
    ``ic_to_2d`` / ``diagnostic_fields`` flags that don't fit the
    cross-domain paper layout, so they live here.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"

    # ── error / cosine curves (styled) ──────────────────────────────────────
    fig_c = _fd_check_figure(
        cfg, exp_key=exp_key, suffix=suffix, save=save, out_dir=out_dir
    )

    # ── gradient magnitude fields ─────────────────────────────────────────────
    # The field panels only need per-solver display labels; that's a
    # presentation concern local to this problem (paper plots don't
    # produce field grids), so we keep the lightweight per-suite
    # ``solver_styles`` here rather than depend on paper-side styling.
    styles = solver_styles(cfg)
    fields_path = out_dir / "gradient_fields.npz"
    if not fields_path.exists():
        return fig_c

    npz = try_load_npz(fields_path)
    solver_names = npz["solver_names"].tolist()
    ic = npz["ic"]

    # Determine IC transform: use ic_to_2d when set (density/IC field),
    # fall back to vorticity_2d for 2-D velocity fields (ndim==4), or skip
    # entirely if neither applies.
    if ic_to_2d is not None:
        f_ic = ic_to_2d
    elif diagnostic_fields:
        f_ic = vorticity_2d if ic.ndim == 4 else None
    else:
        f_ic = None
    if f_ic is None:
        return fig_c

    ic_arr = f_ic(ic)
    ic_label = f"IC ({ic_key})"
    # Non-negative IC (e.g. density ρ ∈ [0,1]): sequential gray_r, no symmetric range.
    # Signed IC (e.g. density contrast δ₀): symmetric RdBu_r.
    if ic_arr.min() >= -1e-6:
        ic_panel_kw = {
            "cmap": "gray_r",
            "vmin": 0.0,
            "vmax": max(float(ic_arr.max()), 1.0),
        }
    else:
        ic_vmax = float(np.abs(ic_arr).max())
        ic_panel_kw = {"cmap": "RdBu_r", "vmin": -ic_vmax, "vmax": ic_vmax}
    panels = [
        (ic_label, ic_arr, ic_panel_kw),
    ]
    for j, name in enumerate(solver_names):
        key = f"grad_{j}"
        if key in npz:
            raw = npz[key]
            # Use the same spatial transform as the IC when available (same per-cell shape).
            # Fall back to the generic grad_magnitude_2d for problems without ic_to_2d.
            if ic_to_2d is not None:
                g = np.abs(f_ic(raw)).astype(np.float32)
            else:
                g = grad_magnitude_2d(raw)
            vmax = g.max() or 1.0
            panels.append(
                (
                    f"∂L/∂IC  {styles.get(name, {}).get('label', name)}",
                    g,
                    {"cmap": "viridis", "vmin": 0, "vmax": vmax},
                )
            )

    fig_g = field_grid(
        panels,
        f"{cfg.name} — G0 gradient magnitude fields",
        shared_scale=False,
        ncols=min(len(panels), 4),
    )
    if save:
        save_fig(fig_g, "gradient_fields", out_dir)
    return fig_c


# ── per-solver error plot helper ──────────────────────────────────────────────


def _plot_error_per_solver(
    by_solver: dict,
    styles: dict,
    title_prefix: str,
    x_keys: Any,
    x_to_float: Any,
    x_label: str,
    x_scale: str = "linear",
) -> plt.Figure:
    """One panel per solver: rel_error_mean vs x_keys, one line per ε value.

    ε values are colour-coded with a sequential palette (small ε = light,
    large ε = dark) so the U-curve shape becomes visible across panels.
    """
    solver_names = list(by_solver.keys())
    n = len(solver_names)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )

    # ε values from the first solver/key entry
    _first = by_solver[solver_names[0]]
    eps_keys = sorted(_first[next(iter(_first))]["eps_sweep"], key=float)
    eps_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(eps_keys)))
    markers = ["o", "s", "^", "D"]

    for idx, name in enumerate(solver_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        xs = [x_to_float(k) for k in x_keys]

        for ei, eps in enumerate(eps_keys):
            re_m = [
                by_solver[name][k]["eps_sweep"][eps]["rel_error_mean"] for k in x_keys
            ]
            ax.semilogy(
                xs,
                re_m,
                color=eps_colors[ei],
                marker=markers[ei % len(markers)],
                markersize=4,
                linewidth=1.5,
                label=f"ε={float(eps):.0e}",
            )

        ax.set_xscale(x_scale)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Relative FD error")
        ax.set_title(
            styles[name]["label"], color=styles[name]["color"], fontweight="bold"
        )

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title_prefix, fontsize=10)
    fig_shared_legend(fig, axes)
    return fig


# ── best-ε overlay helper ─────────────────────────────────────────────────────


def _plot_best_eps_overlay(
    by_solver: dict,
    styles: dict,
    title: str,
    x_keys: Any,
    x_to_float: Any,
    x_label: str,
    x_scale: str = "linear",
) -> plt.Figure:
    """All solvers overlaid: best-ε rel_error_mean vs x_keys."""
    fig, ax = plt.subplots(figsize=(7, 4))

    def _best_re(eps_sweep: dict) -> float:
        finite = [
            v
            for v in (e["rel_error_mean"] for e in eps_sweep.values())
            if np.isfinite(v)
        ]
        return float(min(finite)) if finite else float("nan")

    for name, results in by_solver.items():
        xs = [x_to_float(k) for k in x_keys]
        best_re = [_best_re(results[k]["eps_sweep"]) for k in x_keys]

        pairs = [(x, v) for x, v in zip(xs, best_re, strict=False) if np.isfinite(v)]
        if not pairs:
            continue
        px, py = zip(*pairs, strict=False)
        ax.semilogy(
            px,
            py,
            label=styles[name]["label"],
            **solver_plot_props(styles[name]),
        )

    ax.set_xscale(x_scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Relative FD error (best ε)")
    ax.set_title(title)
    fig_shared_legend(fig, [ax])
    return fig


# ── shared U-curve helper ─────────────────────────────────────────────────────


def _best_eps_series(param_results: dict, param_keys: Any, metric: str) -> list[float]:
    """For each param key pick the best-ε value of `metric` across the eps sweep."""
    return [
        min(param_results[k]["eps_sweep"].values(), key=lambda v: v["rel_error_mean"])[
            metric
        ]
        for k in param_keys
    ]


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
            if name not in styles:
                continue
            entry = by_solver.get(name, {}).get(key)
            sweep = entry.get("eps_sweep") if isinstance(entry, dict) else None
            if not sweep:
                continue
            props = solver_plot_props(styles[name])
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
    units: dict | None = None,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "param_sweep",
    **_kw: Any,
) -> Any:
    """Two files: summary curves (grad norm + best-ε error + cosine) and U-curve grid."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    result_path = out_dir / "result.json"
    data = v1_to_legacy(load_json(result_path))
    styles = solver_styles(cfg)
    sweep_key = data.get("sweep_key", "param")

    # ── summary: grad norm, best-ε rel error, best-ε cosine vs sweep param ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    all_cosines_sweep: list[float] = []
    for name, param_results in data["by_solver"].items():
        param_vals = sorted(param_results.keys(), key=float)
        param_f = [float(v) for v in param_vals]
        norms = [param_results[v]["grad_norm"] for v in param_vals]
        re_mean = _best_eps_series(param_results, param_vals, "rel_error_mean")
        cosines = _best_eps_series(param_results, param_vals, "cosine_mean")
        all_cosines_sweep.extend(c for c in cosines if np.isfinite(c))
        props = solver_plot_props(styles[name])

        axes[0].loglog(param_f, norms, label=styles[name]["label"], **props)
        axes[1].loglog(param_f, re_mean, label=styles[name]["label"], **props)
        axes[2].semilogx(param_f, cosines, label=styles[name]["label"], **props)

    xlbl = unit_label(sweep_key, units)
    axes[0].set_xlabel(xlbl)
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title(f"G2a — gradient norm vs {sweep_key}")
    axes[1].set_xlabel(xlbl)
    axes[1].set_ylabel("Relative FD error (best ε)")
    axes[1].set_title(f"G2a — FD error vs {sweep_key}")
    axes[2].set_xlabel(xlbl)
    axes[2].set_ylabel("Subspace cosine (best ε)")
    axes[2].set_title(f"G2a — direction accuracy vs {sweep_key}")
    min_cos_sweep = min(all_cosines_sweep) if all_cosines_sweep else 0.0
    if min_cos_sweep > 0.8:
        axes[2].set_ylim(min(min_cos_sweep, 0.999) - 0.001, 1.001)
    else:
        axes[2].set_ylim(-0.05, 1.05)
    fig_shared_legend(fig, axes)
    if save:
        save_fig(fig, "param_sweep", out_dir)

    # ── U-curve overlay: all solvers per sweep value ──────────────────────────
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

    # ── FD error vs param, one line per ε (per solver) ────────────────────────
    fig_s = _plot_error_per_solver(
        data["by_solver"],
        styles,
        f"{cfg.name} — G2a FD error vs {sweep_key} (per solver)",
        x_keys=param_vals,
        x_to_float=float,
        x_label=sweep_key,
        x_scale="log",
    )
    if save:
        save_fig(fig_s, "error_vs_param", out_dir)

    # ── Best-ε FD error vs param, all solvers overlaid ────────────────────────
    fig_b = _plot_best_eps_overlay(
        data["by_solver"],
        styles,
        f"{cfg.name} — G2a best-ε FD error vs {sweep_key}",
        x_keys=param_vals,
        x_to_float=float,
        x_label=sweep_key,
        x_scale="log",
    )
    if save:
        save_fig(fig_b, "best_eps_vs_param", out_dir)

    return fig


# ── G3: Jacobian SVD ──────────────────────────────────────────────────────────


def plot_jacobian_svd(
    cfg: Problem,
    *,
    ic_to_2d: Any = None,
    ic_key: str = "ic",
    diagnostic_fields: bool = True,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "jacobian_svd",
    **_kw: Any,
) -> Any:
    """Jacobian-SVD experiment plot: styled spectra grid + cross-cosine + fields.

    The per-solver singular spectrum grid uses the inlined styling
    helpers (:func:`_jacobian_svd_figure`) so the per-experiment
    ``jacobian_svd.pdf`` stays in lockstep with the figure renderer. The cross-solver cosine heatmap, scalar-output
    gradient-norm bar chart, and gradient-field panels are this
    problem's extras: they don't fit the cross-domain paper layout, so
    they live here as additional ``save_fig`` calls.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = v1_to_legacy(load_json(out_dir / "result.json"))
    styles = solver_styles(cfg)

    solver_names = data["solver_names"]
    cross_cos = np.array(data["cross_cosine"])

    per_solver_spectra: dict = data.get("per_solver_spectra", {})
    per_solver_grad_norm: dict = data.get("per_solver_grad_norm", {})

    # Detect scalar output: all per-solver spectra have exactly 1 singular value.
    _scalar_output = per_solver_spectra and all(
        len(v) == 1 for v in per_solver_spectra.values()
    )

    # ── Singular-value spectra grid (styled) ────────────────────────────────
    # For scalar outputs the per-solver spectrum is trivially [1.0] and
    # the helper short-circuits returning None; we fall back to a
    # per-solver gradient-norm bar chart in that case.
    fig_c = _jacobian_svd_figure(
        cfg, exp_key=exp_key, suffix=suffix, save=save, out_dir=out_dir
    )

    if _scalar_output:
        fig_bar, ax = plt.subplots(figsize=(6, 4))
        names = list(per_solver_grad_norm or per_solver_spectra)
        norms = [per_solver_grad_norm.get(n, float("nan")) for n in names]
        colors = [styles.get(n, {}).get("color", "#888888") for n in names]
        labels = [styles.get(n, {}).get("label", n) for n in names]
        ax.bar(range(len(names)), norms, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("‖∇L‖  (gradient norm)")
        ax.set_title(f"{cfg.name} — G3 per-solver gradient norm")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        fig_bar.tight_layout()
        if save:
            save_fig(fig_bar, "gradient_norms", out_dir)
        if fig_c is None:
            fig_c = fig_bar

    # ── Cross-solver cosine similarity heatmap ───────────────────────────────
    fig_h, ax = plt.subplots(figsize=(6, 5))
    n = len(solver_names)
    im = ax.imshow(cross_cos, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [styles.get(s, {}).get("label", s) for s in solver_names]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_title(f"{cfg.name} — G3 cross-solver cosine similarity")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_h.tight_layout()
    if save:
        save_fig(fig_h, "cross_cosine", out_dir)
    if fig_c is None:
        fig_c = fig_h

    # ── Figure 2: IC + per-solver gradient magnitude + top singular direction ─
    fields_path = out_dir / "jacobian_svd.npz"
    if not fields_path.exists():
        return fig_c

    npz = try_load_npz(fields_path)
    npz_solvers = npz["solver_names"].tolist()
    ic = npz["ic"]

    if ic_to_2d is not None:
        f_ic = ic_to_2d
    elif diagnostic_fields:
        f_ic = vorticity_2d if ic.ndim == 4 else None
    else:
        f_ic = None
    if f_ic is None:
        return fig_c

    ic_arr = f_ic(ic)
    if ic_arr.min() >= -1e-6:
        ic_panel_kw = {
            "cmap": "gray_r",
            "vmin": 0.0,
            "vmax": max(float(ic_arr.max()), 1.0),
        }
    else:
        ic_vmax = float(np.abs(ic_arr).max())
        ic_panel_kw = {"cmap": "RdBu_r", "vmin": -ic_vmax, "vmax": ic_vmax}
    panels = [(f"IC ({ic_key})", ic_arr, ic_panel_kw)]

    for j, name in enumerate(npz_solvers):
        key = f"grad_{j}"
        if key in npz:
            raw = npz[key]
            # Use ic_to_2d when available (same per-cell transform as IC).
            if ic_to_2d is not None:
                g = np.abs(f_ic(raw)).astype(np.float32)
            else:
                g = grad_magnitude_2d(raw)
            vmax = g.max() or 1.0
            panels.append(
                (
                    f"∂L/∂IC  {styles.get(name, {}).get('label', name)}",
                    g,
                    {"cmap": "viridis", "vmin": 0, "vmax": vmax},
                )
            )

    # Top singular direction as a spatial field
    if "singular_vectors" in npz and "grad_0" in npz:
        d_top_flat = npz["singular_vectors"][0]
        ref_shape = npz["grad_0"].shape
        if d_top_flat.size == np.prod(ref_shape):
            d_top = d_top_flat.reshape(ref_shape)
            if ic_to_2d is not None:
                d_top_2d = np.abs(f_ic(d_top)).astype(np.float32)
            else:
                d_top_2d = grad_magnitude_2d(d_top)
            vmax = d_top_2d.max() or 1.0
            panels.append(
                (
                    "Top singular direction",
                    d_top_2d,
                    {"cmap": "plasma", "vmin": 0, "vmax": vmax},
                )
            )

    fig_g = field_grid(
        panels,
        f"{cfg.name} — G3 gradient fields + singular direction",
        shared_scale=False,
        ncols=min(len(panels), 4),
    )
    if save:
        save_fig(fig_g, "gradient_fields", out_dir)
    return fig_c


# ── G2c: horizon sweep ────────────────────────────────────────────────────────


def plot_horizon_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "horizon_sweep",
    **_kw: Any,
) -> Any:
    """Horizon-sweep experiment plot: styled curves + auxiliary diagnostics.

    The grad-norm / best-ε error / cosine curves use the inlined
    styling helpers (:func:`_horizon_sweep_figure`) so the
    per-experiment ``horizon_sweep.pdf`` stays in lockstep with the
    figure renderer. The U-curve grid, per-solver error panels
    and gradient-magnitude field panels are this problem's extras and
    live here.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = v1_to_legacy(load_json(out_dir / "result.json"))
    styles = solver_styles(cfg)
    # A ``steps`` sweep lands under ``by_steps`` (not ``by_solver``) post-v1;
    # collapse whichever group is present.
    by_solver = legacy_by_solver(data)

    # ── summary curves (styled) ─────────────────────────────────────────────
    fig_c = _horizon_sweep_figure(
        cfg, exp_key=exp_key, suffix=suffix, save=save, out_dir=out_dir
    )

    if not by_solver:
        return fig_c

    # The ε-based panels below (U-curves, per-ε error, best-ε overlay) only
    # apply when entries carry an ``eps_sweep``. Some horizon experiments
    # (e.g. rollout-limit variants) record different metrics — emit just the
    # styled summary for those rather than crashing on a missing key.
    has_eps_sweep = any(
        isinstance(v, dict) and "eps_sweep" in v
        for sweep in by_solver.values()
        if isinstance(sweep, dict)
        for v in sweep.values()
    )
    if not has_eps_sweep:
        return fig_c

    # ── U-curve overlay: all solvers per horizon ─────────────────────────────
    step_keys = sorted(next(iter(by_solver.values())).keys(), key=int)
    fig_u = _plot_ucurve_overlay(
        by_solver,
        step_keys,
        "steps",
        styles,
        f"{cfg.name} — G2c ε U-curves (horizon sweep)",
        ncols=4,
    )
    if save:
        save_fig(fig_u, "ucurves", out_dir)

    # ── FD error vs steps, one line per ε (per solver) ───────────────────────
    fig_s = _plot_error_per_solver(
        by_solver,
        styles,
        f"{cfg.name} — G2c FD error vs horizon (per solver)",
        x_keys=step_keys,
        x_to_float=int,
        x_label="Steps (horizon)",
    )
    if save:
        save_fig(fig_s, "error_vs_steps", out_dir)

    # ── Best-ε FD error vs steps, all solvers overlaid ───────────────────────
    fig_b = _plot_best_eps_overlay(
        by_solver,
        styles,
        f"{cfg.name} — G2c best-ε FD error vs horizon",
        x_keys=step_keys,
        x_to_float=int,
        x_label="Steps (horizon)",
    )
    if save:
        save_fig(fig_b, "best_eps_vs_steps", out_dir)

    # ── gradient magnitude fields at representative horizons ──────────────────
    fields_path = out_dir / "gradient_fields.npz"
    if not fields_path.exists():
        return fig_c

    npz = try_load_npz(fields_path)
    solver_names = npz["solver_names"].tolist()
    horizons = npz["horizons"].tolist()

    for j, name in enumerate(solver_names):
        panels = []
        for k, h in enumerate(horizons):
            key = f"grad_{j}_{k}"
            if key in npz:
                g = grad_magnitude_2d(npz[key])
                vmax = g.max() or 1.0
                panels.append(
                    (f"T={h}", g, {"cmap": "viridis", "vmin": 0, "vmax": vmax})
                )
        if not panels:
            continue
        lbl = styles.get(name, {}).get("label", name)
        fig_g = field_grid(
            panels,
            f"{cfg.name} — G2c ∂L/∂IC magnitude | {lbl}",
            shared_scale=True,
            symmetric=False,
            ncols=len(panels),
        )
        if save:
            save_fig(fig_g, f"gradient_fields_{name}", out_dir)
    return fig_c


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
        data = v1_to_legacy(load_json(result_path))
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
