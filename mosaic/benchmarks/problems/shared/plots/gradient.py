"""Plots for the gradient evaluation suite (fd_check, param_sweep, jacobian_svd)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.style import (
    apply_style,
    field_grid,
    fig_shared_legend,
    grad_magnitude_2d,
    save_fig,
    solver_plot_props,
    solver_styles,
    unit_label,
    vorticity_2d,
)

apply_style()

_SUITE = "gradient"


# ── G0: finite-difference check ───────────────────────────────────────────────


def plot_fd_check(
    cfg: Problem,
    *,
    ic_to_2d=None,
    ic_key: str = "ic",
    diagnostic_fields: bool = True,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "fd_check",
    **_kw,
):
    """Two files: FD error + subspace cosine curves, and gradient magnitude field panels."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    # ── error / cosine curves ─────────────────────────────────────────────────
    fig_c, axes = plt.subplots(1, 2, figsize=(11, 4))
    all_cosines: list[float] = []
    for name, solver_data in data["by_solver"].items():
        eps_sweep = solver_data["eps_sweep"]
        epsilons = sorted(eps_sweep.keys(), key=float)
        eps_f = [float(e) for e in epsilons]
        props = solver_plot_props(styles[name])

        rel_mean = np.array([np.mean(eps_sweep[e]["rel_error"]) for e in epsilons])
        # "cosine" is now a scalar subspace cosine per eps value
        cos_vals = np.array([eps_sweep[e]["cosine"] for e in epsilons])
        all_cosines.extend(cos_vals.tolist())

        axes[0].loglog(eps_f, rel_mean, label=styles[name]["label"], **props)
        axes[1].semilogx(eps_f, cos_vals, label=styles[name]["label"], **props)

    axes[0].set_xlabel("ε (relative to IC RMS)")
    axes[0].set_ylabel("Relative FD error")
    axes[0].set_title("G0 — gradient verification")
    axes[1].set_xlabel("ε (relative to IC RMS)")
    axes[1].set_ylabel("Subspace cosine similarity")
    axes[1].set_title("G0 — direction accuracy")
    # Zoom y-axis when all cosines are near 1 — minimum span 0.003 to avoid noise zoom
    min_cos = min(all_cosines) if all_cosines else 0.0
    if min_cos > 0.8:
        axes[1].set_ylim(min(min_cos, 0.999) - 0.001, 1.001)
    else:
        axes[1].set_ylim(-0.05, 1.05)
    fig_shared_legend(fig_c, axes)
    if save:
        save_fig(fig_c, "fd_check", out_dir)

    # ── gradient magnitude fields ─────────────────────────────────────────────
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
    x_keys,
    x_to_float,
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
    x_keys,
    x_to_float,
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


def _best_eps_series(param_results: dict, param_keys, metric: str) -> list[float]:
    """For each param key pick the best-ε value of `metric` across the eps sweep."""
    return [
        min(param_results[k]["eps_sweep"].values(), key=lambda v: v["rel_error_mean"])[
            metric
        ]
        for k in param_keys
    ]


def _plot_ucurve_overlay(
    by_solver: dict,
    sweep_keys,
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
    units: dict | None = None,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "param_sweep",
    **_kw,
):
    """Two files: summary curves (grad norm + best-ε error + cosine) and U-curve grid."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    result_path = out_dir / "result.json"
    # When multiple ICs are used, each IC lands in a subdir; plot each one.
    if not result_path.exists():
        subdirs = (
            sorted(
                p
                for p in out_dir.iterdir()
                if p.is_dir() and (p / "result.json").exists()
            )
            if out_dir.exists()
            else []
        )
        if subdirs:
            for sub in subdirs:
                plot_param_sweep(
                    cfg,
                    units=units,
                    save=save,
                    suffix=f"{suffix}/{sub.name}",
                    exp_key=exp_key,
                )
            return
        else:
            raise FileNotFoundError(str(result_path))
    data = load_json(result_path)
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


# ── G2b: resolution sweep ────────────────────────────────────────────────────


def plot_resolution_sweep(
    cfg: Problem,
    *,
    resolution_key: str = "N",
    units: dict | None = None,
    save: bool = True,
    suffix: str = "",
    **_kw,
):
    """Summary curves (grad norm + best-ε error + cosine) and U-curve grid vs N."""
    out_dir = results_dir() / cfg.name / _SUITE / f"resolution_sweep{suffix}"
    result_path = out_dir / "result.json"
    # When --ics <name> is used the run lands in a subdir; fall back to the
    # first IC subdir that contains a result.json.
    if not result_path.exists():
        subdirs = (
            sorted(
                p
                for p in out_dir.iterdir()
                if p.is_dir() and (p / "result.json").exists()
            )
            if out_dir.exists()
            else []
        )
        if not subdirs:
            raise FileNotFoundError(str(result_path))
        result_path = subdirs[0] / "result.json"
    data = load_json(result_path)
    styles = solver_styles(cfg)

    # ── summary: grad norm, best-ε rel error, cosine vs N ────────────────────
    fig_c, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, n_results in data["by_solver"].items():
        N_keys = sorted(n_results.keys(), key=int)
        N_f = [int(k) for k in N_keys]
        norms = [n_results[k]["grad_norm"] for k in N_keys]
        re_mean = _best_eps_series(n_results, N_keys, "rel_error_mean")
        cosines = _best_eps_series(n_results, N_keys, "cosine_mean")
        props = solver_plot_props(styles[name])

        axes[0].loglog(N_f, norms, label=styles[name]["label"], **props)
        axes[1].loglog(N_f, re_mean, label=styles[name]["label"], **props)
        axes[2].semilogx(N_f, cosines, label=styles[name]["label"], **props)

    res_lbl = unit_label(resolution_key, units)
    axes[0].set_xlabel(res_lbl)
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title("G2b — gradient norm vs resolution")
    axes[1].set_xlabel(res_lbl)
    axes[1].set_ylabel("Relative FD error (best ε)")
    axes[1].set_title("G2b — FD error vs resolution")
    axes[2].set_xlabel(res_lbl)
    axes[2].set_ylabel("Subspace cosine (best ε)")
    axes[2].set_title("G2b — direction accuracy vs resolution")
    axes[2].set_ylim(-0.05, 1.05)
    fig_shared_legend(fig_c, axes)
    if save:
        save_fig(fig_c, "resolution_sweep", out_dir)

    # ── U-curve overlay: all solvers per N ───────────────────────────────────
    N_keys = sorted(next(iter(data["by_solver"].values())).keys(), key=int)
    fig_u = _plot_ucurve_overlay(
        data["by_solver"],
        N_keys,
        "N",
        styles,
        f"{cfg.name} — G2b ε U-curves (resolution sweep)",
        ncols=len(N_keys),
    )
    if save:
        save_fig(fig_u, "ucurves", out_dir)

    # ── FD error vs N, one line per ε (per solver) ───────────────────────────
    fig_s = _plot_error_per_solver(
        data["by_solver"],
        styles,
        f"{cfg.name} — G2b FD error vs N (per solver)",
        x_keys=N_keys,
        x_to_float=int,
        x_label="N",
        x_scale="log",
    )
    if save:
        save_fig(fig_s, "error_vs_N", out_dir)

    # ── Best-ε FD error vs N, all solvers overlaid ───────────────────────────
    fig_b = _plot_best_eps_overlay(
        data["by_solver"],
        styles,
        f"{cfg.name} — G2b best-ε FD error vs N",
        x_keys=N_keys,
        x_to_float=int,
        x_label="N",
        x_scale="log",
    )
    if save:
        save_fig(fig_b, "best_eps_vs_N", out_dir)

    # ── gradient magnitude fields at each N (per solver) ─────────────────────
    fields_path = out_dir / "gradient_fields.npz"
    if not fields_path.exists():
        return fig_c

    npz = try_load_npz(fields_path)
    solver_names = npz["solver_names"].tolist()
    N_arr = npz["N_values"].tolist()

    for j, name in enumerate(solver_names):
        panels = []
        for N in N_arr:
            key = f"grad_{j}_N{N}"
            if key in npz:
                g = grad_magnitude_2d(npz[key])
                vmax = g.max() or 1.0
                panels.append(
                    (f"N={N}", g, {"cmap": "viridis", "vmin": 0, "vmax": vmax})
                )
        if not panels:
            continue
        lbl = styles.get(name, {}).get("label", name)
        fig_g = field_grid(
            panels,
            f"{cfg.name} — G2b ∂L/∂IC magnitude | {lbl}",
            shared_scale=True,
            symmetric=False,
            ncols=len(panels),
        )
        if save:
            save_fig(fig_g, f"gradient_fields_{name}", out_dir)
    return fig_c


# ── G3: Jacobian SVD ──────────────────────────────────────────────────────────


def plot_jacobian_svd(
    cfg: Problem,
    *,
    ic_to_2d=None,
    ic_key: str = "ic",
    diagnostic_fields: bool = True,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "jacobian_svd",
    **_kw,
):
    """Three files:
    - jacobian_svd.png: singular spectrum + explained variance + cosine heatmap
    - landscape.png: 1-D loss slice along the top singular direction
    - gradient_fields.png: IC and per-solver gradient magnitude panels
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    solver_names = data["solver_names"]
    S_norm = data["singular_values"]
    cond = data["condition_number"]
    cross_cos = np.array(data["cross_cosine"])

    n_modes = len(S_norm)
    modes = list(range(1, n_modes + 1))
    per_solver_spectra: dict = data.get("per_solver_spectra", {})
    per_solver_eff_rank: dict = data.get("per_solver_eff_rank", {})
    per_solver_grad_norm: dict = data.get("per_solver_grad_norm", {})

    # Detect scalar output: all per-solver spectra have exactly 1 singular value.
    _scalar_output = per_solver_spectra and all(
        len(v) == 1 for v in per_solver_spectra.values()
    )

    # ── Figure 1: per-solver spectra + cross-cosine heatmap ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    if _scalar_output:
        # Scalar output: spectrum is trivially [1.0]. Show gradient norm bar chart.
        names = list(per_solver_grad_norm or per_solver_spectra)
        norms = [per_solver_grad_norm.get(n, float("nan")) for n in names]
        colors = [styles.get(n, {}).get("color", "#888888") for n in names]
        labels = [styles.get(n, {}).get("label", n) for n in names]
        ax.bar(range(len(names)), norms, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("‖∇L‖  (gradient norm)")
        ax.set_title("G3 — per-solver gradient norm")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    elif per_solver_spectra:
        # Vector output: plot normalised singular value spectra (projection vs LBM).
        for name, spec in per_solver_spectra.items():
            style = styles.get(name, {})
            n_i = len(spec)
            m_i = list(range(1, n_i + 1))
            er = per_solver_eff_rank.get(name, float("nan"))
            label = f"{style.get('label', name)}  (r_eff={er:.1f})"
            mk = "o" if n_i <= 32 else ""
            ax.semilogy(
                m_i,
                spec,
                f"{mk}-",
                color=style.get("color", "#888888"),
                markersize=4 if mk else 0,
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel("Mode index i")
        ax.set_ylabel("σᵢ / σ₁  (log scale)")
        ax.set_title("G3 — per-solver singular value spectra")
        ax.legend(fontsize=8)
    else:
        # Fallback: stacked-Jacobian spectrum
        marker = "o" if n_modes <= 32 else ""
        ax.semilogy(
            modes,
            S_norm,
            f"{marker}-",
            color="steelblue",
            markersize=5 if marker else 0,
            linewidth=1.5,
            label=f"stacked  κ={cond:.1f}",
        )
        ax.set_xlabel("Mode index i")
        ax.set_ylabel("σᵢ / σ₁  (log scale)")
        ax.set_title("G3 — per-solver singular value spectra")
        ax.legend(fontsize=8)

    # Panel 1: cross-solver cosine similarity heatmap
    ax = axes[1]
    n = len(solver_names)
    im = ax.imshow(cross_cos, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [styles.get(s, {}).get("label", s) for s in solver_names]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_title("G3 — cross-solver cosine similarity")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{cfg.name} — G3 Jacobian SVD", fontsize=10)
    fig.tight_layout()
    if save:
        save_fig(fig, "jacobian_svd", out_dir)

    # ── Figure 2: IC + per-solver gradient magnitude + top singular direction ─
    fields_path = out_dir / "jacobian_svd.npz"
    if not fields_path.exists():
        return fig

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
        return fig

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
    return fig


# ── G4: memory sweep ─────────────────────────────────────────────────────────


def _memory_sweep_mem_key(entry: dict, pass_prefix: str) -> float:
    """Return GPU mem if present, else RAM (CPU-only solvers), else NaN."""
    gpu = entry.get(f"{pass_prefix}peak_gpu_mem_mb", np.nan)
    if np.isfinite(gpu) and gpu > 0:
        return gpu
    return entry.get(f"{pass_prefix}peak_ram_mb", np.nan)


def _memory_sweep_has_vjp(by_data: dict) -> bool:
    for solver_data in by_data.values():
        for entry in solver_data.values():
            if "vjp_peak_gpu_mem_mb" in entry or "vjp_peak_ram_mb" in entry:
                return True
    return False


def _memory_sweep_draw_panel(ax, by_data, x_label, x_to_int, title, cfg, styles):
    """Draw fwd (dashed) and VJP (solid) lines on ax."""
    all_keys = sorted(next(iter(by_data.values())).keys(), key=x_to_int)
    x_arr = np.array([x_to_int(k) for k in all_keys], dtype=float)
    has_gpu_label = False
    has_ram_label = False
    for spec in cfg.solvers:
        name = spec.name
        solver_data = by_data.get(name, {})
        if not solver_data:
            continue
        style = styles.get(name, {})
        label = style.get("label", name)
        props_fwd = {**solver_plot_props(style), "linestyle": "--", "alpha": 0.7}
        props_vjp = solver_plot_props(style)

        fwd_mem = [
            _memory_sweep_mem_key(solver_data.get(k, {}), "fwd_") for k in all_keys
        ]
        if any(np.isfinite(v) and v > 0 for v in fwd_mem):
            ax.loglog(x_arr, fwd_mem, label=f"{label} fwd", **props_fwd)
            # track whether GPU or RAM labels needed (for y-axis)
            first_val = next(
                (
                    solver_data[k].get("fwd_peak_gpu_mem_mb", np.nan)
                    for k in all_keys
                    if k in solver_data
                ),
                np.nan,
            )
            if np.isfinite(first_val) and first_val > 0:
                has_gpu_label = True
            else:
                has_ram_label = True

        vjp_mem = [
            _memory_sweep_mem_key(solver_data.get(k, {}), "vjp_") for k in all_keys
        ]
        if any(np.isfinite(v) and v > 0 for v in vjp_mem):
            ax.loglog(x_arr, vjp_mem, label=f"{label} VJP", **props_vjp)

    ax.set_xlabel(x_label)
    y_label = "Peak GPU mem (MB)" if has_gpu_label else "Peak RAM (MB)"
    if has_gpu_label and has_ram_label:
        y_label = "Peak GPU / RAM (MB)"
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return all_keys, x_arr


def _memory_sweep_draw_ratio(
    ax, by_data, all_keys, x_label, title, diff_solvers_with_vjp, styles
):
    x_arr = np.array([int(k) for k in all_keys], dtype=float)
    for name in diff_solvers_with_vjp:
        solver_data = by_data.get(name, {})
        style = styles.get(name, {})
        label = style.get("label", name)
        ratios = []
        for k in all_keys:
            entry = solver_data.get(k, {})
            fwd = _memory_sweep_mem_key(entry, "fwd_")
            vjp = _memory_sweep_mem_key(entry, "vjp_")
            ratios.append(
                vjp / fwd
                if (np.isfinite(fwd) and fwd > 0 and np.isfinite(vjp))
                else np.nan
            )
        if any(np.isfinite(r) for r in ratios):
            ax.plot(x_arr, ratios, label=label, **solver_plot_props(style))
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("VJP mem / forward mem")
    ax.set_title(title)


def _memory_sweep_fig_main(cfg, styles, by_N, by_steps, has_N, has_steps, res_key):
    """Build the fwd-vs-VJP memory figure; return (fig, N_keys, steps_keys)."""
    n_cols = int(has_N) + int(has_steps)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5), squeeze=False)
    col = 0
    N_keys = steps_keys = None

    if has_N:
        N_keys, _ = _memory_sweep_draw_panel(
            axes[0][col], by_N, res_key, int, f"G4 — memory vs {res_key}", cfg, styles
        )
        col += 1

    if has_steps:
        steps_keys, _ = _memory_sweep_draw_panel(
            axes[0][col], by_steps, "steps", int, "G4 — memory vs steps", cfg, styles
        )

    fig.suptitle(
        f"{cfg.name} — G4 forward vs VJP memory  (— solid = VJP, -- dashed = fwd)"
    )
    fig_shared_legend(fig, axes[0])
    fig.tight_layout()
    return fig, N_keys, steps_keys


def _memory_sweep_fig_ratio(
    cfg,
    styles,
    by_N,
    by_steps,
    has_N,
    has_steps,
    N_keys,
    steps_keys,
    diff_solvers_with_vjp,
    res_key,
):
    """Build the VJP / forward ratio figure."""
    n_cols_r = int(has_N and N_keys is not None) + int(
        has_steps and steps_keys is not None
    )
    fig_r, axes_r = plt.subplots(1, n_cols_r, figsize=(6 * n_cols_r, 4), squeeze=False)
    col = 0

    if has_N and N_keys is not None:
        _memory_sweep_draw_ratio(
            axes_r[0][col],
            by_N,
            N_keys,
            res_key,
            f"G4 — autodiff overhead vs {res_key}",
            diff_solvers_with_vjp,
            styles,
        )
        col += 1
    if has_steps and steps_keys is not None:
        _memory_sweep_draw_ratio(
            axes_r[0][col],
            by_steps,
            steps_keys,
            "steps",
            "G4 — autodiff overhead vs steps",
            diff_solvers_with_vjp,
            styles,
        )

    fig_r.suptitle(f"{cfg.name} — G4 VJP / forward memory ratio")
    fig_shared_legend(fig_r, axes_r[0])
    fig_r.tight_layout()
    return fig_r


def plot_memory_sweep(
    cfg: Problem,
    *,
    resolution_key: str = "N",
    save: bool = True,
    suffix: str = "",
    **_kw,
):
    """Two files:
    - memory_sweep.png: fwd vs VJP peak GPU memory vs N and vs steps (all solvers).
    - memory_ratio.png: VJP / forward memory ratio vs N and vs steps (diff solvers).

    Forward lines are dashed; VJP lines are solid.  Solvers without GPU (CPU-only)
    are shown using their peak_ram_mb instead and labelled accordingly.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"memory_sweep{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    by_N = data.get("by_N", {})
    by_steps = data.get("by_steps", {})

    has_N = bool(by_N) and any(bool(v) for v in by_N.values())
    has_steps = bool(by_steps) and any(bool(v) for v in by_steps.values())
    if not has_N and not has_steps:
        return None

    res_key = resolution_key

    # ── Figure 1: forward vs VJP memory ─────────────────────────────────────
    fig, N_keys, steps_keys = _memory_sweep_fig_main(
        cfg, styles, by_N, by_steps, has_N, has_steps, res_key
    )
    if save:
        save_fig(fig, "memory_sweep", out_dir)

    # ── Figure 2: VJP / forward memory ratio (differentiable solvers only) ────
    diff_solvers_with_vjp = [
        s.name
        for s in cfg.solvers
        if _memory_sweep_has_vjp({s.name: by_N.get(s.name, {})})
        or _memory_sweep_has_vjp({s.name: by_steps.get(s.name, {})})
    ]
    if not diff_solvers_with_vjp:
        return fig

    fig_r = _memory_sweep_fig_ratio(
        cfg,
        styles,
        by_N,
        by_steps,
        has_N,
        has_steps,
        N_keys,
        steps_keys,
        diff_solvers_with_vjp,
        res_key,
    )
    if save:
        save_fig(fig_r, "memory_ratio", out_dir)

    return fig


# ── G2c: horizon sweep ────────────────────────────────────────────────────────


def plot_horizon_sweep(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    **_kw,
):
    """Two files: summary curves (grad norm + best-ε error + cosine) and U-curve grid,
    plus gradient field panels."""
    out_dir = results_dir() / cfg.name / _SUITE / f"horizon_sweep{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    # ── summary curves ────────────────────────────────────────────────────────
    fig_c, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, h_results in data["by_solver"].items():
        step_keys = sorted(h_results.keys(), key=int)
        steps_f = [int(s) for s in step_keys]
        norms = [h_results[s]["grad_norm"] for s in step_keys]
        re_mean = _best_eps_series(h_results, step_keys, "rel_error_mean")
        cosines = _best_eps_series(h_results, step_keys, "cosine_mean")
        props = solver_plot_props(styles[name])

        axes[0].semilogy(steps_f, norms, label=styles[name]["label"], **props)
        axes[1].semilogy(steps_f, re_mean, label=styles[name]["label"], **props)
        axes[2].plot(steps_f, cosines, label=styles[name]["label"], **props)

    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title("G2c — gradient norm vs horizon")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Relative FD error (best ε)")
    axes[1].set_title("G2c — FD error vs horizon")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Subspace cosine (best ε)")
    axes[2].set_title("G2c — direction accuracy vs horizon")
    axes[2].set_ylim(-0.05, 1.05)
    fig_shared_legend(fig_c, axes)
    if save:
        save_fig(fig_c, "horizon_sweep", out_dir)

    # ── U-curve overlay: all solvers per horizon ─────────────────────────────
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

    # ── FD error vs steps, one line per ε (per solver) ───────────────────────
    fig_s = _plot_error_per_solver(
        data["by_solver"],
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
        data["by_solver"],
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
    **_kw,
):
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
