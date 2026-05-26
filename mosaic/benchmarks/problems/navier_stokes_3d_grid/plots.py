# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the navier-stokes-3d-grid recovery experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
)
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    RCPARAMS,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    fig_shared_legend,
    imshow_with_cbar,
    make_handle,
    resolve_solver_alias,
    save_fig,
    solver_plot_props,
    solver_props,
    solver_styles,
    subplots_grid,
    vorticity_2d,
)

# Default perturb_sigma to highlight on the single-experiment NS paper figure.
_PAPER_NS_SIGMA = "0.1"


def _resolve_recovery_out_dir(base_dir: Path, ic: str | None) -> Path:
    """Resolve the experiment directory: root-level, explicit IC, or auto-detected."""
    root_result = base_dir / "result.json"
    if root_result.exists() and ic is None:
        return base_dir
    if ic is not None:
        return base_dir / ic
    # Auto-detect: look for IC subdirectories with a result.json
    ic_dirs = sorted(
        p.parent for p in base_dir.glob("*/result.json") if p.parent != base_dir
    )
    if not ic_dirs:
        raise FileNotFoundError(
            f"No result.json found in {base_dir} or its subdirectories."
        )
    return ic_dirs[0]


def _sorted_sweep_vals(by_sweep: dict) -> list:
    """Collect ordered sweep values from the first solver's keys."""
    _first = next(iter(by_sweep.values()), {})
    return sorted(
        _first.keys(),
        key=lambda v: float(v) if str(v).replace(".", "").lstrip("-").isdigit() else 0,
    )


def _compute_fallback_ic_error_init(
    out_dir: Path,
    by_sweep: dict,
    sweep_vals: list,
    sweep_key: str,
) -> dict[float, float]:
    """Estimate ``ic_error_init`` per sweep value from ``recovery_fields.npz``.

    When older runs did not record ``ic_error_init`` and the sweep is a
    ``perturb_sigma`` sweep, recover an approximation by measuring the exact
    error at the representative sigma and scaling linearly to other sigmas.
    Returns an empty dict when the fallback cannot be computed.
    """
    fallback: dict[float, float] = {}
    has_ic_error_init = any(
        (s_results.get(v) or s_results.get(str(v)) or {}).get("ic_error_init")
        is not None
        for v in sweep_vals
        for s_results in by_sweep.values()
    )
    if has_ic_error_init or sweep_key != "perturb_sigma":
        return fallback
    fp = out_dir / "recovery_fields.npz"
    if not fp.exists():
        return fallback
    npz = try_load_npz(fp)
    if "ic_true" not in npz or "ic_init" not in npz:
        return fallback
    ic_t = npz["ic_true"].astype(float)
    ic_i = npz["ic_init"].astype(float)
    rep_v = float((npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0])
    ic_t_norm = float(np.sqrt(np.mean(ic_t**2)))
    if ic_t_norm <= 0 or rep_v <= 0:
        return fallback
    rep_err = float(np.sqrt(np.mean((ic_i - ic_t) ** 2))) / ic_t_norm
    for v in sweep_vals:
        fallback[float(v)] = rep_err * float(v) / rep_v
    return fallback


def _plot_recovery_summary(
    cfg: Problem,
    by_sweep: dict,
    sweep_vals: list,
    sweep_key: str,
    styles: dict,
    fallback_ic_error_init: dict[float, float],
    out_dir: Path,
    save: bool,
):
    """Build ``recovery.png``: IC recovery improvement + min loss vs sweep."""
    fig_r, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for name, s_results in by_sweep.items():
        sty = styles.get(name, {})
        xs_ic, ys_ic = [], []
        xs_loss, ys_loss = [], []
        for v in sweep_vals:
            r = s_results.get(v) or s_results.get(str(v))
            if r is None:
                continue
            xs_ic.append(float(v))
            ic_init_val = r.get("ic_error_init") or fallback_ic_error_init.get(float(v))
            if ic_init_val:
                ys_ic.append((r["final_ic_error"] - ic_init_val) / ic_init_val)
            else:
                ys_ic.append(r["final_ic_error"])
            errors = r.get("errors") or []
            if errors:
                xs_loss.append(float(v))
                ys_loss.append(min(errors))
        if xs_ic:
            ax1.plot(
                xs_ic, ys_ic, label=sty.get("label", name), **solver_plot_props(sty)
            )
        if xs_loss:
            ax2.semilogy(
                xs_loss, ys_loss, label=sty.get("label", name), **solver_plot_props(sty)
            )

    ax1.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax1.axhline(-1, color="gray", ls=":", lw=1, alpha=0.4)
    ax1.set_xlabel(sweep_key)
    ax1.set_ylabel("Normalised Δ IC error  (final − init) / init")
    ax1.set_title(
        f"IC recovery improvement vs {sweep_key}\n(−1 = perfect, 0 = no gain)"
    )
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel(sweep_key)
    ax2.set_ylabel("Min loss (MSE)")
    ax2.set_title(f"Minimum achieved loss vs {sweep_key}")
    ax2.grid(True, which="both", alpha=0.3)

    fig_r.suptitle(f"{cfg.name} — recovery")
    fig_shared_legend(fig_r, [ax1])
    if save:
        save_fig(fig_r, "optimization", out_dir)
    return fig_r


def _draw_convergence_panel(
    ax: Any, v: Any, by_sweep: dict, styles: dict, sweep_key: str
) -> None:
    """Draw one convergence-curve panel for sweep value *v*."""
    for name, s_results in by_sweep.items():
        r = s_results.get(v) or s_results.get(str(v))
        if not (r and r.get("errors")):
            continue
        errors = r["errors"]
        sty = styles.get(name, {})
        # Detect a flat (no-gradient) curve: relative drop < 1 %
        is_flat = (
            len(errors) > 1
            and (errors[0] - errors[-1]) / (abs(errors[0]) + 1e-30) < 0.01
        )
        label_str = sty.get("label", name)
        if is_flat:
            label_str += " (no grad)"
        line_kw = solver_plot_props(sty, marker=False)
        if is_flat:
            line_kw = {**line_kw, "linestyle": ":", "alpha": 0.55}
        ax.semilogy(errors, label=label_str, **line_kw)
        # Annotate final IC error inline at the end of each curve
        final_ic = r.get("final_ic_error")
        if final_ic is not None:
            color = sty.get("color", "gray")
            converged = r.get("converged", False)
            ic_label = f"IC={final_ic:.2f}"
            if not converged:
                ic_label += " ✗"
            ax.annotate(
                ic_label,
                xy=(len(errors) - 1, errors[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=6,
                color=color,
                va="center",
            )
    ax.set_xlabel("Iteration")
    ax.set_title(f"{sweep_key}={v}")


def _plot_convergence_curves(
    cfg: Problem,
    by_sweep: dict,
    sweep_vals: list,
    sweep_key: str,
    styles: dict,
    out_dir: Path,
    save: bool,
) -> None:
    """One panel per sweep value showing optim loss curves for every solver."""
    has_any_errors = any(
        (s_results.get(v) or s_results.get(str(v)) or {}).get("errors")
        for v in sweep_vals
        for s_results in by_sweep.values()
    )
    if not has_any_errors:
        return
    fig_lc, axes_lc = subplots_grid(len(sweep_vals), panel_w=5, panel_h=4, sharey=True)
    for ax, v in zip(axes_lc, sweep_vals, strict=False):
        _draw_convergence_panel(ax, v, by_sweep, styles, sweep_key)
    axes_lc[0].set_ylabel("Optim loss (MSE)")
    fig_lc.suptitle(
        f"{cfg.name} — R1 convergence curves (all {sweep_key} values)\n"
        "IC=X.XX annotated at curve end = final IC recovery error "
        "(✗ means IC error > threshold; dotted line = no gradient / flat loss)"
    )
    fig_shared_legend(fig_lc, axes_lc)
    if save:
        save_fig(fig_lc, "convergence_curves", out_dir)


def _imshow_panel(
    ax: Any, fig: Any, arr: Any, v_use: Any, cmap: str = "RdBu_r"
) -> None:
    """Wrap ``imshow_with_cbar`` with the recovery plots' shared options."""
    imshow_with_cbar(
        ax,
        fig,
        arr.T,
        origin="lower",
        cmap=cmap,
        vmin=-v_use,
        vmax=v_use,
        interpolation="nearest",
    )


def _plot_ic_field_comparison(
    cfg: Problem,
    npz: Any,
    solver_names: list,
    ic_true: np.ndarray,
    ic_init: np.ndarray,
    f_ic: Any,
    styles: dict,
    sweep_key: str,
    rep_horizon_str: str,
    out_dir: Path,
    save: bool,
) -> None:
    """Per-solver row of: true | perturbed | recovered | residual."""
    n_solvers = len(solver_names)
    ncols = 4
    fig_fld, axes_fld = plt.subplots(
        n_solvers, ncols, figsize=(ncols * 2.6, n_solvers * 2.6), squeeze=False
    )

    w_true = f_ic(ic_true)
    w_init = f_ic(ic_init)
    vmax = max(np.abs(w_true).max(), np.abs(w_init).max())

    for j, name in enumerate(solver_names):
        key = f"ic_rec_{j}"
        w_rec = f_ic(npz[key]) if key in npz else None
        lbl = styles.get(name, {}).get("label", name)

        for col, (title, arr, cm, v) in enumerate(
            [
                ("True IC", w_true, "RdBu_r", vmax),
                ("Perturbed IC", w_init, "RdBu_r", vmax),
                (
                    "Recovered IC",
                    w_rec if w_rec is not None else w_init,
                    "RdBu_r",
                    vmax,
                ),
                (
                    "Residual",
                    (w_rec - w_true) if w_rec is not None else np.zeros_like(w_true),
                    "RdBu_r",
                    None,
                ),
            ]
        ):
            ax = axes_fld[j, col]
            v_use = np.abs(arr).max() if v is None else v
            _imshow_panel(ax, fig_fld, arr, v_use, cmap=cm)
            if j == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(lbl, fontsize=8)
            ax.axis("off")

    fig_fld.suptitle(
        f"{cfg.name} — IC recovery fields ({sweep_key}={rep_horizon_str})", y=1.01
    )
    fig_fld.tight_layout()
    if save:
        save_fig(fig_fld, "recovery_fields", out_dir)


def _plot_final_state_comparison(
    cfg: Problem,
    npz: Any,
    solver_names: list,
    f_out: Any,
    styles: dict,
    sweep_key: str,
    rep_horizon: float,
    out_dir: Path,
    save: bool,
) -> None:
    """Final temporal state comparison (GT vs recovered rollout)."""
    n_solvers = len(solver_names)
    has_final = any(f"final_gt_{j}" in npz for j in range(n_solvers))
    if not has_final:
        return
    fig_fin, axes_fin = plt.subplots(
        n_solvers, 3, figsize=(3 * 2.6, n_solvers * 2.6), squeeze=False
    )
    for j, name in enumerate(solver_names):
        gt_key = f"final_gt_{j}"
        rec_key = f"final_rec_{j}"
        frv_key = f"final_rep_val_{j}"
        w_gt = f_out(npz[gt_key]) if gt_key in npz else None
        w_fr = f_out(npz[rec_key]) if rec_key in npz else None
        fin_val = float(npz[frv_key][0]) if frv_key in npz else rep_horizon
        lbl = styles.get(name, {}).get("label", name)
        vmax_fin = np.abs(w_gt).max() if w_gt is not None else 1.0
        panels = [
            ("GT final", w_gt, vmax_fin),
            ("Recovered rollout", w_fr, vmax_fin),
            (
                "Residual",
                (w_fr - w_gt) if (w_gt is not None and w_fr is not None) else None,
                None,
            ),
        ]
        for col, (title, arr, v) in enumerate(panels):
            ax = axes_fin[j, col]
            if arr is None:
                ax.axis("off")
                continue
            v_use = np.abs(arr).max() if v is None else v
            _imshow_panel(ax, fig_fin, arr, v_use)
            if j == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{lbl}\n({sweep_key}={fin_val})", fontsize=8)
            ax.axis("off")
    fig_fin.suptitle(
        f"{cfg.name} — final state comparison (best converged {sweep_key})", y=1.01
    )
    fig_fin.tight_layout()
    if save:
        save_fig(fig_fin, "recovery_final_states", out_dir)


def _draw_per_sigma_row(
    axes_sg: Any,
    fig_sg: Any,
    j: int,
    si: int,
    name: str,
    npz: Any,
    f_vis: Any,
    styles: dict,
    shared: dict,
) -> None:
    """Draw one solver row for the per-sigma all-solver grid.

    ``shared`` carries view-arrays + per-row constants computed once by the
    caller: ``w_ic_true``, ``w_ic_pert``, ``w_final_true``, ``vmax_ic``,
    ``vmax_fin``, ``col_titles``.
    """
    lbl = styles.get(name, {}).get("label", name)
    all_ic_key = f"ic_rec_all_{j}"
    all_fr_key = f"final_rec_all_{j}"
    all_fp_key = f"final_perturbed_all_{j}"
    w_ic_rec = f_vis(npz[all_ic_key][si]) if all_ic_key in npz else None
    w_fr_rec = f_vis(npz[all_fr_key][si]) if all_fr_key in npz else None
    w_fr_pert = f_vis(npz[all_fp_key][si]) if all_fp_key in npz else None
    vmax_ic = shared["vmax_ic"]
    vmax_fin = shared["vmax_fin"]
    panels = [
        (shared["w_ic_true"], vmax_ic),
        (shared["w_ic_pert"], vmax_ic),
        (w_ic_rec, vmax_ic),
        (shared["w_final_true"], vmax_fin),
        (w_fr_pert, vmax_fin),
        (w_fr_rec, vmax_fin),
    ]
    col_titles = shared["col_titles"]
    for col, (arr, vmax_p) in enumerate(panels):
        ax = axes_sg[j, col]
        if arr is None:
            ax.axis("off")
            continue
        v_use = vmax_p if vmax_p else np.abs(arr).max() or 1.0
        _imshow_panel(ax, fig_sg, arr, v_use)
        if j == 0:
            ax.set_title(col_titles[col])
        if col == 0:
            ax.set_ylabel(lbl, fontsize=8)
        ax.axis("off")


def _plot_per_sigma_grid(
    cfg: Problem,
    npz: Any,
    solver_names: list,
    ic_true: np.ndarray,
    f_vis: Any,
    styles: dict,
    sweep_key: str,
    out_dir: Path,
    save: bool,
) -> None:
    """One figure per sigma: rows=solvers, cols=[True/Pert/Rec IC | True/Pert/Rec Final]."""
    n_solvers = len(solver_names)
    has_all = any(f"ic_rec_all_{j}" in npz for j in range(n_solvers))
    if not (has_all and "sweep_values" in npz):
        return
    sweep_vals_arr = npz["sweep_values"]
    w_ic_true = f_vis(ic_true)
    w_final_true = f_vis(npz["final_gt_shared"]) if "final_gt_shared" in npz else None
    ic_perturbed_all = npz.get("ic_perturbed_all", None)
    ncols = 6
    col_titles = [
        "True IC",
        "Pert. IC",
        "Rec IC",
        "True Final",
        "Pert. Final",
        "Rec Final",
    ]
    vmax_ic = np.abs(w_ic_true).max() or 1.0
    vmax_fin = np.abs(w_final_true).max() if w_final_true is not None else 1.0
    for si, sv in enumerate(sweep_vals_arr):
        w_ic_pert = (
            f_vis(ic_perturbed_all[si]) if ic_perturbed_all is not None else None
        )
        fig_sg, axes_sg = plt.subplots(
            n_solvers,
            ncols,
            figsize=(ncols * 2.6, n_solvers * 2.6),
            squeeze=False,
        )
        shared = {
            "w_ic_true": w_ic_true,
            "w_ic_pert": w_ic_pert,
            "w_final_true": w_final_true,
            "vmax_ic": vmax_ic,
            "vmax_fin": vmax_fin,
            "col_titles": col_titles,
        }
        for j, name in enumerate(solver_names):
            _draw_per_sigma_row(
                axes_sg,
                fig_sg,
                j,
                si,
                name,
                npz,
                f_vis,
                styles,
                shared,
            )
        sv_str = f"{sv:.2g}".rstrip("0").rstrip(".")
        fig_sg.suptitle(f"{cfg.name} — {sweep_key}={sv_str} · all solvers", y=1.01)
        fig_sg.tight_layout()
        if save:
            save_fig(fig_sg, f"recovery_sigma_{sv_str}", out_dir)


def _solver_order_for(cfg_name: str) -> list[str]:
    """Heuristic solver-ordering pick by problem name (paper figure)."""
    if "ns" in cfg_name:
        return NS_ORDER
    if "structural" in cfg_name:
        return STRUCTURAL_ORDER
    if "thermal" in cfg_name:
        return THERMAL_ORDER
    return NS_ORDER


def _paper_plot_ns_recovery(
    ax: Any, data: dict, solver_order: list[str], title: str, seen: set[str]
) -> None:
    """Draw NS-style by_sweep convergence into *ax* from ``data``."""
    by_sweep = data.get("by_sweep") or data.get("by_horizon", {})

    # ``by_sweep`` is keyed by spec.name (display form); build alias→display.
    alias_to_display: dict[str, str] = {}
    for display_name in by_sweep:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name

    for alias in solver_order:
        display_name = alias_to_display.get(alias)
        if display_name is None:
            continue
        sweep = by_sweep[display_name]
        if not sweep:
            continue
        sigma_key = (
            _PAPER_NS_SIGMA
            if _PAPER_NS_SIGMA in sweep
            else sorted(sweep.keys())[len(sweep) // 2]
        )
        errors = sweep[sigma_key].get("errors") or sweep[sigma_key].get(
            "ic_error_history", []
        )
        if not errors:
            continue
        _label, color, ls, _mk = solver_props(alias)
        ax.semilogy(
            range(len(errors)), errors, color=color, linestyle=ls, linewidth=1.6
        )
        seen.add(alias)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("IC error")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _plot_paper_recovery_experiment(
    cfg: Problem,
    *,
    exp_key: str,
    suffix: str,
    save: bool,
) -> plt.Figure | None:
    """Single-experiment paper-styled IC-recovery convergence figure.

    Draws per-solver IC-error vs iteration on a log scale (using the
    representative sigma slice, see ``_PAPER_NS_SIGMA``). Reads
    ``result.json`` from the experiment directory and writes
    ``<exp_key>.pdf`` next to it when ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[recovery] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)
    data = load_json(result_path)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.45), dpi=300)
    fig.subplots_adjust(bottom=0.36, top=0.90, left=0.13, right=0.96)

    seen: set[str] = set()
    solver_order = _solver_order_for(cfg.name)

    if "by_sweep" in data or "by_horizon" in data:
        _paper_plot_ns_recovery(
            ax,
            data,
            solver_order,
            f"IC recovery — {cfg.category_label or cfg.name}",
            seen,
        )
    else:
        # FEM-style by_solver layout (errors / losses per solver).
        by_solver = data.get("by_solver", {})
        error_key = (
            "errors" if any("errors" in v for v in by_solver.values()) else "losses"
        )
        # ``by_solver`` keyed by spec.name; bridge to alias for ordering.
        alias_to_display: dict[str, str] = {}
        for display_name in by_solver:
            a = resolve_solver_alias(display_name)
            if a is not None:
                alias_to_display[a] = display_name
        for alias in solver_order:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            sdata = by_solver.get(display_name)
            if not sdata:
                continue
            vals = sdata.get(error_key, [])
            if not vals:
                continue
            _label, color, ls, _mk = solver_props(alias)
            ax.semilogy(
                range(len(vals)), vals, color=color, linestyle=ls, linewidth=1.6
            )
            seen.add(alias)
        ax.set_title(f"Recovery — {cfg.category_label or cfg.name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error" if error_key == "errors" else "Loss")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    handles = dedup_handles([make_handle(s) for s in solver_order if s in seen])
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=7.5,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / f"{exp_key}.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def plot_recovery(
    cfg: Problem,
    threshold: float | None = None,
    *,
    field_to_2d: Any = None,
    ic_to_2d: Any = None,
    save: bool = True,
    suffix: str = "",
    ic: str | None = None,
    exp_key: str = "optimization",
    **_kw: Any,
) -> Any:
    """Recovery per-experiment plot — paper figure + extras.

    The canonical paper-styled IC-recovery convergence figure is generated
    inline by :func:`_plot_paper_recovery_experiment` so the polished paper
    styling lands on the experiment results dir too. This wrapper
    additionally produces:

      * ``optimization`` — recovery improvement + min-loss summary panel.
      * ``convergence_curves`` — per-sweep-value loss curves grid.
      * ``recovery_fields`` — per-solver true/perturbed/recovered/residual.
      * ``recovery_final_states`` — GT vs recovered rollout panels.
      * ``recovery_sigma_<v>`` — all-solvers-per-sigma grid.
      * ``recovery_evolution_<solver>.gif`` — IC reconstruction animation.

    When results live in per-IC subdirectories (from
    ``--experiments <suite>/<exp>/<ic>`` runs), pass ``ic`` to select a specific
    IC (e.g. ``ic="multimode"``).  If the root-level
    ``result.json`` is not found, the function automatically falls back to the
    first available IC subdirectory.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    out_dir = _resolve_recovery_out_dir(base_dir, ic)

    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)

    # Use threshold recorded in the experiment params; fall back to argument default.
    if threshold is None:
        threshold = (
            data.get("params", {}).get("optim", {}).get("failure_threshold", 0.5)
        )

    # Support both new schema (by_sweep/failure_values) and old (by_horizon/failure_horizons)
    by_sweep = data.get("by_sweep") or data.get("by_horizon", {})
    sweep_key = data.get("sweep_key", "steps")
    sweep_vals = _sorted_sweep_vals(by_sweep)

    # When ic_error_init was not recorded (older runs), estimate it from the npz
    # for sigma sweeps: compute exact error at rep_val, scale linearly for others.
    fallback_ic_error_init = _compute_fallback_ic_error_init(
        out_dir, by_sweep, sweep_vals, sweep_key
    )

    # ── Canonical paper figure (inlined) ──────────────────────────────────────
    # When the experiment lives in a per-IC subdir, reflect the suffix so the
    # paper figure lands next to result.json in the IC subdir.
    paper_suffix = suffix
    if out_dir != base_dir and out_dir.name != base_dir.name:
        paper_suffix = f"{suffix}/{out_dir.name}"
    _plot_paper_recovery_experiment(
        cfg, exp_key=exp_key, suffix=paper_suffix, save=save
    )

    # ── recovery.png: 2 panels ─────────────────────────────────────────────────
    fig_r = _plot_recovery_summary(
        cfg,
        by_sweep,
        sweep_vals,
        sweep_key,
        styles,
        fallback_ic_error_init,
        out_dir,
        save,
    )

    # ── convergence curves (all sweep values) + IC error consensus ───────────
    _plot_convergence_curves(
        cfg, by_sweep, sweep_vals, sweep_key, styles, out_dir, save
    )

    # ── IC field comparison ────────────────────────────────────────────────────
    fields_path = out_dir / "recovery_fields.npz"
    if not fields_path.exists():
        return fig_r

    npz = try_load_npz(fields_path)
    rep_horizon = float(
        (npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0]
    )
    rep_horizon_str = f"{rep_horizon:g}"
    solver_names = npz["solver_names"].tolist()
    ic_true = npz["ic_true"]
    ic_init = npz["ic_init"]

    # Use ic_to_2d when set (e.g. n-body density contrast δ₀ slice),
    # then field_to_2d (e.g. 3D vorticity slice), then vorticity_2d for 2-D.
    f_ic = ic_to_2d or field_to_2d or vorticity_2d

    _plot_ic_field_comparison(
        cfg,
        npz,
        solver_names,
        ic_true,
        ic_init,
        f_ic,
        styles,
        sweep_key,
        rep_horizon_str,
        out_dir,
        save,
    )
    if save:
        _render_recovery_evolution_gifs(
            out_dir, npz, solver_names, f_ic, styles, sweep_key, rep_horizon
        )

    # ── Final temporal state comparison (GT vs recovered rollout) ────────────
    f_out = ic_to_2d or field_to_2d or vorticity_2d
    _plot_final_state_comparison(
        cfg,
        npz,
        solver_names,
        f_out,
        styles,
        sweep_key,
        rep_horizon,
        out_dir,
        save,
    )

    # ── Per-sigma all-solver grid ─────────────────────────────────────────────
    f_vis = ic_to_2d or field_to_2d or vorticity_2d
    _plot_per_sigma_grid(
        cfg,
        npz,
        solver_names,
        ic_true,
        f_vis,
        styles,
        sweep_key,
        out_dir,
        save,
    )

    return fig_r


def _render_recovery_evolution_gifs(
    out_dir: Path,
    npz: Any,
    solver_names: list,
    f_ic: Any,
    styles: dict,
    sweep_key: str,
    rep_horizon: Any,
) -> None:
    """Write ``recovery_evolution_<solver>.gif`` per solver from ``ic_history_<j>``.

    Each frame is the 2-D scalar view of the IC at snapshot ``frame`` (same
    mapping used in the static ``recovery_fields`` panel: ``ic_to_2d`` or
    vorticity).  Shared vmin/vmax across frames keeps colouring stable so the
    viewer can see the IC re-form rather than a flicker from autoscaling.
    Silently skips solvers without a recorded history.
    """
    for j, name in enumerate(solver_names):
        hist_key = f"ic_history_{j}"
        if hist_key not in npz.files:
            continue
        history = np.asarray(npz[hist_key])  # (n_frames, *ic_shape)
        if history.ndim < 2 or history.shape[0] == 0:
            continue
        n_frames = int(history.shape[0])

        # Collapse IC to 2-D per frame using the same vorticity/slice helper.
        frames_2d = [f_ic(history[i]) for i in range(n_frames)]
        vmax = float(max(np.abs(arr).max() for arr in frames_2d)) or 1.0

        label = styles.get(name, {}).get("label", name)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            frames_2d[0].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        title = ax.set_title(
            f"{label} — snapshot 1 / {n_frames}  ({sweep_key}={rep_horizon})",
            fontsize=9,
        )
        ax.axis("off")
        fig.tight_layout()

        def _update(
            idx: Any,
            _im: Any = im,
            _title: Any = title,
            _frames: Any = frames_2d,
            _label: Any = label,
            _n: Any = n_frames,
            _sk: Any = sweep_key,
            _sv: Any = rep_horizon,
        ) -> Any:
            _im.set_data(_frames[idx].T)
            _title.set_text(f"{_label} — snapshot {idx + 1} / {_n}  ({_sk}={_sv})")
            return _im, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"recovery_evolution_{name}", out_dir, fps=4)
