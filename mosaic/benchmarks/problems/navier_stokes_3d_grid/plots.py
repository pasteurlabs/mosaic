"""Per-problem plots for the navier-stokes-3d-grid recovery experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
    fig_shared_legend,
    imshow_with_cbar,
    save_fig,
    solver_plot_props,
    solver_styles,
    subplots_grid,
    vorticity_2d,
)


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
    ax, v, by_sweep: dict, styles: dict, sweep_key: str
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


def _imshow_panel(ax, fig, arr, v_use, cmap="RdBu_r") -> None:
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
    npz,
    solver_names: list,
    ic_true: np.ndarray,
    ic_init: np.ndarray,
    f_ic,
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
    npz,
    solver_names: list,
    f_out,
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
    axes_sg,
    fig_sg,
    j: int,
    si: int,
    name: str,
    npz,
    f_vis,
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
    npz,
    solver_names: list,
    ic_true: np.ndarray,
    f_vis,
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


def plot_recovery(
    cfg: Problem,
    threshold: float | None = None,
    *,
    field_to_2d=None,
    ic_to_2d=None,
    save: bool = True,
    suffix: str = "",
    ic: str | None = None,
    exp_key: str = "optimization",
    **_kw,
):
    """Three files: error vs horizon + failure bars, loss curves, IC field comparison.

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
    npz,
    solver_names: list,
    f_ic,
    styles: dict,
    sweep_key: str,
    rep_horizon,
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
            idx,
            _im=im,
            _title=title,
            _frames=frames_2d,
            _label=label,
            _n=n_frames,
            _sk=sweep_key,
            _sv=rep_horizon,
        ):
            _im.set_data(_frames[idx].T)
            _title.set_text(f"{_label} — snapshot {idx + 1} / {_n}  ({_sk}={_sv})")
            return _im, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"recovery_evolution_{name}", out_dir, fps=4)
