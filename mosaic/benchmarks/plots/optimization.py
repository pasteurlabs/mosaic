"""Plots for the recovery suite (R1, R2, R3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.console import print_saved
from mosaic.benchmarks.core.utils import load_json, results_dir
from mosaic.benchmarks.plots.style import (
    apply_style,
    fig_shared_legend,
    imshow_with_cbar,
    save_fig,
    solver_plot_props,
    solver_styles,
    subplots_grid,
    vorticity_2d,
)

# ── Evolution-GIF helper ──────────────────────────────────────────────────────


def _save_animation(
    anim: manimation.FuncAnimation,
    stem: str,
    out_dir: Path,
    *,
    fps: int = 4,
) -> None:
    """Write *anim* to ``out_dir/<stem>.gif`` using Pillow, then close the figure.

    Wraps ``FuncAnimation.save`` with ``PillowWriter`` so callers do not have to
    import matplotlib.animation directly. Mirrors :func:`save_fig`'s close-on-
    save convention so each helper can build an animation and forget the figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{stem}.gif"
    writer = manimation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    print_saved(str(gif_path))
    plt.close(anim._fig)


apply_style()

_SUITE = "optimization"


def _rho_to_2d(
    rho: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """Per-cell density → 2D slice for visualisation.

    If ``params`` provides ``nx, ny, nz`` whose product equals ``n_cells``:
      - For ``nz == 1`` (quasi-2D slab), returns the full (ny, nx) top-down view.
      - For true 3-D (``nz > 1``), returns the mid-``y`` ``(nz, nx)`` cross-section.
    Otherwise falls back to the legacy heuristic assuming
    ``nx = 2·ny, nz = 1, n_cells = 2·ny²`` → returns ``(ny, nx)``.
    """
    n_cells = len(rho)
    if params is not None:
        nx = int(params.get("nx", 0))
        ny = int(params.get("ny", 0))
        nz = int(params.get("nz", 0))
        if nx * ny * nz == n_cells and nx > 0 and ny > 0 and nz > 0:
            # Storage convention matches _plot_topopt_3d: (nz, ny, nx)
            rho_xyz = rho.reshape(nz, ny, nx)
            if nz == 1:
                # Quasi-2D slab: return (ny, nx) top-down view.
                return rho_xyz[0]
            # True 3-D: mid-y cross-section → (nz, nx).
            return rho_xyz[:, ny // 2, :]
    ny_ = max(1, round((n_cells / 2) ** 0.5))
    nx_ = max(1, n_cells // ny_)
    return rho.reshape(ny_, nx_)


# ── R1 + R2: recovery vs horizon ─────────────────────────────────────────────


def plot_recovery(
    cfg: ProblemConfig,
    threshold: float | None = None,
    save: bool = True,
    suffix: str = "",
    ic: str | None = None,
    exp_key: str = "optimization",
):
    """Three files: error vs horizon + failure bars, loss curves, IC field comparison.

    When results live in per-IC subdirectories (from ``--ics`` runs), pass ``ic``
    to select a specific IC (e.g. ``ic="multimode"``).  If the root-level
    ``result.json`` is not found, the function automatically falls back to the
    first available IC subdirectory.
    """
    base_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    root_result = base_dir / "result.json"

    # Resolve the experiment directory: root-level, explicit IC, or auto-detected.
    if root_result.exists() and ic is None:
        out_dir = base_dir
    elif ic is not None:
        out_dir = base_dir / ic
    else:
        # Auto-detect: look for IC subdirectories with a result.json
        ic_dirs = sorted(
            p.parent for p in base_dir.glob("*/result.json") if p.parent != base_dir
        )
        if not ic_dirs:
            raise FileNotFoundError(
                f"No result.json found in {base_dir} or its subdirectories."
            )
        out_dir = ic_dirs[0]

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

    # Collect ordered sweep values from first solver's keys
    _first = next(iter(by_sweep.values()), {})
    sweep_vals = sorted(
        _first.keys(),
        key=lambda v: float(v) if str(v).replace(".", "").lstrip("-").isdigit() else 0,
    )

    # When ic_error_init was not recorded (older runs), estimate it from the npz
    # for sigma sweeps: compute exact error at rep_val, scale linearly for others.
    _fallback_ic_error_init: dict[float, float] = {}
    _has_ic_error_init = any(
        (s_results.get(v) or s_results.get(str(v)) or {}).get("ic_error_init")
        is not None
        for v in sweep_vals
        for s_results in by_sweep.values()
    )
    if not _has_ic_error_init and sweep_key == "perturb_sigma":
        _fp = out_dir / "recovery_fields.npz"
        if _fp.exists():
            _npz = np.load(_fp)
            if "ic_true" in _npz and "ic_init" in _npz:
                _ic_t = _npz["ic_true"].astype(float)
                _ic_i = _npz["ic_init"].astype(float)
                _rep_v = float(
                    (_npz.get("rep_val") or _npz.get("rep_horizon", np.array([0])))[0]
                )
                _ic_t_norm = float(np.sqrt(np.mean(_ic_t**2)))
                if _ic_t_norm > 0 and _rep_v > 0:
                    _rep_err = (
                        float(np.sqrt(np.mean((_ic_i - _ic_t) ** 2))) / _ic_t_norm
                    )
                    for _v in sweep_vals:
                        _fallback_ic_error_init[float(_v)] = (
                            _rep_err * float(_v) / _rep_v
                        )

    # ── recovery.png: 2 panels ─────────────────────────────────────────────────
    # Left:  IC recovery improvement vs sweep value (one line per solver)
    # Right: minimum achieved optimizer loss vs sweep value
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
            _ic_init_val = r.get("ic_error_init") or _fallback_ic_error_init.get(
                float(v)
            )
            if _ic_init_val:
                ys_ic.append((r["final_ic_error"] - _ic_init_val) / _ic_init_val)
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

    # ── convergence curves (all sweep values) + IC error consensus ───────────
    # One panel per sweep value showing the optimisation loss curves for every
    # solver.  Each panel also marks the final IC error as a text annotation so
    # the viewer can see at a glance which solvers actually converge in IC space
    # (small final_ic_error) vs those whose loss descends but IC error stays high.
    # Solvers that plateau immediately (flat loss = zero gradient, e.g. VJP=0)
    # appear as horizontal lines and are labelled with "(no gradient)".

    has_any_errors = any(
        (s_results.get(v) or s_results.get(str(v)) or {}).get("errors")
        for v in sweep_vals
        for s_results in by_sweep.values()
    )
    if has_any_errors:
        fig_lc, axes_lc = subplots_grid(
            len(sweep_vals), panel_w=5, panel_h=4, sharey=True
        )
        for ax, v in zip(axes_lc, sweep_vals):
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
        axes_lc[0].set_ylabel("Optim loss (MSE)")
        fig_lc.suptitle(
            f"{cfg.name} — R1 convergence curves (all {sweep_key} values)\n"
            "IC=X.XX annotated at curve end = final IC recovery error "
            "(✗ means IC error > threshold; dotted line = no gradient / flat loss)"
        )
        fig_shared_legend(fig_lc, axes_lc)
        if save:
            save_fig(fig_lc, "convergence_curves", out_dir)

    # ── IC field comparison ────────────────────────────────────────────────────
    fields_path = out_dir / "recovery_fields.npz"
    if not fields_path.exists():
        return fig_r

    npz = np.load(fields_path)
    rep_horizon = float(
        (npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0]
    )
    rep_horizon_str = f"{rep_horizon:g}"
    solver_names = npz["solver_names"].tolist()
    ic_true = npz["ic_true"]
    ic_init = npz["ic_init"]

    # Use cfg.ic_to_2d when set (e.g. n-body density contrast δ₀ slice),
    # then cfg.field_to_2d (e.g. 3D vorticity slice), then vorticity_2d for 2-D.
    f_ic = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d

    # One row per solver: true | perturbed | recovered | residual
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
            imshow_with_cbar(
                ax,
                fig_fld,
                arr.T,
                origin="lower",
                cmap=cm,
                vmin=-v_use,
                vmax=v_use,
                interpolation="nearest",
            )
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
        _render_recovery_evolution_gifs(
            out_dir, npz, solver_names, f_ic, styles, sweep_key, rep_horizon
        )

    # ── Final temporal state comparison (GT vs recovered rollout) ────────────
    has_final = any(f"final_gt_{j}" in npz for j in range(len(solver_names)))
    if has_final:
        f_out = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d
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
                imshow_with_cbar(
                    ax,
                    fig_fin,
                    arr.T,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-v_use,
                    vmax=v_use,
                    interpolation="nearest",
                )
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

    # ── Per-sigma all-solver grid ─────────────────────────────────────────────
    # One figure per sigma: rows=solvers, cols=[True IC | Rec IC | True Final | Rec Final]
    has_all = any(f"ic_rec_all_{j}" in npz for j in range(len(solver_names)))
    if has_all and "sweep_values" in npz:
        sweep_vals_arr = npz["sweep_values"]
        f_vis = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d
        w_ic_true = f_vis(ic_true)
        w_final_true = (
            f_vis(npz["final_gt_shared"]) if "final_gt_shared" in npz else None
        )
        ic_perturbed_all = (
            npz["ic_perturbed_all"] if "ic_perturbed_all" in npz else None
        )
        ncols = 6
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
            vmax_ic = np.abs(w_ic_true).max() or 1.0
            vmax_fin = np.abs(w_final_true).max() if w_final_true is not None else 1.0
            col_titles = [
                "True IC",
                "Pert. IC",
                "Rec IC",
                "True Final",
                "Pert. Final",
                "Rec Final",
            ]
            for j, name in enumerate(solver_names):
                lbl = styles.get(name, {}).get("label", name)
                all_ic_key = f"ic_rec_all_{j}"
                all_fr_key = f"final_rec_all_{j}"
                all_fp_key = f"final_perturbed_all_{j}"
                w_ic_rec = f_vis(npz[all_ic_key][si]) if all_ic_key in npz else None
                w_fr_rec = f_vis(npz[all_fr_key][si]) if all_fr_key in npz else None
                w_fr_pert = f_vis(npz[all_fp_key][si]) if all_fp_key in npz else None
                panels = [
                    (w_ic_true, vmax_ic),
                    (w_ic_pert, vmax_ic),
                    (w_ic_rec, vmax_ic),
                    (w_final_true, vmax_fin),
                    (w_fr_pert, vmax_fin),
                    (w_fr_rec, vmax_fin),
                ]
                for col, (arr, vmax_p) in enumerate(panels):
                    ax = axes_sg[j, col]
                    if arr is None:
                        ax.axis("off")
                        continue
                    v_use = vmax_p if vmax_p else np.abs(arr).max() or 1.0
                    imshow_with_cbar(
                        ax,
                        fig_sg,
                        arr.T,
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=-v_use,
                        vmax=v_use,
                        interpolation="nearest",
                    )
                    if j == 0:
                        ax.set_title(col_titles[col])
                    if col == 0:
                        ax.set_ylabel(lbl, fontsize=8)
                    ax.axis("off")
            sv_str = f"{sv:.2g}".rstrip("0").rstrip(".")
            fig_sg.suptitle(f"{cfg.name} — {sweep_key}={sv_str} · all solvers", y=1.01)
            fig_sg.tight_layout()
            if save:
                save_fig(fig_sg, f"recovery_sigma_{sv_str}", out_dir)

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
    mapping used in the static ``recovery_fields`` panel: ``cfg.ic_to_2d`` or
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


def plot_recovery_evolution_sidebyside(
    cfg: ProblemConfig,
    exp_key: str = "optimization",
    suffix: str = "",
    save: bool = True,
) -> None:
    """Single GIF with all solver evolutions displayed side by side."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    fields_path = out_dir / "recovery_fields.npz"
    if not fields_path.exists():
        return

    npz = np.load(fields_path)
    solver_names = npz["solver_names"].tolist()
    rep_val = float((npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0])
    sweep_key = "perturb_sigma"

    f_ic = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d
    styles = solver_styles(cfg, differentiable_only=True)

    all_frames: list[list] = []
    active_names: list[str] = []
    for j, name in enumerate(solver_names):
        hist_key = f"ic_history_{j}"
        if hist_key not in npz.files:
            continue
        history = np.asarray(npz[hist_key])
        if history.ndim < 2 or history.shape[0] == 0:
            continue
        all_frames.append([f_ic(history[i]) for i in range(history.shape[0])])
        active_names.append(name)

    if not all_frames:
        return

    n_frames = min(len(f) for f in all_frames)
    vmax = (
        float(max(np.abs(arr).max() for frames in all_frames for arr in frames)) or 1.0
    )
    labels = [styles.get(n, {}).get("label", n) for n in active_names]

    n_solvers = len(active_names)
    fig, axes = plt.subplots(1, n_solvers, figsize=(n_solvers * 3, 3.5), squeeze=False)
    axes = axes[0]

    ims = []
    for ax, frames, label in zip(axes, all_frames, labels):
        im = ax.imshow(
            frames[0].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        ims.append(im)

    sup = fig.suptitle(
        f"{cfg.name} — IC evolution ({sweep_key}={rep_val:.2f}) — 1/{n_frames}",
        fontsize=9,
    )
    fig.tight_layout()

    def _update(idx):
        for im, frames in zip(ims, all_frames):
            im.set_data(frames[idx].T)
        sup.set_text(
            f"{cfg.name} — IC evolution ({sweep_key}={rep_val:.2f}) — {idx + 1}/{n_frames}"
        )
        return ims + [sup]

    anim = manimation.FuncAnimation(
        fig, _update, frames=n_frames, interval=250, blit=False
    )
    if save:
        _save_animation(anim, "recovery_evolution_all", out_dir, fps=4)


def plot_recovery_2panel(
    cfg: ProblemConfig,
    exp_key: str = "optimization",
    suffix: str = "",
    threshold: float | None = None,
    save: bool = True,
):
    """Two-panel figure: IC error vs sweep (left) | loss reduction vs sweep (right)."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)

    if threshold is None:
        threshold = (
            data.get("params", {}).get("optim", {}).get("failure_threshold", 0.5)
        )

    by_sweep = data.get("by_sweep") or data.get("by_horizon", {})
    sweep_key = data.get("sweep_key", "steps")

    _first = next(iter(by_sweep.values()), {})
    sweep_vals = sorted(
        _first.keys(),
        key=lambda v: float(v) if str(v).replace(".", "").lstrip("-").isdigit() else 0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for name, s_results in by_sweep.items():
        sty = styles.get(name, {})
        xs, ic_errors, loss_diffs = [], [], []
        for v in sweep_vals:
            r = s_results.get(v) or s_results.get(str(v))
            if r is None:
                continue
            xs.append(float(v))
            ic_errors.append(r["final_ic_error"])
            errors = r.get("errors") or []
            loss_diffs.append(
                float(errors[-1] - errors[0]) if len(errors) >= 2 else 0.0
            )

        if xs:
            kw = solver_plot_props(sty)
            ax1.semilogy(xs, ic_errors, label=sty.get("label", name), **kw)
            ax2.plot(xs, loss_diffs, label=sty.get("label", name), **kw)

    ax1.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold={threshold}")
    ax1.set_xlabel(sweep_key)
    ax1.set_ylabel("Final IC error (rel. L2)")
    ax1.set_title(f"IC recovery vs {sweep_key}")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel(sweep_key)
    ax2.set_ylabel("Final loss − Initial loss")
    ax2.set_title(f"Loss change vs {sweep_key}")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"{cfg.name} — {exp_key}")
    fig_shared_legend(fig, [ax1])

    if save:
        save_fig(fig, "recovery_2panel", out_dir)
    return fig


def plot_recovery_field_grid(
    cfg: ProblemConfig,
    exp_key: str = "optimization",
    suffix: str = "",
    save: bool = True,
):
    """n_solvers × 8 grid: True IC | Pert IC | Pert Res | Rec IC | Rec Res | True Final | Rec Final | Final Res."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    fields_path = out_dir / "recovery_fields.npz"
    if not fields_path.exists():
        return

    npz = np.load(fields_path)
    solver_names = npz["solver_names"].tolist()
    rep_val = float((npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0])

    f_ic = cfg.ic_to_2d or cfg.field_to_2d or vorticity_2d
    styles = solver_styles(cfg, differentiable_only=True)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    w_true = f_ic(npz["ic_true"])
    w_init = f_ic(npz["ic_init"])

    # Collect per-solver arrays and compute shared vmaxes up front.
    solver_arrays = []
    for j in range(len(solver_names)):
        w_rec = f_ic(npz[f"ic_rec_{j}"]) if f"ic_rec_{j}" in npz else None
        w_fgt = f_ic(npz[f"final_gt_{j}"]) if f"final_gt_{j}" in npz else None
        w_frec = f_ic(npz[f"final_rec_{j}"]) if f"final_rec_{j}" in npz else None
        solver_arrays.append((w_rec, w_fgt, w_frec))

    n_solvers = len(solver_names)
    ncols = 7
    col_titles = [
        "True IC",
        "Perturbed IC",
        "Recovered IC",
        "Rec Residual",
        "True Final",
        "Rec Final",
        "Final Residual",
    ]

    cell = 2.2
    label_w = 0.8
    fig, axes = plt.subplots(
        n_solvers,
        ncols,
        figsize=(label_w + ncols * cell, n_solvers * cell),
        squeeze=False,
    )

    for j, name in enumerate(solver_names):
        lbl = styles.get(name, {}).get("label", name)
        w_rec, w_fgt, w_frec = solver_arrays[j]

        panels = [
            w_true,
            w_init,
            w_rec,
            w_rec - w_true if w_rec is not None else None,
            w_fgt,
            w_frec,
            w_frec - w_fgt if (w_frec is not None and w_fgt is not None) else None,
        ]

        for col, arr in enumerate(panels):
            ax = axes[j, col]
            ax.axis("off")
            if j == 0:
                ax.set_title(col_titles[col], fontsize=7, pad=2)
            if col == 0:
                ax.text(
                    -0.12,
                    0.5,
                    lbl,
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="right",
                    va="center",
                )
            if arr is None:
                continue
            ax.axis("on")
            v_use = np.abs(arr).max() or 1.0
            cmap = "PuOr_r" if col in (3, 6) else "RdBu_r"
            im = ax.imshow(
                arr.T,
                origin="lower",
                cmap=cmap,
                vmin=-v_use,
                vmax=v_use,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.03)
            fig.colorbar(im, cax=cax)

    fig.suptitle(
        f"{cfg.name} — field grid ({exp_key}, perturb_sigma={rep_val:.2f})", fontsize=9
    )
    fig.tight_layout()
    if save:
        save_fig(fig, "recovery_field_grid", out_dir)
    return fig


# ── R3: parameter recovery ────────────────────────────────────────────────────


def plot_param_recovery(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Two files: final IC error bar chart + optimisation loss curves per sweep param."""
    out_dir = results_dir() / cfg.name / _SUITE / f"param_recovery{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)
    sweep_key = data.get("sweep_key", "param")

    by_param = data["by_param"]
    param_values = sorted(by_param.keys(), key=float)
    solvers = [
        n for n in cfg.solvers if getattr(cfg.solvers[n], "differentiable", True)
    ]
    x = np.arange(len(param_values))
    width = 0.8 / len(solvers)

    # ── final error bar chart ─────────────────────────────────────────────────
    fig_bar, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(solvers):
        errs = [
            by_param[val].get(name, {}).get("final_ic_error", np.nan)
            if isinstance(by_param[val].get(name), dict)
            else np.nan
            for val in param_values
        ]
        ax.bar(
            x + i * width,
            errs,
            width,
            label=styles[name]["label"],
            color=cfg.solvers[name].color,
        )
    ax.set_xticks(x + width * len(solvers) / 2)
    ax.set_xticklabels([f"{sweep_key}={v}" for v in param_values])
    ax.set_ylabel("Final IC error (rel. L2)")
    ax.set_title(f"{cfg.name} — R3 recovery vs {sweep_key}")
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    fig_shared_legend(fig_bar, [ax])
    if save:
        save_fig(fig_bar, "param_recovery", out_dir)

    # ── loss convergence curves ────────────────────────────────────────────────
    has_curves = any(
        isinstance(by_param[val].get(n), dict) and by_param[val][n].get("errors")
        for val in param_values
        for n in solvers
    )
    if not has_curves:
        return fig_bar

    fig_lc, axes_lc = subplots_grid(
        len(param_values), panel_w=5, panel_h=4, sharey=True
    )
    for ax, val in zip(axes_lc, param_values):
        for name in solvers:
            r = by_param[val].get(name)
            if isinstance(r, dict) and r.get("errors"):
                ax.semilogy(
                    r["errors"],
                    label=styles[name]["label"],
                    **solver_plot_props(styles[name], marker=False),
                )
        ax.set_xlabel("Iteration")
        ax.set_title(f"{sweep_key}={val}")
    axes_lc[0].set_ylabel("MSE loss")
    fig_lc.suptitle(f"{cfg.name} — R3 convergence curves by {sweep_key}")
    fig_shared_legend(fig_lc, axes_lc)
    if save:
        save_fig(fig_lc, "convergence_curves", out_dir)
    return fig_bar


# ── R4: viscosity recovery ────────────────────────────────────────────────────


def plot_viscosity_recovery(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Three files: μ recovery paths, loss curves, and final error bar chart."""
    out_dir = results_dir() / cfg.name / _SUITE / f"viscosity_recovery{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)
    solvers = list(data["by_solver"].keys())
    mu_trues = sorted(data["mu_true_values"], key=float)
    mu_init = data["mu_init"]

    # ── μ recovery paths: one panel per μ_true ────────────────────────────────
    fig_p, axes_p = subplots_grid(len(mu_trues), panel_w=4.5, panel_h=3.5)
    for ax, mu_true in zip(axes_p, mu_trues):
        ax.axhline(mu_true, color="k", linewidth=1.2, linestyle="--", label="μ_true")
        ax.axhline(mu_init, color="gray", linewidth=0.8, linestyle=":", label="μ_init")
        for name in solvers:
            solver_data = data["by_solver"][name]
            r = solver_data.get(str(mu_true)) or solver_data.get(mu_true)
            if not isinstance(r, dict):
                continue
            ax.semilogy(
                r["mu_path"],
                label=styles[name]["label"],
                **solver_plot_props(styles[name], marker=False),
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("μ (log scale)")
        ax.set_title(f"μ_true = {mu_true}")
    fig_p.suptitle(f"{cfg.name} — R4 viscosity recovery paths")
    fig_shared_legend(fig_p, axes_p)
    if save:
        save_fig(fig_p, "mu_paths", out_dir)

    # ── loss convergence curves ────────────────────────────────────────────────
    fig_l, axes_l = subplots_grid(len(mu_trues), panel_w=4.5, panel_h=3.5, sharey=True)
    for ax, mu_true in zip(axes_l, mu_trues):
        for name in solvers:
            solver_data = data["by_solver"][name]
            r = solver_data.get(str(mu_true)) or solver_data.get(mu_true)
            if isinstance(r, dict) and r.get("errors"):
                ax.semilogy(
                    r["errors"],
                    label=styles[name]["label"],
                    **solver_plot_props(styles[name], marker=False),
                )
        ax.set_xlabel("Iteration")
        ax.set_title(f"μ_true = {mu_true}")
    axes_l[0].set_ylabel("MSE loss")
    fig_l.suptitle(f"{cfg.name} — R4 loss convergence")
    fig_shared_legend(fig_l, axes_l)
    if save:
        save_fig(fig_l, "loss_curves", out_dir)

    # ── final relative error bar chart ────────────────────────────────────────
    x = np.arange(len(mu_trues))
    width = 0.8 / max(len(solvers), 1)
    fig_b, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(solvers):
        solver_data = data["by_solver"][name]
        errs = [
            r["mu_rel_error"]
            if isinstance(r := solver_data.get(str(mu)) or solver_data.get(mu), dict)
            else np.nan
            for mu in mu_trues
        ]
        ax.bar(
            x + i * width,
            errs,
            width,
            label=styles[name]["label"],
            color=cfg.solvers[name].color,
        )
    ax.set_xticks(x + width * len(solvers) / 2)
    ax.set_xticklabels([f"μ={v}" for v in mu_trues])
    ax.set_ylabel("Relative error  |μ_hat − μ_true| / μ_true")
    ax.set_title(f"{cfg.name} — R4 final viscosity recovery error")
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    fig_shared_legend(fig_b, [ax])
    if save:
        save_fig(fig_b, "final_error", out_dir)

    return fig_p


# ── R5: recovery vs perturbation sigma ───────────────────────────────────────


def plot_recovery_sigma_sweep(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Two files: final IC error vs σ (phase-transition plot) + loss curves.

    Produces:
      - sigma_recovery.png: final IC error vs perturbation σ for each solver;
        horizontal threshold line marks the success/failure boundary.
      - convergence_curves.png: optimisation loss curves at each σ.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"recovery_sigma_sweep{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=True)
    sigma_values = sorted(data.get("sigma_values", []), key=float)
    by_sigma = data["by_sigma"]
    threshold = data.get("params", {}).get("optim", {}).get("failure_threshold", 0.1)
    solvers = list({name for sigma_dict in by_sigma.values() for name in sigma_dict})

    # ── final IC error vs σ ───────────────────────────────────────────────────
    fig_s, ax = plt.subplots(figsize=(8, 5))
    for name in solvers:
        errs = []
        for sigma in sigma_values:
            r = (
                by_sigma[str(sigma)]
                if str(sigma) in by_sigma
                else by_sigma.get(sigma, {})
            )
            entry = r.get(name) if isinstance(r, dict) else None
            if isinstance(entry, dict):
                errs.append(entry.get("final_ic_error", np.nan))
            else:
                errs.append(np.nan)
        sty = styles.get(name, {})
        ax.semilogx(
            sigma_values,
            errs,
            label=sty.get("label", name),
            **solver_plot_props(sty),
        )
    ax.axhline(
        threshold,
        color="gray",
        ls="--",
        lw=1,
        label=f"threshold={threshold}",
    )
    ax.set_xlabel("Perturbation σ")
    ax.set_ylabel("Final IC error (rel. L2)")
    ax.set_title(f"{cfg.name} — R5 IC recovery vs perturbation σ")
    ax.grid(True, which="both", alpha=0.3)
    fig_shared_legend(fig_s, [ax])
    if save:
        save_fig(fig_s, "sigma_recovery", out_dir)

    # ── loss curves at each σ ─────────────────────────────────────────────────
    has_curves = any(
        isinstance(
            (by_sigma[str(s)] if str(s) in by_sigma else by_sigma.get(s, {})).get(n),
            dict,
        )
        and (by_sigma[str(s)] if str(s) in by_sigma else by_sigma.get(s, {}))
        .get(n, {})
        .get("errors")
        for s in sigma_values
        for n in solvers
    )
    if has_curves:
        fig_lc, axes_lc = subplots_grid(
            len(sigma_values), panel_w=5, panel_h=4, sharey=True
        )
        for ax_lc, sigma in zip(axes_lc, sigma_values):
            sigma_dict = (
                by_sigma[str(sigma)]
                if str(sigma) in by_sigma
                else by_sigma.get(sigma, {})
            )
            for name in solvers:
                r = sigma_dict.get(name) if isinstance(sigma_dict, dict) else None
                if isinstance(r, dict) and r.get("errors"):
                    sty = styles.get(name, {})
                    ax_lc.semilogy(
                        r["errors"],
                        label=sty.get("label", name),
                        **solver_plot_props(sty, marker=False),
                    )
            ax_lc.set_xlabel("Iteration")
            ax_lc.set_title(f"σ={sigma}")
        axes_lc[0].set_ylabel("MSE loss")
        fig_lc.suptitle(f"{cfg.name} — R5 loss curves by σ")
        fig_shared_legend(fig_lc, axes_lc)
        if save:
            save_fig(fig_lc, "convergence_curves", out_dir)

    return fig_s


# ── T1: topology optimisation ─────────────────────────────────────────────────


def plot_topopt(
    cfg: ProblemConfig, save: bool = True, suffix: str = "", exp_key: str = "topopt"
):
    """Two files: compliance + volume fraction convergence; initial + final density fields."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=False)
    # params is the full run dict {ic, physics, optim, ...}; look for v_frac in
    # physics sub-dict first (structural-mesh layout), then directly at top level.
    _params_raw = data.get("params", {}) or {}
    _phys = _params_raw.get("physics", {}) or {}
    v_frac = _phys.get("v_frac") or _params_raw.get("v_frac", 0.5)
    # For field visualisation (_rho_to_2d), pass the physics sub-dict so that
    # nx/ny/nz are found at the correct nesting level.
    params_all = _phys if _phys else _params_raw

    by_solver = data["by_solver"]

    # ── compliance + volume fraction convergence ──────────────────────────────
    fig_c, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(12, 4))
    for name, res in by_solver.items():
        kw = solver_plot_props(styles[name], marker=False)
        ax_c.semilogy(res["compliances"], label=styles[name]["label"], **kw)
        ax_v.plot(res["vol_fracs"], label=styles[name]["label"], **kw)
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Compliance")
    ax_c.set_title("T1 — compliance minimisation")
    ax_v.axhline(v_frac, color="gray", ls="--", lw=1, label=f"target={v_frac}")
    ax_v.set_xlabel("Iteration")
    ax_v.set_ylabel("Volume fraction")
    ax_v.set_title("T1 — volume fraction")
    fig_c.suptitle(f"{cfg.name} — topology optimisation")
    fig_shared_legend(fig_c, [ax_c, ax_v])
    if save:
        save_fig(fig_c, "topopt_convergence", out_dir)

    # ── density field panels ──────────────────────────────────────────────────
    fields_path = out_dir / "topopt_fields.npz"
    if not fields_path.exists():
        return fig_c

    npz = np.load(fields_path)
    solver_names = npz["solver_names"].tolist()
    n_panels = 1 + len(solver_names)
    fig_f, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3, 3), squeeze=False)

    panels = [("Initial ρ", npz["rho_init"])] + [
        (styles.get(n, {}).get("label", n) + " final", npz[f"rho_final_{j}"])
        for j, n in enumerate(solver_names)
        if f"rho_final_{j}" in npz
    ]
    im = None
    for ax, (title, rho) in zip(axes[0], panels):
        im = ax.imshow(
            _rho_to_2d(rho, params_all),
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    if im is not None:
        plt.colorbar(im, ax=axes[0][-1], fraction=0.04)
    fig_f.suptitle(f"{cfg.name} — optimised density fields")
    fig_f.tight_layout()
    if save:
        save_fig(fig_f, "topopt_fields", out_dir)

    # ── density evolution GIFs ────────────────────────────────────────────────
    if save:
        _render_topopt_evolution_gifs(out_dir, npz, solver_names, params_all, styles)

    # ── 3D voxel plots of final density ──────────────────────────────────────
    if save:
        _plot_topopt_3d(cfg, npz, solver_names, by_solver, out_dir, styles, params_all)

    return fig_c


def _plot_topopt_3d(
    cfg: ProblemConfig,
    npz,
    solver_names: list,
    by_solver: dict,
    out_dir: Path,
    styles: dict,
    params: dict | None = None,
    threshold: float = 0.35,
) -> None:
    """3-D voxel plots of the final optimised density field, one per solver.

    Saves ``topopt_3d_{solver_name}.png`` in *out_dir*.  The voxel array is
    reshaped from the flat (n_cells,) layout to (nx, ny, nz) so that matplotlib's
    ``Axes3D.voxels()`` maps naturally to (length, width, height).  Voxels with
    ρ > *threshold* are shown; colour is steel-blue (solid material) with alpha
    proportional to density so partially-dense cells appear translucent.

    Problem-specific annotations are inferred from *params*:
    - ``corner_load=True``:  red marker at the corner load point (x=Lx, y=0, z=0).
    - Always:                translucent blue plane on the fixed/clamped face (x=0).

    Works for any problem using a 2:1:1 hex mesh (structural-mesh, thermal-mesh, …).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

    params = params or {}
    corner_load = params.get("corner_load", False)

    # Steel-blue RGB for solid material
    _SOLID_RGB = np.array([0.267, 0.467, 0.667])  # #4477AA

    for j, name in enumerate(solver_names):
        key = f"rho_final_{j}"
        if key not in npz:
            continue

        rho_flat = npz[key]
        n_cells = len(rho_flat)

        # Prefer explicit (nx, ny, nz) from params when they match n_cells;
        # otherwise fall back to the quasi-2D heuristic (legacy results).
        nx_p = int(params.get("nx", 0))
        ny_p = int(params.get("ny", 0))
        nz_p = int(params.get("nz", 0))
        if nx_p * ny_p * nz_p == n_cells and nx_p * ny_p * nz_p > 0:
            nx_, ny_, nz_ = nx_p, ny_p, nz_p
        else:
            # Infer quasi-2D layout (nz=1, nx=2·ny): n_cells = 2·ny²
            ny_ = max(1, round((n_cells / 2) ** 0.5))
            nx_ = max(1, n_cells // ny_)
            nz_ = 1

        # Reshape: flat → (nz, ny, nx) → (nx, ny, nz) for matplotlib voxels axes
        rho_xyz = rho_flat.reshape(nz_, ny_, nx_).transpose(2, 1, 0)  # (nx, ny, nz)

        filled = rho_xyz > threshold

        # Face colours: steel-blue for all filled voxels; alpha scales with density
        # so near-threshold cells appear translucent and solid cells opaque.
        fc = np.zeros(rho_xyz.shape + (4,))
        fc[..., :3] = _SOLID_RGB
        # Remap density from [threshold, 1] → [0.35, 0.92] for alpha
        alpha = np.where(
            filled,
            0.35 + 0.57 * (rho_xyz - threshold) / (1.0 - threshold + 1e-8),
            0.0,
        )
        fc[..., 3] = alpha

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.voxels(filled, facecolors=fc, edgecolor="none")

        # Fixed/clamped face: translucent blue plane at x=0
        yy, zz = np.meshgrid([0, ny_], [0, nz_])
        xx = np.zeros_like(yy)
        ax.plot_surface(xx, yy, zz, alpha=0.12, color="#4477AA", linewidth=0)

        # Corner load annotation (structural-mesh only).
        # The load is applied at the bottom-front corner of the right face
        # (x=Lx, y=0, z=0 in physical coords → voxel coords nx_-0.5, 0.5, 0.5).
        # Draw it AFTER the voxels so it renders on top, use depthshade=False to
        # prevent matplotlib from darkening the marker, and add a quiver arrow
        # pointing upward (+z) so the force direction is unambiguous.
        if corner_load:
            ax.scatter(
                [nx_ - 0.5],
                [0.5],
                [0.5],
                color="#EE3333",
                s=180,
                zorder=10,
                depthshade=False,
                label="load (↑z)",
            )
            # Arrow: base slightly below the load point, pointing upward (+z).
            # Length ~15 % of nz_ so it is proportional to the mesh.
            arrow_len = max(0.8, nz_ * 0.15)
            ax.quiver(
                nx_ - 0.5,
                0.5,
                0.5 - arrow_len,
                0,
                0,
                arrow_len,
                color="#EE3333",
                linewidth=2,
                arrow_length_ratio=0.4,
                zorder=10,
            )
            ax.legend(fontsize=8, loc="upper left")

        ax.set_xlabel("x  (length)", labelpad=4)
        ax.set_ylabel("y  (width)", labelpad=4)
        ax.set_zlabel("z  (height)", labelpad=4)
        # View from the right-front-top so both the clamped left face and the
        # bottom-front corner load on the right face are simultaneously visible.
        # azim=45 looks from the right side (load corner is on the near face);
        # elev=30 gives enough height to see the 3-D topology clearly.
        ax.view_init(elev=30, azim=45)

        label = styles.get(name, {}).get("label", name)
        compliance_val = by_solver.get(name, {}).get("final_compliance")
        title = f"{cfg.name} — {label}\noptimised topology  (ρ > {threshold})"
        if compliance_val is not None:
            title += f"    C = {compliance_val:.4e}"
        ax.set_title(title, fontsize=9)

        fig.tight_layout()
        save_fig(fig, f"topopt_3d_{name}", out_dir)
        plt.close(fig)


def _render_topopt_evolution_gifs(
    out_dir: Path,
    npz,
    solver_names: list,
    params_all: dict,
    styles: dict,
) -> None:
    """Write ``topopt_evolution_<solver>.gif`` per solver from ``rho_history_<j>``.

    Each frame is the 2-D view of the density field at snapshot ``frame``.
    Shared ``vmin=0/vmax=1`` keeps colouring comparable across frames.
    Silently skips solvers whose ``rho_history_<j>`` key is missing.
    """
    for j, name in enumerate(solver_names):
        hist_key = f"rho_history_{j}"
        if hist_key not in npz.files:
            continue
        history = np.asarray(npz[hist_key])  # (n_frames, n_cells)
        if history.size == 0 or history.shape[0] == 0:
            continue
        n_frames = int(history.shape[0])

        first = _rho_to_2d(history[0], params_all)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(
            first,
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        label = styles.get(name, {}).get("label", name)
        title = ax.set_title(f"{label} — iter 1 / {n_frames}", fontsize=9)
        ax.axis("off")
        fig.tight_layout()

        def _update(
            idx,
            _im=im,
            _title=title,
            _hist=history,
            _params=params_all,
            _label=label,
            _n=n_frames,
        ):
            _im.set_data(_rho_to_2d(_hist[idx], _params))
            _title.set_text(f"{_label} — iter {idx + 1} / {_n}")
            return _im, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"topopt_evolution_{name}", out_dir, fps=4)


def plot_source_recovery(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Three files: loss curves + final error bar chart, plus optimised source fields.

    The fields figure (``source_recovery_fields.{png,pdf}``) overlays
    ``source_init`` with every solver's ``source_final_<name>`` curve, and a
    second row of heatmaps shows ``source_history_<name>`` (iteration × space)
    for any solver that recorded a history.  Solvers missing from the npz
    render as a "no data" placeholder.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"source_recovery{suffix}"
    data = load_json(out_dir / "result.json")
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return

    styles = solver_styles(cfg, differentiable_only=False)

    fig, (ax_lc, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

    names = list(by_solver.keys())
    final_errors = []
    for name in names:
        res = by_solver[name]
        errors = res.get("errors", [])
        sty = styles.get(name, {})
        if errors:
            ax_lc.semilogy(
                errors,
                label=sty.get("label", name),
                **solver_plot_props(sty, marker=False),
            )
        final_errors.append(float(res.get("final_error", 0.0)))

    ax_lc.set_xlabel("Iteration")
    ax_lc.set_ylabel("Identification error")
    ax_lc.set_title("Source recovery — loss curves")
    ax_lc.legend(fontsize=8)

    colors = [cfg.solvers[n].color if n in cfg.solvers else "#888888" for n in names]
    labels = [styles.get(n, {}).get("label", n) for n in names]
    ax_bar.bar(labels, final_errors, color=colors)
    ax_bar.set_ylabel("Final identification error")
    ax_bar.set_title("Source recovery — final error")
    ax_bar.tick_params(axis="x", rotation=30)
    ax_bar.grid(axis="y")

    fig.suptitle(f"{cfg.name} — source identification recovery")
    fig.tight_layout()
    if save:
        save_fig(fig, "source_recovery", out_dir)

    # ── Optimised source-field visualisation (if npz exists) ─────────────────
    fields_path = out_dir / "source_fields.npz"
    if fields_path.exists():
        _plot_source_recovery_fields(
            cfg, fields_path, out_dir, names, styles, save=save
        )
        if save:
            _render_source_recovery_evolution_gifs(fields_path, out_dir, names, styles)
    return fig


def _render_source_recovery_evolution_gifs(
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
) -> None:
    """Write ``source_recovery_evolution_<solver>.gif`` per solver.

    Each frame is a 1-D line plot of ``source_history_<name>[frame, :]`` with
    a dashed ``source_init`` reference.  Y-range is fixed across frames
    (padded init + history min/max) so animation reads as real evolution.
    Silently skips solvers without a recorded history.
    """
    npz = np.load(fields_path)
    source_init = np.asarray(npz["source_init"]) if "source_init" in npz.files else None

    for name in solver_names:
        hkey = f"source_history_{name}"
        if hkey not in npz.files:
            continue
        hist = np.asarray(npz[hkey])  # (n_frames, n_cells)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        n_frames, n_cells = hist.shape
        xs = np.arange(n_cells)

        # Fix y-range across all frames for stable animation.
        extrema = [float(hist.min()), float(hist.max())]
        if source_init is not None:
            extrema.extend([float(source_init.min()), float(source_init.max())])
        ylo, yhi = min(extrema), max(extrema)
        pad = 0.05 * (yhi - ylo + 1e-12)
        ylo -= pad
        yhi += pad

        sty = styles.get(name, {})
        color = sty.get("color", "#3366CC")
        label = sty.get("label", name)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        if source_init is not None:
            ax.plot(
                xs,
                source_init,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="init",
            )
        (line,) = ax.plot(xs, hist[0], color=color, linewidth=1.8, label=label)
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Source intensity")
        ax.set_ylim(ylo, yhi)
        title = ax.set_title(f"{label} — snapshot 1 / {n_frames}", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        def _update(
            idx, _line=line, _title=title, _hist=hist, _label=label, _n=n_frames
        ):
            _line.set_ydata(_hist[idx])
            _title.set_text(f"{_label} — snapshot {idx + 1} / {_n}")
            return _line, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"source_recovery_evolution_{name}", out_dir, fps=4)


def _plot_source_recovery_fields(
    cfg: ProblemConfig,
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
    *,
    save: bool,
) -> None:
    """Render ``source_recovery_fields.{png,pdf}`` — init vs final per solver.

    Top panel: 1-D line plot with ``source_init``, a dashed ``source_truth``
    overlay (when present), and one line per solver's ``source_final_<name>``.
    Solvers whose ``source_final_<name>`` key is missing from the npz are
    skipped silently (no line drawn); the legend still shows the init.

    Middle row: per-solver ``source_final − source_truth`` error
    panel (line plot) so ill-posedness is visually obvious.

    Bottom row (only if any solver recorded ``source_history_<name>``): one
    heatmap per solver showing iteration-vs-space evolution.  Solvers missing
    from the npz render as a "no data" placeholder axis.
    """
    npz = np.load(fields_path)
    if "source_init" not in npz:
        return

    source_init = np.asarray(npz["source_init"])
    xs = np.arange(source_init.size)

    # Ground-truth source: dashed overlay + error subplot baseline.
    source_truth = (
        np.asarray(npz["source_truth"]) if "source_truth" in npz.files else None
    )

    # Determine which solvers have history data (for the optional heatmap row).
    history_keys = {
        name: f"source_history_{name}"
        for name in solver_names
        if f"source_history_{name}" in npz.files
    }
    has_history = bool(history_keys)
    has_truth = source_truth is not None and source_truth.shape == source_init.shape

    # Figure layout:
    #   - always: one line-plot axes (init + truth + final per solver).
    #   - if truth present: one error-line axes per solver.
    #   - if any history: one history-heatmap axes per solver.
    n_solvers = max(len(solver_names), 1)
    rows = 1 + (1 if has_truth else 0) + (1 if has_history else 0)
    height_ratios = [1.0]
    if has_truth:
        height_ratios.append(0.9)
    if has_history:
        height_ratios.append(1.1)

    fig, ax_grid = plt.subplots(
        rows,
        n_solvers,
        figsize=(max(8.0, n_solvers * 3.0), 3.2 * rows),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    # Merge the top row into a single line-plot axes.
    gs = ax_grid[0, 0].get_gridspec()
    for ax in ax_grid[0, :]:
        ax.remove()
    ax_line = fig.add_subplot(gs[0, :])

    row_cursor = 1
    if has_truth:
        ax_err = list(ax_grid[row_cursor, :])
        row_cursor += 1
    else:
        ax_err = []
    if has_history:
        ax_heat = list(ax_grid[row_cursor, :])
    else:
        ax_heat = []

    # ── Line plot: init + truth + final per solver ───────────────────────────
    ax_line.plot(
        xs,
        source_init,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="init",
    )
    if has_truth:
        ax_line.plot(
            xs,
            source_truth,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="ground truth",
        )
    for name in solver_names:
        key = f"source_final_{name}"
        if key not in npz.files:
            continue
        sty = styles.get(name, {})
        ax_line.plot(
            xs,
            np.asarray(npz[key]),
            label=sty.get("label", name),
            **solver_plot_props(sty, marker=False),
        )
    ax_line.set_xlabel("Cell index")
    ax_line.set_ylabel("Source intensity")
    title = (
        "Optimised source field (init, truth overlay, final per solver)"
        if has_truth
        else "Optimised source field (init vs final)"
    )
    ax_line.set_title(title)
    ax_line.grid(True, alpha=0.3)

    # ── Error row: source_final − source_truth per solver ────────────────────
    if has_truth:
        for j, name in enumerate(solver_names):
            ax = ax_err[j]
            label = styles.get(name, {}).get("label", name)
            key = f"source_final_{name}"
            if key not in npz.files:
                ax.set_title(f"{label}\n(no final)", fontsize=8)
                ax.axis("off")
                continue
            final = np.asarray(npz[key])
            if final.shape != source_truth.shape:
                ax.set_title(f"{label}\n(shape mismatch)", fontsize=8)
                ax.axis("off")
                continue
            err = final - source_truth
            color = styles.get(name, {}).get("color", "#3366CC")
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            ax.plot(xs, err, color=color, linewidth=1.3)
            ax.fill_between(xs, err, 0.0, color=color, alpha=0.18)
            emax = float(np.abs(err).max()) or 1.0
            ax.set_ylim(-1.05 * emax, 1.05 * emax)
            ax.set_title(f"{label}\nfinal − truth", fontsize=8)
            ax.set_xlabel("Cell index")
            if j == 0:
                ax.set_ylabel("Δ source")
            ax.grid(True, alpha=0.25)

    # ── Heatmap row (one per solver) ──────────────────────────────────────────
    for j, name in enumerate(solver_names):
        if not ax_heat:
            break
        ax = ax_heat[j]
        label = styles.get(name, {}).get("label", name)
        hkey = history_keys.get(name)
        if hkey is None:
            ax.set_title(f"{label}\n(no data)", fontsize=8)
            ax.axis("off")
            continue
        hist = np.asarray(npz[hkey])  # (n_snaps, n_cells)
        vmax = float(np.abs(hist).max()) or 1.0
        im = ax.imshow(
            hist,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{label}\nsource history", fontsize=8)
        ax.set_xlabel("Cell index")
        if j == 0:
            ax.set_ylabel("Snapshot")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(f"{cfg.name} — source recovery fields")
    fig_shared_legend(fig, [ax_line])
    if save:
        save_fig(fig, "source_recovery_fields", out_dir)


def plot_drag_opt(
    cfg: ProblemConfig, save: bool = True, suffix: str = "", exp_key: str = "drag_opt"
) -> list:
    """Two-panel plot per run: drag convergence curves + optimised inflow profiles.

    Also produces a separate figure (drag_opt_fields) showing velocity magnitude
    and vorticity of the final simulated flow field for each solver, when a
    ``flow_fields.npz`` file is present in the result directory.  Poor results
    (high drag, non-converged) are visually obvious as disordered flow patterns.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    base_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    styles = solver_styles(cfg)
    figs = []

    def _plot_one(data, profiles_path, out_dir):
        by_solver = data.get("by_solver", {})
        if not by_solver:
            return
        run_name = data.get("run_name", "")
        title_suffix = f" — {run_name}" if run_name else ""

        profiles = np.load(profiles_path) if profiles_path.exists() else {}
        solver_names = list(by_solver.keys())

        # ── Panel 1: drag reduction over initial drag ─────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_drag, ax_prof = axes

        for name in solver_names:
            d = by_solver[name]
            drags = d.get("drags", [])
            sty = styles.get(name, {})
            if drags and drags[0] and not np.isnan(drags[0]) and drags[0] != 0:
                drag_0 = drags[0]
                step = 5
                indices = list(range(0, len(drags), step))
                if (len(drags) - 1) % step != 0:
                    indices.append(len(drags) - 1)
                ax_drag.plot(
                    indices,
                    [drags[i] / drag_0 for i in indices],
                    label=sty.get("label", name),
                    **solver_plot_props(sty),
                )
        ax_drag.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax_drag.set_xlabel("iteration")
        ax_drag.set_ylabel("drag / drag₀")
        ax_drag.set_title(f"Drag reduction{title_suffix}")

        # ── Panel 2: inlet profiles ───────────────────────────────────────────
        if "initial" in profiles:
            N = len(profiles["initial"])
            y = np.linspace(0, 1, N, endpoint=False) + 0.5 / N
            ax_prof.plot(profiles["initial"], y, "k--", lw=1.5, label="initial")
            for name in solver_names:
                key = f"final_{name}"
                sty = styles.get(name, {})
                if key in profiles:
                    ax_prof.plot(
                        profiles[key],
                        y,
                        label=sty.get("label", name),
                        **solver_plot_props(sty),
                    )
        ax_prof.set_xlabel("u_x")
        ax_prof.set_ylabel("y")
        ax_prof.set_title(f"Inlet profile{title_suffix}")

        fig_shared_legend(fig, list(axes))
        fig.suptitle(f"{cfg.name} — drag optimisation{title_suffix}")
        fig.tight_layout()
        if save:
            save_fig(fig, "drag_opt", out_dir)
        figs.append(fig)

        # ── Flow field visualisation (velocity + vorticity) ───────────────────
        _plot_drag_opt_fields(data, out_dir, run_name, title_suffix, styles, save, figs)

        # ── Inflow profile evolution GIFs (one per solver) ───────────────────
        if save and "initial" in profiles:
            _render_drag_opt_evolution_gifs(
                profiles, out_dir, solver_names, styles, run_name
            )

    # Single-run layout
    single_path = base_dir / "result.json"
    single_result = load_json(single_path) if single_path.exists() else None
    if single_result is not None:
        _plot_one(single_result, base_dir / "profiles.npz", base_dir)
        return figs

    # Multi-run layout
    if base_dir.is_dir():
        for sub in sorted(base_dir.iterdir()):
            sub_data = load_json(sub / "result.json")
            if sub_data is not None:
                _plot_one(sub_data, sub / "profiles.npz", sub)
    return figs


def _vel_components_2d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_x, u_y) 2-D arrays from field (N, N, 1, 2) or (N, N, 2)."""
    if v.ndim == 4:
        v = v[:, :, 0, :]
    return v[..., 0], v[..., 1]


def _vel_magnitude_2d(v: np.ndarray) -> np.ndarray:
    """Velocity magnitude for a 2-D field with shape (N, N, 1, 2) or (N, N, 2)."""
    if v.ndim == 4:
        vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    else:
        vx, vy = v[:, :, 0], v[:, :, 1]
    return np.sqrt(vx**2 + vy**2)


def _vorticity_2d_arr(v: np.ndarray) -> np.ndarray:
    """Finite-difference vorticity for 2-D field (N, N, 1, 2) or (N, N, 2)."""
    if v.ndim == 4:
        vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    else:
        vx, vy = v[:, :, 0], v[:, :, 1]
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return dvydx - dvxdy


def _plot_drag_opt_fields(
    data: dict,
    out_dir,
    run_name: str,
    title_suffix: str,
    styles: dict,
    save: bool,
    figs: list,
) -> None:
    """Render u_x and u_y velocity component fields from ``flow_fields.npz``.

    The npz is expected to contain keys ``flow_initial`` and
    ``flow_final_{solver_name}`` with shape (N, N, 1, 2) or (N, N, 2).
    If the file does not exist the function silently returns.

    One row per solver (plus initial), two columns: u_x | u_y.
    Solvers that did not converge (converged=False in by_solver) are annotated
    with a red border so the failure is immediately visible.
    """
    fields_path = Path(out_dir) / "flow_fields.npz"
    if not fields_path.exists():
        return

    npz = np.load(fields_path)
    by_solver = data.get("by_solver", {})
    solver_names = [k for k in npz.files if k.startswith("flow_final_")]
    solver_names_clean = [k[len("flow_final_") :] for k in solver_names]

    if not solver_names:
        return

    # Rows: initial + one per solver.  Columns: velocity magnitude | vorticity.
    all_rows = ["__initial__"] + solver_names_clean
    n_rows = len(all_rows)
    ncols = 2
    fig_fld, axes_fld = plt.subplots(
        n_rows, ncols, figsize=(ncols * 3.5, n_rows * 3.0), squeeze=False
    )

    # Compute shared colour scales from the initial flow so all panels are comparable.
    flow_init = npz.get("flow_initial")
    if flow_init is not None:
        ux_init, uy_init = _vel_components_2d(flow_init)
        ux_vmax = float(np.percentile(np.abs(ux_init), 99)) or 1.0
        uy_vmax = float(np.percentile(np.abs(uy_init), 99)) or 1.0
    else:
        ux_vmax, uy_vmax = 1.0, 0.5

    def _render_row(
        row_idx: int, label: str, field: np.ndarray, converged: bool | None
    ):
        ux, uy = _vel_components_2d(field)

        for col, (arr, cmap, vmin, vmax, col_title) in enumerate(
            [
                (ux, "RdBu_r", -ux_vmax, ux_vmax, "$u_x$"),
                (uy, "RdBu_r", -uy_vmax, uy_vmax, "$u_y$"),
            ]
        ):
            ax = axes_fld[row_idx, col]
            imshow_with_cbar(
                ax,
                fig_fld,
                arr.T,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if row_idx == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            ax.axis("off")

        # Row label as text overlaid on the left side of the first column.
        # axis("off") hides set_ylabel, so use ax.text in axes coordinates.
        ax0 = axes_fld[row_idx, 0]
        ax0.text(
            -0.12,
            0.5,
            label,
            transform=ax0.transAxes,
            fontsize=8,
            va="center",
            ha="right",
            rotation=0,
            wrap=True,
        )

        # Red border annotation for non-converged / poor solvers
        if converged is False:
            for col in range(ncols):
                ax_c = axes_fld[row_idx, col]
                for spine in ax_c.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

    # Initial row
    if flow_init is not None:
        _render_row(0, "Initial flow", flow_init, converged=None)
    else:
        for col in range(ncols):
            axes_fld[0, col].set_visible(False)

    # One row per solver
    for i, sname in enumerate(solver_names_clean):
        key = f"flow_final_{sname}"
        field = npz.get(key)
        if field is None:
            for col in range(ncols):
                axes_fld[i + 1, col].set_visible(False)
            continue
        solver_info = by_solver.get(sname, {})
        converged = solver_info.get("converged")
        final_drag = solver_info.get("final_drag")
        sty = styles.get(sname, {})
        label = sty.get("label", sname)
        if final_drag is not None:
            label += f"\ndrag={final_drag:.4g}"
        if converged is False:
            label += "\n[NOT CONVERGED]"
        _render_row(i + 1, label, field, converged)

    fig_fld.suptitle(
        f"{run_name or 'drag_opt'} — optimised flow fields ($u_x$ | $u_y$)",
        y=1.01,
    )
    fig_fld.tight_layout()
    if save:
        save_fig(fig_fld, "drag_opt_fields", out_dir)
    figs.append(fig_fld)


def _render_drag_opt_evolution_gifs(
    profiles,
    out_dir,
    solver_names: list,
    styles: dict,
    run_name: str,
) -> None:
    """Write ``drag_opt_evolution_<solver>.gif`` per solver.

    Each frame is a 1-D line plot of the inflow profile u_x(y) at one
    optimisation snapshot, with the initial profile drawn dashed as a
    reference.  Skips solvers without a recorded ``profile_history_<name>``.
    """
    initial = np.asarray(profiles["initial"])
    N = initial.size
    y = np.linspace(0, 1, N, endpoint=False) + 0.5 / N

    # Treat np.load NpzFile *and* plain dict uniformly via .files / keys.
    keys = set(profiles.files) if hasattr(profiles, "files") else set(profiles.keys())

    for name in solver_names:
        hkey = f"profile_history_{name}"
        if hkey not in keys:
            continue
        hist = np.asarray(profiles[hkey])  # (n_snaps, N)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        n_frames = int(hist.shape[0])

        extrema = [
            float(hist.min()),
            float(hist.max()),
            float(initial.min()),
            float(initial.max()),
        ]
        xlo, xhi = min(extrema), max(extrema)
        pad = 0.05 * (xhi - xlo + 1e-12)
        xlo -= pad
        xhi += pad

        sty = styles.get(name, {})
        color = sty.get("color", "#3366CC")
        label = sty.get("label", name)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(initial, y, "k--", lw=1.4, label="initial")
        (line,) = ax.plot(hist[0], y, color=color, lw=2.0, label=label)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("u_x")
        ax.set_ylabel("y")
        title = ax.set_title(
            f"{label} — snapshot 1 / {n_frames}"
            + (f"  ({run_name})" if run_name else ""),
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        def _update(
            idx,
            _line=line,
            _title=title,
            _hist=hist,
            _label=label,
            _n=n_frames,
            _rn=run_name,
        ):
            _line.set_xdata(_hist[idx])
            _title.set_text(
                f"{_label} — snapshot {idx + 1} / {_n}" + (f"  ({_rn})" if _rn else "")
            )
            return _line, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"drag_opt_evolution_{name}", out_dir, fps=4)


# ── Conductivity recovery (thermal-mesh) ─────────────────────────────────────


def plot_conductivity_recovery(
    cfg: ProblemConfig,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "conductivity_recovery",
):
    """Two outputs: loss curves + final-error bar; recovered conductivity field comparison.

    Files written to ``results/<problem>/optimization/conductivity_recovery<suffix>/``:
    - ``conductivity_recovery_convergence.{png,pdf}`` — semilogy loss + final-error bar
    - ``conductivity_recovery_fields.{png,pdf}``      — rho_init / rho_truth / rho_final per solver
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    styles = solver_styles(cfg, differentiable_only=False)
    names = list(by_solver.keys())

    # ── 1. Convergence + final-error figure ───────────────────────────────────
    fig_cv, (ax_lc, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

    final_errors: list[float] = []
    for name in names:
        res = by_solver[name]
        errors = res.get("errors", [])
        sty = styles.get(name, {})
        if errors:
            ax_lc.semilogy(
                errors,
                label=sty.get("label", name),
                **solver_plot_props(sty, marker=False),
            )
        final_errors.append(float(res.get("final_error", float("nan"))))

    ax_lc.set_xlabel("Iteration")
    ax_lc.set_ylabel("Identification error")
    ax_lc.set_title("Conductivity recovery — loss curves")
    ax_lc.legend(fontsize=8)
    ax_lc.grid(True, which="both", alpha=0.3)

    colors = [cfg.solvers[n].color if n in cfg.solvers else "#888888" for n in names]
    labels = [styles.get(n, {}).get("label", n) for n in names]
    ax_bar.bar(labels, final_errors, color=colors)
    ax_bar.set_ylabel("Final identification error")
    ax_bar.set_title("Conductivity recovery — final error")
    ax_bar.tick_params(axis="x", rotation=30)
    ax_bar.grid(axis="y", alpha=0.4)

    fig_cv.suptitle(f"{cfg.name} — conductivity recovery")
    fig_cv.tight_layout()
    if save:
        save_fig(fig_cv, "conductivity_recovery_convergence", out_dir)

    # ── 2. Conductivity field comparison + evolution GIFs ─────────────────────
    fields_path = out_dir / "rho_fields.npz"
    if fields_path.exists():
        _plot_conductivity_recovery_fields(
            cfg, fields_path, out_dir, names, styles, save=save
        )
        if save:
            _render_conductivity_recovery_evolution_gifs(
                fields_path, out_dir, names, styles
            )

    return fig_cv


def _plot_conductivity_recovery_fields(
    cfg: ProblemConfig,
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
    *,
    save: bool,
) -> None:
    """Render ``conductivity_recovery_fields.{png,pdf}``.

    Shows rho_init, rho_truth, and rho_final per solver as 1-D line plots
    (conductivity field is a 1-D vector over mesh faces/cells).
    """
    npz = np.load(fields_path)

    rho_init = np.asarray(npz["rho_init"]) if "rho_init" in npz.files else None
    rho_truth = np.asarray(npz["rho_truth"]) if "rho_truth" in npz.files else None

    n_active = sum(1 for n in solver_names if f"rho_final_{n}" in npz.files)
    if n_active == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    xs = np.arange(rho_init.shape[0]) if rho_init is not None else None

    if rho_truth is not None:
        ax.plot(
            np.arange(rho_truth.shape[0]),
            rho_truth,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="truth",
            zorder=10,
        )
    if rho_init is not None:
        ax.plot(
            xs,
            rho_init,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="init",
            zorder=9,
        )

    for name in solver_names:
        key = f"rho_final_{name}"
        if key not in npz.files:
            continue
        rho_f = np.asarray(npz[key])
        sty = styles.get(name, {})
        ax.plot(
            np.arange(rho_f.shape[0]),
            rho_f,
            label=sty.get("label", name),
            **solver_plot_props(sty, marker=False),
        )

    ax.set_xlabel("Cell index")
    ax.set_ylabel("Conductivity ρ")
    ax.set_title(f"{cfg.name} — recovered conductivity field")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        save_fig(fig, "conductivity_recovery_fields", out_dir)


def _render_conductivity_recovery_evolution_gifs(
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
) -> None:
    """Write ``conductivity_recovery_evolution_<solver>.gif`` per solver.

    Each frame is a 1-D line plot of ``rho_history_<name>[frame, :]`` with
    dashed truth and dotted init references.  Y-range is fixed across frames.
    Silently skips solvers without a recorded history.
    """
    npz = np.load(fields_path)
    rho_truth = np.asarray(npz["rho_truth"]) if "rho_truth" in npz.files else None
    rho_init = np.asarray(npz["rho_init"]) if "rho_init" in npz.files else None

    for name in solver_names:
        hkey = f"rho_history_{name}"
        if hkey not in npz.files:
            continue
        hist = np.asarray(npz[hkey])  # (n_frames, n_cells)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        n_frames, n_cells = hist.shape
        xs = np.arange(n_cells)

        extrema = [float(hist.min()), float(hist.max())]
        if rho_truth is not None:
            extrema.extend([float(rho_truth.min()), float(rho_truth.max())])
        if rho_init is not None:
            extrema.extend([float(rho_init.min()), float(rho_init.max())])
        ylo, yhi = min(extrema), max(extrema)
        pad = 0.05 * (yhi - ylo + 1e-12)
        ylo -= pad
        yhi += pad

        sty = styles.get(name, {})
        color = sty.get("color", "#3366CC")
        label = sty.get("label", name)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        if rho_truth is not None:
            ax.plot(
                xs,
                rho_truth,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="truth",
            )
        if rho_init is not None:
            ax.plot(
                xs, rho_init, color="gray", linestyle=":", linewidth=1.0, label="init"
            )
        (line,) = ax.plot(xs, hist[0], color=color, linewidth=1.8, label=label)
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Conductivity ρ")
        ax.set_ylim(ylo, yhi)
        title = ax.set_title(f"{label} — snapshot 1 / {n_frames}", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        def _update(
            idx, _line=line, _title=title, _hist=hist, _label=label, _n=n_frames
        ):
            _line.set_ydata(_hist[idx])
            _title.set_text(f"{_label} — snapshot {idx + 1} / {_n}")
            return _line, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"conductivity_recovery_evolution_{name}", out_dir, fps=4)


# ── Load recovery (structural-mesh) ──────────────────────────────────────────


def plot_load_recovery(
    cfg: ProblemConfig,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "load_recovery",
):
    """Three outputs: loss curves, density field comparison, per-solver evolution GIFs.

    Inspired by ``plot_source_recovery`` but for structural load recovery.
    The recovered field is a 3-D density rho visualised via ``_rho_to_2d``
    (mid-y cross-section for 3-D meshes, top-down for 2-D slabs).

    Files written to ``results/<problem>/recovery/<exp_key><suffix>/``:
    - ``load_recovery_convergence.{png,pdf}`` — loss curves + ρ-error bar chart
    - ``load_recovery_density.{png,pdf}``     — ρ field comparison across solvers
    - ``load_recovery_evolution_<solver>.gif`` — density optimisation animation
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    by_solver = data.get("by_solver", {})
    if not by_solver:
        return None

    physics = data.get("params", {}).get("physics", {})
    styles = solver_styles(cfg, differentiable_only=False)
    names = list(by_solver.keys())

    # ── 1. Convergence + final-error figure ───────────────────────────────────
    fields_path = out_dir / "load_fields.npz"
    npz_cv = np.load(fields_path) if fields_path.exists() else None
    rho_truth_flat = (
        np.asarray(npz_cv["rho_truth"])
        if npz_cv is not None and "rho_truth" in npz_cv.files
        else None
    )
    rho_truth_norm = (
        float(np.linalg.norm(rho_truth_flat)) if rho_truth_flat is not None else None
    )

    fig_cv, (ax_lc, ax_err, ax_bar) = plt.subplots(1, 3, figsize=(18, 4))

    rho_errors: list[float] = []
    for name in names:
        res = by_solver[name]
        losses = res.get("losses", [])
        n_iters = res.get("n_iters", len(losses))
        sty = styles.get(name, {})
        plot_kw = solver_plot_props(sty, marker=False)
        label = sty.get("label", name)
        if losses:
            ax_lc.semilogy(losses, label=label, **plot_kw)
        rho_errors.append(float(res.get("rho_rel_error", float("nan"))))

        # Density error over iterations from rho_history snapshots
        if npz_cv is not None and rho_truth_flat is not None and rho_truth_norm:
            hkey = f"rho_history_{name}"
            if hkey in npz_cv.files:
                hist = np.asarray(npz_cv[hkey])  # (n_snaps, n_cells)
                if hist.ndim == 2 and hist.shape[0] > 0:
                    n_snaps = hist.shape[0]
                    # Map snapshot indices to iteration numbers
                    xs = np.linspace(n_iters / n_snaps, n_iters, n_snaps)
                    errs = (
                        np.linalg.norm(hist - rho_truth_flat[None], axis=1)
                        / rho_truth_norm
                    )
                    ax_err.semilogy(xs, errs, label=label, **plot_kw)

    ax_lc.set_xlabel("Iteration")
    ax_lc.set_ylabel("Recovery loss")
    ax_lc.set_title("Load recovery — loss curves")
    ax_lc.legend(fontsize=8)
    ax_lc.grid(True, which="both", alpha=0.3)

    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("Relative density error ‖Δρ‖ / ‖ρ_truth‖")
    ax_err.set_title("Load recovery — density error over time")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", alpha=0.3)

    colors = [cfg.solvers[n].color if n in cfg.solvers else "#888888" for n in names]
    labels = [styles.get(n, {}).get("label", n) for n in names]
    ax_bar.bar(labels, rho_errors, color=colors)
    ax_bar.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax_bar.set_ylabel("Relative density error ‖Δρ‖ / ‖ρ_truth‖")
    ax_bar.set_title("Load recovery — final density error")
    ax_bar.tick_params(axis="x", rotation=30)
    ax_bar.grid(axis="y", alpha=0.4)

    fig_cv.suptitle(f"{cfg.name} — load recovery")
    fig_cv.tight_layout()
    if save:
        save_fig(fig_cv, "load_recovery_convergence", out_dir)

    # ── 2. Density field comparison ───────────────────────────────────────────
    if fields_path.exists():
        _plot_load_recovery_density(
            cfg, fields_path, out_dir, names, styles, physics, save=save
        )
        if save:
            _render_load_recovery_evolution_gifs(
                fields_path, out_dir, names, styles, physics
            )

    return fig_cv


def _plot_load_recovery_density(
    cfg: ProblemConfig,
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
    physics: dict,
    *,
    save: bool,
) -> None:
    """Render ``load_recovery_density.{png,pdf}``.

    Layout (one figure, 2 rows × n_cols):
      Row 0: ρ_init · ρ_truth · ρ_final per solver
      Row 1: (blank) · (blank) · |ρ_final − ρ_truth| per solver
    Each panel is a 2-D heatmap using ``_rho_to_2d``.
    """
    npz = np.load(fields_path)

    rho_init = (
        _rho_to_2d(np.asarray(npz["rho_init"]), physics)
        if "rho_init" in npz.files
        else None
    )
    rho_truth = (
        _rho_to_2d(np.asarray(npz["rho_truth"]), physics)
        if "rho_truth" in npz.files
        else None
    )
    finals: dict[str, np.ndarray] = {}
    for name in solver_names:
        key = f"rho_final_{name}"
        if key in npz.files:
            finals[name] = _rho_to_2d(np.asarray(npz[key]), physics)

    if not finals and rho_init is None:
        return

    has_truth = rho_truth is not None
    n_ref = 2 if has_truth else (1 if rho_init is not None else 0)
    n_solver_cols = len(finals)
    n_cols = n_ref + n_solver_cols
    n_rows = 1 + (1 if has_truth and finals else 0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(8.0, n_cols * 2.5), n_rows * 2.8),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)

    cmap, vmin, vmax = "gray_r", 0.0, 1.0
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

    # Row 0: reference fields then solver finals
    col = 0
    for label_str, field in [("ρ init", rho_init), ("ρ truth", rho_truth)]:
        if field is None:
            col += 1
            continue
        ax = axes[0, col]
        ax.set_visible(True)
        ax.imshow(field, **kw)
        ax.set_title(label_str, fontsize=9)
        ax.axis("off")
        col += 1

    for i, name in enumerate(finals):
        c = i + n_ref
        if c >= n_cols:
            break
        ax = axes[0, c]
        ax.set_visible(True)
        ax.imshow(finals[name], **kw)
        sty = styles.get(name, {})
        ax.set_title(f"ρ final — {sty.get('label', name)}", fontsize=9)
        ax.axis("off")

    # Row 1: per-solver density error |ρ_final − ρ_truth|
    if has_truth and finals and n_rows > 1:
        err_max = max(float(np.abs(finals[n] - rho_truth).max()) for n in finals)
        for i, name in enumerate(finals):
            c = i + n_ref
            if c >= n_cols:
                break
            ax = axes[1, c]
            ax.set_visible(True)
            err = np.abs(finals[name] - rho_truth)
            ax.imshow(
                err,
                cmap="Reds",
                vmin=0.0,
                vmax=max(err_max, 1e-9),
                origin="lower",
                aspect="auto",
            )
            sty = styles.get(name, {})
            ax.set_title(f"|Δρ| — {sty.get('label', name)}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"{cfg.name} — load recovery density fields", fontsize=11)
    fig.tight_layout()
    if save:
        save_fig(fig, "load_recovery_density", out_dir)


def _render_load_recovery_evolution_gifs(
    fields_path: Path,
    out_dir: Path,
    solver_names: list,
    styles: dict,
    physics: dict,
) -> None:
    """Write ``load_recovery_evolution_<solver>.gif`` per solver.

    Each frame is a 2-D density heatmap of ``rho_history_<name>[frame, :]``
    with a fixed colour range so the animation reads as true evolution.
    """
    npz = np.load(fields_path)
    rho_truth_flat = np.asarray(npz["rho_truth"]) if "rho_truth" in npz.files else None

    for name in solver_names:
        hkey = f"rho_history_{name}"
        if hkey not in npz.files:
            continue
        hist_flat = np.asarray(npz[hkey])  # (n_frames, n_cells)
        if hist_flat.ndim != 2 or hist_flat.shape[0] == 0:
            continue
        n_frames = hist_flat.shape[0]
        hist = np.stack([_rho_to_2d(hist_flat[i], physics) for i in range(n_frames)])

        sty = styles.get(name, {})
        label = sty.get("label", name)

        fig, axes = plt.subplots(
            1,
            1 + (1 if rho_truth_flat is not None else 0),
            figsize=(8 if rho_truth_flat is not None else 4, 3),
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        rho_2d_truth = (
            _rho_to_2d(rho_truth_flat, physics) if rho_truth_flat is not None else None
        )

        if rho_2d_truth is not None:
            axes[0].imshow(
                rho_2d_truth,
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                aspect="auto",
            )
            axes[0].set_title("ρ truth", fontsize=9)
            axes[0].axis("off")
            ax_anim = axes[1]
        else:
            ax_anim = axes[0]

        im = ax_anim.imshow(
            hist[0], cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", aspect="auto"
        )
        ax_anim.axis("off")
        title = ax_anim.set_title(f"{label} — iter 1 / {n_frames}", fontsize=9)
        fig.tight_layout()

        def _update(idx, _im=im, _title=title, _hist=hist, _label=label, _n=n_frames):
            _im.set_data(_hist[idx])
            _title.set_text(f"{_label} — iter {idx + 1} / {_n}")
            return _im, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"load_recovery_evolution_{name}", out_dir, fps=4)
