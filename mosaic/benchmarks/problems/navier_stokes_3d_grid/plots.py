# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the navier-stokes-3d-grid recovery experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    load_json,
    results_dir,
    try_load_npz,
    v1_to_legacy,
)
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
    apply_style,
    dedup_handles,
    fig_shared_legend,
    make_handle,
    paper_image_grid,
    resolve_solver_alias,
    save_fig,
    solver_plot_props,
    solver_styles,
    subplots_grid,
    vorticity_2d,
)

# Default perturb_sigma to highlight on the single-experiment NS figure.
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
    """Recovery per-experiment plot — styled figure + extras.

    This wrapper produces:

      * ``convergence_curves`` — per-sweep-value loss curves grid.
      * ``recovery_fields`` — per-solver true/perturbed/recovered/residual.
      * ``recovery_final_states`` — GT vs recovered rollout panels.
      * ``recovery_sigma_<v>`` — all-solvers-per-sigma grid.
      * ``recovery_evolution.gif`` — combined IC reconstruction animation.

    The per-metric "vs steps" summary panel is intentionally omitted: the
    3D-NS recovery sweeps a single ``steps`` value, so a metric-vs-sweep
    curve degenerates to one point.

    When results live in per-IC subdirectories (from
    ``--experiments <suite>/<exp>/<ic>`` runs), pass ``ic`` to select a specific
    IC (e.g. ``ic="multimode"``).  If the root-level
    ``result.json`` is not found, the function automatically falls back to the
    first available IC subdirectory.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    out_dir = _resolve_recovery_out_dir(base_dir, ic)

    data = v1_to_legacy(load_json(out_dir / "result.json"))
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

    # NOTE: the per-metric "vs steps" summary panel (``optimization.png``,
    # produced by ``_plot_recovery_summary``) is intentionally NOT generated
    # here: the 3D-NS recovery runs a single experiment (one ``steps`` value),
    # so a metric-vs-sweep curve degenerates to a single point and is spurious.

    # ── convergence curves (all sweep values) + IC error consensus ───────────
    _plot_convergence_curves(
        cfg, by_sweep, sweep_vals, sweep_key, styles, out_dir, save
    )

    # ── IC field comparison ────────────────────────────────────────────────────
    fields_path = out_dir / "recovery_fields.npz"
    if not fields_path.exists():
        return None

    npz = try_load_npz(fields_path)
    rep_horizon = float(
        (npz.get("rep_val") or npz.get("rep_horizon", np.array([0])))[0]
    )
    solver_names = npz["solver_names"].tolist()

    # Use ic_to_2d when set (e.g. n-body density contrast δ₀ slice),
    # then field_to_2d (e.g. 3D vorticity slice), then vorticity_2d for 2-D.
    f_ic = ic_to_2d or field_to_2d or vorticity_2d

    if save:
        _render_recovery_evolution_gifs(
            out_dir, npz, solver_names, f_ic, styles, sweep_key, rep_horizon
        )

    return None


def _render_recovery_evolution_gifs(
    out_dir: Path,
    npz: Any,
    solver_names: list,
    f_ic: Any,
    styles: dict,
    sweep_key: str,
    rep_horizon: Any,
) -> None:
    """Write a single combined ``recovery_evolution.gif`` for all solvers.

    Builds one figure with a row of image panels — one panel per solver that
    has a recorded ``ic_history_<j>``. Each panel animates that solver's 2-D
    scalar view of the IC (same ``ic_to_2d`` / vorticity mapping as the static
    ``recovery_fields`` panel) re-forming over optimiser snapshots. Frames are
    synchronised across panels: at frame *k* every panel shows its snapshot-*k*
    state, and solvers with fewer snapshots hold (clamp to) their last frame.
    Per-panel vmin/vmax is fixed across frames so the IC re-forming reads
    clearly rather than flickering from autoscaling. A shared solver legend is
    placed below the row, deduplicated by canonical alias. Silently skips
    solvers without a recorded history; emits nothing when none qualify.
    """
    # ``npz`` is a plain dict from ``try_load_npz`` (not an NpzFile), so
    # membership is a key check — ``.files`` would AttributeError.
    panels: list[dict] = []
    seen_aliases: set[str] = set()
    for j, name in enumerate(solver_names):
        hist_key = f"ic_history_{j}"
        if hist_key not in npz:
            continue
        history = np.asarray(npz[hist_key])  # (n_frames, *ic_shape)
        if history.ndim < 2 or history.shape[0] == 0:
            continue
        alias = resolve_solver_alias(name)
        dedup_key = alias if alias is not None else name
        if dedup_key in seen_aliases:
            continue
        seen_aliases.add(dedup_key)

        n_frames = int(history.shape[0])
        frames_2d = [f_ic(history[i]) for i in range(n_frames)]
        vmax = float(max(np.abs(arr).max() for arr in frames_2d)) or 1.0
        label = styles.get(name, {}).get("label", name)
        panels.append(
            {
                "name": name,
                "alias": alias,
                "label": label,
                "frames": frames_2d,
                "n": n_frames,
                "vmax": vmax,
            }
        )

    if not panels:
        return

    n_panels = len(panels)
    n_frames = max(p["n"] for p in panels)

    apply_style()
    fig, axes = paper_image_grid(1, n_panels, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    images: list = []
    for ax, p in zip(axes, panels, strict=True):
        im = ax.imshow(
            p["frames"][0].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-p["vmax"],
            vmax=p["vmax"],
            interpolation="nearest",
        )
        images.append(im)
        ax.set_title(p["label"], fontsize=8)
        ax.axis("off")

    title = fig.suptitle(
        f"IC recovery evolution — snapshot 1 / {n_frames}  ({sweep_key}={rep_horizon})",
        fontsize=9,
    )
    fig.tight_layout()

    handles = dedup_handles(
        [make_handle(p["alias"]) for p in panels if p["alias"] is not None]
    )
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
        fig.subplots_adjust(bottom=0.18)

    def _update(
        idx: Any,
        _images: Any = images,
        _panels: Any = panels,
        _title: Any = title,
        _n: Any = n_frames,
        _sk: Any = sweep_key,
        _sv: Any = rep_horizon,
    ) -> Any:
        for _im, _p in zip(_images, _panels, strict=True):
            k = min(idx, _p["n"] - 1)  # clamp: hold last frame
            _im.set_data(_p["frames"][k].T)
        _title.set_text(
            f"IC recovery evolution — snapshot {idx + 1} / {_n}  ({_sk}={_sv})"
        )
        return (*_images, _title)

    anim = manimation.FuncAnimation(
        fig, _update, frames=n_frames, interval=250, blit=False
    )
    _save_animation(anim, "recovery_evolution", out_dir, fps=4)
