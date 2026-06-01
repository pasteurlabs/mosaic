# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the navier-stokes-grid drag-optimisation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
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
    SOLVER_STYLES,
    TEXTWIDTH,
    dedup_handles,
    imshow_with_cbar,
    make_handle,
    paper_image_grid,
    paper_row,
    resolve_solver_alias,
    save_fig,
    solver_legend,
    solver_props,
    solver_styles,
)

# Solvers shown in the drag_opt paper panel, in display order.
_DRAG_OPT_SOLVER_ORDER = ["xlb", "phiflow", "pict"]


def plot_drag_opt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "drag_opt",
    **_kw: Any,
) -> list:
    """Drag-optimisation per-experiment plot — styled figure + extras.

    Inlined styled 3-column figure (drag reduction, optimised inlet
    profile, profile history) plus:

      * ``drag_opt_fields`` — per-solver flow field panels (u_x, u_y) when
        ``flow_fields.npz`` is present.
      * ``drag_opt_evolution.gif`` — one combined inflow-profile animation
        with a panel per solver when ``profiles.npz`` carries
        ``profile_history_*``.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    styles = solver_styles(cfg)
    figs: list = []

    def _plot_one(
        data: Any,
        profiles_path: Any,
        out_dir: Any,
        *,
        paper_exp_key: Any,
        paper_suffix: Any,
    ) -> None:
        by_solver = data.get("by_solver", {})
        if not by_solver:
            return
        run_name = data.get("run_name", "")
        title_suffix = f" — {run_name}" if run_name else ""

        # ── Canonical figure (inlined) ───────────────────────────────────────
        fig = _drag_opt_figure(
            cfg,
            exp_key=paper_exp_key,
            suffix=paper_suffix,
            save=save,
        )
        if fig is not None:
            figs.append(fig)

        profiles = try_load_npz(profiles_path) if profiles_path.exists() else {}
        solver_names = list(by_solver.keys())

        # ── Flow field visualisation (velocity + vorticity) ──────────────────
        _plot_drag_opt_fields(data, out_dir, run_name, title_suffix, styles, save, figs)

        # ── Inflow profile evolution GIF (combined, one panel per solver) ───
        if save and "initial" in profiles:
            _render_drag_opt_evolution_gifs(
                profiles, out_dir, solver_names, styles, run_name
            )

    # Single-run layout — figure resolves the experiment dir from cfg.
    single_path = base_dir / "result.json"
    single_result = load_json(single_path) if single_path.exists() else None
    if single_result is not None:
        _plot_one(
            single_result,
            base_dir / "profiles.npz",
            base_dir,
            paper_exp_key=exp_key,
            paper_suffix=suffix,
        )
        return figs

    # Multi-run layout: one canonical figure per run subdir.
    if base_dir.is_dir():
        for sub in sorted(base_dir.iterdir()):
            sub_data = load_json(sub / "result.json")
            if sub_data is not None:
                _plot_one(
                    sub_data,
                    sub / "profiles.npz",
                    sub,
                    paper_exp_key=f"{exp_key}/{sub.name}",
                    paper_suffix=suffix,
                )
    return figs


def _drag_alias_to_display(prefix: str, container: Any) -> dict[str, str]:
    """Build an alias→display-name map from npz/dict keys matching *prefix*."""
    if container is None:
        return {}
    keys = container.files if hasattr(container, "files") else container.keys()
    out: dict[str, str] = {}
    for k in keys:
        if not k.startswith(prefix):
            continue
        display = k[len(prefix) :]
        alias = resolve_solver_alias(display)
        if alias is not None:
            out[alias] = display
    return out


def _drag_panel_drag_reduction(
    ax: Any, data: dict, alias_to_display: dict, present: set[str]
) -> None:
    """Draw the drag-reduction-vs-iteration panel; updates ``present`` aliases."""
    for alias in _DRAG_OPT_SOLVER_ORDER:
        display_name = alias_to_display.get(alias, alias)
        sdata = data["by_solver"].get(display_name)
        if sdata is None:
            continue
        drags = sdata.get("drags", [])
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]
        step = max(1, len(drags) // 50)
        indices = list(range(0, len(drags), step))
        if indices[-1] != len(drags) - 1:
            indices.append(len(drags) - 1)
        reductions = [(drag_0 - drags[i]) / drag_0 * 100 for i in indices]
        _label, color, ls, _mk = solver_props(alias)
        ax.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        present.add(alias)

    ax.set_title("Drag reduction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Drag reduction (%)")
    ax.set_ylim(bottom=0)


def _drag_panel_profiles(
    ax: Any, profiles: Any, alias_to_display: dict, present: set[str]
) -> None:
    """Draw the initial + final inlet profiles panel; updates ``present`` aliases."""
    if profiles is not None and "initial" in profiles:
        y_arr = np.linspace(0, 1, profiles["initial"].shape[0])
        ax.plot(
            profiles["initial"], y_arr, color="#999999", linestyle="--", linewidth=1.4
        )
        for alias in _DRAG_OPT_SOLVER_ORDER:
            display_name = alias_to_display.get(alias, alias)
            if f"final_{display_name}" not in profiles:
                continue
            _label, color, ls, _mk = solver_props(alias)
            ax.plot(
                profiles[f"final_{display_name}"],
                y_arr,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            present.add(alias)

    ax.set_title("Optimised profile")
    ax.set_xlabel(r"$u_x$")
    ax.set_ylabel("$y$")


def _drag_panel_history(
    imshow_axes: Any,
    hist_solvers: Any,
    hist_alias_to_display: Any,
    profiles: Any,
    data: Any,
) -> None:
    """Draw the profile-history imshow panels (one per solver row)."""
    for idx, (ax_im, alias) in enumerate(zip(imshow_axes, hist_solvers, strict=False)):
        display_name = hist_alias_to_display.get(alias, alias)
        hist = profiles[f"profile_history_{display_name}"]
        label, color, _ls, _mk = solver_props(alias)
        ax_im.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )
        n_snaps = hist.shape[0]
        n_iters = len(data["by_solver"].get(display_name, {}).get("drags", [1]))
        snap_step = n_iters / max(n_snaps - 1, 1)
        tick_pos = [0, n_snaps // 2, n_snaps - 1]
        ax_im.set_xticks(tick_pos)
        ax_im.set_xticklabels([f"{int(t * snap_step)}" for t in tick_pos], fontsize=6.5)
        ax_im.tick_params(labelsize=6.5)
        ax_im.set_yticks([])
        ax_im.text(
            0.03,
            0.95,
            label,
            transform=ax_im.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=color,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0},
        )
        if idx == 0:
            ax_im.set_title("Profile history")
        if idx < len(hist_solvers) - 1:
            ax_im.tick_params(labelbottom=False)
        else:
            ax_im.set_xlabel("Iteration", fontsize=7.0)

    for ax_im in imshow_axes[len(hist_solvers) :]:
        ax_im.set_visible(False)


def _drag_opt_figure(
    cfg: Problem,
    *,
    exp_key: str = "drag_opt",
    suffix: str = "",
    save: bool = True,
) -> plt.Figure | None:
    """Single-experiment styled drag-optimisation figure.

    Layout (3-column GridSpec):
      * col 0 — drag reduction (%) vs iteration
      * col 1 — final + initial inlet profiles
      * col 2 — profile-history imshow, one row per solver
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    profiles_path = out_dir / "profiles.npz"

    if not result_path.exists():
        print(f"[drag_opt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)

    data = load_json(result_path)
    profiles = try_load_npz(profiles_path) if profiles_path.exists() else None

    # Both ``by_solver`` and ``profiles`` are keyed by spec.name (display form).
    # Build alias→display maps so the alias-ordered _DRAG_OPT_SOLVER_ORDER loop
    # can index display-keyed data.
    hist_alias_to_display = _drag_alias_to_display("profile_history_", profiles)
    final_alias_to_display = _drag_alias_to_display("final_", profiles)
    by_solver_alias_to_display: dict[str, str] = {}
    for name in data.get("by_solver", {}):
        alias = resolve_solver_alias(name)
        if alias is not None:
            by_solver_alias_to_display[alias] = name
    alias_to_display = {**by_solver_alias_to_display, **final_alias_to_display}

    hist_solvers = [s for s in _DRAG_OPT_SOLVER_ORDER if s in hist_alias_to_display]
    n_rows = max(len(hist_solvers), 1)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * (0.14 + 0.13 * n_rows)), dpi=300)
    gs = gridspec.GridSpec(
        n_rows,
        3,
        figure=fig,
        width_ratios=[1.4, 0.9, 1.1],
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.22,
        hspace=0.12,
        wspace=0.45,
    )
    ax_drag = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[:, 1])
    imshow_axes = [fig.add_subplot(gs[r, 2]) for r in range(n_rows)]

    present: set[str] = set()
    _drag_panel_drag_reduction(ax_drag, data, alias_to_display, present)
    _drag_panel_profiles(ax_prof, profiles, alias_to_display, present)
    _drag_panel_history(
        imshow_axes, hist_solvers, hist_alias_to_display, profiles, data
    )

    handles = [
        mlines.Line2D(
            [], [], color="#999999", linestyle="--", linewidth=1.4, label="Initial"
        ),
        *dedup_handles(
            [make_handle(s) for s in NS_ORDER if s in present and s in SOLVER_STYLES]
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(handles), 5),
        fontsize=7.5,
        framealpha=0.7,
        edgecolor="0.8",
        handlelength=2.0,
    )

    if save:
        out = out_dir / f"{exp_key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def _vel_components_2d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_x, u_y) 2-D arrays from field (N, N, 1, 2) or (N, N, 2)."""
    if v.ndim == 4:
        v = v[:, :, 0, :]
    return v[..., 0], v[..., 1]


def _plot_drag_opt_fields(
    data: dict,
    out_dir: Any,
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

    npz = try_load_npz(fields_path)
    by_solver = data.get("by_solver", {})
    # ``try_load_npz`` returns a plain dict; tolerate both dict and NpzFile.
    npz_keys = set(npz.files if hasattr(npz, "files") else npz.keys())

    # Drive the rows from ``by_solver`` (the canonical solver set for this
    # result) rather than scanning every ``flow_final_*`` npz key.  The npz can
    # accumulate stale per-solver entries from earlier runs (e.g. differently
    # cased aliases like ``flow_final_XLB`` alongside ``flow_final_xlb``); those
    # have no ``by_solver`` metadata and would render as blank/duplicate rows.
    # Additionally dedup by canonical alias: if ``by_solver`` carries both a
    # display name and an alias mapping to the same solver, keep only the first.
    solver_names_clean: list[str] = []
    seen_aliases: set[str] = set()
    for s in by_solver:
        if f"flow_final_{s}" not in npz_keys:
            continue
        alias = resolve_solver_alias(s) or s
        if alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        solver_names_clean.append(s)

    if not solver_names_clean:
        return

    # Rows: initial + one per solver.  Columns: u_x | u_y.
    n_rows = 1 + len(solver_names_clean)
    ncols = 2
    fig_fld, axes_fld = paper_image_grid(n_rows, ncols)
    # Widen the figure to host a dedicated left gutter for the (multi-line) row
    # labels without stealing width from the panels/colorbars.  ``IMG_PANEL`` is
    # ~1.45" per panel; reserve a fixed ~1.5" gutter on the left.
    panel_w = fig_fld.get_size_inches()[0] / ncols
    gutter_w = 1.5
    fig_w = ncols * panel_w + gutter_w
    fig_h = fig_fld.get_size_inches()[1]
    fig_fld.set_size_inches(fig_w, fig_h)
    left_frac = gutter_w / fig_w
    # Reserve the left gutter and keep generous vertical/horizontal spacing so
    # labels, column titles, suptitle and colorbars never collide.  Done before
    # rendering so ``ax.get_position()`` reflects the final layout.
    fig_fld.subplots_adjust(
        left=left_frac,
        right=0.97,
        top=0.90,
        bottom=0.03,
        hspace=0.30,
        wspace=0.55,
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
                ax.set_title(col_title, fontweight="bold", pad=6)
            ax.axis("off")

        # Row label placed in the left figure gutter (reserved via
        # subplots_adjust above).  Right-aligned so the label always ends a
        # fixed gap left of the panel regardless of its length — multi-line
        # solver labels never overlap the panels, colorbars or adjacent rows.
        ax0 = axes_fld[row_idx, 0]
        pos = ax0.get_position()
        fig_fld.text(
            pos.x0 - 0.02,
            (pos.y0 + pos.y1) / 2,
            label,
            fontsize=7,
            va="center",
            ha="right",
            linespacing=1.3,
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

    fig_fld.suptitle("Optimised flow fields", fontweight="bold", y=0.97)
    if save:
        save_fig(fig_fld, "drag_opt_fields", out_dir)
    figs.append(fig_fld)


def _render_drag_opt_evolution_gifs(
    profiles: Any,
    out_dir: Any,
    solver_names: list,
    styles: dict,
    run_name: str,
) -> None:
    """Write a single combined ``drag_opt_evolution.gif`` for all solvers.

    Builds one figure with a row of subplots — one panel per solver that has
    a recorded ``profile_history_<name>``. Each panel animates that solver's
    inflow profile u_x(y) over BFGS iterations, with the initial profile drawn
    dashed as a reference. Frames are synchronised across panels: at frame *k*
    every panel shows its iteration-*k* state, and solvers with fewer snapshots
    hold (clamp to) their last frame. A shared solver legend is placed below
    the row. Solvers are deduplicated by canonical alias so each appears once.
    """
    initial = np.asarray(profiles["initial"])
    N = initial.size
    y = np.linspace(0, 1, N, endpoint=False) + 0.5 / N

    # Treat np.load NpzFile *and* plain dict uniformly via .files / keys.
    keys = set(profiles.files) if hasattr(profiles, "files") else set(profiles.keys())

    # Collect one entry per solver that has a usable history, deduped by alias.
    panels: list[dict] = []
    seen_aliases: set[str] = set()
    for name in solver_names:
        hkey = f"profile_history_{name}"
        if hkey not in keys:
            continue
        hist = np.asarray(profiles[hkey])  # (n_snaps, N)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        alias = resolve_solver_alias(name)
        dedup_key = alias if alias is not None else name
        if dedup_key in seen_aliases:
            continue
        seen_aliases.add(dedup_key)

        label, color, _ls, _mk = solver_props(name)
        panels.append(
            {
                "name": name,
                "alias": alias,
                "label": label,
                "color": color,
                "hist": hist,
                "n": int(hist.shape[0]),
            }
        )

    if not panels:
        return

    n_panels = len(panels)
    n_frames = max(p["n"] for p in panels)

    fig, axes = paper_row(n_panels, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    lines: list = []
    for ax, p in zip(axes, panels, strict=True):
        hist = p["hist"]
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

        ax.plot(initial, y, "k--", lw=1.4, label="initial")
        (line,) = ax.plot(hist[0], y, color=p["color"], lw=2.0, label=p["label"])
        lines.append(line)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$u_x$")
        ax.set_ylabel("$y$")
        ax.set_title(p["label"])

    sup = (
        f"Inflow profile evolution — {run_name}"
        if run_name
        else "Inflow profile evolution"
    )
    fig.suptitle(sup)

    # Shared solver legend below the row (canonical order, deduped by alias).
    initial_handle = mlines.Line2D(
        [], [], color="k", linestyle="--", lw=1.4, label="initial"
    )
    solver_legend(
        fig,
        [p["alias"] for p in panels if p["alias"] is not None],
        order=NS_ORDER,
        extra_handles=[initial_handle],
    )

    def _update(
        idx: Any,
        _lines: Any = lines,
        _panels: Any = panels,
    ) -> Any:
        for _line, _p in zip(_lines, _panels, strict=True):
            k = min(idx, _p["n"] - 1)  # clamp: hold last frame
            _line.set_xdata(_p["hist"][k])
        return tuple(_lines)

    anim = manimation.FuncAnimation(
        fig, _update, frames=n_frames, interval=250, blit=False
    )
    _save_animation(anim, "drag_opt_evolution", out_dir, fps=4)
