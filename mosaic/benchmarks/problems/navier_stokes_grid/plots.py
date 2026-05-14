"""Per-problem plots for the navier-stokes-grid drag-optimisation experiments."""

from __future__ import annotations

from pathlib import Path

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
    PAPER_RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    dedup_handles,
    imshow_with_cbar,
    make_handle,
    save_fig,
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
    **_kw,
) -> list:
    """Drag-optimisation per-experiment plot — paper figure + extras.

    Inlined paper-styled 3-column figure (drag reduction, optimised inlet
    profile, profile history) plus:

      * ``drag_opt_fields`` — per-solver flow field panels (u_x, u_y) when
        ``flow_fields.npz`` is present.
      * ``drag_opt_evolution_<solver>.gif`` — one inflow-profile animation
        per solver when ``profiles.npz`` carries ``profile_history_*``.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    styles = solver_styles(cfg)
    figs: list = []

    def _plot_one(data, profiles_path, out_dir, *, paper_exp_key, paper_suffix):
        by_solver = data.get("by_solver", {})
        if not by_solver:
            return
        run_name = data.get("run_name", "")
        title_suffix = f" — {run_name}" if run_name else ""

        # ── Canonical paper figure (inlined) ─────────────────────────────────
        fig = _paper_drag_opt_figure(
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

        # ── Inflow profile evolution GIFs (one per solver) ──────────────────
        if save and "initial" in profiles:
            _render_drag_opt_evolution_gifs(
                profiles, out_dir, solver_names, styles, run_name
            )

    # Single-run layout — paper figure resolves the experiment dir from cfg.
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

    # Multi-run layout: one canonical paper figure per run subdir.
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


def _paper_drag_opt_figure(
    cfg: Problem,
    *,
    exp_key: str = "drag_opt",
    suffix: str = "",
    save: bool = True,
) -> plt.Figure | None:
    """Single-experiment paper-styled drag-optimisation figure.

    Layout (3-column GridSpec):
      * col 0 — drag reduction (%) vs iteration
      * col 1 — final + initial inlet profiles
      * col 2 — profile-history imshow, one row per solver

    Reads ``result.json`` (+ optional ``profiles.npz``) from the
    experiment directory and writes ``<exp_key>.pdf`` next to them when
    ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    profiles_path = out_dir / "profiles.npz"

    if not result_path.exists():
        print(f"[drag_opt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(PAPER_RCPARAMS)

    data = load_json(result_path)
    profiles = try_load_npz(profiles_path) if profiles_path.exists() else None

    hist_solvers = [
        s
        for s in _DRAG_OPT_SOLVER_ORDER
        if profiles is not None and f"profile_history_{s}" in profiles
    ]
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

    # ── Drag reduction panel ─────────────────────────────────────────────────
    for solver in _DRAG_OPT_SOLVER_ORDER:
        sdata = data["by_solver"].get(solver)
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

        _label, color, ls, _mk = solver_props(solver)
        ax_drag.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        present.add(solver)

    ax_drag.set_title("Drag reduction")
    ax_drag.set_xlabel("Iteration")
    ax_drag.set_ylabel("Drag reduction (%)")
    ax_drag.set_ylim(bottom=0)

    # ── Final inlet profiles panel ───────────────────────────────────────────
    if profiles is not None and "initial" in profiles:
        y_arr = np.linspace(0, 1, profiles["initial"].shape[0])
        ax_prof.plot(
            profiles["initial"], y_arr, color="#999999", linestyle="--", linewidth=1.4
        )
        for solver in _DRAG_OPT_SOLVER_ORDER:
            if f"final_{solver}" not in profiles:
                continue
            _label, color, ls, _mk = solver_props(solver)
            ax_prof.plot(
                profiles[f"final_{solver}"],
                y_arr,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            present.add(solver)

    ax_prof.set_title("Optimised profile")
    ax_prof.set_xlabel(r"$u_x$")
    ax_prof.set_ylabel("$y$")

    # ── Profile history imshow panels ────────────────────────────────────────
    for idx, (ax_im, solver) in enumerate(zip(imshow_axes, hist_solvers, strict=False)):
        hist = profiles[f"profile_history_{solver}"]  # (n_snaps, ny)
        label, color, _ls, _mk = solver_props(solver)

        ax_im.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )

        n_snaps = hist.shape[0]
        n_iters = len(data["by_solver"].get(solver, {}).get("drags", [1]))
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

    # Hide unused imshow rows
    for ax_im in imshow_axes[len(hist_solvers) :]:
        ax_im.set_visible(False)

    # ── Legend ───────────────────────────────────────────────────────────────
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
        out = out_dir / f"{exp_key}.pdf"
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

    npz = try_load_npz(fields_path)
    by_solver = data.get("by_solver", {})
    # ``try_load_npz`` returns a plain dict; tolerate both dict and NpzFile.
    npz_keys = npz.files if hasattr(npz, "files") else list(npz.keys())
    solver_names = [k for k in npz_keys if k.startswith("flow_final_")]
    solver_names_clean = [k[len("flow_final_") :] for k in solver_names]

    if not solver_names:
        return

    # Rows: initial + one per solver.  Columns: velocity magnitude | vorticity.
    all_rows = ["__initial__", *solver_names_clean]
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
