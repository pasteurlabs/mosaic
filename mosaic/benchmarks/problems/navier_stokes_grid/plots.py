"""Per-problem plots for the navier-stokes-grid drag-optimisation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
    imshow_with_cbar,
    save_fig,
    solver_styles,
)


def plot_drag_opt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "drag_opt",
    **_kw,
) -> list:
    """Drag-optimisation per-experiment plot — paper figure + extras.

    The canonical drag-reduction / optimised-profile / profile-history figure
    is delegated to
    :func:`mosaic.benchmarks.plots.paper.drag_opt.plot_experiment` so the
    polished paper styling lands on the experiment results dir too. This
    wrapper additionally produces:

      * ``drag_opt_fields`` — per-solver flow field panels (u_x, u_y) when
        ``flow_fields.npz`` is present.
      * ``drag_opt_evolution_<solver>.gif`` — one inflow-profile animation
        per solver when ``profiles.npz`` carries ``profile_history_*``.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    from mosaic.benchmarks.plots.paper import drag_opt as paper_drag_opt

    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    styles = solver_styles(cfg)
    figs: list = []

    def _plot_one(data, profiles_path, out_dir, *, paper_exp_key, paper_suffix):
        by_solver = data.get("by_solver", {})
        if not by_solver:
            return
        run_name = data.get("run_name", "")
        title_suffix = f" — {run_name}" if run_name else ""

        # ── Canonical paper figure (delegated) ────────────────────────────────
        fig = paper_drag_opt.plot_experiment(
            cfg,
            exp_key=paper_exp_key,
            suffix=paper_suffix,
            save=save,
        )
        if fig is not None:
            figs.append(fig)

        profiles = try_load_npz(profiles_path) if profiles_path.exists() else {}
        solver_names = list(by_solver.keys())

        # ── Flow field visualisation (velocity + vorticity) ───────────────────
        _plot_drag_opt_fields(data, out_dir, run_name, title_suffix, styles, save, figs)

        # ── Inflow profile evolution GIFs (one per solver) ───────────────────
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
    solver_names = [k for k in npz.files if k.startswith("flow_final_")]
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
