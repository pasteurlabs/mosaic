# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for the structural-mesh topology optimisation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
)
from mosaic.benchmarks.problems.shared.plots.optimization import (
    _rho_to_2d,
    _save_animation,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    RCPARAMS,
    SOLVER_STYLES,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    paper_image_grid,
    paper_row,
    resolve_solver_alias,
    save_fig,
    solver_props,
    solver_styles,
)

# ── 3-D voxel rendering helpers ───────────────────────────────────────────────

_THRESH = 0.35
_ELEV = 22
_AZIM = 35
_CLR_FIXED = "#888888"
_CLR_LOAD = "#FF1744"


def _add_bcs(ax: Any, nx: int, ny: int, nz: int, ph: dict) -> None:
    """Overlay fixed-face patch and load arrow on a voxel Axes3D."""
    wall = Poly3DCollection(
        [[(0, 0, 0), (0, ny, 0), (0, ny, nz), (0, 0, nz)]],
        alpha=0.55,
        facecolor=_CLR_FIXED,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.add_collection3d(wall)

    corner_z_high = ph.get("corner_z_high", False)
    corner_y_high = ph.get("corner_y_high", False)
    load_axis = ph.get("load_axis", "z")
    ly = (ny - 0.5) if corner_y_high else 0.5
    lz = (nz - 0.5) if corner_z_high else 0.5

    arrow_len = nz * 0.65
    if load_axis == "y":
        y_sign = -1 if corner_y_high else 1
        dx, dy_a, dz_a = 0, y_sign * arrow_len, 0
    else:
        z_sign = -1 if corner_z_high else 1
        dx, dy_a, dz_a = 0, 0, z_sign * arrow_len

    ox = nx + 0.6 if load_axis != "y" else nx
    ax.quiver(
        ox,
        ly,
        lz,
        dx,
        dy_a,
        dz_a,
        color=_CLR_LOAD,
        linewidth=2.5,
        arrow_length_ratio=0.28,
    )


def _voxel_facecolors(
    rho_xyz: np.ndarray,
    filled: np.ndarray,
    base_color: str,
) -> np.ndarray:
    """Return RGBA facecolor array; empty voxels are fully transparent."""
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(base_color)
    fc = np.zeros((*rho_xyz.shape, 4))
    norm = np.where(filled, (rho_xyz - _THRESH) / (1.0 - _THRESH), 0.0)
    fc[..., 0] = r + (1 - r) * (1 - norm) * 0.45
    fc[..., 1] = g + (1 - g) * (1 - norm) * 0.45
    fc[..., 2] = b + (1 - b) * (1 - norm) * 0.45
    fc[..., 3] = filled.astype(float)
    return fc


def _solver_order_for(cfg_name: str) -> list[str]:
    """Best-effort solver-ordering pick based on problem name."""
    if "structural" in cfg_name:
        return STRUCTURAL_ORDER
    if "thermal" in cfg_name:
        return THERMAL_ORDER
    return STRUCTURAL_ORDER


def _field_solver_names(npz: Any, by_solver: dict | None = None) -> list[str]:
    """Per-field solver names aligned with the ``rho_final_<j>`` arrays.

    The NPZ stores one ``rho_final_<j>`` (and optional ``rho_history_<j>``)
    array per solver, written in ``by_solver`` insertion order. The
    companion ``solver_names`` array is *supposed* to label them, but some
    older result sets carry a truncated ``solver_names`` (e.g. only the
    first solver) while still writing every field array — which makes the
    field/3-D/GIF panels silently collapse to a single solver.

    To stay robust we count the actual ``rho_final_<j>`` arrays and:
      * use ``solver_names`` when it labels *every* field array, otherwise
      * fall back to the ``by_solver`` keys (same generation order) so all
        solvers present in the data are drawn.
    """
    npz_keys = npz.files if hasattr(npz, "files") else list(npz.keys())
    n_fields = sum(1 for k in npz_keys if k.startswith("rho_final_"))

    stored = list(npz["solver_names"]) if "solver_names" in npz_keys else []
    if len(stored) >= n_fields and n_fields > 0:
        return [str(s) for s in stored[:n_fields]]

    by_solver_names = list((by_solver or {}).keys())
    if len(by_solver_names) >= n_fields and n_fields > 0:
        return [str(s) for s in by_solver_names[:n_fields]]

    # Last resort: pad whatever names we have so every field array is drawn.
    names = [str(s) for s in (stored or by_solver_names)]
    while len(names) < n_fields:
        names.append(f"solver_{len(names)}")
    return names


def _plot_topopt_figure(
    cfg: Problem,
    *,
    exp_key: str = "topopt",
    suffix: str = "",
    save: bool = True,
) -> plt.Figure | None:
    """Canonical single-experiment topopt figure.

    Layout: two-row composite —
      * top row spans full width: compliance vs iteration per solver
      * bottom row: 3-D voxel renders of the final optimised density,
        one panel per solver (only when ``topopt_fields.npz`` is present)

    Reads ``result.json`` (+ optional ``topopt_fields.npz``,
    ``params.json``) from the experiment directory and writes
    ``<exp_key>.pdf`` next to them when ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[topopt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)
    data = load_json(result_path)
    by_solver = data.get("by_solver", {})

    solver_order = _solver_order_for(cfg.name)

    # Optional fields npz + physics params
    fields_path = out_dir / "topopt_fields.npz"
    params_path = out_dir / "params.json"
    npz = try_load_npz(fields_path) if fields_path.exists() else None
    params = load_json(params_path) if params_path.exists() else None
    ph = (params or data.get("params", {}) or {}).get("physics", {}) or {}

    have_fields = (
        npz is not None
        and "solver_names" in npz
        and ph.get("nx")
        and ph.get("ny")
        and ph.get("nz")
    )

    if have_fields:
        npz_solvers = _field_solver_names(npz, by_solver)
        n_field_panels = len(npz_solvers)
        ncols = max(1, n_field_panels)
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.62), dpi=300)
        gs = fig.add_gridspec(
            2, ncols, height_ratios=[1.0, 1.35], hspace=0.45, top=0.93, bottom=0.18
        )
        ax_c = fig.add_subplot(gs[0, :])
        field_axes = [
            fig.add_subplot(gs[1, i], projection="3d") for i in range(n_field_panels)
        ]
    else:
        fig, ax_c = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.38), dpi=300)
        fig.subplots_adjust(bottom=0.32, top=0.92)
        field_axes = []
        npz_solvers = []

    # ── Compliance vs iteration ──────────────────────────────────────────────
    present: set[str] = set()
    for solver, sdata in by_solver.items():
        alias = resolve_solver_alias(solver)
        _label, color, ls, _mk = solver_props(alias or solver)
        compliances = sdata.get("compliances", [])
        if compliances:
            ax_c.semilogy(
                range(len(compliances)),
                compliances,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            if alias is not None:
                present.add(alias)

    ax_c.set_title(f"Compliance — {cfg.category_label or cfg.name}")
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Compliance")

    # ── 3-D voxel field panels ───────────────────────────────────────────────
    if have_fields:
        nx, ny, nz = int(ph["nx"]), int(ph["ny"]), int(ph["nz"])
        for i, sname in enumerate(npz_solvers):
            ax = field_axes[i]
            rho_flat = npz[f"rho_final_{i}"]
            rho_xyz = rho_flat.reshape(nz, ny, nx).transpose(2, 1, 0)
            filled = rho_xyz > _THRESH

            alias = resolve_solver_alias(sname)
            _label, color, _ls, _mk = solver_props(alias or sname)
            fc = _voxel_facecolors(rho_xyz, filled, color)
            ax.voxels(filled, facecolors=fc, edgecolors=fc, shade=True)
            _add_bcs(ax, nx, ny, nz, ph)

            label = SOLVER_STYLES.get(alias or sname, (sname,))[0]
            ax.set_title(label, fontsize=7.5, pad=-4)
            ax.view_init(elev=_ELEV, azim=_AZIM)
            ax.set_axis_off()
            if alias is not None:
                present.add(alias)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = dedup_handles(
        [make_handle(s) for s in solver_order if s in present and s in SOLVER_STYLES]
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

    if save:
        out = out_dir / f"{exp_key}.png"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def plot_topopt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "topopt",
    **_kw: Any,
) -> Any:
    """Topopt per-experiment plot — styled figure + extras.

    Produces a publication-quality compliance-convergence + 3-D voxel figure
    (``<exp_key>.pdf``) plus:

      * ``topopt_fields`` — initial + per-solver final 2-D density panels.
      * ``topopt_evolution_<solver>.gif`` — density-field animation per solver.
      * ``topopt_3d_<solver>.png`` — interactive-style 3-D voxel render per solver.
    """
    out_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg, differentiable_only=False)
    # params is the full run dict {ic, physics, optim, ...}; v_frac lives in
    # physics sub-dict (structural-mesh layout) or directly at top level.
    _params_raw = data.get("params", {}) or {}
    _phys = _params_raw.get("physics", {}) or {}
    # For field visualisation (_rho_to_2d), pass the physics sub-dict so that
    # nx/ny/nz are found at the correct nesting level.
    params_all = _phys if _phys else _params_raw

    by_solver = data["by_solver"]

    # ── Canonical figure ────────────────────────────────────────────────────
    fig_c = _plot_topopt_figure(cfg, exp_key=exp_key, suffix=suffix, save=save)

    # ── density field panels ──────────────────────────────────────────────────
    fields_path = out_dir / "topopt_fields.npz"
    if not fields_path.exists():
        return fig_c

    npz = try_load_npz(fields_path)
    solver_names = _field_solver_names(npz, by_solver)
    if not solver_names:
        return fig_c
    n_panels = 1 + len(solver_names)
    fig_f, axes = paper_image_grid(1, n_panels)

    initial_panels = [(r"Initial $\rho$", npz["rho_init"])] if "rho_init" in npz else []
    panels = initial_panels + [
        (styles.get(n, {}).get("label", n), npz[f"rho_final_{j}"])
        for j, n in enumerate(solver_names)
        if f"rho_final_{j}" in npz
    ]
    im = None
    for ax, (title, rho) in zip(axes[0], panels, strict=False):
        im = ax.imshow(
            _rho_to_2d(rho, params_all),
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
    fig_f.suptitle("Optimised density fields")
    fig_f.tight_layout()
    if im is not None:
        # Shared colorbar in its own slim axes so every density panel stays
        # square and equally sized (no per-panel shrink).
        fig_f.subplots_adjust(right=0.90)
        cax = fig_f.add_axes((0.92, 0.18, 0.015, 0.62))
        fig_f.colorbar(im, cax=cax)
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
    cfg: Problem,
    npz: Any,
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
        fc = np.zeros((*rho_xyz.shape, 4))
        fc[..., :3] = _SOLID_RGB
        # Remap density from [threshold, 1] → [0.35, 0.92] for alpha
        alpha = np.where(
            filled,
            0.35 + 0.57 * (rho_xyz - threshold) / (1.0 - threshold + 1e-8),
            0.0,
        )
        fc[..., 3] = alpha

        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.62), dpi=300)
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

        ax.set_xlabel("$x$  (length)", labelpad=4)
        ax.set_ylabel("$y$  (width)", labelpad=4)
        ax.set_zlabel("$z$  (height)", labelpad=4)
        # View from the right-front-top so both the clamped left face and the
        # bottom-front corner load on the right face are simultaneously visible.
        # azim=45 looks from the right side (load corner is on the near face);
        # elev=30 gives enough height to see the 3-D topology clearly.
        ax.view_init(elev=30, azim=45)

        label = styles.get(name, {}).get("label", name)
        compliance_val = by_solver.get(name, {}).get("final_compliance")
        title = rf"{label}  ($\rho > {threshold}$)"
        if compliance_val is not None:
            title += f"\n$C = {compliance_val:.3e}$"
        ax.set_title(title)

        fig.tight_layout()
        save_fig(fig, f"topopt_3d_{name}", out_dir)
        plt.close(fig)


def _render_topopt_evolution_gifs(
    out_dir: Path,
    npz: Any,
    solver_names: list,
    params_all: dict,
    styles: dict,
) -> None:
    """Write ``topopt_evolution_<solver>.gif`` per solver from ``rho_history_<j>``.

    Each frame is the 2-D view of the density field at snapshot ``frame``.
    Shared ``vmin=0/vmax=1`` keeps colouring comparable across frames.
    Silently skips solvers whose ``rho_history_<j>`` key is missing.
    """
    # ``try_load_npz`` returns a plain dict; tolerate both dict and NpzFile.
    npz_keys = npz.files if hasattr(npz, "files") else set(npz.keys())
    for j, name in enumerate(solver_names):
        hist_key = f"rho_history_{j}"
        if hist_key not in npz_keys:
            continue
        history = np.asarray(npz[hist_key])  # (n_frames, n_cells)
        if history.size == 0 or history.shape[0] == 0:
            continue
        n_frames = int(history.shape[0])

        first = _rho_to_2d(history[0], params_all)
        fig, ax = paper_row(1)
        im = ax.imshow(
            first,
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        label = styles.get(name, {}).get("label", name)
        title = ax.set_title(f"{label} — iter 1 / {n_frames}")
        ax.axis("off")

        def _update(
            idx: Any,
            _im: Any = im,
            _title: Any = title,
            _hist: Any = history,
            _params: Any = params_all,
            _label: Any = label,
            _n: Any = n_frames,
        ) -> Any:
            _im.set_data(_rho_to_2d(_hist[idx], _params))
            _title.set_text(f"{_label} — iter {idx + 1} / {_n}")
            return _im, _title

        anim = manimation.FuncAnimation(
            fig, _update, frames=n_frames, interval=250, blit=False
        )
        _save_animation(anim, f"topopt_evolution_{name}", out_dir, fps=4)
