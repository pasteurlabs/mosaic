"""Per-problem plots for the structural-mesh topology optimisation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.optimization import (
    _rho_to_2d,
    _save_animation,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    fig_shared_legend,
    save_fig,
    solver_plot_props,
    solver_styles,
)


def plot_topopt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "topopt",
    **_kw,
):
    """Two files: compliance + volume fraction convergence; initial + final density fields."""
    out_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
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

    npz = try_load_npz(fields_path)
    solver_names = npz["solver_names"].tolist()
    n_panels = 1 + len(solver_names)
    fig_f, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3, 3), squeeze=False)

    panels = [("Initial ρ", npz["rho_init"])] + [
        (styles.get(n, {}).get("label", n) + " final", npz[f"rho_final_{j}"])
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
    cfg: Problem,
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
        fc = np.zeros((*rho_xyz.shape, 4))
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
