"""Cross-domain / cross-experiment extra plots for structural-mesh.

Registered on the :class:`Problem` via :meth:`Problem.add_extra_plot` from
:mod:`structural_mesh.config`. Each plot is wrapped to take the standard
``(cfg, **kw)`` signature used by the runner and writes its outputs under
``results/<cfg.name>/_extra/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.cost_overview import (
    plot_cost_overview_for,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    RCPARAMS,
    SOLVER_STYLES,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    dedup_handles,
    make_handle,
    resolve_solver_alias,
)

_THRESH = 0.35
_ELEV = 22
_AZIM = 35
_CLR_FIXED = "#888888"
_CLR_LOAD = "#FF1744"


# ── 3-D helpers ───────────────────────────────────────────────────────────────


def _add_bcs(ax, nx: int, ny: int, nz: int, ph: dict) -> None:
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

    arrow_len = nz * 0.60
    ox = nx + 1.5 if load_axis != "y" else nx

    # Draw load arrow as Poly3DCollection — same rendering path as the wall,
    # so it is not occluded by voxel surfaces like scatter/quiver can be.
    hw = ny * 0.55  # head half-width (Y)
    hl = nz * 0.15  # head length (Z or Y)
    sw = ny * 0.18  # shaft half-width

    if load_axis == "y":
        y_sign = -1 if corner_y_high else 1
        ey = ly + y_sign * arrow_len
        shaft_end = ly + y_sign * (arrow_len - hl)
        # shaft rectangle in XY plane at z=lz
        shaft_verts = [
            [
                (ox - sw, ly, lz),
                (ox + sw, ly, lz),
                (ox + sw, shaft_end, lz),
                (ox - sw, shaft_end, lz),
            ]
        ]
        head_verts = [
            [
                (ox - hw, shaft_end, lz),
                (ox + hw, shaft_end, lz),
                (ox, ey, lz),
            ]
        ]
    else:
        z_sign = -1 if corner_z_high else 1
        ez = lz + z_sign * arrow_len
        shaft_end = lz + z_sign * (arrow_len - hl)
        # shaft rectangle in YZ plane at x=ox
        shaft_verts = [
            [
                (ox, ly - sw, lz),
                (ox, ly + sw, lz),
                (ox, ly + sw, shaft_end),
                (ox, ly - sw, shaft_end),
            ]
        ]
        head_verts = [
            [
                (ox, ly - hw, shaft_end),
                (ox, ly + hw, shaft_end),
                (ox, ly, ez),
            ]
        ]

    for verts in [shaft_verts, head_verts]:
        ax.add_collection3d(
            Poly3DCollection(
                verts,
                alpha=1.0,
                facecolor=_CLR_LOAD,
                edgecolor=_CLR_LOAD,
            )
        )


def _voxel_facecolors(
    rho_xyz: np.ndarray, filled: np.ndarray, base_color: str
) -> np.ndarray:
    r, g, b, _ = mcolors.to_rgba(base_color)
    fc = np.zeros((*rho_xyz.shape, 4))
    norm = np.where(filled, (rho_xyz - _THRESH) / (1.0 - _THRESH), 0.0)
    fc[..., 0] = r + (1 - r) * (1 - norm) * 0.45
    fc[..., 1] = g + (1 - g) * (1 - norm) * 0.45
    fc[..., 2] = b + (1 - b) * (1 - norm) * 0.45
    fc[..., 3] = filled.astype(float)
    return fc


# ── 2-D engineering view helpers ──────────────────────────────────────────────


def _draw_fixed_support(
    ax, x0: float, y0: float, y1: float, size: float, orient: str = "left"
) -> None:
    """Hatch marks representing a clamped edge."""
    ax.plot(
        [x0, x0],
        [y0, y1],
        color="#222222",
        linewidth=1.2,
        solid_capstyle="butt",
        zorder=4,
    )
    n = max(4, int(abs(y1 - y0) / size * 1.4))
    for yi in np.linspace(y0, y1, n + 1)[:-1] + abs(y1 - y0) / (2 * n):
        if orient == "left":
            ax.plot(
                [x0, x0 - size],
                [yi, yi - size * 0.6],
                color="#555555",
                linewidth=0.7,
                zorder=4,
            )
        else:
            ax.plot(
                [x0, x0 + size],
                [yi, yi - size * 0.6],
                color="#555555",
                linewidth=0.7,
                zorder=4,
            )


def _draw_load_arrow(
    ax, x: float, y: float, dx: float, dy: float, length: float, ref: float = 1.0
) -> None:
    """Arrow in screen-space size so it renders clearly regardless of data aspect."""
    ax.annotate(
        "",
        xy=(x + dx * length, y + dy * length),
        xytext=(x, y),
        arrowprops={
            "arrowstyle": "simple",
            "facecolor": _CLR_LOAD,
            "edgecolor": _CLR_LOAD,
            "mutation_scale": 16,
        },
        zorder=5,
    )


def _ortho_view(
    ax,
    nx: int,
    ny: int,
    nz: int,
    ph: dict,
    view: str,
    rho_xyz: np.ndarray | None = None,
) -> None:
    """
    Draw one orthographic projection of the cantilever domain with BCs.

    view: 'front' = XZ plane (looking from −y, the main elevation)
          'top'   = XY plane (looking from +z, plan view)
          'right' = YZ plane (looking from +x, the load face)
    """
    Lx = ph["Lx"]
    Ly = ph["Ly"]
    Lz = ph["Lz"]
    corner_z_high = ph.get("corner_z_high", False)
    corner_y_high = ph.get("corner_y_high", False)
    load_z = Lz * (1 - 0.5 / nz) if corner_z_high else Lz / (2 * nz)
    load_y = Ly * (1 - 0.5 / ny) if corner_y_high else Ly / (2 * ny)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if view == "front":
        W, H = Lx, Lz
        ax.set_xlim(-0.22 * W, W + 0.08 * W)
        ax.set_ylim(-0.08 * H, H + 0.10 * H)

        ax.add_patch(
            mpatches.Rectangle(
                (0, 0), W, H, fc="#f0f0f0", ec="#333333", lw=0.9, zorder=1
            )
        )

        if rho_xyz is not None:
            proj = rho_xyz.sum(axis=1) / ny
            proj_norm = proj / max(proj.max(), 1e-6)
            ax.imshow(
                proj_norm.T[::-1],
                extent=[0, W, 0, H],
                cmap="gray_r",
                vmin=0,
                vmax=1,
                aspect="auto",
                alpha=0.75,
                zorder=2,
            )

        _draw_fixed_support(ax, x0=0, y0=0, y1=H, size=0.10 * H, orient="left")
        _draw_load_arrow(ax, x=W, y=load_z, dx=0, dy=1, length=0.38 * H, ref=H)
        ax.set_title("Front (XZ)", fontsize=6, pad=2)

    elif view == "top":
        W, H = Lx, Ly
        ax.set_xlim(-0.22 * W, W + 0.08 * W)
        ax.set_ylim(-0.08 * H, H + 0.10 * H)

        ax.add_patch(
            mpatches.Rectangle(
                (0, 0), W, H, fc="#f0f0f0", ec="#333333", lw=0.9, zorder=1
            )
        )

        if rho_xyz is not None:
            proj = rho_xyz.sum(axis=2) / nz
            proj_norm = proj / max(proj.max(), 1e-6)
            ax.imshow(
                proj_norm.T[::-1],
                extent=[0, W, 0, H],
                cmap="gray_r",
                vmin=0,
                vmax=1,
                aspect="auto",
                alpha=0.75,
                zorder=2,
            )

        _draw_fixed_support(ax, x0=0, y0=0, y1=H, size=0.30 * H, orient="left")

        # +z load out-of-plane: circled dot
        cx, cy, r = W - Lx / nx / 2, load_y, 0.10 * H
        ax.add_patch(mpatches.Circle((cx, cy), r, fc=_CLR_LOAD, ec=_CLR_LOAD, zorder=5))
        ax.scatter([cx], [cy], s=8, color="white", zorder=6)
        ax.set_title("Top (XY)", fontsize=6, pad=2)

    elif view == "right":
        W, H = Ly, Lz
        ax.set_xlim(-0.12 * W, W + 0.12 * W)
        ax.set_ylim(-0.08 * H, H + 0.10 * H)

        ax.add_patch(
            mpatches.Rectangle(
                (0, 0), W, H, fc="#f0f0f0", ec="#333333", lw=0.9, zorder=1
            )
        )

        if rho_xyz is not None:
            proj = rho_xyz.sum(axis=0) / nx
            proj_norm = proj / max(proj.max(), 1e-6)
            ax.imshow(
                proj_norm.T[::-1],
                extent=[0, W, 0, H],
                cmap="gray_r",
                vmin=0,
                vmax=1,
                aspect="auto",
                alpha=0.75,
                zorder=2,
            )

        _draw_load_arrow(ax, x=load_y, y=load_z, dx=0, dy=1, length=0.38 * H, ref=H)
        ax.set_title("Right (YZ)", fontsize=6, pad=2)


# ── Convergence plot helper ───────────────────────────────────────────────────


def _plot_convergence(
    ax,
    datasets: list[tuple[str, dict]],
    key: str,
    ylabel: str,
    seen_solvers: set,
    seen_methods: set,
    x_jitter: float = 1.06,
) -> None:
    """Overlay multiple optimizer datasets on ax. datasets = [(m_ls, by_solver), ...]
    x_jitter: multiplicative x offset per line to separate overlapping curves on log scale."""
    line_n = 0
    for m_ls, by_solver in datasets:
        # ``by_solver`` keyed by spec.name (display form); bridge to alias.
        alias_to_display: dict[str, str] = {}
        for display_name in by_solver:
            a = resolve_solver_alias(display_name)
            if a is not None:
                alias_to_display[a] = display_name
        for alias in STRUCTURAL_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            sdata = by_solver.get(display_name)
            if sdata is None:
                continue
            vals = sdata.get(key, [])
            if not vals:
                continue
            _, color, _, _ = SOLVER_STYLES.get(alias, (alias, "#888", "-", "o"))
            step = max(1, len(vals) // 300)
            idx = list(range(0, len(vals), step))
            if idx and idx[-1] != len(vals) - 1:
                idx.append(len(vals) - 1)
            x_mult = x_jitter**line_n
            ax.semilogx(
                [x_mult * (i + 1) for i in idx],
                [vals[i] for i in idx],
                color=color,
                linestyle=m_ls,
                linewidth=1.5,
                alpha=0.9,
            )
            seen_solvers.add(alias)
            seen_methods.add(m_ls)
            line_n += 1
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)


# ── topopt_overview generate body ─────────────────────────────────────────────


def _topopt_overview_generate(out_dir: Path) -> None:
    """Generate ``topopt_overview.pdf`` into *out_dir*.

    Layout:
      Left column (split vertically):
        Top  : 2×2 3-D voxel panels (one per solver)
        Bottom: engineering plan views (front / top / right) with BC symbols
      Right column:
        Top   : compliance convergence  (log x, linear y)
        Bottom: volume-fraction history (log x, linear y, target dashed)
    """
    base = results_dir() / "structural-mesh" / "optimization"
    opt_methods = {
        "adam": ("-", "Adam", base / "topopt" / "result.json"),
        "mma": ("--", "MMA", base / "topopt_mma" / "result.json"),
    }

    result_path = base / "topopt" / "result.json"
    if not result_path.exists():
        print(f"[topopt_overview] {result_path} not found — skipping")
        return

    data = load_json(result_path)
    by_solver = data["by_solver"]

    # Load additional optimizer results where available
    opt_datasets: list[tuple[str, dict]] = []
    for key, (m_ls, _m_label, rp) in opt_methods.items():
        if rp.exists():
            opt_datasets.append((m_ls, load_json(rp)["by_solver"]))
        else:
            print(f"[topopt_overview] {rp} not found — skipping {key}")
    if not opt_datasets:
        opt_datasets = [("-", by_solver)]

    fields_path = base / "topopt" / "topopt_fields.npz"
    params_path = base / "topopt" / "params.json"
    has_fields = fields_path.exists() and params_path.exists()

    if has_fields:
        npz = try_load_npz(fields_path)
        ph = load_json(params_path)["physics"]
        nx, ny, nz = ph["nx"], ph["ny"], ph["nz"]
        npz_solvers = list(npz["solver_names"])
        # ``npz_solvers`` are spec.names (display form); ``STRUCTURAL_ORDER``
        # is alias-keyed.  Build alias→display, then keep alias entries in
        # canonical order while resolving back to the npz key when needed.
        _npz_alias_to_display: dict[str, str] = {}
        for display_name in npz_solvers:
            a = resolve_solver_alias(display_name)
            if a is not None:
                _npz_alias_to_display[a] = display_name
        field_solvers = [s for s in STRUCTURAL_ORDER if s in _npz_alias_to_display]
    else:
        field_solvers = []
        ph = {}
        _npz_alias_to_display = {}

    n_field = len(field_solvers)

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.72))
        outer = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.05, 1.0],
            left=0.03,
            right=0.98,
            top=0.97,
            bottom=0.17,
            wspace=0.28,
        )

        # ── Left column: 3-D panels / ortho views ─────────────────────────
        left_col = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[0],
            height_ratios=[1.35, 0.48],
            hspace=0.08,
        )

        # ── 2×2 3-D voxel panels ──────────────────────────────────────────
        seen3d: set[str] = set()
        if n_field > 0:
            top_gs = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=left_col[0],
                wspace=0.0,
                hspace=-0.05,
            )
            for panel_i, alias in enumerate(field_solvers[:4]):
                r, c = divmod(panel_i, 2)
                ax3d = fig.add_subplot(top_gs[r, c], projection="3d")

                display_name = _npz_alias_to_display[alias]
                npz_i = npz_solvers.index(display_name)
                rho_xyz = (
                    npz[f"rho_final_{npz_i}"].reshape(nz, ny, nx).transpose(2, 1, 0)
                )
                filled = rho_xyz > _THRESH

                _, color, _, _ = SOLVER_STYLES.get(alias, (alias, "#555555", "-", "o"))
                fc = _voxel_facecolors(rho_xyz, filled, color)
                ax3d.voxels(filled, facecolors=fc, edgecolors=fc, shade=True)
                _add_bcs(ax3d, nx, ny, nz, ph)

                label = SOLVER_STYLES.get(alias, (alias,))[0]
                ax3d.set_title(label, fontsize=7.0, pad=-4)
                ax3d.view_init(elev=_ELEV, azim=_AZIM)
                ax3d.set_axis_off()
                seen3d.add(alias)

        # ── Bottom: 3 ortho views ─────────────────────────────────────────
        if n_field > 0:
            ortho_gs = gridspec.GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=left_col[1],
                width_ratios=[2.0, 2.0, 1.0],
                wspace=0.12,
            )

            # representative solution (first solver) for BC-view overlay
            _rep_display = _npz_alias_to_display[field_solvers[0]]
            npz_i0 = npz_solvers.index(_rep_display)
            rho_rep = npz[f"rho_final_{npz_i0}"].reshape(nz, ny, nx).transpose(2, 1, 0)

            for col_i, view in enumerate(["front", "top", "right"]):
                ax_v = fig.add_subplot(ortho_gs[0, col_i])
                _ortho_view(ax_v, nx, ny, nz, ph, view, rho_rep)

        # ── BC legend (left column, bottom) ───────────────────────────────
        bc_handles = [
            mpatches.Patch(
                facecolor=_CLR_FIXED,
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.7,
                label="Fixed ($u=0$)",
            ),
            mlines.Line2D(
                [],
                [],
                color=_CLR_LOAD,
                linewidth=2.0,
                marker=">",
                markersize=4,
                label="Load",
            ),
        ]
        fig.legend(
            handles=bc_handles,
            loc="lower left",
            bbox_to_anchor=(outer.left, 0.01),
            bbox_transform=fig.transFigure,
            ncol=2,
            fontsize=6.5,
            framealpha=0.8,
            handlelength=1.6,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        # ── Right column: convergence plots ───────────────────────────────
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[1],
            hspace=0.32,
        )
        ax_comp = fig.add_subplot(right_gs[0])
        ax_volf = fig.add_subplot(right_gs[1])

        seen_conv: set[str] = set()
        seen_methods: set[str] = set()
        _plot_convergence(
            ax_comp, opt_datasets, "compliances", "Compliance", seen_conv, seen_methods
        )
        _plot_convergence(
            ax_volf,
            opt_datasets,
            "vol_fracs",
            "Volume fraction",
            seen_conv,
            seen_methods,
        )

        v_frac_target = ph.get("v_frac", 0.5) if has_fields else 0.5
        ax_volf.axhline(v_frac_target, color="#555555", linestyle=":", linewidth=1.0)
        ax_volf.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
        ax_volf.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax_comp.set_title("Structural topology optimisation", fontsize=8)

        method_handles = [
            mlines.Line2D(
                [], [], color="0.3", linestyle=m_ls, linewidth=1.5, label=m_label
            )
            for _, (m_ls, m_label, rp) in opt_methods.items()
            if rp.exists()
        ]
        solver_handles = dedup_handles(
            [make_handle(s) for s in STRUCTURAL_ORDER if s in seen_conv]
        )
        ax_comp.legend(
            handles=method_handles + solver_handles,
            loc="upper right",
            ncol=1,
            fontsize=6.5,
            framealpha=0.80,
            handlelength=1.8,
        )

        out = out_dir / "topopt_overview.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


# ── Adapter + registration ────────────────────────────────────────────────────


def _topopt_overview_plot(cfg: Problem, **_kw) -> None:
    """Runner-facing adapter: writes ``topopt_overview.pdf`` under ``_extra/``."""
    out_dir = results_dir() / cfg.name / "_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    _topopt_overview_generate(out_dir)


def _plot_cost_overview(cfg: Problem, **_kw) -> None:
    plot_cost_overview_for(cfg, steady_state=True)


def register(problem: Problem) -> None:
    """Attach cross-experiment extras to *problem*."""
    problem.add_extra_plot("_extra/topopt_overview", _topopt_overview_plot)
    problem.add_extra_plot("_extra/cost_overview", _plot_cost_overview)
