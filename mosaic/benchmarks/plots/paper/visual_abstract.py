"""Generate the Mosaic visual abstract (teaser / Figure 1).

Four-section horizontal flow:
  1  Solver backends — stacked category cards
  2  Standardized interface — single Tesseract with forward/vjp
  3  Benchmark tasks — actual 2D cylinder flow results:
       forward → initial flow field with parabolic inflow
       vjp     → optimized flow field with gradient-optimized inflow profile
  4  Evaluation results — stylized summary table

Output: visual_abstract.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from mosaic.benchmarks.core.utils import results_dir

C = {
    "bg": "#FFFFFF",
    "jax": "#4285F4",
    "pytorch": "#EE4C2C",
    "julia": "#9558B2",
    "cpp": "#6C757D",
    "fenics": "#2CA02C",
    "arrow": "#495057",
    "tess": "#2C3E50",
    "text": "#212529",
    "muted": "#868E96",
    "api_bg": "#E9ECEF",
    "fwd": "#2980B9",
    "inv": "#C0392B",
    "placeholder": "#B0BEC5",
}

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
    "text.color": C["text"],
    "figure.facecolor": C["bg"],
    "savefig.facecolor": C["bg"],
}


def _load_flow_data() -> dict | None:
    """Load flow fields and profiles from the drag_opt benchmark results."""
    base = results_dir() / "ns-grid" / "optimization" / "drag_opt" / "re20"
    flow_path = base / "flow_fields.npz"
    prof_path = base / "profiles.npz"
    if not flow_path.exists() or not prof_path.exists():
        print(f"[visual_abstract] result data not found at {base} — skipping")
        return None
    flows = np.load(flow_path)
    profs = np.load(prof_path)
    return {
        "flow_initial": flows["flow_initial"][:, :, 0, :],
        "flow_final": flows["flow_final_xlb"][:, :, 0, :],
        "profile_initial": profs["initial"],
        "profile_final": profs["final_xlb"],
    }


def _card(ax, cx, cy, w, h, name, fs, alpha=0.85) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            facecolor="#4a5568",
            edgecolor="none",
            alpha=alpha,
            lw=0,
            zorder=2,
        )
    )
    ax.text(
        cx,
        cy,
        name,
        fontsize=fs,
        fontweight="medium",
        color="white",
        ha="center",
        va="center",
        zorder=3,
        fontfamily="monospace",
    )


def _section_title(ax, text: str) -> None:
    ax.text(
        0.50,
        1.05,
        text,
        fontsize=16,
        fontweight="bold",
        color=C["text"],
        ha="center",
        va="top",
        transform=ax.transAxes,
    )


def _flow_arrow(fig, x0, y0, x1, y1) -> None:
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="->,head_width=6,head_length=4.5",
            color=C["arrow"],
            lw=2,
            mutation_scale=1,
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,
        )
    )


def _colored_arrow(fig, x0, y0, x1, y1, color) -> None:
    rad = -0.15 if y1 > y0 else 0.15
    conn = f"arc3,rad={rad}"
    style = "->,head_width=5,head_length=4"
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            color="white",
            lw=3,
            mutation_scale=1,
            connectionstyle=conn,
            transform=fig.transFigure,
            zorder=9,
            clip_on=False,
        )
    )
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            color=color,
            lw=2,
            mutation_scale=1,
            connectionstyle=conn,
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,
        )
    )


def _draw_flow_panel(ax, flow_field, profile, border_color) -> None:
    ux = flow_field[:, :, 0]
    uy = flow_field[:, :, 1]
    speed = np.sqrt(ux**2 + uy**2)
    extent = [0, 1, 0, 1]
    ax.imshow(
        speed.T,
        origin="lower",
        extent=extent,
        cmap="YlGnBu_r",
        vmin=0,
        vmax=0.85,
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(
        plt.Circle((0.33, 0.5), 0.06, fc="#2c3e50", ec="#1a252f", lw=1.0, zorder=3)
    )
    wall_h = 0.015
    ax.add_patch(
        Rectangle((0, 1 - wall_h), 1, wall_h, fc="#555555", ec="none", zorder=2)
    )
    ax.add_patch(Rectangle((0, 0), 1, wall_h, fc="#555555", ec="none", zorder=2))

    ny = len(profile)
    ys = np.linspace(0.03, 0.97, ny)
    max_arrow_len = 0.14
    p_min, p_max = profile.min(), profile.max()
    if p_max > p_min:
        p_norm = (profile - p_min) / (p_max - p_min)
    else:
        p_norm = np.ones_like(profile)

    for y, pn in list(zip(ys, p_norm, strict=False))[::2]:
        length = max_arrow_len * max(pn, 0.05)
        x_tip = length
        x_tail = -0.01
        ax.plot(
            [x_tail, x_tip],
            [y, y],
            color="black",
            lw=3.6,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.plot(
            x_tip + 0.003,
            y,
            marker=">",
            color="black",
            ms=6.0,
            markeredgewidth=0,
            zorder=4,
        )
        ax.plot(
            [x_tail, x_tip],
            [y, y],
            color=border_color,
            lw=3.2,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            x_tip + 0.003,
            y,
            marker=">",
            color=border_color,
            ms=5.5,
            markeredgewidth=0,
            zorder=5,
        )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")


def _draw_backends(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _section_title(ax, "Solver backends")
    solvers = [
        "PhiFlow",
        "XLB",
        "PICT",
        "FEniCS",
        "INS.jl",
        "OpenFOAM",
        "deal.II",
        "···",
    ]
    w, h = 0.72, 0.080
    n = len(solvers)
    total_h = n * h + (n - 1) * 0.022
    y_top = 0.5 + total_h / 2
    for i, name in enumerate(solvers):
        cy = y_top - i * (h + 0.022) - h / 2
        _card(ax, 0.50, cy, w, h, name, 13.0, alpha=0.85 if name != "···" else 0.35)


_FWD_LABEL_FIG: tuple[float, float] | None = None
_VJP_LABEL_FIG: tuple[float, float] | None = None


def _draw_interface(ax) -> None:
    global _FWD_LABEL_FIG, _VJP_LABEL_FIG
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _section_title(ax, "Standardized\ninterface")
    tcx, tcy = 0.50, 0.50
    tw, th = 0.84, 0.22
    ax.add_patch(
        FancyBboxPatch(
            (tcx - tw / 2, tcy - th / 2),
            tw,
            th,
            boxstyle="round,pad=0.018",
            facecolor=C["tess"],
            edgecolor=C["tess"],
            alpha=0.92,
            lw=2.0,
            zorder=2,
        )
    )
    ax.text(
        tcx,
        tcy + 0.06,
        "Tesseract",
        fontsize=16,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        zorder=4,
        fontfamily="monospace",
    )
    ax.text(
        tcx,
        tcy - 0.01,
        "apply(x) → y",
        fontsize=15,
        color="#85C1E9",
        ha="center",
        va="center",
        zorder=4,
        fontfamily="monospace",
        fontweight="medium",
    )
    ax.text(
        tcx,
        tcy - 0.08,
        "vjp(x, v) → g",
        fontsize=15,
        color="#F1948A",
        ha="center",
        va="center",
        zorder=4,
        fontfamily="monospace",
        fontweight="medium",
    )
    _FWD_LABEL_FIG = (tcx + tw / 2 - 0.02, tcy - 0.01)
    _VJP_LABEL_FIG = (tcx + tw / 2 - 0.02, tcy - 0.08)


def _draw_tasks(ax, data) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _section_title(ax, "Benchmark tasks")
    if data is None:
        ax.text(
            0.5,
            0.5,
            "(result data not available)",
            ha="center",
            va="center",
            fontsize=12,
            color=C["muted"],
        )
        return
    iax_top = ax.inset_axes([0.02, 0.57, 0.96, 0.34])
    _draw_flow_panel(iax_top, data["flow_initial"], data["profile_initial"], C["fwd"])
    for sp in iax_top.spines.values():
        sp.set_visible(True)
        sp.set_color(C["fwd"])
        sp.set_linewidth(2.0)
    ax.text(
        0.50,
        0.92,
        "forward solve",
        fontsize=13,
        color=C["fwd"],
        ha="center",
        va="bottom",
        style="italic",
        zorder=10,
    )
    iax_bot = ax.inset_axes([0.02, 0.08, 0.96, 0.34])
    _draw_flow_panel(iax_bot, data["flow_final"], data["profile_final"], C["inv"])
    for sp in iax_bot.spines.values():
        sp.set_visible(True)
        sp.set_color(C["inv"])
        sp.set_linewidth(2.0)
    ax.text(
        0.50,
        0.43,
        "optimized inflow (via gradient)",
        fontsize=13,
        color=C["inv"],
        ha="center",
        va="bottom",
        style="italic",
        zorder=10,
    )


def _draw_results(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _section_title(ax, "Evaluation results")
    solvers = ["PhiFlow", "XLB", "PICT", "JAX-CFD", "INS.jl", "Warp-NS", "OpenFOAM"]
    col_headers = ["Fwd", "VJP", "Grad", "Opt."]
    n_rows = len(solvers)
    x0, y0 = 0.04, 0.08
    tw, th = 0.92, 0.82
    header_h = 0.07
    row_h = (th - header_h) / n_rows
    col_w = [0.26, 0.16, 0.16, 0.16, 0.16]
    total_cw = sum(col_w)
    col_w = [c / total_cw * tw for c in col_w]
    sym = {
        "ok": ("●", "#222222"),
        "pt": ("◑", "#555555"),
        "no": ("✗", "#555555"),
        "na": ("—", "#AAAAAA"),
    }
    cell_data = [
        ["ok", "ok", "ok", "pt"],
        ["ok", "ok", "ok", "ok"],
        ["ok", "ok", "ok", "ok"],
        ["ok", "ok", "ok", "no"],
        ["ok", "ok", "pt", "no"],
        ["ok", "ok", "pt", "pt"],
        ["ok", "na", "na", "na"],
    ]
    hy = y0 + th
    ax.plot([x0, x0 + tw], [hy, hy], color=C["text"], lw=1.6)
    hx = x0 + col_w[0]
    hdr_y = hy - header_h / 2
    for j, hdr in enumerate(col_headers):
        cx = hx + sum(col_w[1 : j + 1]) + col_w[j + 1] / 2
        ax.text(
            cx,
            hdr_y,
            hdr,
            fontsize=10,
            fontweight="bold",
            color=C["text"],
            ha="center",
            va="center",
        )
    rule_y = hy - header_h
    ax.plot([x0, x0 + tw], [rule_y, rule_y], color=C["text"], lw=0.8)
    for i, solver in enumerate(solvers):
        ry = rule_y - (i + 0.5) * row_h
        ax.text(
            x0 + 0.02,
            ry,
            solver,
            fontsize=10,
            color=C["text"],
            ha="left",
            va="center",
            fontfamily="monospace",
        )
        for j, glyph in enumerate(cell_data[i]):
            cx = hx + sum(col_w[1 : j + 1]) + col_w[j + 1] / 2
            symbol, color = sym[glyph]
            ax.text(
                cx,
                ry,
                symbol,
                fontsize=11,
                color=color,
                ha="center",
                va="center",
                zorder=2,
            )
    bot_y = rule_y - n_rows * row_h
    ax.plot([x0, x0 + tw], [bot_y, bot_y], color=C["text"], lw=1.6)


def generate(out_dir: Path) -> None:
    data = _load_flow_data()

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(14, 5.8), dpi=200)
        gap = 0.028
        widths = [0.14, 0.16, 0.22, 0.32]
        left = 0.008
        bottom, h = 0.015, 0.95

        positions = []
        x = left
        for w in widths:
            positions.append(x)
            x += w + gap

        axes = []
        for pos, w in zip(positions, widths, strict=False):
            axes.append(fig.add_axes([pos, bottom, w, h]))

        _draw_backends(axes[0])
        _draw_interface(axes[1])
        _draw_tasks(axes[2], data)
        _draw_results(axes[3])

        my = 0.48
        _flow_arrow(fig, positions[0] + widths[0], my, positions[1], my)
        _flow_arrow(fig, positions[2] + widths[2], my, positions[3], my)

        ax_iface = axes[1]
        ax_tasks = axes[2]

        if _FWD_LABEL_FIG is not None:
            fwd_src = ax_iface.transData.transform(_FWD_LABEL_FIG)
            fwd_src_fig = fig.transFigure.inverted().transform(fwd_src)
            fwd_dst = ax_tasks.transData.transform((0.0, 0.71))
            fwd_dst_fig = fig.transFigure.inverted().transform(fwd_dst)
            _colored_arrow(
                fig,
                fwd_src_fig[0],
                fwd_src_fig[1],
                fwd_dst_fig[0],
                fwd_dst_fig[1],
                C["fwd"],
            )

        if _VJP_LABEL_FIG is not None:
            vjp_src = ax_iface.transData.transform(_VJP_LABEL_FIG)
            vjp_src_fig = fig.transFigure.inverted().transform(vjp_src)
            vjp_dst = ax_tasks.transData.transform((0.0, 0.24))
            vjp_dst_fig = fig.transFigure.inverted().transform(vjp_dst)
            _colored_arrow(
                fig,
                vjp_src_fig[0],
                vjp_src_fig[1],
                vjp_dst_fig[0],
                vjp_dst_fig[1],
                C["inv"],
            )

        for ext in ("pdf", "png"):
            out = out_dir / f"visual_abstract.{ext}"
            fig.savefig(
                out,
                bbox_inches="tight",
                pad_inches=0.04,
                dpi=300 if ext == "png" else 200,
            )
            print(f"Saved {out}")
        plt.close(fig)
