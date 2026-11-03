"""Generate domain illustration figures for all benchmark domains."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import TEXTWIDTH
from .style import RCPARAMS

# ── Shared palette & styling for Figure 2 ──────────────────────────────
CONTROL_COLOR = "#2471a3"  # blue — marks control variables
OBJECTIVE_COLOR = "#c0392b"  # red — marks objectives / loads / targets
PHYS_COLOR = "#333333"  # dark gray — governing eqns, physical labels
OBJ_BOX_KW = {
    "boxstyle": "round,pad=0.3",
    "facecolor": "#F2F2F2",
    "edgecolor": "#AAAAAA",
    "lw": 0.6,
}
OBJ_FONTSIZE = 5
CTRL_FONTSIZE = 4.5
LABEL_FONTSIZE = 4.5
OFFSET_OBJECTIVE = 0.1


def _make_domain1(out_dir: Path) -> None:
    """Domain 1: 2D Cylinder Flow — Inflow Optimization."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.22), dpi=300)
    ax.set_xlim(-0.5, 7.8)
    ax.set_ylim(-1.0, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    font_kw = {"fontfamily": "sans-serif"}

    chan_x0, chan_x1 = 0.0, 6.5
    chan_y0, chan_y1 = 0.0, 2.0
    chan_w = chan_x1 - chan_x0
    chan_h = chan_y1 - chan_y0

    fluid = mpatches.FancyBboxPatch(
        (chan_x0, chan_y0),
        chan_w,
        chan_h,
        boxstyle="square,pad=0",
        facecolor="#dceefb",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(fluid)

    wall_thick = 0.10
    top_wall = mpatches.FancyBboxPatch(
        (chan_x0, chan_y1),
        chan_w,
        wall_thick,
        boxstyle="square,pad=0",
        facecolor="#888888",
        edgecolor="#555555",
        linewidth=0.5,
        zorder=2,
    )
    bot_wall = mpatches.FancyBboxPatch(
        (chan_x0, chan_y0 - wall_thick),
        chan_w,
        wall_thick,
        boxstyle="square,pad=0",
        facecolor="#888888",
        edgecolor="#555555",
        linewidth=0.5,
        zorder=2,
    )
    ax.add_patch(top_wall)
    ax.add_patch(bot_wall)

    # ax.text(
    #    chan_x0 + chan_w / 2,
    #    chan_y1 + wall_thick + 0.12,
    #    "no-slip wall",
    #    ha="center",
    #    va="bottom",
    #    fontsize=4,
    #    color="#555555",
    #    **font_kw,
    # )
    # ax.text(
    #    chan_x0 + chan_w / 2,
    #    chan_y0 - wall_thick - 0.12,
    #    "no-slip wall",
    #    ha="center",
    #    va="top",
    #    fontsize=4,
    #    color="#555555",
    #    **font_kw,
    # )

    cyl_cx = chan_x0 + chan_w / 3
    cyl_cy = chan_y0 + chan_h / 2
    cyl_r = 0.22
    cyl = plt.Circle(
        (cyl_cx, cyl_cy), cyl_r, fc="#3b6fa0", ec="#1a3a5c", linewidth=0.5, zorder=5
    )
    ax.add_patch(cyl)

    drag_arrow_x = cyl_cx - cyl_r - 0.08
    ax.annotate(
        "",
        xy=(drag_arrow_x - 0.7, cyl_cy),
        xytext=(drag_arrow_x, cyl_cy),
        arrowprops={
            "arrowstyle": "->,head_width=0.08,head_length=0.06",
            "color": OBJECTIVE_COLOR,
            "lw": 0.8,
        },
        zorder=6,
    )
    ax.text(
        drag_arrow_x,
        cyl_cy - 0.5,
        r"Drag $F_D$",
        ha="right",
        va="center",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color=OBJECTIVE_COLOR,
        **font_kw,
    )

    n_arrows = 9
    ys = np.linspace(chan_y0 + 0.12, chan_y1 - 0.12, n_arrows)
    y_mid = (chan_y0 + chan_y1) / 2
    max_len = 1.0
    for y in ys:
        frac = 1.0 - ((y - y_mid) / (chan_h / 2)) ** 2
        length = max_len * max(frac, 0.08)
        ax.annotate(
            "",
            xy=(chan_x0 + length, y),
            xytext=(chan_x0 - 0.05, y),
            arrowprops={
                "arrowstyle": "->,head_width=0.08,head_length=0.06",
                "color": CONTROL_COLOR,
                "lw": 0.6,
            },
            zorder=4,
        )

    ax.text(
        chan_x0 - 0.15,
        chan_y1 + 0.45,
        r"Control: inflow profile $u(y)$",
        ha="left",
        va="bottom",
        fontsize=CTRL_FONTSIZE,
        fontweight="bold",
        color=CONTROL_COLOR,
        **font_kw,
    )

    for y in np.linspace(chan_y0 + 0.25, chan_y1 - 0.25, 5):
        ax.annotate(
            "",
            xy=(chan_x1 + 0.35, y),
            xytext=(chan_x1 - 0.15, y),
            arrowprops={
                "arrowstyle": "->,head_width=0.08,head_length=0.06",
                "color": CONTROL_COLOR,
                "lw": 0.6,
            },
            zorder=4,
        )
    # ax.text(
    #    chan_x1 + 0.4,
    #    cyl_cy,
    #    "outflow",
    #    ha="left",
    #    va="center",
    #    fontsize=4,
    #    color="#2471a3",
    #    **font_kw,
    # )

    xs_stream = np.linspace(cyl_cx + cyl_r + 0.25, chan_x1 - 0.3, 200)
    offsets = [0.0, 0.35, -0.35, 0.65, -0.65]
    amplitudes = [0.18, 0.14, 0.14, 0.06, 0.06]
    phases = [0.0, np.pi, 0.0, np.pi / 2, -np.pi / 2]

    for off, amp, ph in zip(offsets, amplitudes, phases, strict=False):
        local_amp = amp * np.clip(
            (xs_stream - xs_stream[0]) / (xs_stream[-1] - xs_stream[0]), 0, 1
        )
        ys_stream = (
            cyl_cy
            + off
            + local_amp
            * np.sin(
                2.5
                * np.pi
                * (xs_stream - xs_stream[0])
                / (xs_stream[-1] - xs_stream[0])
                * 2
                + ph
            )
        )
        ys_stream = np.clip(ys_stream, chan_y0 + 0.05, chan_y1 - 0.05)
        ax.plot(xs_stream, ys_stream, color="#5dade2", lw=0.4, alpha=0.7, zorder=1)

    vortex_colors = ["#e74c3c", "#2471a3"]
    vortex_xs = np.linspace(cyl_cx + cyl_r + 0.8, chan_x1 - 1.0, 4)
    for i, vx in enumerate(vortex_xs):
        sign = 1 if i % 2 == 0 else -1
        vy = cyl_cy + sign * 0.22
        ell = mpatches.Ellipse(
            (vx, vy),
            0.36,
            0.23,
            angle=sign * 10,
            fc="none",
            ec=vortex_colors[i % 2],
            lw=0.6,
            ls="--",
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(ell)
        theta_start = 30 if sign > 0 else 210
        arc_angles = np.linspace(
            np.radians(theta_start), np.radians(theta_start + 260), 40
        )
        rx, ry = 0.12, 0.07
        arc_x = vx + rx * np.cos(arc_angles)
        arc_y = vy + ry * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color=vortex_colors[i % 2], lw=0.6, alpha=0.5, zorder=3)
        ax.annotate(
            "",
            xy=(arc_x[-1], arc_y[-1]),
            xytext=(arc_x[-3], arc_y[-3]),
            arrowprops={
                "arrowstyle": "->,head_width=0.05,head_length=0.04",
                "color": vortex_colors[i % 2],
                "lw": 0.6,
            },
            zorder=3,
        )

    for y_off in [-0.65, -0.35, 0.0, 0.35, 0.65]:
        xs_up = np.linspace(chan_x0 + 1.1, cyl_cx - cyl_r - 0.15, 60)
        ys_up = np.full_like(xs_up, cyl_cy + y_off)
        if abs(y_off) < 0.5:
            ys_up += 0.04 * y_off * np.linspace(0, 1, len(xs_up)) ** 2
        ys_up = np.clip(ys_up, chan_y0 + 0.05, chan_y1 - 0.05)
        ax.plot(xs_up, ys_up, color="#5dade2", lw=0.4, alpha=0.7, zorder=1)

    fig.text(
        0.5,
        OFFSET_OBJECTIVE,
        r"Objective: $\min_{u(y)}\; F_D$",
        ha="center",
        va="bottom",
        fontsize=OBJ_FONTSIZE,
        color=PHYS_COLOR,
        bbox=OBJ_BOX_KW,
    )

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.10)
    fig.savefig(
        out_dir / "domain1_2d_fluids.png",
        dpi=300,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.10)
    fig.savefig(
        out_dir / "domain1_2d_fluids.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain1_2d_fluids.png'}")
    return fig


def _make_domain2a_ic_recovery(out_dir: Path) -> None:
    """Domain 2A: 3D Initial Condition Recovery."""

    def taylor_green_vorticity(n=80):
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)
        return -2 * np.sin(X) * np.sin(Y)

    def decayed_vorticity(n=80):
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)
        angle = 0.4
        ca, sa = np.cos(angle), np.sin(angle)
        Xr = ca * X + sa * Y
        Yr = -sa * X + ca * Y
        return 0.55 * (-2) * np.sin(Xr) * np.sin(Yr)

    def paint_face(ax, vort, face, origin, size, cmap, vmax):
        n = vort.shape[0]
        o = np.array(origin, dtype=float)
        s = size
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        cm = plt.get_cmap(cmap)
        step = max(1, n // 80)
        coords = np.linspace(0, s, n)
        for i in range(0, n - step, step):
            for j in range(0, n - step, step):
                val = vort[i : i + step, j : j + step].mean()
                ci, cj = coords[i], coords[j]
                di = coords[min(i + step, n - 1)] - ci
                dj = coords[min(j + step, n - 1)] - cj
                if face == "xy_bottom":
                    verts = [
                        o + [ci, cj, 0],
                        o + [ci + di, cj, 0],
                        o + [ci + di, cj + dj, 0],
                        o + [ci, cj + dj, 0],
                    ]
                elif face == "xz_front":
                    verts = [
                        o + [ci, 0, cj],
                        o + [ci + di, 0, cj],
                        o + [ci + di, 0, cj + dj],
                        o + [ci, 0, cj + dj],
                    ]
                elif face == "yz_left":
                    verts = [
                        o + [0, ci, cj],
                        o + [0, ci + di, cj],
                        o + [0, ci + di, cj + dj],
                        o + [0, ci, cj + dj],
                    ]
                poly = Poly3DCollection([verts], alpha=0.75, zorder=2)
                poly.set_facecolor(cm(norm(val)))
                poly.set_edgecolor("none")
                ax.add_collection3d(poly)

    def draw_box_edges(ax, origin, size):
        o = np.array(origin)
        s = size
        corners = (
            np.array(
                [
                    [0, 0, 0],
                    [s, 0, 0],
                    [s, s, 0],
                    [0, s, 0],
                    [0, 0, s],
                    [s, 0, s],
                    [s, s, s],
                    [0, s, s],
                ]
            )
            + o
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for i, j in edges:
            ax.plot(
                *zip(corners[i], corners[j], strict=False),
                color="0.5",
                linewidth=0.5,
                linestyle="-",
                zorder=5,
            )

    BLUE = CONTROL_COLOR
    PURPLE = OBJECTIVE_COLOR
    GRAY = "0.4"

    box_size = 1.0
    gap = 1.3

    # --- render 3D scene to buffer ---
    fig3d = plt.figure(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=800)
    ax = fig3d.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    ax.view_init(elev=20, azim=-55)
    ax.set_proj_type("persp", focal_length=0.35)

    omega_ic = taylor_green_vorticity(160)
    omega_tgt = decayed_vorticity(160)

    lo = (0, 0, 0)
    paint_face(ax, omega_ic, "xy_bottom", lo, box_size, "RdBu_r", 2.0)
    paint_face(ax, omega_ic.T, "xz_front", lo, box_size, "RdBu_r", 2.0)
    paint_face(ax, omega_ic, "yz_left", lo, box_size, "RdBu_r", 2.0)
    draw_box_edges(ax, lo, box_size)

    ro = (box_size + gap, 0, 0)
    paint_face(ax, omega_tgt, "xy_bottom", ro, box_size, "PiYG_r", 1.5)
    paint_face(ax, omega_tgt.T, "xz_front", ro, box_size, "PiYG_r", 1.5)
    paint_face(ax, omega_tgt, "yz_left", ro, box_size, "PiYG_r", 1.5)
    draw_box_edges(ax, ro, box_size)

    ax.plot(
        [box_size + 0.12, box_size + gap - 0.12],
        [0.5, 0.5],
        [0.5, 0.5],
        color=GRAY,
        linewidth=0.8,
        zorder=10,
        solid_capstyle="round",
    )
    ax.quiver(
        box_size + gap - 0.25,
        0.5,
        0.5,
        0.12,
        0,
        0,
        color=GRAY,
        arrow_length_ratio=0.55,
        linewidth=0.8,
        zorder=10,
    )
    ax.text(
        box_size + gap / 2,
        0.5,
        0.7,
        "Navier\u2013Stokes\nevolution",
        fontsize=3.5,
        color=PHYS_COLOR,
        ha="center",
        fontweight="bold",
        zorder=12,
    )

    total_w = 2 * box_size + gap
    xlim = (-0.1, total_w + 0.5)
    ylim = (-0.1, 1.15)
    zlim = (-0.05, 1.35)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    ax.set_axis_off()

    buf = BytesIO()
    fig3d.savefig(buf, format="png", dpi=800, facecolor="white")
    buf.seek(0)
    plt.close(fig3d)

    # --- final figure: imshow + 2D labels ---
    img = plt.imread(buf)

    # Auto-crop to content bounding box with small padding
    gray = np.mean(img[:, :, :3], axis=2)
    mask = gray < 0.99
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    pad_r = int(img.shape[0] * 0.02)
    pad_c = int(img.shape[1] * 0.03)
    r0 = max(rows[0] - pad_r, 0)
    r1 = min(rows[-1] + pad_r, img.shape[0])
    c0 = max(cols[0] - pad_c, 0)
    c1 = min(cols[-1] + pad_c, img.shape[1])
    img = img[r0:r1, c0:c1, :]

    fig = plt.figure(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=300)
    ax2d = fig.add_axes((0.0, 0.10, 1.0, 0.76))
    ax2d.imshow(img)
    ax2d.axis("off")

    # control label — above left box
    fig.text(
        0.25,
        0.90,
        r"Control: $\mathbf{v}_0$",
        fontsize=CTRL_FONTSIZE,
        color=BLUE,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    # target label — above right box
    fig.text(
        0.68,
        0.90,
        r"Target: $\mathbf{v}(T)$",
        fontsize=CTRL_FONTSIZE,
        color=PURPLE,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    # objective box at bottom
    fig.text(
        0.5,
        OFFSET_OBJECTIVE,
        r"Objective: $\min_{\mathbf{v}_0}"
        r" \|\mathbf{v}(T;\,\mathbf{v}_0)"
        r" - \mathbf{v}_{\mathrm{target}}\|^2$",
        fontsize=OBJ_FONTSIZE,
        color=PHYS_COLOR,
        ha="center",
        va="bottom",
        bbox=OBJ_BOX_KW,
    )

    fig.savefig(out_dir / "domain2a_3d_ic_recovery.png", dpi=300, facecolor="white")
    fig.savefig(
        out_dir / "domain2a_3d_ic_recovery.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2a_3d_ic_recovery.png'}")
    return fig


def _make_domain3(out_dir: Path) -> None:
    """Domain 3: Cantilever Beam — Compliance Minimization."""
    from matplotlib import patches

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.22), dpi=300)

    x0, y0 = 1.5, 1.0
    W, H = 6.0, 2.0
    nx, ny = 90, 30
    rho = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            xn = i / (nx - 1)
            yn = j / (ny - 1)
            density = 0.0
            top_thickness = 0.12 + 0.06 * (1 - xn)
            if yn > (1.0 - top_thickness):
                density = max(density, 0.85 + 0.15 * (1 - xn))
            bot_thickness = 0.12 + 0.06 * (1 - xn)
            if yn < bot_thickness:
                density = max(density, 0.85 + 0.15 * (1 - xn))
            n_bays = 5
            for k in range(n_bays):
                bay_left = k / n_bays
                bay_right = (k + 1) / n_bays
                bay_mid = (bay_left + bay_right) / 2.0
                strut_half_w = 0.035 + 0.02 * (1 - bay_mid)
                if bay_left <= xn <= bay_right:
                    local_x = (xn - bay_left) / (bay_right - bay_left)
                    for target in [
                        1.0 - local_x * 0.5,
                        local_x * 0.5,
                        0.5 + local_x * 0.5,
                        0.5 - local_x * 0.5,
                    ]:
                        if abs(yn - target) < strut_half_w:
                            density = max(density, 0.7 + 0.2 * (1 - xn))
            if xn < 0.04:
                density = max(density, 1.0)
            dist_to_load = np.sqrt((xn - 1.0) ** 2 + (yn - 0.5) ** 2)
            if dist_to_load < 0.12:
                density = max(density, 0.9)
            rho[j, i] = density

    def smooth_2d(arr, n=3):
        out = arr.copy()
        for _ in range(n):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0
        return out

    rho = smooth_2d(rho, n=3)
    rho = np.clip(rho, 0, 1)

    cmap = LinearSegmentedColormap.from_list(
        "topo", ["#FFFFFF", "#BDD7EE", "#4472C4", "#1F3864"]
    )

    extent = [x0, x0 + W, y0, y0 + H]
    ax.imshow(
        rho,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="bilinear",
    )

    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            W,
            H,
            boxstyle="square,pad=0",
            linewidth=0.6,
            edgecolor="black",
            facecolor="none",
        )
    )

    wall_w = 0.3
    ax.add_patch(
        patches.Rectangle(
            (x0 - wall_w, y0 - 0.1),
            wall_w,
            H + 0.2,
            linewidth=0.4,
            edgecolor="#333333",
            facecolor="#DDDDDD",
            hatch="////",
        )
    )
    ax.plot(
        [x0 - wall_w, x0 - wall_w],
        [y0 - 0.1, y0 + H + 0.1],
        color="#333333",
        linewidth=0.7,
    )
    ax.text(
        x0 - wall_w / 2,
        y0 + H + 0.30,
        "Clamped",
        ha="center",
        va="bottom",
        fontsize=4,
        fontweight="bold",
        color="#333333",
    )

    load_x = x0 + W
    load_y = y0 + H / 2
    arrow_len = 0.8
    ax.annotate(
        "",
        xy=(load_x + 0.15, load_y - arrow_len),
        xytext=(load_x + 0.15, load_y + 0.05),
        arrowprops={
            "arrowstyle": "->",
            "color": OBJECTIVE_COLOR,
            "lw": 1.0,
            "mutation_scale": 6,
        },
    )
    ax.text(
        load_x + 0.45,
        load_y - arrow_len / 2 + 0.05,
        "      $\\mathbf{F}$\n(tip load)",
        ha="left",
        va="center",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color=OBJECTIVE_COLOR,
    )

    ax.text(
        x0 + W / 2,
        y0 + H + 0.45,
        r"Control: element densities $\rho_e$",
        ha="center",
        va="bottom",
        fontsize=CTRL_FONTSIZE,
        fontweight="bold",
        color=CONTROL_COLOR,
    )
    fig.text(
        0.5,
        OFFSET_OBJECTIVE,
        r"Objective: $\min_\rho\; \mathbf{f}^\top \mathbf{u}(\rho)\quad"
        r"\mathrm{s.t.}\quad \sum_e\, v_e \rho_e \leq V_f$",
        ha="center",
        va="bottom",
        fontsize=OBJ_FONTSIZE,
        color=PHYS_COLOR,
        bbox=OBJ_BOX_KW,
    )

    # ax.set_title(
    #    "Domain 3: Cantilever Beam — Compliance Minimization",
    #    fontsize=4.5,
    #    fontweight="bold",
    #    pad=4,
    # )
    ax.set_xlim(x0 - 0.8, x0 + W + 1.6)
    ax.set_ylim(y0 - 0.9, y0 + H + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.18)
    fig.savefig(
        out_dir / "domain3_structures.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(
        out_dir / "domain3_structures.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain3_structures.png'}")
    return fig


def _make_domain4(out_dir: Path) -> None:
    """Domain 4: Steady-State Heat Conduction — Conductivity Inversion."""
    N = 200
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    def gaussian(X, Y, cx, cy, sx, sy, amp):
        return amp * np.exp(
            -((X - cx) ** 2) / (2 * sx**2) - (Y - cy) ** 2 / (2 * sy**2)
        )

    k_field = (
        1.0
        + gaussian(X, Y, 0.3, 0.7, 0.10, 0.10, 2.5)
        + gaussian(X, Y, 0.7, 0.3, 0.12, 0.09, 2.0)
        + gaussian(X, Y, 0.5, 0.55, 0.08, 0.11, 1.5)
    )
    T_obs = (
        0.15 * (1 - X)
        + gaussian(X, Y, 0.55, 0.50, 0.25, 0.25, 0.6)
        + gaussian(X, Y, 0.35, 0.65, 0.18, 0.18, 0.3)
        + 0.08 * np.sin(np.pi * Y)
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=300
    )
    fig.subplots_adjust(wspace=0.7, top=0.92, bottom=0.10, left=0.04, right=0.96)

    # fig.suptitle(
    #    "Domain 4: Steady-State Heat Conduction — Conductivity Inversion",
    #    fontsize=14,
    #    fontweight="bold",
    #    y=0.97,
    # )

    ax = ax_l
    ax.set_title(
        r"Control: $k(x,y)$",
        fontsize=CTRL_FONTSIZE,
        fontweight="bold",
        color=CONTROL_COLOR,
        pad=4,
    )
    ax.imshow(
        k_field[::-1],
        extent=[0, 1, 0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=0.8,
        vmax=4.0,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor="k",
            facecolor="none",
        )
    )

    ax.plot([0, 0], [0, 1], color="0.45", linewidth=1.5, solid_capstyle="butt")
    ax.text(
        -0.05,
        0.5,
        r"$T = T_0$ (Dirichlet)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=3.5,
        color="0.45",
    )

    ax.plot([1, 1], [0, 1], color="darkorange", linewidth=1.5, solid_capstyle="butt")
    for yy in np.linspace(0.15, 0.85, 5):
        n_wave = 40
        t_arr = np.linspace(0, 0.14, n_wave + 6)
        fade_start = int(0.80 * n_wave)
        flat_start = int(0.95 * n_wave)
        envelope = np.ones(len(t_arr))
        envelope[fade_start:flat_start] = 0.5 * (
            1 + np.cos(np.linspace(0, np.pi, flat_start - fade_start))
        )
        envelope[flat_start:] = 0
        wave = yy + 0.030 * envelope * np.sin(2 * np.pi * t_arr / 0.04)
        ax.plot(1 + t_arr, wave, color="darkorange", linewidth=0.9)
        ax.annotate(
            "",
            xy=(1.17, yy),
            xytext=(1.12, yy),
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-|>,head_width=0.14,head_length=0.14",
                "color": "darkorange",
                "lw": 0.75,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    # ax.text(
    #    1.16,
    #    0.5,
    #    "Convective\nBC",
    #    ha="left",
    #    va="center",
    #    fontsize=4,
    #    color="darkorange",
    #    fontweight="bold",
    # )

    ax.text(
        0.47,
        0.82,
        r"$\nabla \cdot (k \nabla T) + q = 0$",
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
        color="white",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.4)],
        transform=ax.transAxes,
    )

    ax.annotate(
        "",
        xy=(1.70, 0.62),
        xytext=(1.10, 0.62),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>,head_width=0.14,head_length=0.18",
            "color": "black",
            "lw": 1,
        },
    )
    ax.text(
        1.40,
        0.67,
        "Steady-state\nsolve",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.5,
        fontstyle="italic",
    )

    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(-0.12, 1.14)
    ax.set_box_aspect(1)
    ax.set_axis_off()

    ax = ax_r
    ax.set_title(
        "Observed\ntemperature field",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color=PHYS_COLOR,
        pad=-10,
    )
    ax.imshow(
        T_obs[::-1],
        extent=[0, 1, 0, 1],
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=1.0,
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor="k",
            facecolor="none",
        )
    )

    ax.text(
        0.50,
        0.75,
        r"$T_{\mathrm{obs}} = T(x,y;\, k^*)$",
        ha="center",
        va="bottom",
        fontsize=4.7,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(-0.60, 0.38),
        xytext=(0.00, 0.38),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>,head_width=0.14,head_length=0.18",
            "color": OBJECTIVE_COLOR,
            "lw": 1,
        },
    )
    ax.text(
        -0.30,
        0.32,
        r"Invert: find $k(x,y)$" + "\n" + r"that produces $T_{\mathrm{obs}}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.2,
        color=OBJECTIVE_COLOR,
        fontstyle="italic",
    )
    fig.text(
        0.5,
        OFFSET_OBJECTIVE,
        r"Objective:  $\min_k \; \| T(k) - T_{\mathrm{obs}} \|^2$",
        ha="center",
        va="bottom",
        fontsize=OBJ_FONTSIZE,
        color=PHYS_COLOR,
        bbox=OBJ_BOX_KW,
    )

    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(-0.12, 1.14)
    ax.set_box_aspect(1)
    ax.set_axis_off()

    fig.savefig(out_dir / "domain4_heat.png", dpi=300, facecolor="white")
    fig.savefig(out_dir / "domain4_heat.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_dir / 'domain4_heat.png'}")
    return fig


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _make_domain1(out_dir)
        _make_domain2a_ic_recovery(out_dir)
        _make_domain3(out_dir)
        _make_domain4(out_dir)
