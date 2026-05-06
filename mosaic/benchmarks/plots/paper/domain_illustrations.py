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

from mosaic.benchmarks.plots.paper import TEXTWIDTH

# ── Shared palette & styling for Figure 2 ──────────────────────────────
CONTROL_COLOR = "#2471a3"  # blue — marks control variables
OBJECTIVE_COLOR = "#c0392b"  # red — marks objectives / loads / targets
PHYS_COLOR = "#333333"  # dark gray — governing eqns, physical labels
OBJ_BOX_KW = dict(
    boxstyle="round,pad=0.3",
    facecolor="#F2F2F2",
    edgecolor="#AAAAAA",
    lw=0.6,
)
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

    font_kw = dict(fontfamily="sans-serif")

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
        arrowprops=dict(
            arrowstyle="->,head_width=0.08,head_length=0.06",
            color=OBJECTIVE_COLOR,
            lw=0.8,
        ),
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
            arrowprops=dict(
                arrowstyle="->,head_width=0.08,head_length=0.06",
                color=CONTROL_COLOR,
                lw=0.6,
            ),
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
            arrowprops=dict(
                arrowstyle="->,head_width=0.08,head_length=0.06",
                color=CONTROL_COLOR,
                lw=0.6,
            ),
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

    for off, amp, ph in zip(offsets, amplitudes, phases):
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
            arrowprops=dict(
                arrowstyle="->,head_width=0.05,head_length=0.04",
                color=vortex_colors[i % 2],
                lw=0.6,
            ),
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
                *zip(corners[i], corners[j]),
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
    img = img[
        int(img.shape[0] * 0.30) : -int(img.shape[0] * 0.05),
        int(img.shape[1] * 0.20) : -int(img.shape[1] * 0.20),
        :,
    ]

    fig = plt.figure(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=300)
    ax2d = fig.add_axes((0.0, 0.10, 1.0, 0.76))
    ax2d.imshow(img)
    ax2d.axis("off")

    # control label — above left box
    fig.text(
        0.28,
        0.92,
        r"Control: $\mathbf{v}_0$",
        fontsize=CTRL_FONTSIZE,
        color=BLUE,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    # fig.text(
    #    0.18,
    #    0.60,
    #    "Fourier basis, ~2k DOFs",
    #    fontsize=4,
    #    color=BLUE,
    #    ha="center",
    #    va="top",
    #    alpha=0.8,
    # )

    # target label — above right box
    fig.text(
        0.72,
        0.92,
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


def _make_domain2a_cavity(out_dir: Path) -> None:
    """Domain 2A: 3D Lid-Driven Cavity — Lid Velocity Optimization."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-50)

    edges = [
        ([0, 1], [0, 0], [0, 0]),
        ([0, 1], [1, 1], [0, 0]),
        ([0, 0], [0, 1], [0, 0]),
        ([1, 1], [0, 1], [0, 0]),
        ([0, 1], [0, 0], [1, 1]),
        ([0, 1], [1, 1], [1, 1]),
        ([0, 0], [0, 1], [1, 1]),
        ([1, 1], [0, 1], [1, 1]),
        ([0, 0], [0, 0], [0, 1]),
        ([1, 1], [0, 0], [0, 1]),
        ([0, 0], [1, 1], [0, 1]),
        ([1, 1], [1, 1], [0, 1]),
    ]
    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, color="0.3", linewidth=1.0, zorder=1)

    top_face = Poly3DCollection(
        [[(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]],
        alpha=0.25,
        facecolor="#4C9BE8",
        edgecolor="#2070C0",
        linewidths=1.5,
        zorder=5,
    )
    ax.add_collection3d(top_face)

    np.random.seed(42)
    n_arrows = 6
    gx = np.linspace(0.15, 0.85, n_arrows)
    gy = np.linspace(0.15, 0.85, n_arrows)
    GX, GY = np.meshgrid(gx, gy)
    GX = GX.ravel()
    GY = GY.ravel()

    base_angle = np.pi * 0.15
    U = (
        0.06
        * (1 + 0.5 * np.sin(2 * np.pi * GY))
        * np.cos(base_angle + 0.4 * np.sin(np.pi * GX))
    )
    V = 0.06 * 0.4 * np.sin(2 * np.pi * GX) * np.cos(np.pi * GY)

    for i in range(len(GX)):
        ax.quiver(
            GX[i],
            GY[i],
            1.0,
            U[i],
            V[i],
            0,
            color="#1A5FAF",
            arrow_length_ratio=0.35,
            linewidth=1.0,
            zorder=6,
        )

    meas_plane = Poly3DCollection(
        [[(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5)]],
        alpha=0.15,
        facecolor="#F5A623",
        edgecolor="#D4850F",
        linewidths=1.0,
        linestyle="--",
        zorder=3,
    )
    ax.add_collection3d(meas_plane)

    def draw_curve(ax, pts, color="#888888", lw=1.0):
        pts = np.array(pts)
        t = np.linspace(0, 1, len(pts))
        t_fine = np.linspace(0, 1, 88)
        xs = np.interp(t_fine, t, pts[:, 0])
        ys = np.interp(t_fine, t, pts[:, 1])
        zs = np.interp(t_fine, t, pts[:, 2])
        ax.plot(xs, ys, zs, color=color, linewidth=lw, zorder=2)
        idx = 55
        ax.quiver(
            xs[idx],
            ys[idx],
            zs[idx],
            xs[idx + 1] - xs[idx],
            ys[idx + 1] - ys[idx],
            zs[idx + 1] - zs[idx],
            color=color,
            arrow_length_ratio=0.6,
            linewidth=lw,
            zorder=2,
        )

    draw_curve(
        ax,
        [
            (0.50, 0.50, 0.92),
            (0.80, 0.50, 0.75),
            (0.80, 0.50, 0.40),
            (0.50, 0.50, 0.15),
            (0.20, 0.50, 0.40),
            (0.20, 0.50, 0.75),
            (0.45, 0.50, 0.90),
        ],
        color="#666666",
        lw=1.2,
    )
    draw_curve(
        ax,
        [
            (0.50, 0.25, 0.88),
            (0.70, 0.25, 0.65),
            (0.65, 0.25, 0.30),
            (0.35, 0.25, 0.25),
            (0.25, 0.25, 0.55),
            (0.40, 0.25, 0.85),
        ],
        color="#999999",
        lw=0.9,
    )

    ax.text(
        0.50,
        -0.12,
        1.08,
        r"Control: lid velocity $\mathbf{v}_{\mathrm{lid}}(x,y)$",
        fontsize=10,
        color="#1A5FAF",
        ha="center",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        1.08,
        1.05,
        0.50,
        "Measurement plane\n$z = 0.5$",
        fontsize=9,
        color="#C06A00",
        ha="left",
        va="center",
        zorder=10,
    )
    ax.text(
        0.50,
        -0.18,
        -0.08,
        "No-slip walls (5 faces)",
        fontsize=9,
        color="0.35",
        ha="center",
        style="italic",
        zorder=10,
    )
    ax.text(
        -0.05,
        1.15,
        -0.05,
        "Re = 100–400",
        fontsize=10,
        color="0.25",
        ha="left",
        fontweight="bold",
        zorder=10,
    )

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_zlim(-0.05, 1.15)
    ax.set_xlabel("x", fontsize=10, labelpad=2)
    ax.set_ylabel("y", fontsize=10, labelpad=2)
    ax.set_zlabel("z", fontsize=10, labelpad=2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.85")
    ax.yaxis.pane.set_edgecolor("0.85")
    ax.zaxis.pane.set_edgecolor("0.85")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(
        "Task 2A: 3D Lid-Driven Cavity — Lid Velocity Optimization",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(
        out_dir / "domain2a_3d_cavity.png",
        dpi=150,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2a_3d_cavity.png'}")


def _make_domain2b_topology(out_dir: Path) -> None:
    """Domain 2B: 3D Topology Optimization for Flow Devices."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    _Lx, Ly, Lz = 3.0, 1.0, 1.0
    x_inlet = 0.0
    x_design_start = 0.6
    x_design_end = 2.4
    x_outlet = 3.0

    def box_faces(x0, x1, y0, y1, z0, z1):
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]

    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_inlet, x_design_start, 0, Ly, 0, Lz),
            alpha=0.15,
            facecolor="#a8d8ea",
            edgecolor="#4a90a4",
            linewidth=0.6,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_design_start, x_design_end, 0, Ly, 0, Lz),
            alpha=0.12,
            facecolor="#ffe0a0",
            edgecolor="#c49030",
            linewidth=0.6,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_design_end, x_outlet, 0, Ly, 0, Lz),
            alpha=0.15,
            facecolor="#a8d8ea",
            edgecolor="#4a90a4",
            linewidth=0.6,
        )
    )

    def draw_ellipsoid(ax, cx, cy, cz, rx, ry, rz, color="#555555", alpha=0.7):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = cx + rx * np.outer(np.cos(u), np.sin(v))
        y = cy + ry * np.outer(np.sin(u), np.sin(v))
        z = cz + rz * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True, linewidth=0)

    draw_ellipsoid(ax, 1.1, 0.35, 0.5, 0.15, 0.12, 0.30, color="#666666", alpha=0.75)
    draw_ellipsoid(ax, 1.5, 0.70, 0.35, 0.20, 0.10, 0.15, color="#555555", alpha=0.75)
    draw_ellipsoid(ax, 1.9, 0.30, 0.65, 0.12, 0.18, 0.20, color="#666666", alpha=0.75)
    draw_ellipsoid(ax, 1.6, 0.55, 0.75, 0.10, 0.15, 0.12, color="#777777", alpha=0.70)
    draw_ellipsoid(ax, 2.1, 0.65, 0.45, 0.14, 0.10, 0.18, color="#555555", alpha=0.75)

    for yy in [0.3, 0.7]:
        for zz in [0.3, 0.7]:
            ax.quiver(
                -0.15,
                yy,
                zz,
                0.55,
                0,
                0,
                arrow_length_ratio=0.25,
                color="#1565C0",
                linewidth=1.8,
                alpha=0.8,
            )
    for yy in [0.3, 0.7]:
        for zz in [0.3, 0.7]:
            ax.quiver(
                x_outlet - 0.1,
                yy,
                zz,
                0.45,
                0,
                0,
                arrow_length_ratio=0.25,
                color="#1565C0",
                linewidth=1.8,
                alpha=0.8,
            )

    arr_y = -0.15
    arr_z = -0.15
    ax.plot(
        [0.05, 2.95],
        [arr_y, arr_y],
        [arr_z, arr_z],
        color="#B71C1C",
        linewidth=2.0,
        linestyle="-",
    )
    ax.quiver(
        0.05,
        arr_y,
        arr_z,
        -0.001,
        0,
        0,
        arrow_length_ratio=100,
        color="#B71C1C",
        linewidth=2.0,
    )
    ax.quiver(
        2.95,
        arr_y,
        arr_z,
        0.001,
        0,
        0,
        arrow_length_ratio=100,
        color="#B71C1C",
        linewidth=2.0,
    )
    ax.text(
        1.5,
        arr_y - 0.08,
        arr_z - 0.12,
        r"$\Delta p = p_{\mathrm{in}} - p_{\mathrm{out}}$",
        fontsize=11,
        color="#B71C1C",
        ha="center",
        va="top",
        fontweight="bold",
    )

    ax.text(
        0.3,
        0.5,
        1.15,
        "Inlet",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#0D47A1",
        fontweight="bold",
    )
    ax.text(
        2.7,
        0.5,
        1.15,
        "Outlet",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#0D47A1",
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.5,
        1.20,
        "Design region",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#BF360C",
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.5,
        -0.45,
        r"Control: density field $\rho(x,y,z) \in [0,1]$",
        fontsize=9.5,
        ha="center",
        va="top",
        color="#6D4C00",
        fontstyle="italic",
    )
    ax.text(
        1.5,
        1.25,
        -0.30,
        "Brinkman penalization on regular grid",
        fontsize=9,
        ha="center",
        va="top",
        color="#37474F",
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#90A4AE", alpha=0.85
        ),
    )
    ax.text(
        1.5, -0.05, 1.02, "wall", fontsize=7, ha="center", color="#607D8B", alpha=0.7
    )
    ax.text(
        1.5, 1.05, 1.02, "wall", fontsize=7, ha="center", color="#607D8B", alpha=0.7
    )

    fig.suptitle(
        "Task 2B: 3D Topology Optimization for Flow Devices",
        fontsize=14,
        fontweight="bold",
        y=0.94,
    )

    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_zlim(-0.3, 1.3)
    ax.set_xlabel("x", fontsize=9, labelpad=2)
    ax.set_ylabel("y", fontsize=9, labelpad=2)
    ax.set_zlabel("z", fontsize=9, labelpad=2)
    ax.set_box_aspect([3, 1, 1])
    ax.view_init(elev=22, azim=-55)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(
        out_dir / "domain2b_3d_topology.png",
        dpi=150,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2b_3d_topology.png'}")


def _make_domain3(out_dir: Path) -> None:
    """Domain 3: Cantilever Beam — Compliance Minimization."""
    import matplotlib.patches as patches

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
        arrowprops=dict(
            arrowstyle="->", color=OBJECTIVE_COLOR, lw=1.0, mutation_scale=6
        ),
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
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.14,head_length=0.14",
                color="darkorange",
                lw=0.75,
                shrinkA=0,
                shrinkB=0,
            ),
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
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.14,head_length=0.18", color="black", lw=1
        ),
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
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.14,head_length=0.18",
            color=OBJECTIVE_COLOR,
            lw=1,
        ),
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
    _make_domain1(out_dir)
    _make_domain2a_ic_recovery(out_dir)
    _make_domain2a_cavity(out_dir)
    _make_domain2b_topology(out_dir)
    _make_domain3(out_dir)
    _make_domain4(out_dir)
