"""Figure: Drag optimisation — inlet profiles + velocity field + drag reduction.

Two-row layout per Re:
  Row 1: inlet profile lines (left) + x-velocity final field (right)
  Row 2: drag / drag_0 vs iteration

Outputs: drag_opt_overview_re20.{pdf,png}, drag_opt_overview_re100.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np

from benchmarks.plots.paper import TEXTWIDTH
from benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

RESULTS      = Path(__file__).parent.parent.parent / "results"
SOLVER_ORDER = ["xlb", "phiflow", "pict", "jax_cfd", "su2"]


def _draw_domain_illus(ax: plt.Axes) -> None:
    """Draw a compact square cylinder-flow domain illustration onto ax."""
    W = H = 2.0
    ax.set_xlim(-0.75, 3.1)
    ax.set_ylim(-0.45, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")

    fkw = dict(fontfamily="sans-serif")
    FS = 7.5   # base font size for labels

    # Fluid domain — orange background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), W, H, boxstyle="square,pad=0",
        facecolor="#F4A261", edgecolor="black", linewidth=0.8, zorder=0,
    ))

    # Walls (no-slip)
    wt = 0.13
    for y0, label_y, va in [(-wt, -wt - 0.07, "top"), (H, H + wt + 0.07, "bottom")]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y0), W, wt, boxstyle="square,pad=0",
            facecolor="#888888", edgecolor="#555555", linewidth=0.5, zorder=2,
        ))
        ax.text(W / 2, label_y, "no-slip  ($u = 0$)",
                ha="center", va=va, fontsize=FS - 0.5, color="black", **fkw)

    # Cylinder — white, no border
    cx, cy, cr = W / 2, H / 2, 0.26
    ax.add_patch(plt.Circle((cx, cy), cr, fc="white", ec="none", zorder=5))

    # Drag arrow — black
    ax.annotate("", xy=(cx - cr - 0.55, cy), xytext=(cx - cr, cy),
                arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.12",
                                color="#7B2D8B", lw=2.0), zorder=6)
    ax.text(cx - cr - 0.28, cy - 0.18, r"$F_D$",
            ha="center", va="top", fontsize=FS + 2.0,
            fontweight="bold", color="#7B2D8B", **fkw)

    # Inflow arrows
    y_mid = H / 2
    for y in np.linspace(0.15, H - 0.15, 6):
        frac = max(1.0 - ((y - y_mid) / (H / 2)) ** 2, 0.08)
        L = 0.65 * frac
        ax.annotate("", xy=(L, y), xytext=(0.0, y),
                    arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.09",
                                    color="black", lw=1.2), zorder=4)

    # Inflow BC label
    ax.text(-0.55, H / 2, "Inflow\n$u(y)$",
            ha="center", va="center", fontsize=FS, fontweight="bold",
            color="black", **fkw)

    # Outflow arrows
    for y in np.linspace(0.35, H - 0.35, 4):
        ax.annotate("", xy=(W + 0.38, y), xytext=(W, y),
                    arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.09",
                                    color="black", lw=1.0), zorder=4)
    ax.text(W + 0.60, H / 2, "Outflow",
            ha="center", va="center", fontsize=FS, color="black", **fkw)

    # Streamlines downstream of cylinder
    xs = np.linspace(cx + cr + 0.15, W - 0.05, 100)
    for off, amp, ph in [(0, 0.14, 0), (0.28, 0.10, np.pi), (-0.28, 0.10, 0)]:
        t = (xs - xs[0]) / (xs[-1] - xs[0])
        ys = cy + off + amp * t * np.sin(2 * np.pi * 2 * t + ph)
        ys = np.clip(ys, 0.05, H - 0.05)
        ax.plot(xs, ys, color="black", lw=0.5, alpha=0.5, zorder=1)

    # Control label above
    ax.text(W / 2, H + wt + 0.55, "Control: inflow $u(y)$",
            ha="center", va="bottom", fontsize=FS, fontweight="bold",
            color="black", **fkw)
    ax.annotate("", xy=(0.15, H + 0.02), xytext=(W / 2 - 0.35, H + wt + 0.55),
                arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.10",
                                connectionstyle="arc3,rad=0.3",
                                color="black", lw=1.2), zorder=4)



def _draw_tesseract_interface(ax: plt.Axes) -> None:
    """Draw the Tesseract interface diagram in the centre column."""
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fkw  = dict(fontfamily="monospace")
    fkwt = dict(fontfamily="sans-serif")

    # ── central box ──────────────────────────────────────────────────────────
    bx0, by0, bw, bh = 0.18, 0.25, 0.64, 0.50
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx0, by0), bw, bh,
        boxstyle="round,pad=0.03",
        facecolor="#f8f8f8", edgecolor="black", linewidth=1.4, zorder=1,
    ))

    # "Tesseract" title above box
    ax.text(0.5, by0 + bh + 0.06, "Tesseract",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="black", **fkwt)

    # Two method names inside box
    ax.text(0.5, by0 + bh * 0.70, "forward()",
            ha="center", va="center", fontsize=7.5,
            color="black", fontfamily="monospace")
    ax.text(0.5, by0 + bh * 0.30, "vjp()",
            ha="center", va="center", fontsize=7.5,
            color="black", fontfamily="monospace")

    # divider
    div_y = by0 + bh * 0.50
    ax.plot([bx0 + 0.05, bx0 + bw - 0.05], [div_y, div_y],
            color="#aaaaaa", lw=0.7, zorder=2)

    # ── left arrows: two inputs + dots ───────────────────────────────────────
    y_in1 = by0 + bh * 0.78
    y_in2 = by0 + bh * 0.55

    x_tip   = bx0 - 0.05          # tip stops just before box
    x_start = bx0 - 0.28          # doubled length
    for y, lbl in [(y_in1, r"$u(y)$"), (y_in2, r"$\nu$")]:
        ax.annotate("", xy=(x_tip, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.10",
                                    color="black", lw=1.6))
        ax.text((x_start + x_tip) / 2, y + 0.06, lbl,
                ha="center", va="bottom", fontsize=9, color="black", **fkwt)

    ax.text((x_start + x_tip) / 2, by0 + bh * 0.28, r"$\vdots$",
            ha="center", va="center", fontsize=10, color="black", **fkwt)

    # ── right arrow: output centred on box ───────────────────────────────────
    y_out       = by0 + bh * 0.50
    x_out_start = bx0 + bw + 0.05
    x_out_end   = bx0 + bw + 0.33          # doubled length

    ax.annotate("", xy=(x_out_end, y_out), xytext=(x_out_start, y_out),
                arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.10",
                                color="black", lw=1.6))
    ax.text((x_out_start + x_out_end) / 2, y_out + 0.06, r"$\mathbf{u}$",
            ha="center", va="bottom", fontsize=9, color="black", **fkwt)


_SOLVER_BOX_INFO = [
    # (key, display, color, discr., numerics)
    ("xlb",     "XLB",     "#2171B5", "JAX  ·  LBM",     "Streaming"),
    ("phiflow", "PhiFlow", "#CC3311", "JAX  ·  FD",      "Semi-Lagrangian"),
    ("pict",    "PICT",    "#43A047", "PyTorch  ·  FV",  "PISO, BDF1"),
]


def _draw_solver_boxes(ax: plt.Axes) -> None:
    """Draw three solver info boxes stacked vertically."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n      = len(_SOLVER_BOX_INFO)
    margin = 0.07                                   # top/bottom padding in axes
    gap    = 0.07                                   # gap between boxes
    bh     = (1.0 - 2 * margin - (n - 1) * gap) / n

    for i, (_, label, color, discr, numerics) in enumerate(_SOLVER_BOX_INFO):
        y0 = (1.0 - margin) - (i + 1) * bh - i * gap
        bg = color + "1A"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y0), 0.96, bh,
            boxstyle="round,pad=0.01",
            facecolor=bg, edgecolor=color, linewidth=1.2, zorder=1,
            transform=ax.transData,
        ))
        # solver name centred in upper half, details centred in lower half
        ax.text(0.5, y0 + bh * 0.65, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=color, transform=ax.transData)
        ax.text(0.5, y0 + bh * 0.28, f"{discr}  ·  {numerics}",
                ha="center", va="center", fontsize=6.0,
                color="#444444", transform=ax.transData)


def _plot_re(re_tag: str, out_dir: Path) -> None:
    base         = RESULTS / "ns-grid" / "optimization" / "drag_opt" / re_tag
    result_path  = base / "result.json"
    profiles_npz = base / "profiles.npz"
    fields_npz   = base / "flow_fields.npz"

    if not result_path.exists():
        print(f"[drag_opt_overview] missing {result_path}, skipping")
        return

    data     = json.loads(result_path.read_text())
    profiles = np.load(profiles_npz)
    fields   = np.load(fields_npz)

    N   = data["params"]["physics"]["N"]
    cx  = data["params"]["physics"]["obstacle"]["center"][0]
    cy  = data["params"]["physics"]["obstacle"]["center"][1]
    r   = data["params"]["physics"]["obstacle"]["radius"]
    y   = np.linspace(0, 1, N)

    ux_field_key = "flow_final_phiflow" if "flow_final_phiflow" in fields else "flow_initial"
    ux_init = fields[ux_field_key][:, :, 0, 0]
    vel_max = float(ux_init.max()) * 1.05
    vel_min = float(ux_init.min())

    # ── figure: outer 2-row × 3-col; right col spans both rows ─────────────
    plt.rcParams.update(RCPARAMS)
    fig = plt.figure(figsize=(TEXTWIDTH * 1.7, TEXTWIDTH * 0.75))

    gs_outer = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 0.8, 2.0],
        height_ratios=[1.3, 0.7],
        left=0.03, right=0.97, bottom=0.14, top=0.93,
        hspace=0.10, wspace=0.18,
    )
    # inner: 2 rows × 2 cols (profile | field); colorbar as inset on field
    # right col spans both outer rows so plots fill the same height as before
    gs_inner = gs_outer[:, 2].subgridspec(
        2, 2,
        width_ratios=[0.45, 1.0],
        height_ratios=[1.3, 0.7],
        hspace=0.60, wspace=0.03,
    )

    ax_illus  = fig.add_subplot(gs_outer[0, 0])
    ax_boxes  = fig.add_subplot(gs_outer[1, 0])
    ax_center = fig.add_subplot(gs_outer[:, 1])
    ax_prof  = fig.add_subplot(gs_inner[0, 0])
    ax_vel   = fig.add_subplot(gs_inner[0, 1], sharey=ax_prof)
    ax_drag  = fig.add_subplot(gs_inner[1, :])

    _draw_domain_illus(ax_illus)
    _draw_solver_boxes(ax_boxes)
    _draw_tesseract_interface(ax_center)

    # ── velocity field ───────────────────────────────────────────────────────
    im = ax_vel.imshow(
        ux_init.T,
        origin="lower", extent=[0, 1, 0, 1],
        cmap="plasma", vmin=vel_min, vmax=vel_max,
        aspect="equal", interpolation="bilinear",
    )
    cax = ax_vel.inset_axes([1.03, 0, 0.06, 1.0])
    cb_vel = fig.colorbar(im, cax=cax)
    cb_vel.set_label(r"$u_x$", fontsize=7, labelpad=2)
    cb_vel.ax.tick_params(labelsize=6, length=0)

    cyl = plt.Circle((cx, cy), r, color="white", zorder=5)
    ax_vel.add_patch(cyl)

    ax_vel.set_anchor("W")
    ax_vel.set_xlabel("$x$", fontsize=8)
    ax_vel.tick_params(labelsize=7)
    ax_vel.grid(False)
    ax_vel.yaxis.set_visible(False)
    ax_vel.spines["left"].set_visible(False)

    # ── inlet profiles ───────────────────────────────────────────────────────
    p_ref = float(profiles["initial"].max())
    p_max = max(
        float(profiles[f"final_{s}"].max())
        for s in SOLVER_ORDER if f"final_{s}" in profiles
    )
    ax_prof.axvline(p_ref, color="0.5", lw=0.7, ls="--")
    ax_prof.plot(profiles["initial"], y, color="#999999", lw=1.2, ls="--")

    present: set[str] = set()
    for solver in SOLVER_ORDER:
        key = f"final_{solver}"
        if key not in profiles:
            continue
        label, color, ls, mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
        ax_prof.plot(profiles[key], y, color=color, lw=1.5, linestyle=ls)
        present.add(solver)

    ax_prof.spines["right"].set_visible(False)
    ax_prof.set_xlim(0.1, p_max * 1.05)
    ax_prof.set_ylim(0, 1)
    ax_prof.set_xlabel(r"$u_x$", fontsize=8)
    ax_prof.set_ylabel("$y$", fontsize=8)
    ax_prof.tick_params(labelsize=7)
    ax_prof.xaxis.set_major_locator(mticker.MaxNLocator(3))

    field_label = "PhiFlow final" if ux_field_key == "flow_final_phiflow" else "initial"
    title_str = f"Inlet profiles  ·  $u_x$ {field_label}"
    ax_vel.set_title(title_str, fontweight="bold")
    ax_vel.title.set_position((0.29, 1.02))

    # ── drag reduction ───────────────────────────────────────────────────────
    step = 5
    for solver in SOLVER_ORDER:
        sdata = data["by_solver"].get(solver)
        if sdata is None:
            continue
        drags = sdata.get("drags", [])
        if not drags or drags[0] in (None, 0) or np.isnan(drags[0]):
            continue
        drag_0 = drags[0]
        indices = list(range(0, len(drags), step))
        if (len(drags) - 1) % step != 0:
            indices.append(len(drags) - 1)
        rel = [abs(drags[i]) / abs(drag_0) for i in indices]
        label, color, ls, mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
        ax_drag.plot(indices, rel, color=color, ls=ls, lw=1.6)
        present.add(solver)

    ax_drag.axhline(1.0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax_drag.set_title("Drag reduction", fontweight="bold")
    ax_drag.set_xlabel("Iteration", fontsize=8)
    ax_drag.set_ylabel(r"$|D| / |D_0|$", fontsize=8)
    ax_drag.tick_params(labelsize=7)
    ax_drag.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── legend ───────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color="#999999", ls="--", lw=1.2, label="Initial")
    ] + [
        mlines.Line2D([], [],
                      color=SOLVER_STYLES[s][1], ls=SOLVER_STYLES[s][2], lw=1.6,
                      label=SOLVER_STYLES[s][0])
        for s in SOLVER_ORDER if s in present
    ]
    ax_drag.legend(handles=handles,
                   loc="lower center", bbox_to_anchor=(0.5, 1.18),
                   ncol=len(handles), fontsize=7, framealpha=0.8, handlelength=2.0,
                   borderpad=0.4, labelspacing=0.25, columnspacing=1.0)

    # Align drag axes to span from ax_prof left edge to colorbar right edge
    fig.canvas.draw()
    p_prof = ax_prof.get_position()
    p_cax  = cax.get_position()
    p_drag = ax_drag.get_position()
    ax_drag.set_position([p_prof.x0, p_drag.y0, p_cax.x1 - p_prof.x0, p_drag.height])

    for ext in ("pdf", "png"):
        out = out_dir / f"drag_opt_overview_{re_tag}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _plot_re("re20",  out_dir)
        _plot_re("re100", out_dir)


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
