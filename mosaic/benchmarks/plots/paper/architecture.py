"""Generate Figure 1: Mosaic architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.8, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- colors ----------
    C_TESS = "#4A90D9"       # Tesseract blue
    C_TESS_LIGHT = "#D6E6F5"
    C_API = "#2ECC71"         # API green
    C_API_LIGHT = "#D5F5E3"
    C_EVAL = "#E67E22"        # Eval orange
    C_EVAL_LIGHT = "#FDEBD0"
    C_USER = "#8E44AD"        # User purple
    C_USER_LIGHT = "#E8DAEF"
    C_SOLVER = "#95A5A6"      # Solver gray
    C_SOLVER_LIGHT = "#EAEDED"
    C_ARROW = "#2C3E50"

    ROUND = "round,pad=0.15"
    FONT = dict(fontfamily="sans-serif")

    def box(ax, x, y, w, h, text, fc, ec, fontsize=8, bold=False, alpha=1.0):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=ROUND,
            facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=alpha
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=ec if ec != "white" else "#333",
            **FONT
        )
        return rect

    def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.8):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                connectionstyle="arc3,rad=0"
            )
        )

    # ==========================================================================
    # Column 1: Solver backends (left)
    # ==========================================================================
    col1_x = 0.0
    solvers = [
        ("OpenFOAM, deal.II", "Reference"),
        ("Warp-NS", "CUDA kernel AD"),
        ("FEniCS, Firedrake", "FE + adjoint"),
        ("XLB, PICT", "LBM / FV"),
        ("JAX-CFD, PhiFlow,\nExponax", "JAX / PyTorch"),
    ]

    solver_h = 0.55
    solver_w = 1.8
    solver_gap = 0.12
    total_h = len(solvers) * solver_h + (len(solvers) - 1) * solver_gap
    y_start = (3.2 - total_h) / 2 + 0.1

    for i, (name, label) in enumerate(solvers):
        y = y_start + i * (solver_h + solver_gap)
        box(ax, col1_x, y, solver_w, solver_h, name,
            fc=C_SOLVER_LIGHT, ec="#666", fontsize=6.5)

    ax.text(
        col1_x + solver_w / 2, y_start + total_h + 0.35,
        "Solver backends",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#555", **FONT
    )

    # ==========================================================================
    # Column 2: Tesseract wrapping
    # ==========================================================================
    col2_x = 3.6
    tess_w = 2.0
    tess_h = total_h + 0.3
    tess_y = y_start - 0.15

    outer = mpatches.FancyBboxPatch(
        (col2_x, tess_y), tess_w, tess_h, boxstyle="round,pad=0.2",
        facecolor=C_TESS_LIGHT, edgecolor=C_TESS, linewidth=2.0,
        linestyle="-"
    )
    ax.add_patch(outer)

    ax.text(
        col2_x + tess_w / 2, tess_y + tess_h - 0.28,
        "Tesseract",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=C_TESS, **FONT
    )

    inner_items = [
        "vjp(inputs, grad)\n→ input_grads",
        "forward(inputs)\n→ outputs",
        "Container\n(Docker / Apptainer)",
    ]
    inner_h = 0.52
    inner_w = 1.6
    inner_gap = 0.18
    inner_x = col2_x + (tess_w - inner_w) / 2
    inner_y_start = tess_y + 0.2

    for i, txt in enumerate(inner_items):
        y = inner_y_start + i * (inner_h + inner_gap)
        box(ax, inner_x, y, inner_w, inner_h, txt,
            fc="white", ec=C_TESS, fontsize=6.5)

    ax.text(
        col2_x + tess_w / 2, tess_y + tess_h + 0.2,
        "Uniform interface",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_TESS, **FONT
    )

    # ==========================================================================
    # Arrows: solvers → tesseract
    # ==========================================================================
    for i in range(len(solvers)):
        y = y_start + i * (solver_h + solver_gap) + solver_h / 2
        arrow(ax, col1_x + solver_w + 0.6, y, col2_x - 0.1, y)

    ax.text(
        (col1_x + solver_w + 0.6 + col2_x - 0.1) / 2, y_start - 0.45,
        "wrap",
        ha="center", va="center", fontsize=7, fontstyle="italic",
        color=C_ARROW, **FONT
    )

    # ==========================================================================
    # Column 3: Evaluation protocol
    # ==========================================================================
    col3_x = 6.8
    eval_w = 2.4
    eval_h = total_h + 0.3
    eval_y = tess_y

    eval_outer = mpatches.FancyBboxPatch(
        (col3_x, eval_y), eval_w, eval_h, boxstyle="round,pad=0.2",
        facecolor=C_EVAL_LIGHT, edgecolor=C_EVAL, linewidth=2.0
    )
    ax.add_patch(eval_outer)

    ax.text(
        col3_x + eval_w / 2, eval_y + eval_h - 0.28,
        "Evaluation suite",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=C_EVAL, **FONT
    )

    eval_items = [
        "Gradient coverage",
        "Gradient accuracy (vs. FD)",
        "Gradient overhead",
        "Optimization convergence",
        "Forward accuracy",
        "Scaling behavior",
    ]
    item_h = 0.32
    item_w = 2.0
    item_x = col3_x + (eval_w - item_w) / 2
    item_y_start = eval_y + 0.2

    for i, txt in enumerate(eval_items):
        y = item_y_start + i * (item_h + 0.06)
        box(ax, item_x, y, item_w, item_h, txt,
            fc="white", ec=C_EVAL, fontsize=6.5)

    mid_y = tess_y + tess_h / 2
    arrow(ax, col2_x + tess_w + 0.1, mid_y, col3_x - 0.1, mid_y)

    ax.text(
        (col2_x + tess_w + 0.1 + col3_x - 0.1) / 2, mid_y + 0.2,
        "auto",
        ha="center", va="center", fontsize=7, fontstyle="italic",
        color=C_ARROW, **FONT
    )

    # ==========================================================================
    # Column 4: Outputs / user-facing
    # ==========================================================================
    col4_x = 10.3
    out_w = 2.0
    out_items = [
        ("Coverage\nheatmap", C_USER_LIGHT, C_USER),
        ("Scaling\nplots", C_USER_LIGHT, C_USER),
        ("Convergence\ncurves", C_USER_LIGHT, C_USER),
        ("Solver\nselection", C_API_LIGHT, C_API),
    ]
    out_h = 0.6
    out_gap = 0.12
    out_total = len(out_items) * out_h + (len(out_items) - 1) * out_gap
    out_y_start = eval_y + (eval_h - out_total) / 2

    for i, (txt, fc, ec) in enumerate(out_items):
        y = out_y_start + i * (out_h + out_gap)
        box(ax, col4_x, y, out_w, out_h, txt, fc=fc, ec=ec, fontsize=7)

    ax.text(
        col4_x + out_w / 2, eval_y + eval_h + 0.2,
        "Outputs",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_USER, **FONT
    )

    arrow(ax, col3_x + eval_w + 0.1, mid_y, col4_x - 0.1, mid_y)

    # ==========================================================================
    # Bottom annotation
    # ==========================================================================
    ax.text(
        6.0, -0.55,
        "pip install mosaic-benchmark    ·    Add a solver: write a Tesseract wrapper → full eval runs automatically",
        ha="center", va="center", fontsize=7, color="#666",
        fontstyle="italic", **FONT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8",
                  edgecolor="#CCC", alpha=0.8)
    )

    out_path = out_dir / "architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")
