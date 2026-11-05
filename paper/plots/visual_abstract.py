"""Generate the Mosaic visual abstract (teaser / Figure 1).

Four columns, read left to right:
  1  Solvers — the backends in the release, grouped by domain family
  2  Interface — the Tesseract with its apply and vjp endpoints
  3  Tasks — 2D cylinder flow results, upstream part of the DFG channel:
       apply → flow field with the parabolic inflow
       vjp   → flow field with the gradient-optimized inflow
  4  Results — status per suite for 2D Navier–Stokes, from ``status.json``

Output: visual_abstract.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

from . import by_solver, is_ok, load_result, results_dir
from .style import NS_ORDER, solver_props

# Docs palette, see docs/figures/make_domain_figures.py.
INK = "#1d1e20"
MUTED = "#5f6368"
BORDER = "#d4d5d8"
SAND = "#f1f1ef"
# Results table: state -> (tile fill, glyph colour).
TILES = {
    "ok": ("#d5efe2", "#1f7a50"),
    "pt": ("#fdeec4", "#a06f00"),
    "no": ("#f8d3cf", "#b3261e"),
    "na": ("#f4f4f2", None),
}
INDIGO = "#2f4b8f"  # the control the optimizer changes
PLUM = "#6e3f8f"  # gradients

SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed",
    ["#2e1e3b", "#413d7b", "#37659e", "#348fa7", "#40b7ad", "#8bdab2", "#def5e5"],
)

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.monospace": ["Menlo", "DejaVu Sans Mono"],
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.5,
}

# DFG 2D-1 channel and cylinder (Schäfer & Turek 1996); the task panels show
# its upstream part, x < _X_SHOW.
_LX, _LY = 2.2, 0.41
_CYL_CENTER, _CYL_RADIUS = (0.2, 0.2), 0.05
_X_SHOW = 0.65
# Results-table columns: status.json suites and their headers.
_SUITES = ["forward", "cost", "gradient", "optimization"]
_SUITE_HEADERS = ["Fwd", "Cost", "Grad", "Opt."]

# Layout grid, in drawing units. All four columns share ROWS rows of height ROW,
# so text, table rows, panels and arrows line up by construction. Gutters hold
# only the connecting arrows, and PAD is the one inset used inside the grid.
MARGIN, GUTTER, ROW, PAD, ROWS = 8, 52, 40, 6, 10
BOTTOM = 8
COLUMNS = (300, 240, 152, 420)
W = 2 * MARGIN + sum(COLUMNS) + 3 * GUTTER
H = BOTTOM + ROWS * ROW + 58
HEADING_Y = BOTTOM + ROWS * ROW + 18


def _column_x(i: int) -> float:
    return MARGIN + sum(COLUMNS[:i]) + i * GUTTER


def _row(k: int) -> float:
    """Center height of grid row *k*, counted from 1 at the top."""
    return BOTTOM + (ROWS + 0.5 - k) * ROW


FLUID_SOLVERS = [
    "JAX-CFD",
    "PhiFlow",
    "XLB",
    "PICT",
    "Warp-NS",
    "Exponax",
    "INS.jl",
    "OpenFOAM",
]
FEM_SOLVERS = ["FEniCS", "Firedrake", "JAX-FEM", "torch-fem", "TopOpt.jl", "deal.II"]


def _load_flow_data() -> dict:
    """Flow fields and inflow profiles of the drag_opt solver with the largest reduction."""
    experiment = "optimization/drag_opt"
    runs = {
        s: m["None"] for s, m in by_solver(load_result("ns-grid", experiment)).items()
    }
    best = max(
        (s for s, m in runs.items() if is_ok(m)),
        key=lambda s: runs[s]["drag_reduction_pct"],
    )
    base = results_dir() / "ns-grid" / experiment
    flows = np.load(base / "flow_fields.npz")
    profs = np.load(base / "profiles.npz")
    nx_show = round(flows[f"flow_initial_{best}"].shape[0] * _X_SHOW / _LX)
    flow_initial = flows[f"flow_initial_{best}"][:nx_show, :, 0, :]
    flow_final = flows[f"flow_final_{best}"][:nx_show, :, 0, :]
    return {
        "flow_initial": flow_initial,
        "flow_final": flow_final,
        "profile_initial": profs["initial"],
        "profile_final": profs[f"final_{best}"],
        "vmax": float(
            max(
                np.hypot(f[..., 0], f[..., 1]).max() for f in (flow_initial, flow_final)
            )
        ),
    }


def _status_glyphs(problem: str) -> dict[str, list[str]]:
    """Per solver, one glyph key per suite of *problem* from ``status.json``.

    A suite is "ok" when every experiment passes its checks, "pt" when some do
    and "no" when none do. Experiments that were not run and categorical
    exclusions (the method cannot express the task) do not count; a suite with
    nothing left is "na".
    """
    status = json.loads((results_dir().parent / "status.json").read_text())
    rows = status["problems"][problem]["rows"]
    out = {}
    for solver in status["problems"][problem]["solvers"]:
        glyphs = []
        for suite in _SUITES:
            cells = [
                r["cells"][solver]
                for r in rows
                if r["suite"] == suite and solver in r["cells"]
            ]
            counted = [
                c["status"] == "ok"
                for c in cells
                if c["status"] != "not_run" and c["category"] != "categorical"
            ]
            if not counted:
                glyphs.append("na")
            elif all(counted):
                glyphs.append("ok")
            elif any(counted):
                glyphs.append("pt")
            else:
                glyphs.append("no")
        out[solver] = glyphs
    return out


def _heading(ax, x: float, text: str) -> None:
    ax.text(x, HEADING_Y, text, fontsize=15, fontweight="bold", va="baseline")


def _arrow(ax, p0, p1, color=INK, lw=1.4, style="-|>", connection="arc3") -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=13,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connection,
            zorder=6,
        )
    )


def _solver_group(ax, x: float, row: int, title: str, names: list[str]) -> None:
    ax.text(x, _row(row), title, fontsize=10.5, color=MUTED, va="center")
    for i, name in enumerate(names):
        ax.text(
            x + (i % 2) * COLUMNS[0] / 2,
            _row(row + 1 + i // 2),
            name,
            fontsize=12,
            family="monospace",
            va="center",
        )


def _draw_solvers(ax, x: float) -> None:
    _heading(ax, x, "Solvers")
    _solver_group(ax, x, 1, "Grid and spectral", FLUID_SOLVERS)
    _solver_group(ax, x, 7, "Finite element", FEM_SOLVERS)


def _draw_interface(ax, x: float, w: float) -> None:
    """Draw the Tesseract node on rows 5 to 7: a title bar and one row per endpoint."""
    _heading(ax, x, "Interface")
    y0 = _row(7) - ROW / 2
    radius = 8
    ax.add_patch(
        FancyBboxPatch(
            (x, y0),
            w,
            3 * ROW,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc="white",
            ec=INK,
            lw=1.2,
            zorder=2,
        )
    )
    y_bar = _row(5) - ROW / 2
    ax.add_patch(
        FancyBboxPatch(
            (x, y_bar),
            w,
            ROW,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=INK,
            ec=INK,
            lw=1.2,
            zorder=3,
        )
    )
    ax.add_patch(plt.Rectangle((x, y_bar), w, radius, fc=INK, ec=INK, lw=1.2, zorder=3))
    inset = x + 3 * PAD
    ax.text(
        inset,
        _row(5),
        "Tesseract",
        fontsize=13,
        fontweight="bold",
        color="white",
        va="center",
        zorder=5,
    )
    ax.plot([x, x + w], [_row(6) - ROW / 2] * 2, color=BORDER, lw=0.7, zorder=3)
    for row, label, color in (
        (6, "apply(x) → y", INK),
        (7, "vjp(x, v) → g", PLUM),
    ):
        ax.text(
            inset,
            _row(row),
            label,
            fontsize=12.5,
            family="monospace",
            color=color,
            va="center",
            zorder=5,
        )
        ax.add_patch(
            Circle((x + w, _row(row)), PAD, fc=color, ec="white", lw=1.2, zorder=7)
        )
    ax.text(
        x + w / 2,
        _row(8),
        "one container per solver",
        fontsize=10.5,
        color=MUTED,
        ha="center",
        va="center",
    )


def _draw_flow(ax, x, row, w, flow, profile, vmax) -> None:
    """Draw a flow panel filling rows *row* to *row* + 2, inset for the wall hatching."""
    y = _row(row + 2) - ROW / 2 + 2 * PAD
    h = 3 * ROW - 4 * PAD
    speed = np.hypot(flow[:, :, 0], flow[:, :, 1])
    ax.imshow(
        speed.T,
        origin="lower",
        extent=[x, x + w, y, y + h],
        cmap=SPEED_CMAP,
        vmin=0,
        vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )
    for wall, side in ((y + h, 1), (y, -1)):
        ax.add_patch(
            plt.Rectangle(
                (x, wall),
                w,
                side * 2 * PAD,
                fc="#dcdee3",
                ec=MUTED,
                lw=0,
                hatch="/////",
                zorder=3,
            )
        )
        ax.plot([x, x + w], [wall, wall], color=INK, lw=1.3, zorder=4)
    ax.add_patch(
        Ellipse(
            (x + w * _CYL_CENTER[0] / _X_SHOW, y + h * _CYL_CENTER[1] / _LY),
            2 * w * _CYL_RADIUS / _X_SHOW,
            2 * h * _CYL_RADIUS / _LY,
            fc=INK,
            ec="none",
            zorder=3,
        )
    )
    scale = 0.17 * w / max(float(profile.max()), 1e-9)
    ys = y + h * np.linspace(0.04, 0.96, len(profile))
    rows = np.linspace(0, len(profile) - 1, 9).astype(int)[1:-1]
    for yy, u in zip(ys[rows], profile[rows], strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (x + 3, yy),
                (x + 3 + max(u * scale, 11), yy),
                arrowstyle="simple,tail_width=1.5,head_width=5,head_length=5",
                mutation_scale=1,
                fc=INDIGO,
                ec="white",
                lw=0.5,
                shrinkA=0,
                shrinkB=0,
                zorder=6,
            )
        )


def _draw_tasks(ax, x, w, data) -> None:
    _heading(ax, x, "Tasks")
    for row, flow, profile, color, text in (
        (1, data["flow_initial"], data["profile_initial"], INK, "Forward solve"),
        (6, data["flow_final"], data["profile_final"], PLUM, "Optimized inflow"),
    ):
        ax.text(x, _row(row), text, fontsize=11, color=color, va="center")
        _draw_flow(ax, x, row + 1, w, flow, profile, data["vmax"])
    ax.text(
        x, _row(10), "plus H, S and F3 tasks", fontsize=10.5, color=MUTED, va="center"
    )


def _tile(ax, x: float, y: float, w: float, h: float, kind: str) -> None:
    """One table cell; fill and glyph both carry the state, so print stays readable."""
    fill, color = TILES[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=4",
            fc=fill,
            ec="none",
        )
    )
    line = {"color": color, "lw": 2.0, "solid_capstyle": "round"}
    if kind == "ok":
        ax.plot(
            [x - 8, x - 2, x + 9], [y, y - 6, y + 6], solid_joinstyle="round", **line
        )
    elif kind == "pt":
        ax.plot([x - 8, x + 8], [y, y], **line)
    elif kind == "no":
        for s in (1, -1):
            ax.plot([x - 6, x + 6], [y - s * 6, y + s * 6], **line)


def _draw_results(ax, x: float, w: float) -> None:
    _heading(ax, x, "Results")
    ax.text(
        x + w,
        HEADING_Y,
        "2D Navier–Stokes, checks per suite",
        fontsize=10.5,
        color=MUTED,
        ha="right",
        va="baseline",
    )
    glyphs = _status_glyphs("ns-grid")
    solvers = [s for s in NS_ORDER if s in glyphs]
    name_w = 128
    col_w = (w - name_w) / len(_SUITE_HEADERS)
    for j, header in enumerate(_SUITE_HEADERS):
        ax.text(
            x + name_w + (j + 0.5) * col_w,
            _row(1),
            header,
            fontsize=10.5,
            color=MUTED,
            ha="center",
            va="center",
        )
    for i, solver in enumerate(solvers):
        y = _row(i + 2)
        ax.text(
            x, y, solver_props(solver)[0], fontsize=12, family="monospace", va="center"
        )
        for j, kind in enumerate(glyphs[solver]):
            _tile(ax, x + name_w + (j + 0.5) * col_w, y, col_w - PAD, ROW - PAD, kind)
    labels = {"ok": "pass", "pt": "partly", "no": "fail", "na": "not applicable"}
    used = [k for k in labels if any(k in glyphs[s] for s in solvers)]
    for j, kind in enumerate(used):
        lx = x + j * w / len(used)
        _tile(ax, lx + 13, _row(10), 26, 20, kind)
        ax.text(
            lx + 26 + PAD,
            _row(10),
            labels[kind],
            fontsize=10.5,
            color=MUTED,
            va="center",
        )


def generate(out_dir: Path) -> None:
    data = _load_flow_data()

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(10.6, 10.6 * H / W), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])

        x_solvers, x_iface, x_tasks, x_results = (_column_x(i) for i in range(4))
        _draw_solvers(ax, x_solvers)
        _draw_interface(ax, x_iface, COLUMNS[1])
        _draw_tasks(ax, x_tasks, COLUMNS[2], data)
        _draw_results(ax, x_results, COLUMNS[3])

        # Gutter arrows sit on the middle row, PAD clear of both columns.
        y_mid = _row(6)
        for x_next in (x_iface, x_results):
            _arrow(ax, (x_next - GUTTER + PAD, y_mid), (x_next - PAD, y_mid))
        x_elbow = x_tasks - GUTTER / 2
        for row_from, row_to, color in ((6, 3, INK), (7, 8, PLUM)):
            ax.plot(
                [x_iface + COLUMNS[1], x_elbow, x_elbow],
                [_row(row_from), _row(row_from), _row(row_to)],
                color=color,
                lw=1.4,
                solid_joinstyle="round",
            )
            _arrow(
                ax, (x_elbow, _row(row_to)), (x_tasks - PAD, _row(row_to)), color=color
            )

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")

        for ext in ("pdf", "png"):
            out = out_dir / f"visual_abstract.{ext}"
            fig.savefig(out, dpi=300)
            print(f"Saved {out}")
        plt.close(fig)
