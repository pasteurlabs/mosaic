"""Generate Figure: Gradient coverage heatmap from actual benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH

RESULTS = Path(__file__).parent.parent.parent / "results"

OK = 2
NO_GRAD = 1
ERROR = 0
NA = -1

OVERRIDES = {
    ("phiflow", "ns-3d-grid", "input/v0"): OK,
}


def _load_domain(subdir: str) -> dict:
    path = RESULTS / subdir / "gradient/differentiability_table/result.json"
    with open(path) as f:
        return json.load(f)["by_solver"]


def _status(field_data: dict) -> int:
    s = field_data.get("status", "")
    if s == "ok":
        return OK
    if s == "not_differentiable":
        return NO_GRAD
    return ERROR


def _lookup(domain_data: dict, solver_key, domain_subdir: str, *field_keys) -> int:
    if solver_key is None or solver_key not in domain_data:
        return NA
    solver_fields = domain_data[solver_key]
    for k in field_keys:
        if k in solver_fields:
            override_key = (solver_key, domain_subdir, k)
            if override_key in OVERRIDES:
                return OVERRIDES[override_key]
            return _status(solver_fields[k])
    return NA


def generate(out_dir: Path) -> None:
    ns2d = _load_domain("ns-grid")
    ns3d = _load_domain("ns-3d-grid")
    struct = _load_domain("structural-mesh")
    therm = _load_domain("thermal-mesh")

    columns = [
        ("2D NS", "ns-grid", ns2d, ("input/v0",), "I.C."),
        ("2D NS", "ns-grid", ns2d, ("input/viscosity",), "visc."),
        ("2D NS", "ns-grid", ns2d, ("input/dt",), "dt"),
        ("3D NS", "ns-3d-grid", ns3d, ("input/v0",), "I.C."),
        ("3D NS", "ns-3d-grid", ns3d, ("input/viscosity",), "visc."),
        ("3D NS", "ns-3d-grid", ns3d, ("input/dt",), "dt"),
        ("Structural", "structural-mesh", struct, ("input/rho",), "density"),
        (
            "Structural",
            "structural-mesh",
            struct,
            ("input/E_max", "input/E"),
            "Young's\nmod.",
        ),
        ("Heat", "thermal-mesh", therm, ("input/rho",), "k-field"),
        ("Heat", "thermal-mesh", therm, ("input/source",), "source"),
        ("Heat", "thermal-mesh", therm, ("input/k_max",), "k_max"),
    ]

    solvers = [
        ("JAX-CFD", "jax_cfd", None, None, None),
        ("PhiFlow", "phiflow", "phiflow", None, None),
        ("INS.jl", "ins_jl", "ins_jl", None, None),
        ("XLB", "xlb", "xlb", None, None),
        ("PICT", "pict", "pict", None, None),
        ("Warp-NS", "warp_ns", "warp_ns", None, None),
        ("Exponax", None, "exponax", None, None),
        ("FEniCS", "fenics_ns", None, "fenics_structural", "fenics_heat"),
        ("Firedrake", None, None, "firedrake_structural", "firedrake_heat"),
        ("JAX-FEM", None, None, "jax_fem", "jax_fem"),
        ("TopOpt.jl", None, None, "topopt_jl", None),
        ("TorchFEM", None, None, None, "torch_fem_thermal"),
        ("deal.II", None, None, "dealii_structural", "dealii_heat"),
    ]

    n_solvers = len(solvers)
    n_cols = len(columns)
    data = np.zeros((n_solvers, n_cols), dtype=int)

    for i, solver_tuple in enumerate(solvers):
        name, ns2d_key, ns3d_key, struct_key, therm_key = solver_tuple
        solver_keys = {
            "2D NS": (ns2d, ns2d_key),
            "3D NS": (ns3d, ns3d_key),
            "Structural": (struct, struct_key),
            "Heat": (therm, therm_key),
        }
        for j, (domain_label, subdir, _, field_keys, _) in enumerate(columns):
            domain_data, solver_key = solver_keys[domain_label]
            data[i, j] = _lookup(domain_data, solver_key, subdir, *field_keys)

    # ── Layout constants (all in coordinate units) ───────────────────────────
    cell_w, cell_h = 1.0, 0.65
    label_col_w = 3.8  # left margin for solver names
    right_pad = 0.5

    colors = {OK: "#2ecc71", NO_GRAD: "#f39c12", ERROR: "#e74c3c", NA: "#ecf0f1"}
    symbols = {OK: "✓", NO_GRAD: "○", ERROR: "✗", NA: "—"}
    sym_colors = {OK: "#1a7a42", NO_GRAD: "#7d5006", ERROR: "#922b21", NA: "#bdc3c7"}

    # Wider figure so column labels have breathing room
    FIG_W = TEXTWIDTH * 1.2  # ~6.6 inches; include in LaTeX at width=1.2\linewidth
    coord_w = n_cols * cell_w + label_col_w + right_pad
    scale = FIG_W / coord_w

    # Vertical layout:  cells | field-label gap | domain-title gap | top pad
    field_label_h = 0.55  # space for rotated labels
    domain_title_h = 0.45
    legend_h = 1.0
    coord_h = n_solvers * cell_h + field_label_h + domain_title_h + 0.2 + legend_h + 0.3

    fig, ax = plt.subplots(figsize=(FIG_W, coord_h * scale), dpi=200)

    # ── Cell grid ────────────────────────────────────────────────────────────
    x0_grid = 0.0  # left edge of column grid (solver names drawn to the left)
    for i in range(n_solvers):
        row_y = (n_solvers - 1 - i) * cell_h
        for j in range(n_cols):
            val = int(data[i, j])
            rect = mpatches.FancyBboxPatch(
                (x0_grid + j * cell_w, row_y),
                cell_w - 0.06,
                cell_h - 0.06,
                boxstyle="round,pad=0.02",
                facecolor=colors[val],
                edgecolor="white",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                x0_grid + j * cell_w + (cell_w - 0.06) / 2,
                row_y + (cell_h - 0.06) / 2,
                symbols[val],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=sym_colors[val],
            )

    # ── Solver row labels (left of grid) ─────────────────────────────────────
    for i, (name, *_) in enumerate(solvers):
        row_y = (n_solvers - 1 - i) * cell_h + (cell_h - 0.06) / 2
        ax.text(
            x0_grid - 0.15,
            row_y,
            name,
            ha="right",
            va="center",
            fontsize=8,
            fontfamily="sans-serif",
        )

    # ── Column field labels (rotated 45°, above grid) ────────────────────────
    label_base_y = n_solvers * cell_h + 0.12
    for j, (_, _, _, _, field_label) in enumerate(columns):
        cx = x0_grid + j * cell_w + (cell_w - 0.06) / 2
        ax.text(
            cx,
            label_base_y,
            field_label,
            ha="left",
            va="bottom",
            fontsize=7,
            fontfamily="sans-serif",
            rotation=40,
            rotation_mode="anchor",
        )

    # ── Domain group headers (above field labels) ─────────────────────────────
    domain_spans: dict[str, list[int]] = {}
    for j, (domain, *_) in enumerate(columns):
        domain_spans.setdefault(domain, [j, j])
        domain_spans[domain][1] = j

    header_y = n_solvers * cell_h + field_label_h + domain_title_h - 0.1
    for domain, (j0, j1) in domain_spans.items():
        x_left = x0_grid + j0 * cell_w
        x_right = x0_grid + j1 * cell_w + cell_w - 0.06
        cx = (x_left + x_right) / 2
        ax.text(
            cx,
            header_y,
            domain,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            fontfamily="sans-serif",
        )
        ax.plot(
            [x_left, x_right],
            [header_y - 0.08, header_y - 0.08],
            color="#555",
            linewidth=1.2,
            clip_on=False,
        )

    # ── Legend (two rows of two items each) ──────────────────────────────────
    legend_y = -legend_h + 0.15
    legend_items = [
        (OK, "Correct gradient"),
        (NO_GRAD, "Not differentiable"),
        (ERROR, "Incorrect / error"),
        (NA, "N/A (domain not supported)"),
    ]
    items_per_row = 2
    lx_step = (n_cols * cell_w) / items_per_row
    for k, (val, label) in enumerate(legend_items):
        row, col = divmod(k, items_per_row)
        lx = col * lx_step
        ly = legend_y - row * 0.45
        rect = mpatches.FancyBboxPatch(
            (lx, ly),
            0.35,
            0.32,
            boxstyle="round,pad=0.02",
            facecolor=colors[val],
            edgecolor="#ccc",
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            lx + 0.52,
            ly + 0.16,
            label,
            ha="left",
            va="center",
            fontsize=7.5,
            fontfamily="sans-serif",
            color="#333",
        )

    ax.set_xlim(-label_col_w, n_cols * cell_w + right_pad)
    ax.set_ylim(legend_y - 0.2, header_y + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    out = out_dir / "coverage_heatmap_results.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
