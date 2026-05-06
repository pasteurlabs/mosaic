"""Shared visual style for all Mosaic paper figures.

Import RCPARAMS, SOLVER_STYLES, solver ordering lists, and helper
functions from here rather than duplicating them across plot scripts.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Base rcParams — 8 pt sans-serif, NeurIPS scale (TEXTWIDTH = 5.5")
# ---------------------------------------------------------------------------

RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}

# ---------------------------------------------------------------------------
# Solver styles: solver_key → (display_label, color, linestyle, marker)
# ---------------------------------------------------------------------------

SOLVER_STYLES: dict[str, tuple] = {
    # ── Fluid / NS ──────────────────────────────────────────────────────────
    # Paul Tol Vibrant (7) for main differentiable solvers — distinguishable
    # under all forms of colorblindness and in greyscale (linestyle+marker).
    # label              color       linestyle          marker
    "jax_cfd": ("JAX-CFD", "#0077BB", "-", "o"),  # Vibrant blue
    "phiflow": ("PhiFlow", "#CC3311", "--", "s"),  # Vibrant red
    "ins_jl": ("INS.jl", "#33BBEE", "-.", "^"),  # Vibrant cyan
    "pict": ("PICT", "#EE3377", ":", "D"),  # Vibrant magenta
    "xlb": ("XLB", "#009988", (0, (4, 1)), "v"),  # Vibrant teal
    "warp_ns": ("Warp-NS", "#EE7733", (0, (1, 1)), "P"),  # Vibrant orange
    "exponax": ("Exponax", "#CCBB44", (0, (5, 1)), "<"),  # Muted yellow
    # Reference / excluded solvers — muted tones
    "openfoam": ("OpenFOAM", "#DDCC77", "--", "h"),  # Muted sand
    # ── FEM / Structural ────────────────────────────────────────────────────
    "jax_fem": ("JAX-FEM", "#0077BB", "-", "o"),  # blue (JAX family)
    "topopt_jl": ("TopOpt.jl", "#009988", "--", "s"),  # Vibrant teal
    "dealii_structural": ("deal.II", "#DDCC77", "-.", "^"),  # Muted sand
    "fenics_structural": ("FEniCS", "#CC3311", ":", "D"),  # Vibrant red
    "firedrake_structural": (
        "Firedrake",
        "#EE7733",
        (0, (3, 1)),
        "v",
    ),  # Vibrant orange
    # ── FEM / Thermal ───────────────────────────────────────────────────────
    "dealii_heat": ("deal.II", "#DDCC77", "-.", "^"),  # Muted sand
    "fenics_heat": ("FEniCS", "#CC3311", ":", "D"),  # Vibrant red
    "firedrake_heat": ("Firedrake", "#EE7733", (0, (3, 1)), "v"),  # Vibrant orange
    "torch_fem_thermal": (
        "TorchFEM",
        "#009988",
        (0, (5, 1, 1, 1)),
        "<",
    ),  # Vibrant teal
}

# ---------------------------------------------------------------------------
# Canonical solver ordering for legend construction
# ---------------------------------------------------------------------------

NS_ORDER: list[str] = [
    "jax_cfd",
    "phiflow",
    "ins_jl",
    "xlb",
    "pict",
    "warp_ns",
    "exponax",
    "openfoam",
]

FEM_ORDER: list[str] = [
    "jax_fem",
    "topopt_jl",
    "dealii_structural",
    "fenics_structural",
    "firedrake_structural",
    "dealii_heat",
    "fenics_heat",
    "firedrake_heat",
    "torch_fem_thermal",
]

STRUCTURAL_ORDER: list[str] = [
    "jax_fem",
    "topopt_jl",
    "dealii_structural",
    "fenics_structural",
    "firedrake_structural",
]

THERMAL_ORDER: list[str] = [
    "firedrake_heat",
    "jax_fem",
    "fenics_heat",
    "dealii_heat",
    "torch_fem_thermal",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def solver_props(name: str) -> tuple:
    """Return (label, color, linestyle, marker) for *name*, with fallback."""
    return SOLVER_STYLES.get(name, (name, "#888888", "-", "o"))


def make_handle(solver: str) -> mlines.Line2D:
    """Return a legend Line2D proxy for *solver*."""
    label, color, ls, mk = SOLVER_STYLES[solver]
    return mlines.Line2D(
        [],
        [],
        color=color,
        linestyle=ls,
        marker=mk,
        markersize=5,
        markeredgewidth=0,
        linewidth=1.6,
        label=label,
    )


def dedup_handles(handles: list) -> list:
    """Remove duplicate legend handles (same label keeps first occurrence)."""
    seen: set[str] = set()
    out = []
    for h in handles:
        lbl = h.get_label()
        if lbl not in seen:
            out.append(h)
            seen.add(lbl)
    return out


def rc_context():
    """Return ``plt.rc_context(RCPARAMS)`` for use as a context manager."""
    return plt.rc_context(RCPARAMS)
