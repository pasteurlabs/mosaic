# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot-independent styling and field helpers shared by all benchmark charts.

Solver labels, colours, line styles and markers (:data:`SOLVER_STYLES`), the
canonical solver orderings used for legends, alias resolution, and the 2-D
field projections (vorticity, gradient magnitude) that the
chart builders in :mod:`~mosaic.benchmarks.problems.shared.plots.charts`
plot. Nothing here depends on a plotting library.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SOLVER_STYLES: dict[str, tuple] = {
    # ── Fluid / NS ──────────────────────────────────────────────────────────
    "jax_cfd": ("JAX-CFD", "#0077BB", "-", "o"),
    "phiflow": ("PhiFlow", "#CC3311", "--", "s"),
    "ins_jl": ("INS.jl", "#33BBEE", "-.", "^"),
    "pict": ("PICT", "#EE3377", ":", "D"),
    "xlb": ("XLB", "#CCBB44", (0, (4, 1)), "v"),
    "warp_ns": ("Warp-NS", "#EE7733", (0, (1, 1)), "P"),
    "exponax": ("Exponax", "#009988", (0, (5, 1)), "<"),
    "openfoam": ("OpenFOAM", "#DDCC77", "--", "h"),
    "waterlily": ("WaterLily", "#228833", "-.", "^"),
    "oceananigans": ("Oceananigans", "#AA3377", "-", "D"),
    # ── FEM / Structural ────────────────────────────────────────────────────
    "jax_fem": ("JAX-FEM", "#0077BB", "-", "o"),
    "topopt_jl": ("TopOpt.jl", "#009988", "--", "s"),
    "dealii_structural": ("deal.II", "#DDCC77", "-.", "^"),
    "fenics_structural": ("FEniCS", "#CC3311", ":", "D"),
    "firedrake_structural": ("Firedrake", "#EE7733", (0, (3, 1)), "v"),
    # ── FEM / Thermal ───────────────────────────────────────────────────────
    "dealii_heat": ("deal.II", "#DDCC77", "-.", "^"),
    "fenics_heat": ("FEniCS", "#CC3311", ":", "D"),
    "firedrake_heat": ("Firedrake", "#EE7733", (0, (3, 1)), "v"),
    "torch_fem_thermal": ("TorchFEM", "#009988", (0, (5, 1, 1, 1)), "<"),
}


# Canonical solver ordering — drives legend construction so multi-panel figures
# show solvers in the same order even when individual panels are subsets.
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


BUOYANCY_ORDER: list[str] = [
    "phiflow",
    "jax_cfd",
    "ins_jl",
    "waterlily",
    "oceananigans",
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


def _norm_solver_name(s: str) -> str:
    """Case-insensitive, separator-insensitive normaliser for solver names."""
    return s.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def resolve_solver_alias(name: str, prefer: Sequence[str] | None = None) -> str | None:
    """Map a solver string to its canonical alias in :data:`SOLVER_STYLES`.

    Handles three forms:
      * the alias key itself (``"openfoam"``)        → returned as-is
      * the display label  (``"OpenFOAM"``)          → resolved via styles
      * the spec.name from YAML (``"jax-cfd"``)      → resolved via normalisation

    Returns ``None`` if no entry matches. Used to bridge ``result.json``
    keys (which are ``spec.name``) and ``NS_ORDER`` / ``FEM_ORDER`` /
    ``SOLVER_STYLES`` (which are alias-keyed).

    ``prefer`` breaks label ties between domains: structural and thermal
    both label their FEniCS solver "FEniCS", so a bare lookup resolves to
    whichever entry comes first in :data:`SOLVER_STYLES` (the structural
    one) — and thermal plots that filter against :data:`THERMAL_ORDER`
    then silently drop the solver. Pass the domain's canonical order
    (e.g. ``THERMAL_ORDER``) so matches from that domain win.
    """
    if name in SOLVER_STYLES:
        return name
    target = _norm_solver_name(name)
    matches = [
        k
        for k, v in SOLVER_STYLES.items()
        if _norm_solver_name(k) == target or _norm_solver_name(v[0]) == target
    ]
    if not matches:
        return None
    if prefer:
        for p in prefer:
            if p in matches:
                return p
    return matches[0]


def solver_order_for_problem(name: str) -> list[str]:
    """Canonical solver ordering for a problem, picked by problem name."""
    if name == "buoyancy-grid":
        return BUOYANCY_ORDER
    if "thermal" in name:
        return THERMAL_ORDER
    if "structural" in name:
        return STRUCTURAL_ORDER
    return NS_ORDER


def solver_props(name: str) -> tuple:
    """``(label, color, linestyle, marker)`` for *name*, with grey fallback.

    Accepts either an alias key (``"openfoam"``), a display label
    (``"OpenFOAM"``), or a spec.name (``"jax-cfd"``). Falls back to a
    neutral grey when no match is found.
    """
    alias = resolve_solver_alias(name)
    if alias is not None:
        return SOLVER_STYLES[alias]
    return (name, "#888888", "-", "o")


def vorticity_2d(v: np.ndarray) -> np.ndarray:
    """Vorticity of a 2-D velocity field, shape (N, N, 1, 2) → (N, N)."""
    vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return dvydx - dvxdy


def grad_magnitude_2d(g: np.ndarray) -> np.ndarray:
    """Gradient magnitude slice for multiple field shapes.

    Handles: 2-D velocity (N,N,1,2), 3-D velocity (N,N,N,3),
    3-D scalar (N,N,N), and 1-D flat field (N,).
    """
    if g.ndim == 4 and g.shape[-1] == 2:
        return np.sqrt(g[:, :, 0, 0] ** 2 + g[:, :, 0, 1] ** 2)
    if g.ndim == 4 and g.shape[-1] == 3:
        # 3-D velocity field: take absolute value of z=0 slice vorticity
        return np.abs(vorticity_2d(g)).astype(np.float32)
    if g.ndim == 1:
        # 1-D flat field: infer 3-D shape assuming canonical 2:1:1 cantilever geometry
        n_cells = len(g)
        nx = round(float(n_cells) ** 0.5)  # n_cells = nx² for ny=2, nz=nx//2
        ny = 2
        nz = max(1, nx // 2)
        g3d = g.reshape(nz, ny, nx)
        return np.abs(g3d[:, ny // 2, :]).astype(np.float32)
    # 3-D scalar field (e.g. n-body IC gradient): take middle-z slice magnitude
    return np.abs(g[:, :, g.shape[2] // 2]).astype(np.float32)
