# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body force from a control-volume momentum balance on the canonical grid.

For a steady incompressible flow (ρ = 1), the force the fluid exerts on
everything inside a closed contour S equals the momentum flux and stress
integrated over S:

    F = ∮_S ( σ·n − u (u·n) ) dS,   σ = −p I + ν (∇u + ∇uᵀ),

with n the outward normal of S. The contour can sit anywhere in the fluid, so
the result does not depend on how a solver represents the body surface. That
makes it a solver-independent cross-check of each solver's own body force.

The implementation uses only slicing, arithmetic and ``.sum()``, so it works
unchanged on NumPy, JAX and PyTorch arrays.
"""

from __future__ import annotations

from typing import Any

__all__ = ["control_volume_force"]


def control_volume_force(
    u: Any,
    v: Any,
    p: Any,
    *,
    viscosity: float,
    dx: float,
    box: tuple[int, int, int, int],
) -> tuple[Any, Any]:
    """Force (F_x, F_y) on the body enclosed by ``box``, per unit depth and density.

    Args:
        u: x-velocity at the cell centers, shape (nx, ny).
        v: y-velocity at the cell centers, shape (nx, ny).
        p: kinematic pressure p/ρ at the cell centers, shape (nx, ny).
        viscosity: kinematic viscosity ν.
        dx: cell size (the grid is isotropic).
        box: ``(i0, i1, j0, j1)``, the cell-center indices of the contour's
            left, right, bottom and top sides. Every cell on the contour must
            be fluid, and the contour needs one cell of margin on each side
            for the central differences: ``1 <= i0 < i1 <= nx - 2`` and
            ``1 <= j0 < j1 <= ny - 2``.

    Returns:
        ``(F_x, F_y)``: the force on the body, positive along +x and +y.
        Valid for a steady flow only, since the momentum-storage term is not
        included.
    """
    i0, i1, j0, j1 = box
    nx, ny = u.shape
    if not (1 <= i0 < i1 <= nx - 2 and 1 <= j0 < j1 <= ny - 2):
        raise ValueError(f"box {box} needs one cell of margin inside a {nx}x{ny} grid")
    nu = viscosity
    h = 2.0 * dx

    def trapezoid(f: Any) -> Any:
        return (f.sum() - 0.5 * (f[0] + f[-1])) * dx

    def vertical_side(i: int, sign: float) -> tuple[Any, Any]:
        js = slice(j0, j1 + 1)
        uu, vv, pp = u[i, js], v[i, js], p[i, js]
        dudx = (u[i + 1, js] - u[i - 1, js]) / h
        dvdx = (v[i + 1, js] - v[i - 1, js]) / h
        dudy = (u[i, j0 + 1 : j1 + 2] - u[i, j0 - 1 : j1]) / h
        sxx = -pp + 2.0 * nu * dudx
        sxy = nu * (dudy + dvdx)
        return trapezoid(sign * (sxx - uu * uu)), trapezoid(sign * (sxy - vv * uu))

    def horizontal_side(j: int, sign: float) -> tuple[Any, Any]:
        is_ = slice(i0, i1 + 1)
        uu, vv, pp = u[is_, j], v[is_, j], p[is_, j]
        dudy = (u[is_, j + 1] - u[is_, j - 1]) / h
        dvdy = (v[is_, j + 1] - v[is_, j - 1]) / h
        dvdx = (v[i0 + 1 : i1 + 2, j] - v[i0 - 1 : i1, j]) / h
        sxy = nu * (dudy + dvdx)
        syy = -pp + 2.0 * nu * dvdy
        return trapezoid(sign * (sxy - uu * vv)), trapezoid(sign * (syy - vv * vv))

    sides = (
        vertical_side(i0, -1.0),
        vertical_side(i1, 1.0),
        horizontal_side(j0, -1.0),
        horizontal_side(j1, 1.0),
    )
    fx = sides[0][0] + sides[1][0] + sides[2][0] + sides[3][0]
    fy = sides[0][1] + sides[1][1] + sides[2][1] + sides[3][1]
    return fx, fy
