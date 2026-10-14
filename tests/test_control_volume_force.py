# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control-volume body force on fields with a known answer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from mosaic_shared.problems.navier_stokes_grid.forces import control_volume_force


def _poiseuille(nx: int = 120, ny: int = 41, h: float = 0.41, um: float = 0.3):
    dx = h / ny
    x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    nu = 1e-3
    u = 4 * um * Y * (h - Y) / h**2
    p = -8 * nu * um / h**2 * X
    return u, np.zeros_like(u), p, nu, dx


def _stokeslet(n: int, f: tuple[float, float], center=(0.503, 0.497)):
    """Stokes flow driven by the point force ``f`` on the fluid at ``center``."""
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    rx, ry = X - center[0], Y - center[1]
    r2 = rx**2 + ry**2
    fr = f[0] * rx + f[1] * ry
    c = 1 / (4 * np.pi)  # viscosity 1
    u = c * (-f[0] * 0.5 * np.log(r2) + fr * rx / r2)
    v = c * (-f[1] * 0.5 * np.log(r2) + fr * ry / r2)
    p = fr / (2 * np.pi * r2)
    return u, v, p, dx


def test_cv_force_balances_poiseuille_flow():
    """No body in the box: pressure drop and wall shear cancel exactly.

    The shear on the top and bottom of the box carries the whole balance, so a
    formula that drops ∂u/∂y fails here.
    """
    u, v, p, nu, dx = _poiseuille()
    fx, fy = control_volume_force(u, v, p, viscosity=nu, dx=dx, box=(10, 90, 5, 30))
    assert abs(fx) < 1e-12 and abs(fy) < 1e-12


@pytest.mark.parametrize("box", [(16, 48, 16, 48), (8, 38, 21, 56)])
def test_cv_force_recovers_point_force_with_sign(box):
    """The body feels minus the force it exerts on the fluid.

    A body that slows the flow in +x pushes the fluid towards −x, so the
    force on it is +x: the drag sign the benchmark reports.
    """
    f = (-1e-3, 4e-4)
    u, v, p, dx = _stokeslet(64, f)
    fx, fy = control_volume_force(u, v, p, viscosity=1.0, dx=dx, box=box)
    assert fx > 0
    np.testing.assert_allclose([fx, fy], [-f[0], -f[1]], rtol=1e-4)


def test_cv_force_runs_on_jax_arrays():
    u, v, p, dx = _stokeslet(32, (-1e-3, 0.0))
    ref = control_volume_force(u, v, p, viscosity=1.0, dx=dx, box=(8, 24, 8, 24))
    out = control_volume_force(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(p),
        viscosity=1.0, dx=dx, box=(8, 24, 8, 24),
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-9)


def test_cv_force_rejects_a_box_without_margin():
    u, v, p, nu, dx = _poiseuille()
    with pytest.raises(ValueError, match="margin"):
        control_volume_force(u, v, p, viscosity=nu, dx=dx, box=(0, 90, 5, 30))
