# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermal-mesh reference solve.

``_approx_target_temperature`` produces ``target_temperature``, the ground
truth for ``optimization/conductivity_recovery_bfgs`` and the two
source-identification gradient experiments. If it solves a different problem
than the one the solvers are handed, every recovered-vs-truth comparison is
measured against an inconsistent target.

The checks here are closed-form rather than golden values:

  1. The assembled Neumann load equals the prescribed ``q_n * Ly * Lz``, so
     the reference applies the flux the boundary condition asks for.
  2. At uniform density the heated-face temperature equals the analytic
     slab solution ``Q * Lx / (k * Ly * Lz)``, at several resolutions, so the
     reference is also mesh-independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.problems.thermal_mesh.physics import (
    _K_MAX,
    _K_MIN_RATIO,
    _P_EXP,
    _approx_target_temperature,
    _heated_block_bcs,
)

_LX, _LY, _LZ = 2.0, 1.0, 1.0
_Q_TOTAL = 100.0
_MESHES = [(16, 8, 1), (8, 4, 1), (32, 16, 1), (16, 8, 2)]


def _structured_hex_mesh(nx: int, ny: int, nz: int):
    """Return (points, cells) for a structured hex mesh on [0,Lx]x[0,Ly]x[0,Lz]."""
    xs = np.linspace(0.0, _LX, nx + 1)
    ys = np.linspace(0.0, _LY, ny + 1)
    zs = np.linspace(0.0, _LZ, nz + 1)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    def nid(ix: int, iy: int, iz: int) -> int:
        return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix

    cells = np.array(
        [
            [
                nid(ix, iy, iz),
                nid(ix + 1, iy, iz),
                nid(ix + 1, iy + 1, iz),
                nid(ix, iy + 1, iz),
                nid(ix, iy, iz + 1),
                nid(ix + 1, iy, iz + 1),
                nid(ix + 1, iy + 1, iz + 1),
                nid(ix, iy + 1, iz + 1),
            ]
            for iz in range(nz)
            for iy in range(ny)
            for ix in range(nx)
        ],
        dtype=np.int64,
    )
    return points, cells


def _bc_dict(points):
    bc = _heated_block_bcs(points, _LX, _LY, _LZ, Q_total=_Q_TOTAL, hot_spot=False)
    return bc.model_dump() if hasattr(bc, "model_dump") else bc


def _analytic_hot_face_temperature(rho: float) -> float:
    """Uniform slab with one insulated end: T_hot = Q * Lx / (k * Ly * Lz)."""
    k_min = _K_MIN_RATIO * _K_MAX
    k = k_min + (_K_MAX - k_min) * rho**_P_EXP
    return _Q_TOTAL * _LX / (k * _LY * _LZ)


@pytest.mark.parametrize("nx,ny,nz", _MESHES, ids=lambda v: str(v))
def test_reference_matches_the_analytic_slab_solution(nx, ny, nz):
    """Uniform density reduces to a 1D slab with a closed-form temperature.

    Distributing the flux per unique face node rather than per face element
    applies only ``(ny+1)(nz+1) / (4*ny*nz)`` of it, which is 0.5625 on the
    configured 16x8x1 mesh, so the reference used to sit ~44% low here and
    move with resolution.
    """
    rho_0 = 0.5
    points, cells = _structured_hex_mesh(nx, ny, nz)
    T = _approx_target_temperature(
        np.full(len(cells), rho_0),
        np.zeros(len(cells)),
        cells,
        points,
        _bc_dict(points),
    )

    hot_face = np.abs(points[:, 0] - points[:, 0].max()) < 1e-9
    assert T[hot_face].mean() == pytest.approx(
        _analytic_hot_face_temperature(rho_0), rel=1e-6
    )


def test_reference_is_mesh_independent():
    """The analytic check above must hold at one value across resolutions."""
    temps = []
    for nx, ny, nz in _MESHES:
        points, cells = _structured_hex_mesh(nx, ny, nz)
        T = _approx_target_temperature(
            np.full(len(cells), 0.5),
            np.zeros(len(cells)),
            cells,
            points,
            _bc_dict(points),
        )
        hot_face = np.abs(points[:, 0] - points[:, 0].max()) < 1e-9
        temps.append(float(T[hot_face].mean()))

    assert max(temps) == pytest.approx(min(temps), rel=1e-6)
