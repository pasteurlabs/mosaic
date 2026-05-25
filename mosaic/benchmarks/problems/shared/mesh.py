"""Shared mesh-construction helpers for hex-mesh problem domains.

Used by ``structural_mesh`` and ``thermal_mesh`` (both consume the same
structured HEX8 mesh on a box). Each domain still owns its
domain-specific BC builder (``_cantilever_bcs`` / ``_heated_block_bcs``).
"""

from __future__ import annotations

import numpy as np


def hex_mesh_arrays(
    nx: int,
    ny: int,
    nz: int,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Structured HEX8 mesh on [0,Lx]×[0,Ly]×[0,Lz].

    Returns:
        points: ``(n_nodes, 3) float32`` — node coordinates.
        cells:  ``(n_cells, 8) int32``  — HEX8 connectivity, 0-based Abaqus
                                          ordering.
    """
    xs = np.linspace(0.0, Lx, nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny + 1, dtype=np.float32)
    zs = np.linspace(0.0, Lz, nz + 1, dtype=np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    def _nid(ix: int, iy: int, iz: int) -> int:
        return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix

    cells = np.array(
        [
            [
                _nid(ix, iy, iz),
                _nid(ix + 1, iy, iz),
                _nid(ix + 1, iy + 1, iz),
                _nid(ix, iy + 1, iz),
                _nid(ix, iy, iz + 1),
                _nid(ix + 1, iy, iz + 1),
                _nid(ix + 1, iy + 1, iz + 1),
                _nid(ix, iy + 1, iz + 1),
            ]
            for iz in range(nz)
            for iy in range(ny)
            for ix in range(nx)
        ],
        dtype=np.int32,
    )
    return points, cells
