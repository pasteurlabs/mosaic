# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical InputSchema / OutputSchema for thermal-mesh tesseracts.

All solvers perform SIMP thermal topology optimisation on a padded hexahedral
mesh (steady-state heat conduction) and return the thermal compliance and the
source-identification error.

Canonical interface
-------------------
  Inputs:  rho (N,), source (N,), target_temperature (N_verts,),
           boundary_conditions (MeshBC), hex_mesh (HexMesh)
  Outputs: thermal_compliance (), identification_error ()

The base schemas carry plain (non-`Differentiable`) array types.  Each solver
wraps the fields it actually supports gradients on via
``mosaic_shared.schema_types.make_differentiable``::

    from mosaic_shared.problems.thermal_mesh import (
        InputSchema as _Base,
        OutputSchema as _BaseOut,
    )
    from mosaic_shared.schema_types import make_differentiable

    InputSchema = make_differentiable(_Base, ["rho", "source"])
    OutputSchema = make_differentiable(
        _BaseOut, ["thermal_compliance", "identification_error"]
    )

Solvers with additional material parameters (k_max, p_exp) should subclass
``InputSchema`` (after ``make_differentiable``) and add their additional fields.
"""

import numpy as np
from mosaic_shared.schema_types import HexMesh, MeshBC, MeshDirichletBC, MeshNeumannBC
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float32


class InputSchema(BaseModel):
    """Canonical inputs for thermal-mesh (SIMP heat conduction) tesseracts."""

    rho: Array[(None,), Float32] = Field(
        description=(
            "Per-cell density field, shape (n_max_cells,). "
            "Active slice = rho[:hex_mesh.n_faces]. "
            "ρ=1 → fully conducting, ρ=0 → insulating."
        )
    )
    source: Array[(None,), Float32] = Field(
        default_factory=lambda: np.zeros(1, dtype=np.float32),
        description=(
            "Per-element volumetric heat source (W/m³), shape (n_cells,). "
            "Active slice = source[:hex_mesh.n_faces]. "
            "Zero by default (topology-optimisation mode; no body heat source)."
        ),
    )
    target_temperature: Array[(None,), Float32] = Field(
        default_factory=lambda: np.zeros(1, dtype=np.float32),
        description=(
            "Per-node target temperature for the inverse (source-identification) "
            "objective, shape (n_nodes,). "
            "identification_error = ||T - target_temperature||²_2. "
            "Zero by default (not used in topology-optimisation mode)."
        ),
    )
    boundary_conditions: MeshBC = Field(
        description=(
            "Mesh boundary conditions. dirichlet.mask/values prescribe temperature "
            "(values shape (n_groups, 1), None → zero); neumann.mask/values prescribe "
            "heat flux (values shape (n_groups, 1))."
        )
    )
    hex_mesh: HexMesh = Field(
        description=(
            "Hexahedral mesh (8-node hex elements). "
            "n_points / n_faces give the active slice sizes."
        )
    )


def make_default_inputs(
    nx: int = 8,
    ny: int = 4,
    nz: int = 1,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    rho_0: float = 0.5,
    Q_total: float = 1.0,
) -> dict:
    """Build a minimal valid input dict for thermal-mesh solvers.

    Creates a structured hex mesh on [0,Lx]x[0,Ly]x[0,Lz] with heated-block
    boundary conditions (cold left face, heat flux on right face) and uniform
    density.  The returned dict is ready to pass to ``Tesseract.apply()`` or
    ``apply_tesseract()``.

    Args:
        nx: Number of elements in the x direction.
        ny: Number of elements in the y direction.
        nz: Number of elements in the z direction.
        Lx: Domain length in the x direction.
        Ly: Domain length in the y direction.
        Lz: Domain length in the z direction.
        rho_0: Uniform density (0 = insulating, 1 = conducting).
        Q_total: Total heat flux on the right face.

    Returns:
        Dict with keys: rho, source, target_temperature, boundary_conditions,
        hex_mesh.
    """
    # ── Structured hex mesh ──────────────────────────────────────────────
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

    n_cells = nx * ny * nz
    n_nodes = len(points)

    # ── Heated-block BCs: cold left face (T=0), heat flux on right face ──
    tol = 1e-6 * max(Lx, Ly, Lz)
    d_mask = np.zeros(n_nodes, dtype=np.int32)
    d_mask[points[:, 0] < tol] = 1

    n_mask = np.zeros(n_nodes, dtype=np.int32)
    n_mask[points[:, 0] > (Lx - tol)] = 1
    q_n = float(Q_total) / (Ly * Lz)

    bc = MeshBC(
        dirichlet=MeshDirichletBC(
            mask=d_mask,
            values=np.array([[0.0]], dtype=np.float32),
        ),
        neumann=MeshNeumannBC(
            mask=n_mask,
            values=np.array([[q_n]], dtype=np.float32),
        ),
    )

    hex_mesh = HexMesh(
        points=points,
        faces=cells,
        n_points=int(n_nodes),
        n_faces=int(n_cells),
    )

    return {
        "rho": np.full(n_cells, rho_0, dtype=np.float32),
        "source": np.zeros(n_cells, dtype=np.float32),
        "target_temperature": np.zeros(n_nodes, dtype=np.float32),
        "boundary_conditions": bc.model_dump(),
        "hex_mesh": hex_mesh.model_dump(),
    }


class OutputSchema(BaseModel):
    """Canonical outputs for thermal-mesh (SIMP heat conduction) tesseracts."""

    thermal_compliance: Array[(), Float32] = Field(
        description=(
            "Total thermal compliance C = ∮_Γ_N q_n · T dΓ (scalar). "
            "Equals the work done by the Neumann heat flux on the temperature field."
        )
    )
    identification_error: Array[(), Float32] = Field(
        default=np.float32(0.0),
        description=(
            "Source-identification objective: ||T - target_temperature||²_2 (scalar). "
            "Computed as sum((T_nodes - target_temperature[:n_nodes])²). "
            "Zero when target_temperature is all-zeros (topology-optimisation mode)."
        ),
    )
