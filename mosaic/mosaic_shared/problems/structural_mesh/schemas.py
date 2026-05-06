"""Canonical InputSchema / OutputSchema for structural-mesh tesseracts.

All solvers perform SIMP (Solid Isotropic Material with Penalization) topology
optimisation on a padded hexahedral mesh and return at minimum the structural
compliance.

Canonical interface
-------------------
  Inputs:  rho (N,), boundary_conditions (MeshBC), hex_mesh (HexMesh)
  Outputs: compliance (), von_mises_stress (n_cells,), displacement (n_nodes, 3)

Solvers with additional material parameters (E, nu, xmin) should subclass
InputSchema::

    from mosaic_shared.problems.structural_mesh import InputSchema as _Base

    class InputSchema(_Base):
        E:    float = Field(default=1.0,   description="Young's modulus.")
        nu:   float = Field(default=0.3,   description="Poisson's ratio.")
        xmin: float = Field(default=0.001, description="Minimum (void) stiffness.")
"""

import numpy as np
from mosaic_shared.types import HexMesh, MeshBC, MeshDirichletBC, MeshNeumannBC
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    """Canonical inputs for structural-mesh (SIMP) tesseracts."""

    rho: Differentiable[Array[(None,), Float32]] = Field(
        description=(
            "Per-cell density field, shape (n_max_cells,). "
            "Active slice = rho[:hex_mesh.n_faces]. "
            "Values in [0, 1]: 0 = void, 1 = solid."
        )
    )
    boundary_conditions: MeshBC = Field(
        description=(
            "Mesh boundary conditions. dirichlet.mask/values prescribe displacement "
            "(values shape (n_groups, 3), None → zero); neumann.mask/values prescribe "
            "surface traction (values shape (n_groups, 3))."
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
    ny: int = 2,
    nz: int = 4,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    rho_0: float = 0.5,
    F_total: float = 1.0,
) -> dict:
    """Build a minimal valid input dict for structural-mesh solvers.

    Creates a structured hex mesh on [0,Lx]x[0,Ly]x[0,Lz] with cantilever
    boundary conditions (clamped left face, downward traction on right face)
    and uniform density.  The returned dict is ready to pass to
    ``Tesseract.apply()`` or ``apply_tesseract()``.

    Args:
        nx, ny, nz: Number of elements in each direction.
        Lx, Ly, Lz: Domain dimensions.
        rho_0: Uniform density (0 = void, 1 = solid).
        F_total: Total force on the right face.

    Returns:
        Dict with keys: rho, boundary_conditions, hex_mesh.
    """
    # ── Structured hex mesh ──────────────────────────────────────────────
    xs = np.linspace(0.0, Lx, nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny + 1, dtype=np.float32)
    zs = np.linspace(0.0, Lz, nz + 1, dtype=np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    def _nid(ix, iy, iz):
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

    # ── Cantilever BCs: clamp left face, downward traction on right face ─
    tol = 1e-6 * max(Lx, Ly, Lz)
    d_mask = np.zeros(n_nodes, dtype=np.int32)
    d_mask[points[:, 0] < tol] = 1  # clamp at x=0

    n_mask = np.zeros(n_nodes, dtype=np.int32)
    n_mask[points[:, 0] > (Lx - tol)] = 1  # load at x=Lx
    traction = F_total / (Ly * Lz)  # force per unit area

    bc = MeshBC(
        dirichlet=MeshDirichletBC(mask=d_mask, values=None),  # zero displacement
        neumann=MeshNeumannBC(
            mask=n_mask,
            values=np.array([[0.0, 0.0, -traction]], dtype=np.float32),  # −z
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
        "boundary_conditions": bc.model_dump(),
        "hex_mesh": hex_mesh.model_dump(),
    }


class OutputSchema(BaseModel):
    """Canonical outputs for structural-mesh (SIMP) tesseracts."""

    compliance: Differentiable[Array[(), Float32]] = Field(
        description=(
            "Structural compliance C = F^T U (scalar). "
            "Minimising compliance is equivalent to maximising stiffness."
        )
    )
    von_mises_stress: Differentiable[Array[(None,), Float32]] = Field(
        description="Per-cell von Mises stress at element centroids, shape (n_cells,)."
    )
    displacement: Differentiable[Array[(None, 3), Float32]] = Field(
        description="Nodal displacement field, shape (n_nodes, 3)."
    )
