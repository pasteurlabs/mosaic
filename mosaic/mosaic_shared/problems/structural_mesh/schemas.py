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

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

from mosaic_shared.types import HexMesh, MeshBC


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
