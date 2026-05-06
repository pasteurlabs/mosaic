"""Canonical InputSchema / OutputSchema for structural-mesh tesseracts.

All solvers perform SIMP (Solid Isotropic Material with Penalization) topology
optimisation on a padded hexahedral mesh and return the structural compliance.

Canonical interface
-------------------
  Inputs:  rho (N,), boundary_conditions (MeshBC), hex_mesh (HexMesh)
  Outputs: compliance ()

The base schemas carry plain (non-`Differentiable`) array types.  Each solver
wraps the fields it actually supports gradients on via
``mosaic_shared.types.make_differentiable``::

    from mosaic_shared.problems.structural_mesh import (
        InputSchema as _Base,
        OutputSchema as _BaseOut,
    )
    from mosaic_shared.types import make_differentiable

    InputSchema = make_differentiable(_Base, ["rho"])
    OutputSchema = make_differentiable(_BaseOut, ["compliance"])

Solvers with additional material parameters (E, nu, xmin) should subclass
``InputSchema`` (after ``make_differentiable``) and add their additional fields.
"""

from mosaic_shared.types import HexMesh, MeshBC
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float32


class InputSchema(BaseModel):
    """Canonical inputs for structural-mesh (SIMP) tesseracts."""

    rho: Array[(None,), Float32] = Field(
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

    compliance: Array[(), Float32] = Field(
        description=(
            "Structural compliance C = F^T U (scalar). "
            "Minimising compliance is equivalent to maximising stiffness."
        )
    )
