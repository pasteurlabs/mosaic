"""Canonical InputSchema / OutputSchema for thermal-mesh tesseracts.

All solvers perform SIMP thermal topology optimisation on a padded hexahedral
mesh (steady-state heat conduction) and return at minimum the thermal compliance
and the temperature field.

Canonical interface
-------------------
  Inputs:  rho (N,), source (N,), target_temperature (N_verts,),
           boundary_conditions (MeshBC), hex_mesh (HexMesh)
  Outputs: thermal_compliance (), temperature (N_verts,),
           identification_error ()

Solvers with additional material parameters (k_max, p_exp) should subclass
InputSchema::

    from mosaic_shared.problems.thermal_mesh import InputSchema as _Base

    class InputSchema(_Base):
        k_max: float = Field(default=1.0, description="Max thermal conductivity.")
        p_exp: float = Field(default=3.0, description="SIMP penalisation exponent.")
"""

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

from mosaic_shared.types import HexMesh, MeshBC


class InputSchema(BaseModel):
    """Canonical inputs for thermal-mesh (SIMP heat conduction) tesseracts."""

    rho: Differentiable[Array[(None,), Float32]] = Field(
        description=(
            "Per-cell density field, shape (n_max_cells,). "
            "Active slice = rho[:hex_mesh.n_faces]. "
            "ρ=1 → fully conducting, ρ=0 → insulating."
        )
    )
    source: Differentiable[Array[(None,), Float32]] = Field(
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


class OutputSchema(BaseModel):
    """Canonical outputs for thermal-mesh (SIMP heat conduction) tesseracts."""

    thermal_compliance: Differentiable[Array[(), Float32]] = Field(
        description=(
            "Total thermal compliance C = ∮_Γ_N q_n · T dΓ (scalar). "
            "Equals the work done by the Neumann heat flux on the temperature field."
        )
    )
    temperature: Differentiable[Array[(None,), Float32]] = Field(
        description="Temperature field at mesh vertices/cells, shape (n_vertices,)."
    )
    identification_error: Differentiable[Array[(), Float32]] = Field(
        default=np.float32(0.0),
        description=(
            "Source-identification objective: ||T - target_temperature||²_2 (scalar). "
            "Computed as sum((T_nodes - target_temperature[:n_nodes])²). "
            "Zero when target_temperature is all-zeros (topology-optimisation mode)."
        ),
    )
