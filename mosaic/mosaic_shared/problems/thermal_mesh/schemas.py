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
``mosaic_shared.types.make_differentiable``::

    from mosaic_shared.problems.thermal_mesh import (
        InputSchema as _Base,
        OutputSchema as _BaseOut,
    )
    from mosaic_shared.types import make_differentiable

    InputSchema = make_differentiable(_Base, ["rho", "source"])
    OutputSchema = make_differentiable(
        _BaseOut, ["thermal_compliance", "identification_error"]
    )

Solvers with additional material parameters (k_max, p_exp) should subclass
``InputSchema`` (after ``make_differentiable``) and add their additional fields.
"""

import numpy as np
from mosaic_shared.types import HexMesh, MeshBC
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
