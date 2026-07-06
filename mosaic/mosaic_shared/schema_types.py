# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh and boundary condition type definitions."""

import types
from enum import StrEnum
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, create_model
from tesseract_core.runtime import Array, Differentiable, Float32, Int32

# ---------------------------------------------------------------------------
# Boundary conditions for Cartesian grid solvers
# ---------------------------------------------------------------------------


class BCType(StrEnum):
    """Boundary condition type for a single domain face."""

    PERIODIC = "periodic"
    NO_SLIP = "no_slip"  # homogeneous Dirichlet — zero velocity at wall
    NEUMANN = "neumann"  # zero normal derivative — outflow / symmetry
    DIRICHLET = "dirichlet"  # prescribed velocity — inflow


class FaceBC(BaseModel):
    """Boundary condition for one face of the domain."""

    type: BCType = BCType.PERIODIC
    value: list[float] | None = Field(
        default=None,
        description=(
            "Velocity vector for DIRICHLET type (ndim components). "
            "None is equivalent to NO_SLIP (zero velocity)."
        ),
    )


class GridBC(BaseModel):
    """Per-face boundary conditions for a Cartesian grid.

    Default: all PERIODIC — matches the current behaviour of every grid solver
    so existing call sites require no changes.

    Face ordering::

        x_lo / x_hi  — left / right  (x-axis)
        y_lo / y_hi  — front / back  (y-axis)
        z_lo / z_hi  — bottom / top  (z-axis)

    Example — 2-D channel flow (periodic in x, no-slip walls in y)::

        from mosaic_shared.schema_types import BCType, FaceBC, GridBC

        bc = GridBC(
            y_lo=FaceBC(type=BCType.NO_SLIP),
            y_hi=FaceBC(type=BCType.NO_SLIP),
        )
    """

    x_lo: FaceBC = Field(default_factory=FaceBC)
    x_hi: FaceBC = Field(default_factory=FaceBC)
    y_lo: FaceBC = Field(default_factory=FaceBC)
    y_hi: FaceBC = Field(default_factory=FaceBC)
    z_lo: FaceBC = Field(default_factory=FaceBC)
    z_hi: FaceBC = Field(default_factory=FaceBC)

    @property
    def is_fully_periodic(self) -> bool:
        """True iff all six faces are PERIODIC."""
        return all(
            f.type == BCType.PERIODIC
            for f in [self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.z_lo, self.z_hi]
        )


# ---------------------------------------------------------------------------
# Embedded obstacles for Cartesian grid solvers
# ---------------------------------------------------------------------------


class ObstacleShape(StrEnum):
    """Shape of an embedded obstacle in the flow domain."""

    CYLINDER = "cylinder"  # infinite cylinder (circle in 2-D cross-section)
    BOX = "box"  # axis-aligned rectangular box


class GridObstacle(BaseModel):
    """Geometric obstacle embedded in the flow domain (no-slip wall).

    Coordinates are expressed as fractions of domain_extent so the spec is
    resolution-independent.  Solvers that work on a rasterized lattice (XLB,
    Lettuce) convert center/radius to grid indices internally.
    """

    shape: ObstacleShape = ObstacleShape.CYLINDER
    center: list[float] = Field(
        default=[0.5, 0.5],
        description="Obstacle center as fraction of domain_extent per axis.",
    )
    radius: float | None = Field(
        default=None,
        description="Cylinder radius as fraction of domain_extent (CYLINDER only).",
    )
    half_widths: list[float] | None = Field(
        default=None,
        description="Box half-widths per axis as fractions of domain_extent (BOX only).",
    )
    bc: Literal["no_slip"] = "no_slip"


# ---------------------------------------------------------------------------
# Common grid field type aliases
# ---------------------------------------------------------------------------

# Velocity / force field on a Cartesian grid.
# Shape convention: (nx, ny, nz, ndim) — use nz=1 for 2-D simulations.
GridVectorField = Array[(None, None, None, None), Float32]

# Scalar field on a Cartesian grid.
# Shape convention: (nx, ny, nz).
GridScalarField = Array[(None, None, None), Float32]


# ---------------------------------------------------------------------------
# Boundary conditions for mesh-based solvers
# ---------------------------------------------------------------------------


class MeshDirichletBC(BaseModel):
    """Dirichlet (prescribed value) BCs for mesh-based solvers.

    Node groups are identified by ``mask``:

    * ``mask[i] = 0`` — node *i* is free.
    * ``mask[i] = k`` (k ≥ 1) — node *i* belongs to group *k*; its prescribed
      value is ``values[k-1, :]``.

    ``values`` may be ``None`` for homogeneous (zero) Dirichlet BCs, e.g.
    clamped walls or no-slip velocity conditions.
    """

    mask: Array[(None,), Int32] = Field(
        description="Per-node group index. 0 = free, k ≥ 1 → prescribe values[k-1]."
    )
    values: Array[(None, None), Float32] | None = Field(
        default=None,
        description=(
            "Prescribed value per group, shape (n_groups, n_dofs). "
            "None → homogeneous zero for all marked nodes."
        ),
    )


class MeshNeumannBC(BaseModel):
    """Neumann (prescribed flux / traction) BCs for mesh-based solvers.

    Same group-index convention as :class:`MeshDirichletBC`:

    * ``mask[i] = 0`` — no prescribed flux at node *i*.
    * ``mask[i] = k`` (k ≥ 1) — node *i* belongs to Neumann group *k*;
      the flux/traction is ``values[k-1, :]``.
    """

    mask: Array[(None,), Int32] = Field(
        description="Per-node Neumann group index. 0 = no flux, k ≥ 1 → prescribe values[k-1]."
    )
    values: Array[(None, None), Float32] = Field(
        description="Prescribed flux / traction per group, shape (n_groups, n_dofs)."
    )


class MeshBC(BaseModel):
    """Combined boundary conditions for mesh-based solvers.

    Either component may be ``None`` when not required.

    Example — 3-D Stokes no-slip walls::

        import numpy as np
        from mosaic_shared.schema_types import MeshBC, MeshDirichletBC

        bc = MeshBC(
            dirichlet=MeshDirichletBC(
                mask=wall_node_mask,  # int32, 1 at wall nodes
                values=np.zeros((1, 3), dtype=np.float32),
            )
        )
    """

    dirichlet: MeshDirichletBC | None = Field(
        default=None, description="Dirichlet (prescribed value) BCs."
    )
    neumann: MeshNeumannBC | None = Field(
        default=None, description="Neumann (prescribed flux / traction) BCs."
    )


class TriangularMesh(BaseModel):
    """Triangular surface mesh."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Array[(None, 3), Float32]
    faces: Array[(None, 3), Int32]
    point_data: Annotated[dict[str, Array] | None, SkipValidation()] = None
    cell_data: Annotated[dict[str, Array] | None, SkipValidation()] = None
    n_points: int | None = None
    n_cells: int | None = None


class VolumetricMesh(BaseModel):
    """Unstructured volumetric mesh (tetrahedra, hexahedra, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Array[(None, 3), Float32]
    cell_connectivity: Array[(None,), Int32]
    cell_types: Array[(None,), Int32]
    point_data: Annotated[dict[str, Array] | None, SkipValidation()] = None
    cell_data: Annotated[dict[str, Array] | None, SkipValidation()] = None
    n_points: int | None = None
    n_cells: int | None = None


class TetMesh(BaseModel):
    """Tetrahedral mesh (4-node tet elements)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Array[(None, 3), Float32]
    faces: Array[(None, 4), Int32]
    n_points: Int32 = 0
    n_faces: Int32 = 0


class HexMesh(BaseModel):
    """Hexahedral mesh (8-node hex elements)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Array[(None, 3), Float32]
    faces: Array[(None, 8), Int32]
    n_points: Int32 = 0
    n_faces: Int32 = 0


def _wrap_differentiable(field_type: Any) -> Any:
    """Wrap a field annotation in `Differentiable[...]`, handling Optional and dict."""
    origin = get_origin(field_type)

    if origin is Annotated:
        return field_type

    if origin is Union or origin is types.UnionType:
        args = get_args(field_type)
        non_none = tuple(a for a in args if a is not type(None))
        has_none = len(non_none) != len(args)
        if len(non_none) == 1:
            inner = non_none[0]
            inner_origin = get_origin(inner)
            if inner_origin is dict:
                k, v = get_args(inner)
                wrapped = dict[k, Differentiable[v]]
            else:
                wrapped = Differentiable[inner]
            return wrapped | None if has_none else wrapped
        return field_type

    if origin is dict:
        key_type, value_type = get_args(field_type)
        return dict[key_type, Differentiable[value_type]]

    return Differentiable[field_type]


def make_differentiable(
    base_class: type[BaseModel],
    differentiable_fields: list[str],
) -> type[BaseModel]:
    """Create a Pydantic subclass with the specified fields wrapped in `Differentiable`.

    Field metadata (defaults, default_factory, descriptions, validators) is
    preserved by reusing each field's `FieldInfo` from the base class.

    Args:
        base_class: Pydantic BaseModel subclass to extend.
        differentiable_fields: Field names to wrap in `Differentiable[...]`.

    Returns:
        A new Pydantic model subclass with the requested fields differentiable.

    Example:
        >>> DiffOutput = make_differentiable(OutputSchema, ["compliance"])
    """
    field_defs: dict[str, tuple] = {}

    for field_name, field_info in base_class.model_fields.items():
        field_type = base_class.__annotations__.get(field_name)
        if field_type is None:
            # Field is inherited from a parent class — leave it untouched.
            continue

        if field_name in differentiable_fields:
            field_type = _wrap_differentiable(field_type)

        field_defs[field_name] = (field_type, field_info)

    return create_model(
        f"Differentiable{base_class.__name__}",
        __base__=base_class,
        **field_defs,
    )
