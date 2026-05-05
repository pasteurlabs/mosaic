"""Mesh and boundary condition type definitions."""

from enum import Enum
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, SkipValidation
from tesseract_core.runtime import Array, Differentiable, Float32, Int32

# ---------------------------------------------------------------------------
# Boundary conditions for Cartesian grid solvers
# ---------------------------------------------------------------------------


class BCType(str, Enum):
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

        from mosaic_shared.types import BCType, FaceBC, GridBC

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


class ObstacleShape(str, Enum):
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
        from mosaic_shared.types import MeshBC, MeshDirichletBC

        bc = MeshBC(
            dirichlet=MeshDirichletBC(
                mask=wall_node_mask,           # int32, 1 at wall nodes
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


def make_differentiable(
    base_class: type[BaseModel],
    differentiable_fields: list[str],
) -> type[BaseModel]:
    """Create a mesh class with specified fields marked as differentiable.

    Args:
        base_class: Base mesh class (TriangularMesh or VolumetricMesh)
        differentiable_fields: List of field names to mark as differentiable

    Returns:
        New Pydantic model class with differentiable fields

    Example:
        >>> DiffMesh = make_differentiable(TriangularMesh, ["points"])
        >>> class InputSchema(BaseModel):
        ...     mesh: DiffMesh
    """
    from pydantic_core import PydanticUndefined

    new_annotations = {}
    new_namespace = {}

    # Get field defaults from the base class
    for field_name, field_info in base_class.model_fields.items():
        if field_info.default is not PydanticUndefined:
            new_namespace[field_name] = field_info.default
        elif field_info.default_factory is not None:
            new_namespace[field_name] = field_info.default_factory

    for field_name, field_type in base_class.__annotations__.items():
        if field_name not in differentiable_fields:
            new_annotations[field_name] = field_type
            continue

        # Handle Annotated types (e.g. Annotated[dict[str, Array] | None, SkipValidation()])
        origin = get_origin(field_type)
        if origin is Annotated:
            # Can't make Annotated dict fields differentiable - keep as is
            new_annotations[field_name] = field_type
            continue

        # Check if it's Optional (Union with None)
        if origin is type(None):  # Optional type: T | None
            args = get_args(field_type)
            inner_type = args[0]
            inner_origin = get_origin(inner_type)

            if inner_origin is dict:
                # dict[str, Array] | None -> dict[str, Differentiable[Array]] | None
                key_type, value_type = get_args(inner_type)
                new_annotations[field_name] = (
                    dict[key_type, Differentiable[value_type]] | None
                )
            else:
                # Array | None -> Differentiable[Array] | None
                new_annotations[field_name] = Differentiable[inner_type] | None
        elif origin is dict:
            # dict[str, Array] -> dict[str, Differentiable[Array]]
            key_type, value_type = get_args(field_type)
            new_annotations[field_name] = dict[key_type, Differentiable[value_type]]
        else:
            # Simple case: Array -> Differentiable[Array]
            new_annotations[field_name] = Differentiable[field_type]

    new_namespace["__annotations__"] = new_annotations

    return type(
        f"Differentiable{base_class.__name__}",
        (base_class,),
        new_namespace,
    )
