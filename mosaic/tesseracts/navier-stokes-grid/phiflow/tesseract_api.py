# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import (
    check_periodic_or_channel,
    solid_cells,
)
from mosaic_shared.schema_types import make_differentiable
from phi.jax.flow import (
    Box,
    CenteredGrid,
    Field,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    field,
    fluid,
    math,
)
from phiml.backend._numpy_backend import NumPyBackend
from pydantic import model_validator
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# mosaic-intervention: int64-sparse-indices
_numpy_ravel_multi_index = NumPyBackend.ravel_multi_index


def _ravel_multi_index_int64(
    self: NumPyBackend,
    multi_index: np.ndarray,
    shape: tuple[int, ...],
    mode: str | int = "undefined",
) -> np.ndarray:
    """Flatten indices in int64 when the flat index space exceeds int32.

    phiml casts the flat index back to the dtype of ``multi_index``. Sparse
    pressure matrices are (cells x cells), so with int32 indices the flat index
    wraps at 2D N>=216 and 3D N>=36 and matrix compression fails with "index
    -2147... is out of bounds". Upstream fix: tum-pbs/PhiML#16.
    """
    if np.prod(shape, dtype=np.int64) > np.iinfo(multi_index.dtype).max:
        multi_index = multi_index.astype(np.int64)
    return _numpy_ravel_multi_index(self, multi_index, shape, mode)


NumPyBackend.ravel_multi_index = _ravel_multi_index_int64
# mosaic-intervention-end: int64-sparse-indices


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    """PhiFlow Navier-Stokes input schema with differentiable velocity and physics params."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "phiflow")
        return self


class OutputSchema(
    make_differentiable(_CanonicalOutputSchema, ["result", "body_force"])
):
    """PhiFlow Navier-Stokes output schema with differentiable result and body force."""


# The periodic pressure solve stops at a relative and absolute residual of
# 1e-7, the loosest tolerance whose results match a fully converged solve.
# phiflow's default 1e-5 leaves enough noise in the loss that the ns-3d
# recovery stalls at ~3x the IC error. Much tighter (1e-10) is below float32's
# residual floor, so CG runs to max_iterations at 2-3x the cost without
# improving any result.
_CG_TOL = 1e-7
# The channel's pressure solve stops at 1e-2 (relative and absolute), the
# loosest at which Cd, Cl and the pressure drop stay within 0.1% of a 1e-7
# solve at 10 and 20 cells per diameter, and drag_opt ends at the same drag.
# At 1e-1 the Cd error at 10 cells per diameter moves by 4%.
_CHANNEL_CG_TOL = 1e-2


def periodic_fwd(
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
) -> jnp.ndarray:
    """Run a 2D or 3D incompressible Navier-Stokes simulation in a periodic box.

    2nd-order central-difference advection and diffusion integrated with
    phiflow's ``fluid.incompressible_rk4``, which projects every stage with a
    CG pressure solve. Dimensionality is inferred from v0.shape[-1]
    (2 → 2D, 3 → 3D).

    Args:
        v0: Initial velocity field, shape (*spatial, ndim).
        viscosity: Kinematic viscosity.
        dt: Timestep size.
        steps: Number of simulation steps.
        domain_extent: Side length of the periodic square/cubic domain.

    Returns:
        Final velocity at the cell centers, same shape as v0.
    """
    ndim = v0.shape[-1]  # 2 for 2D, 3 for 3D
    if ndim == 2:
        # squeeze the dummy nz=1 axis: (nx, ny, 1, 2) → (nx, ny, 2)
        v0 = v0[:, :, 0, :]

    if ndim == 2:
        nx, ny = v0.shape[:2]
        spatial_str = "x,y"
        grid_kwargs = {"x": nx, "y": ny}
        bounds = Box["x,y", 0:domain_extent, 0:domain_extent]
    else:
        nx, ny, nz = v0.shape[:3]
        spatial_str = "x,y,z"
        grid_kwargs = {"x": nx, "y": ny, "z": nz}
        bounds = Box["x,y,z", 0:domain_extent, 0:domain_extent, 0:domain_extent]
    ext = extrapolation.PERIODIC
    _spatial_names = spatial_str.split(",")

    def faces_to_staggered(face_arr: jnp.ndarray) -> StaggeredGrid:
        """Face-centred (ndim, *spatial) carry → StaggeredGrid."""
        component_tensors = {
            _spatial_names[i]: math.tensor(face_arr[i], math.spatial(spatial_str))
            for i in range(ndim)
        }
        staggered_tensor = math.stack(component_tensors, dim=math.dual("vector"))
        return StaggeredGrid(staggered_tensor, ext, bounds=bounds, **grid_kwargs)

    def staggered_to_faces(vel: StaggeredGrid) -> jnp.ndarray:
        """StaggeredGrid → face-centred (ndim, *spatial) carry."""
        return jnp.stack(
            [vel.vector[i].values.native(spatial_str) for i in range(ndim)], axis=0
        )

    # Use explicit CG pressure solver to prevent numerical divergence in 3D
    # at step counts ≥ 20 (default solver produces NaN in 3D; 2D unaffected).
    _cg_solve = math.Solve("CG", _CG_TOL, _CG_TOL)

    def momentum(vel: StaggeredGrid) -> StaggeredGrid:
        return advect.finite_difference(vel, vel, order=2) + diffuse.finite_difference(
            vel, viscosity, order=2
        )

    # phiflow's documented pairing for accurate periodic flow: finite
    # differences under incompressible_rk4 instead of operator splitting
    # ("Higher-Order Simulations" in its fluid simulation guide). The pressure
    # gradient matches the 2nd-order stencils. The scheme carries the pressure
    # from step to step; each step makes four CG pressure projections.
    def step(
        carry: tuple[jnp.ndarray, jnp.ndarray], _: None
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        face_arr, p_arr = carry
        vel = faces_to_staggered(face_arr)
        p = CenteredGrid(
            math.tensor(p_arr, math.spatial(spatial_str)),
            ext,
            bounds=bounds,
            **grid_kwargs,
        )
        vel, p = fluid.incompressible_rk4(
            momentum, vel, p, dt, pressure_order=2, pressure_solve=_cg_solve
        )
        return (staggered_to_faces(vel), p.values.native(spatial_str)), None

    # Convert IC from collocated to staggered face values once using PhiFlow's
    # native CenteredGrid→StaggeredGrid resampling.  The scan carry then stays
    # in face-value space so no smoothing accumulates across steps.
    v_arr = jnp.moveaxis(v0, -1, 0)  # (*spatial, ndim) → (ndim, *spatial)
    channel = math.stack(
        [math.tensor(v_arr[i], math.spatial(spatial_str)) for i in range(ndim)],
        dim=math.channel("vector"),
    )
    vel_init = StaggeredGrid(
        CenteredGrid(channel, ext, bounds=bounds, **grid_kwargs),
        ext,
        bounds=bounds,
        **grid_kwargs,
    )
    face_init = staggered_to_faces(vel_init)

    # fluid.make_incompressible's CG while_loop is wrapped with custom_vjp,
    # so the backward pass does an adjoint solve rather than differentiating
    # through the loop — both forward and VJP are safe inside lax.scan.
    p_init = jnp.zeros(face_init.shape[1:], face_init.dtype)
    (face_final, _), _ = jax.lax.scan(step, (face_init, p_init), None, length=steps)

    # Convert final face values back to collocated once.
    vel_final = faces_to_staggered(face_final)
    centered_final = CenteredGrid(vel_final, ext, bounds=bounds, **grid_kwargs)
    v_final = jnp.stack(
        [centered_final.vector[i].values.native(spatial_str) for i in range(ndim)],
        axis=0,
    )

    result = jnp.moveaxis(v_final, 0, -1)  # (ndim, *spatial) → (*spatial, ndim)
    if ndim == 2:
        # restore the dummy nz=1 axis: (nx, ny, 2) → (nx, ny, 1, 2)
        result = result[:, :, None, :]
    return result


# Boundary types of the DFG channel with a zero inlet velocity. The masks and
# the pressure operator depend only on the types, so they are built from these,
# without the inlet profile.
_CHANNEL_VELOCITY_TYPES = extrapolation.combine_sides(
    x=(extrapolation.ZERO, extrapolation.ZERO_GRADIENT), y=extrapolation.ZERO
)
# Cells beyond the inlet and the walls are solid, beyond the outlet fluid.
_CHANNEL_ACCESSIBLE = extrapolation.combine_sides(
    x=(extrapolation.ZERO, extrapolation.ONE), y=extrapolation.ZERO
)
# Zero normal pressure gradient at the inlet and the walls, p = 0 at the outlet.
_CHANNEL_PRESSURE = extrapolation.combine_sides(
    x=(extrapolation.ZERO_GRADIENT, extrapolation.ZERO), y=extrapolation.ZERO_GRADIENT
)


def channel_fwd(
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: list[float],
    obstacle: dict,
    inflow_profile: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Run the 2-D DFG channel: profile inlet, outflow, no-slip walls, cylinder.

    Operator splitting with phiflow's default semi-Lagrangian self-advection,
    explicit diffusion and a CG pressure projection (``incompressible_rk4``
    takes no obstacles). Wind-tunnel boundaries, and the cylinder as a staircase of the
    cells whose center lies strictly inside it. Every face touching a solid cell
    is held at zero velocity, and the pressure is solved in the fluid cells only.

    The force on the cylinder is the momentum the solid faces remove from the
    fluid in a step (from advection and diffusion) plus the pressure on the
    faces between fluid and solid cells.

    Returns:
        ``result`` (nx, ny, 1, 2) at the cell centers, ``pressure`` (nx, ny, 1),
        kinematic with p = 0 at the outlet, and ``body_force`` (steps, 2).
    """
    nx, ny = v0.shape[:2]
    dx = domain_extent[0] / nx
    grid_kwargs = {"x": nx, "y": ny, "bounds": Box(x=nx * dx, y=ny * dx)}

    solid = solid_cells(nx, ny, dx, obstacle["center"], obstacle["radius"])
    # mosaic-intervention: inlined-obstacle-projection

    inlet = extrapolation.combine_by_direction(
        normal=extrapolation.ConstantExtrapolation(
            math.tensor(inflow_profile, math.spatial("y"))
        ),
        tangential=extrapolation.ZERO,
    )
    ext = extrapolation.combine_sides(
        x=(inlet, extrapolation.ZERO_GRADIENT), y=extrapolation.ZERO
    )
    with math.NUMPY:
        accessible = CenteredGrid(
            math.tensor(~solid, math.spatial("x,y")), _CHANNEL_ACCESSIBLE, **grid_kwargs
        )
        # 1 on faces between two fluid cells, 0 on faces touching a solid cell.
        open_faces = field.stagger(
            accessible,
            math.minimum,
            _CHANNEL_VELOCITY_TYPES,
            at="face",
            dims=("x", "y"),
        )
    active = accessible.with_extrapolation(extrapolation.NONE)
    # x faces sit at x = (i + 1) dx (the last is the outlet), y faces at
    # y = (j + 1) dx. The inlet and wall faces are boundary values.
    open_x = jnp.asarray(open_faces.vector["x"].values.numpy("x,y"))
    open_y = jnp.asarray(open_faces.vector["y"].values.numpy("x,y"))
    fluid_cells = jnp.asarray(~solid)
    # mosaic-intervention-end: inlined-obstacle-projection

    def to_grid(ux: jnp.ndarray, uy: jnp.ndarray) -> StaggeredGrid:
        components = {
            "x": math.tensor(ux, math.spatial("x,y")),
            "y": math.tensor(uy, math.spatial("x,y")),
        }
        return StaggeredGrid(
            math.stack(components, math.dual("vector")), ext, **grid_kwargs
        )

    # mosaic-intervention: inlined-obstacle-projection
    # mosaic-intervention: body-force-output
    def close(vel: StaggeredGrid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Zero the faces touching the cylinder and return the momentum removed."""
        ux = vel.vector["x"].values.native("x,y")
        uy = vel.vector["y"].values.native("x,y")
        removed = jnp.stack([(ux * (1 - open_x)).sum(), (uy * (1 - open_y)).sum()])
        return ux * open_x, uy * open_y, removed * dx**2

    # mosaic-intervention-end: inlined-obstacle-projection

    def pressure_force(p: jnp.ndarray) -> jnp.ndarray:
        """Pressure on the faces between fluid and solid cells, from the fluid side."""
        p = jnp.where(fluid_cells, p, 0.0)
        shut_x = 1 - open_x[:-1]  # interior x faces
        shut_y = 1 - open_y
        fx = ((p[:-1] - p[1:]) * shut_x).sum()
        fy = ((p[:, :-1] - p[:, 1:]) * shut_y).sum()
        return jnp.stack([fx, fy]) * dx

    # mosaic-intervention-end: body-force-output

    def step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: None
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        ux, uy, p_prev = carry
        vel = to_grid(ux, uy)
        vel = advect.semi_lagrangian(vel, vel, dt)
        # mosaic-intervention: inlined-obstacle-projection
        ux, uy, removed = close(vel)
        vel = diffuse.explicit(to_grid(ux, uy), viscosity, dt)
        ux, uy, removed_viscous = close(vel)
        vel = to_grid(ux, uy)
        div = field.divergence(vel) * active
        # The projection subtracts grad(p dt), so phiflow's pressure is p dt.
        # Solve for the change from the previous step's pressure, which keeps
        # the CG iterations low in the near-steady flow. (Passing that pressure
        # as the solve's x0 instead leaks a tracer into the VJP inside lax.scan.)
        laplace_args = (_CHANNEL_VELOCITY_TYPES, open_faces, active)
        laplace_kwargs = {
            "wide_stencil": False,
            "order": 2,
            "implicit": None,
            "upwind": None,
            "correct_skew": False,
        }
        # mosaic-intervention: channel-pressure-warm-start
        p_guess = Field(
            div.geometry,
            math.tensor(p_prev * dt, math.spatial("x,y")),
            _CHANNEL_PRESSURE,
        )
        residual = div - fluid.masked_laplace(p_guess, *laplace_args, **laplace_kwargs)
        zero = Field(div.geometry, 0, _CHANNEL_PRESSURE)
        p_dt = p_guess + math.solve_linear(
            fluid.masked_laplace,
            residual,
            math.Solve("CG", _CHANNEL_CG_TOL, _CHANNEL_CG_TOL, x0=zero),
            *laplace_args,
            **laplace_kwargs,
        )
        # mosaic-intervention-end: channel-pressure-warm-start
        grad = field.spatial_gradient(
            p_dt, ext, at="face", order=2, scheme="green-gauss"
        )
        vel = vel - grad * open_faces
        # mosaic-intervention-end: inlined-obstacle-projection
        p = p_dt.values.native("x,y") / dt
        # mosaic-intervention: body-force-output
        force = (removed + removed_viscous) / dt + pressure_force(p)
        # mosaic-intervention-end: body-force-output
        ux = vel.vector["x"].values.native("x,y")
        uy = vel.vector["y"].values.native("x,y")
        return (ux, uy, p), force

    v_init = math.tensor(
        v0[:, :, 0, :], math.spatial("x,y"), math.channel(vector="x,y")
    )
    vel_init = StaggeredGrid(
        CenteredGrid(v_init, ext, **grid_kwargs), ext, **grid_kwargs
    )
    ux, uy, _ = close(vel_init)
    p_init = jnp.zeros((nx, ny), v0.dtype)
    (ux, uy, p), force = jax.lax.scan(step, (ux, uy, p_init), None, length=steps)

    centered = CenteredGrid(to_grid(ux, uy), ext, **grid_kwargs)
    velocity = jnp.stack([centered.vector[d].values.native("x,y") for d in "xy"], -1)
    velocity = jnp.where(fluid_cells[:, :, None], velocity, 0.0)
    return {
        "result": velocity[:, :, None, :],
        "pressure": jnp.where(fluid_cells, p, 0.0)[:, :, None],
        "body_force": force,
    }


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled forward pass: the channel when an obstacle is set, else the periodic box."""
    if inputs["obstacle"] is not None:
        return channel_fwd(
            inputs["v0"],
            inputs["viscosity"],
            inputs["dt"],
            inputs["steps"],
            inputs["domain_extent"],
            inputs["obstacle"],
            inputs["inflow_profile"],
        )
    result = periodic_fwd(
        inputs["v0"],
        inputs["viscosity"],
        inputs["dt"],
        inputs["steps"],
        inputs["domain_extent"],
    )
    return {"result": result}


_SCALAR_KEYS = ("viscosity", "dt")


def _unpack_scalars(d: dict) -> dict:
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in _SCALAR_KEYS:
        if key in d:
            d[key] = float(d[key][0])
    return d


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Run the PhiFlow Navier-Stokes forward simulation."""
    return apply_jit(_unpack_scalars(inputs.model_dump()))


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the vector-Jacobian product via JAX autodiff."""
    return vjp_jit(
        _unpack_scalars(inputs.model_dump()),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: Any) -> dict:
    """Shapes of ``result`` (as v0) and, with an obstacle, pressure and body_force."""
    v0_info = abstract_inputs.v0
    shape = tuple(v0_info["shape"] if isinstance(v0_info, dict) else v0_info.shape)
    out = {"result": {"shape": shape, "dtype": "float32"}}
    if abstract_inputs.obstacle is not None:
        out["pressure"] = {"shape": (*shape[:2], 1), "dtype": "float32"}
        out["body_force"] = {"shape": (abstract_inputs.steps, 2), "dtype": "float32"}
    return out


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
) -> dict[str, Any]:
    """JIT-compiled vector-Jacobian product computation."""
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    grads = vjp_func(cotangent_vector)[0]
    # Scalar physics params (viscosity, dt, etc.) are unpacked to Python floats
    # before the VJP, so their gradients come back as 0D arrays.  Tesseract
    # expects shape (1,) to match the input schema; reshape any 0D grad here.
    return jax.tree.map(jnp.atleast_1d, grads)
