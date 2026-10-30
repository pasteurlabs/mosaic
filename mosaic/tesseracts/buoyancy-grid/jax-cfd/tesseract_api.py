# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native JAX-CFD closed-wall momentum and scalar transport on a MAC grid."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
from mosaic_shared.problems.buoyancy_grid import InputSchema as BaseInput
from mosaic_shared.problems.buoyancy_grid import OutputSchema as BaseOutput
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator
from tesseract_core.runtime import ShapeDType
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Native Van Leer flux ratios need float64 to keep their reverse denominators
# from underflowing in the tiny tails of smooth float32 plume inputs.
jax.config.update("jax_enable_x64", True)


class InputSchema(
    make_differentiable(
        BaseInput, ["state0", "viscosity", "diffusivity", "buoyancy", "dt"]
    )
):
    """Coupled inputs with explicit diffusion and initial advection bounds."""

    @model_validator(mode="after")
    def _check_cfl(self) -> "InputSchema":
        if isinstance(self.state0, (dict, ShapeDType)):
            return self
        state = np.asarray(self.state0)
        inverse_dx = state.shape[0] / self.domain_extent
        dt = float(np.asarray(self.dt)[0])
        diffusion = max(
            float(np.asarray(self.viscosity)[0]), float(np.asarray(self.diffusivity)[0])
        )
        if dt * diffusion * inverse_dx**2 > 0.25:
            raise ValueError(
                "Explicit diffusion requires dt * max(viscosity, diffusivity) * N² / L² <= 0.25."
            )
        speed = np.max(np.abs(state[:-1, :, 0])) + np.max(np.abs(state[:, :-1, 1]))
        if dt * inverse_dx * speed > 0.5:
            raise ValueError("Initial advective CFL must be <= 0.5; reduce dt.")
        return self


class OutputSchema(make_differentiable(BaseOutput, ["result"])):
    """Final state in the canonical staggered packing."""


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """Advance coupled native finite-volume fluxes, diffusion, and projection."""
    n = inputs["state0"].shape[0]
    extent = inputs["domain_extent"]
    grid = cfd.grids.Grid((n, n), domain=((0.0, extent), (0.0, extent)))
    velocity_bc = cfd.boundaries.HomogeneousBoundaryConditions(
        (("dirichlet", "dirichlet"),) * 2
    )
    scalar_bc = cfd.boundaries.HomogeneousBoundaryConditions(
        (("neumann", "neumann"),) * 2
    )
    dt = inputs["dt"].astype(jnp.float64)[0]
    viscosity = inputs["viscosity"].astype(jnp.float64)[0]
    diffusivity = inputs["diffusivity"].astype(jnp.float64)[0]
    buoyancy = inputs["buoyancy"].astype(jnp.float64)[0]

    def unpack(state: jax.Array):
        # Canonical offsets already match JAX-CFD. Impose the normal wall values
        # instead of resampling between cell centres and faces.
        velocity = tuple(
            velocity_bc.impose_bc(cfd.grids.GridArray(state[..., axis], offset, grid))
            for axis, offset in enumerate(grid.cell_faces)
        )
        scalar = scalar_bc.impose_bc(
            cfd.grids.GridArray(state[..., 2], grid.cell_center, grid)
        )
        return velocity, scalar

    def step(state: jax.Array, _: None):
        velocity, scalar = unpack(state)
        scalar_rhs = cfd.advection.advect_van_leer(
            scalar, velocity, dt
        ) + cfd.diffusion.diffuse(scalar, diffusivity)
        next_scalar = scalar_bc.impose_bc(scalar.array + dt * scalar_rhs)
        vertical_force = (
            buoyancy * cfd.interpolation.linear(scalar, grid.cell_faces[1]).array
        )

        def forcing(v: tuple):
            return (
                cfd.grids.GridArray(jnp.zeros_like(v[0].data), v[0].offset, grid),
                vertical_force,
            )

        # The default advect_van_leer_using_limiters only supports periodic BCs.
        # This native Van Leer implementation explicitly handles closed walls.
        def convect(v: tuple):
            return tuple(
                cfd.advection.advect_van_leer(component, v, dt) for component in v
            )

        def pressure_solve(v: tuple, q0: Any, pressure_bc: Any):
            # Buoyancy and diffusion can change provisional wall faces. Enforce
            # impermeability before forming the Poisson RHS, not only after
            # projection, so the boundary flux and pressure equation agree.
            bounded = tuple(u.bc.impose_bc(u.array) for u in v)
            return cfd.pressure.solve_fast_diag(bounded, q0, pressure_bc)

        momentum_step = cfd.equations.semi_implicit_navier_stokes(
            density=1.0,
            viscosity=viscosity,
            dt=dt,
            grid=grid,
            convect=convect,
            forcing=forcing,
            pressure_solve=pressure_solve,
        )
        next_velocity = momentum_step(velocity)
        result = jnp.stack(
            [next_velocity[0].data, next_velocity[1].data, next_scalar.data], axis=-1
        )
        return result, None

    result, _ = jax.lax.scan(
        step, inputs["state0"].astype(jnp.float64), None, length=inputs["steps"]
    )
    return {"result": result.astype(jnp.float32)}


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Run the coupled solver, retaining only its exact recurrent state."""
    return apply_jit(inputs.model_dump())


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str, ...],
    vjp_outputs: tuple[str, ...],
    cotangent_vector: dict,
) -> dict:
    """Compute selected input cotangents through the coupled solver."""
    selected = filter_func(apply_jit, inputs, vjp_outputs)
    _, pullback = jax.vjp(
        selected, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return pullback(cotangent_vector)[0]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Differentiate all coupled timesteps using JAX reverse-mode AD."""
    return vjp_jit(
        inputs.model_dump(),
        tuple(sorted(vjp_inputs)),
        tuple(sorted(vjp_outputs)),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Infer metadata without executing a numerical step."""
    state = abstract_inputs.state0
    shape = tuple(state["shape"]) if isinstance(state, dict) else state.shape
    return {"result": {"shape": shape, "dtype": "float32"}}
