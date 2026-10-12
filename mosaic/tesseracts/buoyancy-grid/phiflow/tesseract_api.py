# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable coupled buoyancy flow with exact staggered-state recurrence."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.buoyancy_grid import InputSchema as BaseInput
from mosaic_shared.problems.buoyancy_grid import OutputSchema as BaseOutput
from mosaic_shared.schema_types import make_differentiable
from phi.jax.flow import (
    Box,
    CenteredGrid,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
    math,
)
from pydantic import model_validator
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(
    make_differentiable(
        BaseInput, ["state0", "viscosity", "diffusivity", "buoyancy", "dt"]
    )
):
    """PhiFlow inputs with the four-substep explicit diffusion stability bound."""

    @model_validator(mode="after")
    def _check_diffusion_cfl(self) -> "InputSchema":
        if isinstance(self.state0, dict):
            return self
        n = np.asarray(self.state0).shape[0]
        dt = float(np.asarray(self.dt)[0])
        coefficient = max(
            float(np.asarray(self.viscosity)[0]), float(np.asarray(self.diffusivity)[0])
        )
        if dt * coefficient * (n / self.domain_extent) ** 2 > 1.0:
            raise ValueError(
                "Explicit diffusion requires dt * max(viscosity, diffusivity) * N² / L² <= 1 (four substeps)."
            )
        return self


class OutputSchema(make_differentiable(BaseOutput, ["result"])):
    """Differentiable final state with the runtime-required schema name."""


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """Advect scalar and velocity, diffuse, add buoyancy, then project velocity."""
    state0 = inputs["state0"]
    n = state0.shape[0]
    extent = inputs["domain_extent"]
    bounds = Box["x,y", 0:extent, 0:extent]
    ext = extrapolation.ZERO
    scalar_ext = extrapolation.ZERO_GRADIENT
    dt = inputs["dt"][0]
    viscosity = inputs["viscosity"][0]
    diffusivity = inputs["diffusivity"][0]
    buoyancy = inputs["buoyancy"][0]
    # Four fixed diffusion substeps; the runtime schema enforces the CFL bound.
    solve = math.Solve("CG", 1e-6, 1e-7, max_iterations=1000)

    def unpack(state: jax.Array) -> tuple[StaggeredGrid, CenteredGrid]:
        faces = math.stack(
            {
                "x": math.tensor(state[:-1, :, 0], math.spatial("x,y")),
                "y": math.tensor(state[:, :-1, 1], math.spatial("x,y")),
            },
            math.dual("vector"),
        )
        velocity = StaggeredGrid(faces, ext, bounds=bounds, x=n, y=n)
        scalar = CenteredGrid(
            math.tensor(state[..., 2], math.spatial("x,y")),
            scalar_ext,
            bounds=bounds,
            x=n,
            y=n,
        )
        return velocity, scalar

    def pack(velocity: StaggeredGrid, scalar: CenteredGrid) -> jax.Array:
        ux = jnp.pad(velocity.vector["x"].values.native("x,y"), ((0, 1), (0, 0)))
        uy = jnp.pad(velocity.vector["y"].values.native("x,y"), ((0, 0), (0, 1)))
        return jnp.stack([ux, uy, scalar.values.native("x,y")], axis=-1)

    def step(state: jax.Array, _: None) -> tuple[jax.Array, None]:
        velocity, scalar = unpack(state)
        scalar = advect.semi_lagrangian(scalar, velocity, dt)
        scalar = diffuse.explicit(scalar, diffusivity, dt, substeps=4)
        velocity = advect.semi_lagrangian(velocity, velocity, dt)
        velocity = diffuse.explicit(velocity, viscosity, dt, substeps=4)
        force = scalar * math.tensor([0.0, 1.0], math.channel(vector="x,y"))
        velocity = velocity + dt * buoyancy * (force @ velocity)
        velocity, _ = fluid.make_incompressible(velocity, solve=solve)
        return pack(velocity, scalar), None

    result, _ = jax.lax.scan(step, state0, None, length=inputs["steps"])
    return {"result": result}


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Run the coupled flow and return the final recurrent state."""
    return apply_jit(inputs.model_dump())


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str, ...],
    vjp_outputs: tuple[str, ...],
    cotangent_vector: dict,
) -> dict:
    """Compute selected input cotangents through the coupled simulation."""
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
    """Expose the runtime vector-Jacobian product endpoint."""
    return vjp_jit(
        inputs.model_dump(),
        tuple(sorted(vjp_inputs)),
        tuple(sorted(vjp_outputs)),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, dict[str, Any]]:
    """Infer output metadata without running a simulation."""
    state = abstract_inputs.state0
    if isinstance(state, dict):
        shape, dtype = tuple(state["shape"]), state["dtype"]
    else:
        shape, dtype = state.shape, str(state.dtype)
    return {"result": {"shape": shape, "dtype": dtype}}
