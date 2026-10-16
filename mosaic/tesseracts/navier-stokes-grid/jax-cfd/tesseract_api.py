# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import check_periodic
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["v0", "viscosity", "dt"])
):
    """Input schema for jax-cfd Navier-Stokes solver."""

    density: Differentiable[Array[(1,), Float32]] = Field(
        description="Density of the fluid", default=1.0
    )
    inner_steps: int = Field(
        description="Solver sub-iterations per timestep (higher = more accurate pressure solve)",
        default=1,
    )

    @model_validator(mode="after")
    def _check_periodic(self) -> "InputSchema":
        check_periodic(self, "jax-cfd")
        return self


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    """Output schema for jax-cfd Navier-Stokes solver."""


def cfd_fwd(
    v0: jnp.ndarray,
    density: float,
    viscosity: float,
    dt: float,
    steps: int,
    inner_steps: int,
    domain_extent: float,
    **_kwargs: Any,
) -> jax.Array:
    """Compute the final velocity field of the incompressible Navier-Stokes equations.

    Supports 2D and 3D on a fully periodic domain, inferred from
    v0.shape[-1]. The result has the same shape as v0.
    """
    ndim = v0.shape[-1]  # 2 for 2D, 3 for 3D
    if ndim == 2:
        # squeeze the dummy nz=1 axis: (nx, ny, 1, 2) → (nx, ny, 2)
        v0 = v0[:, :, 0, :]
    domain_sizes = (domain_extent,) * ndim

    bc = cfd.boundaries.periodic_boundary_conditions(ndim)

    spatial_shape = v0.shape[:-1]
    grid = cfd.grids.Grid(
        spatial_shape,
        domain=tuple((0.0, L) for L in domain_sizes),
    )

    # --- staggered grid helper -------------------------------------------
    def interp_to_face(v_comp: Any, own_axis: int) -> Any:
        return 0.5 * (v_comp + jnp.roll(v_comp, -1, axis=own_axis))

    def make_grid_vars(v_2d: jnp.ndarray) -> Any:
        return tuple(
            cfd.grids.GridVariable(
                cfd.grids.GridArray(
                    interp_to_face(v_2d[..., i], own_axis=i),
                    grid=grid,
                    offset=tuple(1.0 if j == i else 0.5 for j in range(ndim)),
                ),
                bc,
            )
            for i in range(ndim)
        )

    def interp_to_coloc(v_stag: Any, own_axis: int) -> Any:
        return 0.5 * (v_stag + jnp.roll(v_stag, 1, axis=own_axis))

    # --- NS step function ------------------------------------------------
    # The library's default (van Leer + forward Euler) uses a Lax-Wendroff
    # flux whose time correction assumes pure advection. In incompressible
    # flow the pressure gradient balances advection, so that correction
    # becomes an O(dt) error. Central advection avoids it, but is unstable
    # under forward Euler and under RK2 at CFL ≈ 0.8, hence RK4.
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density,
            viscosity=viscosity,
            dt=dt,
            grid=grid,
            convect=cfd.advection.convect_linear,
            time_stepper=cfd.time_stepping.classic_rk4,
        ),
        steps=inner_steps,
    )

    v_grid = make_grid_vars(v0)

    rollout_fn = cfd.funcutils.trajectory(step_fn, steps)
    _, trajectory = rollout_fn(v_grid)
    stag_components = [trajectory[i].array.data[-1] for i in range(ndim)]

    # --- interpolate back to collocated grid ---
    coloc_components = [
        interp_to_coloc(stag_components[i], own_axis=i) for i in range(ndim)
    ]
    result = jnp.stack(coloc_components, axis=-1)

    if ndim == 2:
        result = result[:, :, None, :]  # (nx, ny, 2) → (nx, ny, 1, 2)
    return result


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled jax-cfd forward pass."""
    return {"result": cfd_fwd(**inputs)}


def _unpack_scalars(d: dict) -> dict:
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in ("density", "viscosity", "dt"):
        if key in d:
            d[key] = float(d[key][0])
    return d


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the jax-cfd Navier-Stokes solver on the given inputs."""
    return apply_jit(_unpack_scalars(inputs.model_dump()))


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """Compute the vector-Jacobian product for jax-cfd via autodiff."""
    return vjp_jit(
        _unpack_scalars(inputs.model_dump()),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Calculate output shape of apply from the shape of its inputs.

    For jax-cfd the output ``result`` always has the same shape as ``v0``.
    """
    raw = abstract_inputs.model_dump()
    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        v0_shape = tuple(v0["shape"])
        v0_dtype = v0["dtype"]
    else:
        arr = jnp.asarray(v0)
        v0_shape = arr.shape
        v0_dtype = str(arr.dtype)
    return {"result": {"shape": v0_shape, "dtype": v0_dtype}}


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
) -> dict:
    """JIT-compiled VJP: compute gradients via JAX autodiff."""
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    grads = vjp_func(cotangent_vector)[0]
    # Scalar physics params (viscosity, dt, etc.) are unpacked to Python floats
    # before the VJP, so their gradients come back as 0D arrays.  Tesseract
    # expects shape (1,) to match the input schema; reshape any 0D grad here.
    return jax.tree.map(jnp.atleast_1d, grads)
