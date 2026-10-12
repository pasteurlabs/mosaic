# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions and canonical inputs for the buoyancy benchmark."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.buoyancy_grid import make_plume_ic

from mosaic.benchmarks.core.config import SolverSpec


def plume(N: int = 24, L: float = 1.0, seed: int = 0, **_: Any) -> jax.Array:
    """A warm Gaussian blob at rest, with position and width set by seed."""
    return jnp.asarray(make_plume_ic(N=N, seed=seed))


def diffusion_ic(N: int = 24, **_: Any) -> jax.Array:
    """Neumann cosine mode for an analytic scalar-diffusion check."""
    x = (jnp.arange(N) + 0.5) / N
    scalar = 0.5 + 0.25 * jnp.cos(jnp.pi * x[:, None]) * jnp.cos(jnp.pi * x[None, :])
    return jnp.zeros((N, N, 3), dtype=jnp.float32).at[..., 2].set(scalar)


def diffusion_reference(
    ic: jax.Array, t: float, L: float, diffusivity: float, **_: Any
) -> jax.Array:
    """Exact diffusion of the cosine mode with zero buoyancy and velocity."""
    decay = jnp.exp(-2 * (jnp.pi / L) ** 2 * diffusivity * t)
    return ic.at[..., 2].set(0.5 + (ic[..., 2] - 0.5) * decay)


diffusion_reference.supported_ics = {"diffusion"}


def scalar_field(state: np.ndarray) -> np.ndarray:
    """Extract the cell-centered scalar for initial-condition plots."""
    return state[..., 2]


def make_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    nu: float = 0.001,
    diffusivity: float = 0.001,
    buoyancy: float = 1.0,
    dt: float = 0.04,
    steps: int = 2,
    domain_extent: float = 1.0,
    **_: Any,
) -> dict:
    """Build the common input contract without changing the physical budget."""
    return {
        "state0": ic,
        "viscosity": jnp.asarray([nu], dtype=jnp.float32),
        "diffusivity": jnp.asarray([diffusivity], dtype=jnp.float32),
        "buoyancy": jnp.asarray([buoyancy], dtype=jnp.float32),
        "dt": jnp.asarray([dt], dtype=jnp.float32),
        "steps": steps,
        "domain_extent": domain_extent,
        **spec.input_overrides,
    }
