# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal pseudo-spectral viscous-diffusion solver for CI testing."""

from typing import Any

import jax
import jax.numpy as jnp
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable


class InputSchema(make_differentiable(_CanonicalInputSchema, ["v0", "viscosity"])):
    """Test spectral solver inputs; v0 and viscosity carry gradients."""


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    """Test spectral solver outputs."""


def _spectral_step(
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
) -> jnp.ndarray:
    """Evolve v(x, 0) -> v(x, dt*steps) under 2D viscous diffusion.

    Pure spectral update on a periodic [0, L]^2 grid:
        v_hat(k, t) = v_hat(k, 0) * exp(-nu * |k|^2 * t)
    """
    N = v0.shape[0]
    k = jnp.fft.fftfreq(N, d=domain_extent / N) * 2 * jnp.pi
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    decay = jnp.exp(-viscosity * (kx**2 + ky**2) * dt * steps)
    v_hat = jnp.fft.fft2(v0[:, :, 0, :], axes=(0, 1))  # (N, N, 2)
    v_new = jnp.fft.ifft2(v_hat * decay[..., None], axes=(0, 1)).real
    return v_new[:, :, None, :].astype(jnp.float32)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the spectral viscous-diffusion forward solver."""
    d = inputs.model_dump()
    result = _spectral_step(
        v0=d["v0"],
        viscosity=float(d["viscosity"][0]),
        dt=float(d["dt"][0]),
        steps=int(d["steps"]),
        domain_extent=float(d["domain_extent"]),
    )
    return {"result": result}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Calculate output shape from input shapes."""
    v0_info = abstract_inputs.v0
    shape = tuple(v0_info["shape"]) if isinstance(v0_info, dict) else v0_info.shape
    return {"result": {"shape": shape, "dtype": "float32"}}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode VJP w.r.t. v0 and viscosity."""
    d = inputs.model_dump()
    v0 = jnp.asarray(d["v0"])
    viscosity = jnp.asarray(d["viscosity"])
    dt = float(d["dt"][0])
    steps = int(d["steps"])
    L = float(d["domain_extent"])

    def fwd(v0: jnp.ndarray, viscosity: jnp.ndarray) -> jnp.ndarray:
        return _spectral_step(v0, viscosity[0], dt, steps, L)

    _, vjp_fn = jax.vjp(fwd, v0, viscosity)
    g_v0, g_visc = vjp_fn(cotangent_vector["result"])

    out: dict[str, Any] = {}
    if "v0" in vjp_inputs:
        out["v0"] = g_v0
    if "viscosity" in vjp_inputs:
        out["viscosity"] = jnp.atleast_1d(g_visc)
    return out
