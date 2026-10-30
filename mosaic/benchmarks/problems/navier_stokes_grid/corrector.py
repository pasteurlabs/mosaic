# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small periodic CNN that corrects a coarse Kolmogorov-flow state.

The network reads the velocity and the phase of the forcing, sin(κy) and
cos(κy), since the force breaks the translation symmetry in y. It returns a
single stream-function channel ψ, and the correction is its spectral curl
(∂ᵧψ, −∂ₓψ): exactly divergence-free, with zero mean and no Nyquist mode.
The readout starts at zero, so an untrained corrector leaves the solver's
state unchanged.

The parameters are a tuple of ``{"w", "b"}`` dicts with HWIO kernels, in plain
JAX, and every solver trains the same architecture from the same
initialisation.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .kolmogorov import AMPLITUDE, DOMAIN_EXTENT, INTERVAL, WAVENUMBER, cell_centres

# Scales of the inputs and output, fixed by the forcing rather than by any
# solver's data. √(A/κ) balances advection against the force at the forcing
# scale, and A·INTERVAL/κ is the stream function of the velocity the force
# adds over one correction interval.
_KAPPA = 2 * np.pi * WAVENUMBER / DOMAIN_EXTENT
VELOCITY_SCALE = float(np.sqrt(AMPLITUDE / _KAPPA))
STREAM_SCALE = float(AMPLITUDE * INTERVAL / _KAPPA)

_INPUT_CHANNELS = 4


def init_corrector(
    seed: int, hidden_channels: int = 8, layers: int = 3, kernel_size: int = 5
) -> tuple[dict[str, jax.Array], ...]:
    """He-initialised hidden layers and a zero readout."""
    channels = (_INPUT_CHANNELS, *([hidden_channels] * (layers - 1)), 1)
    keys = jax.random.split(jax.random.PRNGKey(seed), layers)
    params = []
    for i, (key, cin, cout) in enumerate(
        zip(keys, channels[:-1], channels[1:], strict=True)
    ):
        shape = (kernel_size, kernel_size, cin, cout)
        if i < layers - 1:
            w = jax.random.normal(key, shape, dtype=jnp.float32) * math.sqrt(
                2.0 / (kernel_size * kernel_size * cin)
            )
        else:
            w = jnp.zeros(shape, dtype=jnp.float32)
        params.append({"w": w, "b": jnp.zeros((cout,), dtype=jnp.float32)})
    return tuple(params)


def _features(v: jax.Array, domain_extent: float) -> jax.Array:
    n = v.shape[0]
    phase = 2 * np.pi * WAVENUMBER * cell_centres(n, domain_extent) / domain_extent
    sin = jnp.broadcast_to(jnp.asarray(np.sin(phase), v.dtype)[None, :], (n, n))
    cos = jnp.broadcast_to(jnp.asarray(np.cos(phase), v.dtype)[None, :], (n, n))
    return jnp.concatenate(
        [v[:, :, 0, :] / VELOCITY_SCALE, sin[..., None], cos[..., None]], axis=-1
    )


def stream_function(
    params: Any, v: jax.Array, domain_extent: float = DOMAIN_EXTENT
) -> jax.Array:
    """The network's stream function ψ for a (N, N, 1, 2) state, shape (N, N)."""
    v = jnp.asarray(v, dtype=jnp.float32)
    x = _features(v, domain_extent)[None]
    for i, layer in enumerate(params):
        pad = layer["w"].shape[0] // 2
        x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
        x = (
            jax.lax.conv_general_dilated(
                x,
                layer["w"],
                (1, 1),
                "VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + layer["b"]
        )
        if i < len(params) - 1:
            x = jax.nn.gelu(x)
    return x[0, :, :, 0] * STREAM_SCALE


def curl(psi: jax.Array, domain_extent: float = DOMAIN_EXTENT) -> jax.Array:
    """Spectral (∂ᵧψ, −∂ₓψ) of an (N, N) stream function, shape (N, N, 1, 2).

    The Nyquist modes are zeroed on even grids, where a real field has no
    derivative there, and so is the mean.
    """
    n = psi.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=domain_extent / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    keep = (k != 0) | (np.arange(n) == 0)
    mask = jnp.asarray(np.outer(keep, keep))
    kx, ky = np.meshgrid(k, k, indexing="ij")
    hat = jnp.fft.fft2(psi) * mask
    u = jnp.real(jnp.fft.ifft2(1j * ky * hat))
    v = jnp.real(jnp.fft.ifft2(-1j * kx * hat))
    return jnp.stack([u, v], axis=-1)[:, :, None, :].astype(psi.dtype)


def apply_corrector(
    params: Any, v: jax.Array, domain_extent: float = DOMAIN_EXTENT
) -> jax.Array:
    """The corrected state v + curl ψ."""
    v = jnp.asarray(v, dtype=jnp.float32)
    return v + curl(stream_function(params, v, domain_extent), domain_extent)
