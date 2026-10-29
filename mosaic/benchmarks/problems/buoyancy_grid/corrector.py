# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-aware velocity correction with optional temperature correction."""

import math
from typing import Any

import jax
import jax.numpy as jnp

from mosaic.mosaic_shared.problems.buoyancy_grid import restrict_state  # noqa: F401


def init_corrector(
    seed: int = 0, hidden_channels: int = 8, correct_temperature: bool = False
) -> tuple[dict[str, jax.Array], ...]:
    """Zero readout makes all training arms start from the uncorrected solver."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    channels = (3, hidden_channels, hidden_channels, 2 if correct_temperature else 1)
    return tuple(
        {
            "w": (
                jax.random.normal(key, (3, 3, cin, cout), dtype=jnp.float32)
                * math.sqrt(2 / (9 * cin))
                if i < 2
                else jnp.zeros((3, 3, cin, cout), dtype=jnp.float32)
            ),
            "b": jnp.zeros(cout, dtype=jnp.float32),
        }
        for i, (key, cin, cout) in enumerate(
            zip(keys, channels[:-1], channels[1:], strict=True)
        )
    )


def apply_corrector(params: Any, state: jax.Array, scales: jax.Array) -> jax.Array:
    """Curl a zero-boundary streamfunction into divergence-free face updates."""
    # Eager Tesseract calls return NumPy arrays; traced calls return JAX values.
    state = jnp.asarray(state, dtype=jnp.float32)
    scales = jnp.asarray(scales, dtype=state.dtype)
    x = (state / scales)[None]
    for i, layer in enumerate(params):
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
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
            x = jax.nn.tanh(x)
    # psi is at grid corners; its constant (zero) wall value enforces zero
    # normal correction. Mixed differences commute, so divergence is zero.
    psi = jnp.pad(x[0, :-1, :-1, 0], ((1, 1), (1, 1))) * scales[0]
    delta = jnp.stack(
        (jnp.diff(psi, axis=1)[1:], -jnp.diff(psi, axis=0)[:, 1:]), axis=-1
    )
    corrected = state.at[..., :2].add(delta)
    if x.shape[-1] == 2:
        corrected = corrected.at[..., 2].add(x[0, ..., 1] * scales[2])
    return corrected


def normalized_loss(
    state: jax.Array, target: jax.Array, scales: jax.Array
) -> jax.Array:
    """Give velocity and buoyancy scalar equal weight after training-set scaling."""
    error = ((state - target) / scales) ** 2
    return 0.5 * (jnp.mean(error[..., :2]) + jnp.mean(error[..., 2]))
