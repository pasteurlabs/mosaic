# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small wall-aware velocity corrector; scalar evolves only through the solver."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.buoyancy_grid import restrict_state  # noqa: F401


def init_corrector(
    seed: int = 0, hidden_channels: int = 8
) -> tuple[dict[str, jax.Array], ...]:
    """Zero readout makes all training arms start from the uncorrected solver."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    channels = (3, hidden_channels, hidden_channels, 1)
    return tuple(
        {
            "w": (
                jax.random.normal(key, (3, 3, cin, cout)) * np.sqrt(2 / (9 * cin))
                if i < 2
                else jnp.zeros((3, 3, cin, cout))
            ),
            "b": jnp.zeros(cout),
        }
        for i, (key, cin, cout) in enumerate(
            zip(keys, channels[:-1], channels[1:], strict=True)
        )
    )


def apply_corrector(params: Any, state: jax.Array, scales: jax.Array) -> jax.Array:
    """Curl a zero-boundary streamfunction into divergence-free face updates."""
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
    return state.at[..., :2].add(delta)


def normalized_loss(
    state: jax.Array, target: jax.Array, scales: jax.Array
) -> jax.Array:
    """Give velocity and buoyancy scalar equal weight after training-set scaling."""
    error = ((state - target) / scales) ** 2
    return 0.5 * (jnp.mean(error[..., :2]) + jnp.mean(error[..., 2]))
