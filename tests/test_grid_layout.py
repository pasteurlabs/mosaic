# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mosaic_shared.problems.navier_stokes_grid import (
    collocated_to_staggered_periodic,
    lift_collocated_to_staggered_periodic,
    staggered_to_collocated_periodic,
)


@pytest.mark.parametrize("shape", [(8, 10, 1, 2), (6, 8, 10, 3)])
def test_periodic_layout_conversions_follow_high_face_convention(shape):
    velocity = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    staggered = collocated_to_staggered_periodic(velocity)
    expected_staggered = np.stack(
        [
            0.5
            * (
                velocity[..., component]
                + np.roll(velocity[..., component], -1, axis=component)
            )
            for component in range(shape[-1])
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(staggered, expected_staggered)

    collocated = staggered_to_collocated_periodic(staggered)
    expected_collocated = np.stack(
        [
            0.5
            * (
                staggered[..., component]
                + np.roll(staggered[..., component], 1, axis=component)
            )
            for component in range(shape[-1])
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(collocated, expected_collocated)


@pytest.mark.parametrize("shape", [(16, 12, 1, 2), (8, 10, 12, 3)])
def test_periodic_lift_is_right_inverse_on_representable_modes(shape):
    spatial = np.indices(shape[:-1], dtype=np.float32)
    collocated = np.stack(
        [
            np.cos(2 * np.pi * (component + 1) * spatial[component] / shape[component])
            for component in range(shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)

    lifted = lift_collocated_to_staggered_periodic(collocated)
    reconstructed = staggered_to_collocated_periodic(lifted)

    np.testing.assert_allclose(reconstructed, collocated, rtol=2e-5, atol=2e-5)
    assert lifted.dtype == collocated.dtype


def test_periodic_lift_drops_unrepresentable_nyquist_mode():
    sign = (-1.0) ** np.arange(16, dtype=np.float32)
    correction = np.zeros((16, 8, 1, 2), dtype=np.float32)
    correction[..., 0] = sign[:, None, None]

    lifted = lift_collocated_to_staggered_periodic(correction)
    reconstructed = staggered_to_collocated_periodic(lifted)

    np.testing.assert_allclose(reconstructed[..., 0], 0.0, atol=1e-6)


def test_periodic_lift_enforces_gain_bound():
    n = 128
    mode = np.cos(2 * np.pi * 63 * np.arange(n, dtype=np.float32) / n)
    correction = np.zeros((n, 8, 1, 2), dtype=np.float32)
    correction[..., 0] = mode[:, None, None]

    bounded = lift_collocated_to_staggered_periodic(correction)
    recovered = staggered_to_collocated_periodic(bounded)
    np.testing.assert_allclose(recovered[..., 0], 0.0, atol=2e-5)

    expanded = lift_collocated_to_staggered_periodic(correction, max_gain=64.0)
    recovered = staggered_to_collocated_periodic(expanded)
    np.testing.assert_allclose(recovered, correction, rtol=2e-5, atol=2e-5)


def test_periodic_layout_helpers_are_jax_differentiable():
    velocity = jnp.arange(8 * 8 * 2, dtype=jnp.float32).reshape(8, 8, 1, 2)

    def loss(value):
        faces = collocated_to_staggered_periodic(value, xp=jnp)
        collocated = staggered_to_collocated_periodic(faces, xp=jnp)
        lifted = lift_collocated_to_staggered_periodic(collocated, xp=jnp)
        return jnp.mean(lifted**2)

    gradient = jax.grad(loss)(velocity)
    assert gradient.shape == velocity.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
