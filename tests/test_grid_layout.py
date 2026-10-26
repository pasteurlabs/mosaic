# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maps between the canonical cell-centred velocity and periodic staggered faces."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mosaic_shared")

from mosaic_shared.problems.navier_stokes_grid.grid_layout import (
    collocated_to_staggered_periodic,
    lift_collocated_to_staggered_periodic,
    staggered_to_collocated_periodic,
)


def _field(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, n, 2))


def test_averages_follow_the_high_face_convention():
    v = _field(8)
    faces = collocated_to_staggered_periodic(v)
    np.testing.assert_allclose(faces[3, 5, 0], 0.5 * (v[3, 5, 0] + v[4, 5, 0]))
    np.testing.assert_allclose(faces[3, 5, 1], 0.5 * (v[3, 5, 1] + v[3, 6, 1]))
    back = staggered_to_collocated_periodic(faces)
    np.testing.assert_allclose(back[3, 5, 0], 0.5 * (faces[2, 5, 0] + faces[3, 5, 0]))


def test_lift_inverts_the_face_to_centre_average_below_nyquist():
    n = 16
    v = _field(n)
    # Remove the Nyquist mode along each component's own axis, which the
    # average maps to zero and the lift therefore drops.
    for i in range(2):
        hat = np.fft.fft(v[..., i], axis=i)
        index = [slice(None)] * 2
        index[i] = n // 2
        hat[tuple(index)] = 0
        v[..., i] = np.real(np.fft.ifft(hat, axis=i))
    lifted = lift_collocated_to_staggered_periodic(v)
    np.testing.assert_allclose(staggered_to_collocated_periodic(lifted), v, atol=1e-12)


def test_lift_drops_the_nyquist_mode():
    n = 16
    x = np.arange(n)
    v = np.zeros((n, n, 2))
    v[..., 0] = np.cos(np.pi * x)[:, None]
    np.testing.assert_allclose(lift_collocated_to_staggered_periodic(v), 0, atol=1e-12)


def test_lift_works_on_jax_arrays():
    jnp = pytest.importorskip("jax.numpy")
    v = _field(8)
    np.testing.assert_allclose(
        np.asarray(lift_collocated_to_staggered_periodic(jnp.asarray(v), xp=jnp)),
        lift_collocated_to_staggered_periodic(v),
        atol=1e-6,
    )
