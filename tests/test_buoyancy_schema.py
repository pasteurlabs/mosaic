# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical state geometry and schema validation for buoyancy flow."""

import numpy as np
import pytest
from mosaic_shared.problems.buoyancy_grid import (
    InputSchema,
    make_interacting_plumes_ic,
    make_plume_ic,
    restrict_state,
)


def test_plume_matches_across_resolutions():
    coarse = make_plume_ic(8, seed=7)
    fine = make_plume_ic(16, seed=7)
    assert coarse.dtype == np.float32
    assert coarse.shape == (8, 8, 3)
    assert np.count_nonzero(coarse[..., :2]) == 0
    # Same physical plume sampled at different cell centers.
    assert np.mean(np.abs(coarse[..., 2] - restrict_state(fine, 2)[..., 2])) < 0.01


def test_face_restriction_preserves_discrete_divergence():
    rng = np.random.default_rng(9)
    state = rng.normal(size=(12, 12, 3)).astype(np.float32)
    state[-1, :, 0] = 0
    state[:, -1, 1] = 0

    def divergence(value):
        n = value.shape[0]
        ux = np.pad(value[:-1, :, 0], ((1, 1), (0, 0)))
        uy = np.pad(value[:, :-1, 1], ((0, 0), (1, 1)))
        return n * (np.diff(ux, axis=0) + np.diff(uy, axis=1))

    coarse = restrict_state(state, 3)
    expected = divergence(state).reshape(4, 3, 4, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(divergence(coarse), expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "field,value",
    [("viscosity", -1.0), ("diffusivity", -1.0), ("dt", 0.0), ("buoyancy", np.nan)],
)
def test_invalid_physics_rejected(field, value):
    with pytest.raises(ValueError):
        InputSchema(**{field: np.array([value], np.float32)})


def test_non_square_grid_rejected():
    with pytest.raises(ValueError, match="square"):
        InputSchema(state0=np.zeros((8, 9, 3), np.float32))


@pytest.mark.parametrize("seed", [0, 1, 100])
def test_interacting_plumes_preserve_initial_state_under_refinement(seed):
    coarse = make_interacting_plumes_ic(64, seed=seed)
    fine = make_interacting_plumes_ic(128, seed=seed)
    assert coarse.shape == (64, 64, 3)
    assert coarse.dtype == np.float32
    assert np.isfinite(coarse).all()
    np.testing.assert_array_equal(coarse[..., :2], 0)
    assert np.min(coarse[..., 2]) >= 0
    assert 0.9 < np.max(coarse[:32, :, 2]) < 1.1
    assert 0.7 < np.max(coarse[32:, :, 2]) < 0.9
    assert 0.045 < np.mean(coarse[..., 2]) < 0.06
    np.testing.assert_allclose(coarse, restrict_state(fine, 2), rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        np.mean(coarse[..., 2], dtype=np.float64),
        np.mean(fine[..., 2], dtype=np.float64),
        rtol=1e-7,
    )


def test_interacting_plumes_are_seeded_without_touching_global_rng():
    before = np.random.get_state()  # noqa: NPY002 - explicitly inspect global RNG
    first = make_interacting_plumes_ic(64, seed=7)
    np.testing.assert_array_equal(first, make_interacting_plumes_ic(64, seed=7))
    assert not np.array_equal(first, make_interacting_plumes_ic(64, seed=8))
    after = np.random.get_state()  # noqa: NPY002 - check it was not modified
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
