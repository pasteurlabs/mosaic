# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical state geometry and schema validation for buoyancy flow."""

import math

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


def _original_interacting_plumes_ic(N: int = 64, seed: int = 0) -> np.ndarray:
    """Two unequal warm plumes at rest, averaged over each physical grid cell.

    Normalized centers are near (0.36, 0.18) and (0.64, 0.25), with Gaussian
    standard deviations 0.06 and 0.075 and peak amplitudes 1.0 and 0.8.
    Small seeded position/width changes give distinct members of the same
    family. Analytic cell averages preserve the initial mass under refinement.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.36, 0.18], [0.64, 0.25]]) + rng.uniform(
        -0.015, 0.015, size=(2, 2)
    )
    widths = np.array([0.06, 0.075]) * rng.uniform(0.95, 1.05, size=2)
    edges = np.arange(N + 1, dtype=np.float64) / N
    scalar = np.zeros((N, N), dtype=np.float64)
    for center, width, amplitude in zip(centers, widths, (1.0, 0.8), strict=True):
        averages = []
        for coordinate in center:
            scaled_edges = (edges - coordinate) / (math.sqrt(2) * width)
            primitive = np.fromiter(
                (math.erf(value) for value in scaled_edges), dtype=np.float64
            )
            averages.append(N * width * math.sqrt(math.pi / 2) * np.diff(primitive))
        scalar += amplitude * averages[0][:, None] * averages[1][None, :]
    state = np.zeros((N, N, 3), dtype=np.float32)
    state[..., 2] = scalar
    return state


@pytest.mark.parametrize("seed", [0, 2, 100])
def test_default_plumes_match_original_algorithm_bitwise(seed):
    expected = _original_interacting_plumes_ic(64, seed)
    np.testing.assert_array_equal(make_interacting_plumes_ic(64, seed), expected)
    np.testing.assert_array_equal(
        make_interacting_plumes_ic(
            64, seed, position_jitter=0.015, width_jitter=0.05, amplitude_jitter=0.0
        ),
        expected,
    )


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_broader_plumes_preserve_refinement_and_determinism(seed):
    options = {
        "seed": seed,
        "position_jitter": 0.04,
        "width_jitter": 0.15,
        "amplitude_jitter": 0.15,
    }
    coarse = make_interacting_plumes_ic(64, **options)
    fine = make_interacting_plumes_ic(128, **options)
    np.testing.assert_array_equal(coarse, make_interacting_plumes_ic(64, **options))
    assert not np.array_equal(coarse, make_interacting_plumes_ic(64, seed=seed))
    assert np.isfinite(coarse).all()
    assert np.min(coarse[..., 2]) >= 0
    np.testing.assert_array_equal(coarse[..., :2], 0)
    np.testing.assert_allclose(coarse, restrict_state(fine, 2), rtol=2e-7, atol=1e-7)
    np.testing.assert_allclose(
        np.mean(coarse[..., 2], dtype=np.float64),
        np.mean(fine[..., 2], dtype=np.float64),
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    "name", ["position_jitter", "width_jitter", "amplitude_jitter"]
)
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_plume_variations_reject_negative_or_nonfinite(name, value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_interacting_plumes_ic(8, **{name: value})


@pytest.mark.parametrize("name", ["width_jitter", "amplitude_jitter"])
@pytest.mark.parametrize("value", [1.0, 1.1])
def test_relative_plume_variations_keep_widths_and_amplitudes_positive(name, value):
    with pytest.raises(ValueError, match="less than one"):
        make_interacting_plumes_ic(8, **{name: value})


def test_plume_positions_must_remain_inside_domain():
    with pytest.raises(ValueError, match="inside the unit box"):
        make_interacting_plumes_ic(8, seed=0, position_jitter=1000)


def test_zero_plume_variations_remove_seed_dependence():
    options = {"position_jitter": 0.0, "width_jitter": 0.0, "amplitude_jitter": 0.0}
    np.testing.assert_array_equal(
        make_interacting_plumes_ic(8, seed=0, **options),
        make_interacting_plumes_ic(8, seed=123, **options),
    )


def test_plume_physics_wrapper_forwards_variation_parameters(monkeypatch):
    from mosaic.benchmarks.problems.buoyancy_grid import physics

    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return np.zeros((8, 8, 3), dtype=np.float32)

    monkeypatch.setattr(physics, "make_interacting_plumes_ic", record)
    physics.interacting_plumes(
        N=8, seed=2, position_jitter=0.04, width_jitter=0.15, amplitude_jitter=0.15
    )
    assert calls == [
        {
            "N": 8,
            "seed": 2,
            "position_jitter": 0.04,
            "width_jitter": 0.15,
            "amplitude_jitter": 0.15,
        }
    ]
