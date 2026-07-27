# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fixed-task XLB 3D full-field surrogate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tesseract_core.runtime")
pytest.importorskip("mosaic_shared")

_API_PATH = (
    Path(__file__).parents[1]
    / "mosaic"
    / "tesseracts"
    / "navier-stokes-grid"
    / "xlb-3d-surrogate"
    / "tesseract_api.py"
)
_DATA_PATH = _API_PATH.with_name("training_data.py")
_SPEC = importlib.util.spec_from_file_location("xlb_3d_surrogate_api", _API_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_API = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_API_PATH.parent))
try:
    _SPEC.loader.exec_module(_API)
finally:
    sys.path.pop(0)
_DATA_SPEC = importlib.util.spec_from_file_location(
    "xlb_3d_surrogate_training_data",
    _DATA_PATH,
)
assert _DATA_SPEC is not None and _DATA_SPEC.loader is not None
_DATA = importlib.util.module_from_spec(_DATA_SPEC)
_DATA_SPEC.loader.exec_module(_DATA)


def _weights(width: int = 4, modes: int = 2):
    rng = np.random.default_rng(2026)
    weights = {
        "input_scale": jnp.asarray(0.2, dtype=jnp.float32),
        "correction_scale": jnp.full((3,), 0.1, dtype=jnp.float32),
        "w_lift": jnp.asarray(
            rng.normal(scale=0.05, size=(3, width)),
            dtype=jnp.float32,
        ),
        "w_out": jnp.asarray(
            rng.normal(scale=0.05, size=(width, 3)),
            dtype=jnp.float32,
        ),
        "w_local_0": jnp.eye(width, dtype=jnp.float32),
    }
    shape = (modes, modes, modes, width, width)
    for quadrant in range(4):
        weights[f"spec_0_{quadrant}_real"] = jnp.asarray(
            rng.normal(scale=0.01, size=shape),
            dtype=jnp.float32,
        )
        weights[f"spec_0_{quadrant}_imag"] = jnp.asarray(
            rng.normal(scale=0.01, size=shape),
            dtype=jnp.float32,
        )
    return weights


def test_full_field_shape_and_zero_state(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    result = _API._surrogate_forward(initial, _weights())
    assert result.shape == initial.shape
    assert np.all(np.isfinite(np.asarray(result)))
    assert np.allclose(np.asarray(result), 0.0, atol=1e-7)


def test_full_field_vjp_is_finite(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    weights = _weights()
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    initial = initial.at[1, 2, 3, 0].set(0.1)

    def field_sum(value):
        return jnp.sum(_API._surrogate_forward(value, weights) ** 2)

    gradient = jax.grad(field_sum)(initial)
    assert gradient.shape == initial.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_forward_reuses_one_macro_step_operator(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    weights = _weights()
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    initial = initial.at[1, 2, 3, 0].set(0.1)
    expected = initial[None]
    for _ in range(3):
        expected = _API._one_step(expected, weights)
    result = _API._surrogate_forward(initial, weights)
    assert np.allclose(np.asarray(result), np.asarray(expected[0]), atol=1e-7)


def test_training_distribution_is_reproducible_and_full_field():
    first = _DATA.make_inputs(10, 123)
    second = _DATA.make_inputs(10, 123)

    fields, amplitudes, families = first
    assert fields.shape == (10, _API._N, _API._N, _API._N, 3)
    assert amplitudes.shape == (10,)
    assert families.shape == (10,)
    assert np.all(np.isfinite(fields))
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
