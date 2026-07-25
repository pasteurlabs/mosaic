# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the task-specific XLB full-field surrogate."""

from __future__ import annotations

import importlib.util
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
    / "xlb-surrogate"
    / "tesseract_api.py"
)
_SPEC = importlib.util.spec_from_file_location("xlb_surrogate_api", _API_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_API = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_API)


def _weights(rank: int = 4, hidden: int = 8) -> dict[str, jax.Array]:
    """Construct a small deterministic parameter tree for shape/VJP tests."""
    rng = np.random.default_rng(2026)

    def normal(shape):
        return jnp.asarray(rng.normal(scale=0.01, size=shape), dtype=jnp.float32)

    n_field = _API._N * _API._N * 4
    weights = {
        "profile_mean": jnp.full((_API._N,), 0.5, dtype=jnp.float32),
        "profile_scale": jnp.ones((_API._N,), dtype=jnp.float32),
        "field_mean": jnp.zeros((n_field,), dtype=jnp.float32),
        "field_scale": jnp.ones((4,), dtype=jnp.float32),
        "field_basis": normal((rank, n_field)),
        "w_in": normal((_API._N, hidden)),
        "b_in": normal((hidden,)),
        "w_out": normal((hidden, rank)),
        "b_out": normal((rank,)),
    }
    for idx in range(3):
        weights[f"w_res_{idx}"] = normal((hidden, hidden))
        weights[f"b_res_{idx}"] = normal((hidden,))
    return weights


def test_full_field_and_drag_shapes():
    output = _API._surrogate_forward(
        jnp.full((_API._N,), 0.5, dtype=jnp.float32),
        _weights(),
    )
    assert output["result"].shape == (_API._N, _API._N, 1, 2)
    assert output["drag"].shape == (1,)
    assert np.all(np.isfinite(np.asarray(output["result"])))
    assert np.all(np.isfinite(np.asarray(output["drag"])))


def test_drag_has_finite_profile_gradient():
    weights = _weights()

    def drag_sum(profile):
        return jnp.sum(_API._surrogate_forward(profile, weights)["drag"])

    gradient = jax.grad(drag_sum)(jnp.full((_API._N,), 0.5, dtype=jnp.float32))
    assert gradient.shape == (_API._N,)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert gradient[0] == 0.0
    assert gradient[-1] == 0.0
