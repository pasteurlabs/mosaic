# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-NS spectral Poisson solve against NumPy.

Covers both FFT paths: native tile FFTs for power-of-two ``N`` and Bluestein
for every other ``N``. The pressure-correction kernels divide the solver's
output by ``N**dim``, so both paths must return the same unnormalized scale.

Needs ``warp`` (not a harness dependency, so the module is skipped where it is
missing); runs on CPU when no GPU is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

_API = (
    Path(__file__).parents[1]
    / "mosaic/tesseracts/navier-stokes-grid/warp-ns/tesseract_api.py"
)


@pytest.fixture(scope="module")
def api():
    """Load the Warp-NS tesseract module straight from source."""
    spec = importlib.util.spec_from_file_location("warp_ns_tesseract_api", _API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_complex(values: np.ndarray, device: str) -> wp.array:
    packed = np.stack([values, np.zeros_like(values)], axis=-1).astype(np.float32)
    return wp.array(packed, dtype=wp.vec2f, device=device, requires_grad=True)


@pytest.mark.parametrize(
    ("dim", "n"),
    [(2, 16), (2, 20), (2, 48), (3, 8), (3, 12)],
)
def test_spectral_poisson_matches_numpy(api, dim: int, n: int) -> None:
    """Solve and VJP match an inverse-Laplacian built from ``np.fft``."""
    device = api._warp_device()
    length = 2 * np.pi
    if dim == 2:
        solve, inv_lambda = api._spectral_poisson_2d_core, api._inv_lambda_2d
    else:
        solve, inv_lambda = api._spectral_poisson_3d_core, api._inv_lambda_3d
    inv_lambda = inv_lambda(n, length, device).numpy()

    def expected(rhs: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(np.fft.fftn(rhs) * inv_lambda).real

    rng = np.random.default_rng(0)
    rhs = rng.normal(size=(n,) * dim)
    rhs_c = _to_complex(rhs, device)
    tape = wp.Tape()
    with tape:
        p_c = solve(rhs_c, length, device)
    pressure = p_c.numpy()[..., 0] / n**dim
    np.testing.assert_allclose(pressure, expected(rhs), rtol=0, atol=1e-5)

    # The operator is real and symmetric, so its adjoint is itself.
    cotangent = rng.normal(size=(n,) * dim)
    tape.backward(grads={p_c: _to_complex(cotangent / n**dim, device)})
    grad = rhs_c.grad.numpy()[..., 0]
    np.testing.assert_allclose(grad, expected(cotangent), rtol=0, atol=1e-5)
