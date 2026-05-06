"""Integration tests that build and run a real Tesseract container.

These tests require Docker and are skipped by default.  Run explicitly with::

    pytest -m integration

The Exponax solver is used because it is small, fast, JAX-only, and
exercises the full apply / VJP path without GPU requirements.
"""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.integration

_SOLVER_DIR = "mosaic/tesseracts/navier-stokes-grid/exponax"
_IMAGE = "exponax_navier_stokes_grid:latest"


@pytest.fixture(scope="module")
def built_solver():
    """Build the Exponax tesseract once for all tests in this module."""
    result = subprocess.run(
        ["tesseract", "build", _SOLVER_DIR],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.skip(f"tesseract build failed:\n{result.stderr[-500:]}")
    return _IMAGE


def test_apply_returns_finite_output(built_solver):
    """apply() with default inputs returns a finite result array."""
    from tesseract_core import Tesseract

    with Tesseract.from_image(built_solver) as t:
        out = t.apply({})
        assert "result" in out, f"Missing 'result' key. Got: {list(out.keys())}"
        import numpy as np

        arr = np.asarray(out["result"])
        assert arr.size > 0, "result array is empty"
        assert np.all(np.isfinite(arr)), "result contains NaN or Inf"


def test_vjp_returns_finite_gradient(built_solver):
    """vector_jacobian_product() returns a finite gradient for v0."""
    from tesseract_core import Tesseract

    with Tesseract.from_image(built_solver) as t:
        # Get output shape from a forward pass
        out = t.apply({})
        import numpy as np

        result = np.asarray(out["result"])
        cotangent = {"result": np.ones_like(result)}
        grad = t.vector_jacobian_product({}, cotangent)
        assert "v0" in grad, f"Missing 'v0' gradient. Got: {list(grad.keys())}"
        g = np.asarray(grad["v0"])
        assert g.size > 0, "gradient is empty"
        assert np.all(np.isfinite(g)), "gradient contains NaN or Inf"
