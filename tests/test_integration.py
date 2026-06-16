# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that build and run a real Tesseract container.

These tests build a real Docker image, so they only run where Docker (and the
``tesseract`` CLI) are available; on hosts without a working Docker daemon the
whole module is skipped rather than failing.

The Exponax solver is used because it is small, fast, JAX-only, and
exercises the full apply / VJP path without GPU requirements.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _docker_available() -> bool:
    """True if both the ``tesseract`` CLI and a responsive Docker daemon exist."""
    if shutil.which("tesseract") is None or shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon and/or tesseract CLI unavailable",
)

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
        # tesseract_core's VJP endpoint takes ``(inputs, vjp_inputs,
        # vjp_outputs, cotangent_vector)`` — the two ``vjp_*`` lists pick
        # which input/output pair the gradient is computed against. Lists
        # rather than sets because the SDK serialises them over JSON.
        grad = t.vector_jacobian_product({}, ["v0"], ["result"], cotangent)
        assert "v0" in grad, f"Missing 'v0' gradient. Got: {list(grad.keys())}"
        g = np.asarray(grad["v0"])
        assert g.size > 0, "gradient is empty"
        assert np.all(np.isfinite(g)), "gradient contains NaN or Inf"
