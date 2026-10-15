# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PhiFlow ns-grid tesseract on the DFG 2D-1 channel.

The tesseract module imports phiflow, so every test here runs against the
``phiflow_navier_stokes_grid:latest`` image built from this checkout
(``mosaic build -p ns-grid -s PhiFlow``) and is skipped where that image or
Docker is missing. It is never built here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

from .helpers import channel_inputs, image_available, unsupported_bcs

_IMAGE = "phiflow_navier_stokes_grid:latest"


pytestmark = pytest.mark.skipif(
    not image_available(_IMAGE), reason=f"{_IMAGE} not built"
)


@pytest.fixture(scope="module")
def phiflow():
    from tesseract_core import Tesseract

    with Tesseract.from_image(_IMAGE) as t:
        yield t


def test_cylinder_res10(phiflow):
    """A res-10 run: positive drag within 10% of the reference, steady."""
    out = phiflow.apply(channel_inputs(10))
    force = np.asarray(out["body_force"])
    assert force.shape == (cylinder.time_step(10)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0
    metrics = cylinder.dfg_metrics(out["result"], out["pressure"], force, 10)
    # The staircase cylinder underestimates the drag at first order: about
    # -7% at 10 cells per diameter, -4% at 20.
    assert metrics["error"] < 0.1
    assert metrics["cd_drift"] < 1e-3
    assert abs(metrics["cd_cv"] - metrics["cd"]) / metrics["cd"] < 0.02


def test_drag_gradient_matches_fd(phiflow):
    """d(mean F_x)/d(inflow_profile) from the VJP against a central difference.

    Over drag_opt's horizon the agreement is 2e-4. Over a shorter, still
    transient horizon (t_end = 1) it is 6e-3, from the CG tolerance.
    """
    inputs = channel_inputs(10, t_end=4.0)
    steps = inputs["steps"]
    window = cylinder.window_size(steps)

    def mean_drag(profile: np.ndarray) -> float:
        force = phiflow.apply({**inputs, "inflow_profile": profile.astype(np.float32)})
        return float(np.asarray(force["body_force"], np.float64)[-window:, 0].mean())

    cotangent = np.zeros((steps, 2), np.float32)
    cotangent[-window:, 0] = 1.0 / window
    grad = phiflow.vector_jacobian_product(
        inputs, ["inflow_profile"], ["body_force"], {"body_force": cotangent}
    )["inflow_profile"]

    profile = np.asarray(inputs["inflow_profile"], np.float64)
    direction = np.sin(np.linspace(0, 3 * np.pi, profile.size)) * 0.05
    h = 0.1
    fd = (mean_drag(profile + h * direction) - mean_drag(profile - h * direction)) / (
        2 * h
    )
    assert np.dot(grad, direction) == pytest.approx(fd, rel=2e-3)


@unsupported_bcs
def test_rejects_unsupported_bcs(phiflow, change):
    with pytest.raises(Exception, match="phiflow supports"):
        phiflow.apply({**channel_inputs(10, t_end=0.1), **change})
