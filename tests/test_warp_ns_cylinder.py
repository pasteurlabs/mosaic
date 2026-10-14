# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-NS on the DFG 2D-1 channel.

Needs ``warp`` (not a harness dependency, so the module is skipped where it is
missing); runs on CPU when no GPU is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

pytest.importorskip("warp")

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


def _channel_inputs(resolution: int, t_end: float = cylinder.T_END) -> dict:
    inputs = cylinder.channel_inputs(
        cylinder.inflow_profile(resolution), resolution, t_end
    )
    return {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()}


def test_cylinder_res10(api):
    """A res-10 run: positive drag within 5% of the reference, steady."""
    out = api.apply(api.InputSchema(**_channel_inputs(10)))
    force = np.asarray(out.body_force)
    assert force.shape == (cylinder.time_step(10)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0
    metrics = cylinder.dfg_metrics(out.result, out.pressure, force, 10)
    # The staircase cylinder underestimates the drag at first order: -2.8% at
    # 10 cells per diameter, -1.8% at 20.
    assert metrics["error"] < 0.05
    assert metrics["cd_drift"] < 1e-3
    assert abs(metrics["cd_cv"] - metrics["cd"]) / metrics["cd"] < 0.01


def test_drag_gradient_matches_fd(api):
    """d(mean F_x)/d(inflow_profile) from the VJP against a central difference.

    Over drag_opt's horizon the agreement is about 1e-4.
    """
    inputs = _channel_inputs(10, t_end=4.0)
    steps = inputs["steps"]
    window = cylinder.window_size(steps)

    def mean_drag(profile: np.ndarray) -> float:
        out = api.apply(
            api.InputSchema(**{**inputs, "inflow_profile": profile.astype(np.float32)})
        )
        return float(np.asarray(out.body_force, np.float64)[-window:, 0].mean())

    cotangent = np.zeros((steps, 2), np.float32)
    cotangent[-window:, 0] = 1.0 / window
    grad = api.vector_jacobian_product(
        api.InputSchema(**inputs),
        {"inflow_profile"},
        {"body_force"},
        {"body_force": cotangent},
    )["inflow_profile"]

    profile = np.asarray(inputs["inflow_profile"], np.float64)
    direction = np.sin(np.linspace(0, 3 * np.pi, profile.size)) * 0.05
    h = 0.1
    fd = (mean_drag(profile + h * direction) - mean_drag(profile - h * direction)) / (
        2 * h
    )
    assert np.dot(grad, direction) == pytest.approx(fd, rel=1e-3)


@pytest.mark.parametrize(
    "change",
    [
        {
            "boundary_conditions": {
                **cylinder.BOUNDARY_CONDITIONS,
                "x_hi": {"type": "no_slip"},
            }
        },
        {
            "boundary_conditions": {
                **cylinder.BOUNDARY_CONDITIONS,
                "y_lo": {"type": "neumann"},
            }
        },
        {"boundary_conditions": {}, "inflow_profile": None},
    ],
    ids=["outlet_no_slip", "wall_neumann", "periodic_with_obstacle"],
)
def test_rejects_unsupported_bcs(api, change):
    with pytest.raises(ValueError, match="warp-ns supports"):
        api.InputSchema(**{**_channel_inputs(10, t_end=0.1), **change})
