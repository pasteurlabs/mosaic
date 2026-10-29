# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFoam ns-grid tesseract on the DFG 2D-1 channel.

The schema and cotangent checks load the tesseract module straight from
source. The solver runs need the ``dafoam_navier_stokes_grid:latest`` image
built from this checkout (``mosaic build -p ns-grid -s DAFoam``) and are
skipped where that image or Docker is missing. It is never built here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

from .helpers import channel_inputs, image_available, unsupported_bcs

_API = (
    Path(__file__).parents[1]
    / "mosaic/tesseracts/navier-stokes-grid/dafoam/tesseract_api.py"
)
_IMAGE = "dafoam_navier_stokes_grid:latest"
# 20 steps at resolution 10, short enough for a gradient check on a CPU.
_T_END = 0.2


@pytest.fixture(scope="module")
def api():
    """Load the DAFoam tesseract module straight from source."""
    spec = importlib.util.spec_from_file_location("dafoam_tesseract_api", _API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accepts_dfg_channel(api):
    api.InputSchema(**channel_inputs(10))


@unsupported_bcs
def test_rejects_unsupported_bcs(api, change):
    with pytest.raises(ValueError, match="dafoam supports"):
        api.InputSchema(**{**channel_inputs(10), **change})


def test_rejects_periodic_box(api):
    with pytest.raises(ValueError, match="dafoam supports only the 2-D channel"):
        api.InputSchema()


def test_trailing_window_of_drag_cotangent(api):
    steps = cylinder.time_step(10)[1]
    cotangent = np.zeros((steps, 2))
    w = cylinder.window_size(steps)
    cotangent[-w:, 0] = 0.25
    assert api._trailing_window(cotangent) == (w, 0.25)
    assert api._trailing_window(np.zeros((steps, 2))) == (0, 0.0)


@pytest.mark.parametrize(
    "rows, column",
    [(slice(None), 1), (slice(0, 5), 0), ([-3, -1], 0)],
    ids=["lift", "leading_window", "gap_in_window"],
)
def test_rejects_other_cotangents(api, rows, column):
    cotangent = np.zeros((100, 2))
    cotangent[rows, column] = 1.0
    with pytest.raises(NotImplementedError, match="trailing window"):
        api._trailing_window(cotangent)


@pytest.fixture(scope="module")
def dafoam():
    if not image_available(_IMAGE):
        pytest.skip(f"{_IMAGE} not built")
    from tesseract_core import Tesseract

    with Tesseract.from_image(_IMAGE) as t:
        yield t


def test_cylinder_short(dafoam):
    """A short res-10 run: finite force history with positive drag."""
    out = dafoam.apply(channel_inputs(10, t_end=_T_END))
    force = np.asarray(out["body_force"])
    assert force.shape == (cylinder.time_step(10, _T_END)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0


def test_drag_vjp_matches_finite_differences(dafoam):
    """The adjoint drag gradient agrees with central differences along a direction."""
    inputs = channel_inputs(10, t_end=_T_END)
    profile = inputs["inflow_profile"].astype(np.float64)
    steps = cylinder.time_step(10, _T_END)[1]

    def cd(p):
        out = dafoam.apply({**inputs, "inflow_profile": p.astype(np.float32)})
        return float(cylinder.drag_coefficient(np.asarray(out["body_force"])))

    # dCd/d(body_force) is uniform on F_x over the averaging window.
    w = cylinder.window_size(steps)
    cotangent = np.zeros((steps, 2), dtype=np.float32)
    cotangent[-w:, 0] = 2.0 / (cylinder.U_MEAN**2 * cylinder.DIAMETER * w)
    grad = np.asarray(
        dafoam.vector_jacobian_product(
            inputs, ["inflow_profile"], ["body_force"], {"body_force": cotangent}
        )["inflow_profile"]
    )

    v = np.random.default_rng(0).standard_normal(profile.shape)
    v /= np.linalg.norm(v)
    # body_force comes back in float32, so a smaller step loses Cd to rounding.
    h = 1e-2 * np.sqrt(np.mean(profile**2))
    fd = (cd(profile + h * v) - cd(profile - h * v)) / (2 * h)
    assert np.dot(grad, v) == pytest.approx(fd, rel=1e-3)
