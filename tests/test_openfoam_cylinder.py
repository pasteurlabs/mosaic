# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM ns-grid tesseract on the DFG 2D-1 channel.

The boundary-condition checks load the tesseract module straight from source.
The solver run needs the ``openfoam_navier_stokes_grid:latest`` image built
from this checkout (``mosaic build -p ns-grid -s OpenFOAM``) and is skipped
where that image or Docker is missing; it is never built here.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

_API = (
    Path(__file__).parents[1]
    / "mosaic/tesseracts/navier-stokes-grid/openfoam/tesseract_api.py"
)
_IMAGE = "openfoam_navier_stokes_grid:latest"


@pytest.fixture(scope="module")
def api():
    """Load the OpenFOAM tesseract module straight from source."""
    spec = importlib.util.spec_from_file_location("openfoam_tesseract_api", _API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _channel_inputs(resolution: int) -> dict:
    inputs = cylinder.channel_inputs(cylinder.inflow_profile(resolution), resolution)
    return {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()}


def test_accepts_dfg_channel(api):
    api.InputSchema(**_channel_inputs(10))


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
    with pytest.raises(ValueError, match="openfoam supports"):
        api.InputSchema(**{**_channel_inputs(10), **change})


def _image_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "image", "inspect", _IMAGE],
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _image_available(), reason=f"{_IMAGE} not built")
def test_cylinder_res10():
    """A res-10 run: positive drag within the forward/cylinder threshold, steady."""
    from tesseract_core import Tesseract

    with Tesseract.from_image(_IMAGE) as t:
        out = t.apply(_channel_inputs(10))
    force = np.asarray(out["body_force"])
    assert force.shape == (cylinder.time_step(10)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0
    metrics = cylinder.dfg_metrics(out["result"], out["pressure"], force, 10)
    assert metrics["error"] < 0.05
    assert metrics["cd_drift"] < 1e-3
