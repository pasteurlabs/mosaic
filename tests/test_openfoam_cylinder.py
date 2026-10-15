# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM ns-grid tesseract on the DFG 2D-1 channel.

The boundary-condition checks load the tesseract module straight from source.
The solver run needs the ``openfoam_navier_stokes_grid:latest`` image built
from this checkout (``mosaic build -p ns-grid -s OpenFOAM``) and is skipped
where that image or Docker is missing. It is never built here.
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


def test_accepts_dfg_channel(api):
    api.InputSchema(**channel_inputs(10))


@unsupported_bcs
def test_rejects_unsupported_bcs(api, change):
    with pytest.raises(ValueError, match="openfoam supports"):
        api.InputSchema(**{**channel_inputs(10), **change})


@pytest.mark.skipif(not image_available(_IMAGE), reason=f"{_IMAGE} not built")
def test_cylinder_res10():
    """A res-10 run: positive drag within the forward/cylinder threshold, steady."""
    from tesseract_core import Tesseract

    with Tesseract.from_image(_IMAGE) as t:
        out = t.apply(channel_inputs(10))
    force = np.asarray(out["body_force"])
    assert force.shape == (cylinder.time_step(10)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0
    metrics = cylinder.dfg_metrics(out["result"], out["pressure"], force, 10)
    assert metrics["error"] < 0.05
    assert metrics["cd_drift"] < 1e-3
