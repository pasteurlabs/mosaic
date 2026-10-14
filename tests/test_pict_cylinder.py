# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PICT ns-grid tesseract on the DFG 2D-1 channel.

PICT's module needs PISOtorch and a GPU, so every test here runs against the
``pict_navier_stokes_grid:latest`` image built from this checkout
(``mosaic build -p ns-grid -s PICT``) and is skipped where that image, Docker
or a GPU is missing; it is never built here.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

from .helpers import channel_inputs, image_available, unsupported_bcs

_IMAGE = "pict_navier_stokes_grid:latest"


def _gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return (
            subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=30, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(
    not (image_available(_IMAGE) and _gpu_available()),
    reason=f"{_IMAGE} or GPU missing",
)


@pytest.fixture(scope="module")
def pict():
    from tesseract_core import Tesseract

    with Tesseract.from_image(_IMAGE, gpus=["all"]) as t:
        yield t


def test_cylinder_res10(pict):
    """A res-10 run: positive drag within 5% of the reference, steady."""
    out = pict.apply(channel_inputs(10))
    force = np.asarray(out["body_force"])
    assert force.shape == (cylinder.time_step(10)[1], 2)
    assert np.all(np.isfinite(force))
    assert force[-1, 0] > 0
    metrics = cylinder.dfg_metrics(out["result"], out["pressure"], force, 10)
    # The orthogonal scheme on the O-grid overestimates the drag by about 3%.
    assert metrics["error"] < 0.05
    assert metrics["cd_drift"] < 1e-3


@unsupported_bcs
def test_rejects_unsupported_bcs(pict, change):
    with pytest.raises(Exception, match="pict supports"):
        pict.apply({**channel_inputs(10, t_end=0.1), **change})
