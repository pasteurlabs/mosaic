# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PICT's VJP on the periodic box is linear in the cotangent.

Runs against the ``pict_navier_stokes_grid:latest`` image built from this
checkout (``mosaic build -p ns-grid -s PICT``) and is skipped where that image,
Docker or a GPU is missing. It is never built here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _multimode

from .helpers import image_available
from .test_pict_cylinder import _gpu_available

_IMAGE = "pict_navier_stokes_grid:latest"

pytestmark = pytest.mark.skipif(
    not (image_available(_IMAGE) and _gpu_available()),
    reason=f"{_IMAGE} or GPU missing",
)


def test_vjp_is_linear_in_the_cotangent():
    """A cotangent scaled down by 1e5 gives the gradient scaled down by 1e5.

    PICT's adjoint solves stop at an absolute residual, so without the
    tesseract's rescaling a small cotangent cuts them short.
    """
    from tesseract_core import Tesseract

    inputs = get_config("ns-grid").make_inputs(
        "PICT", _multimode(32), nu=1e-3, dt=0.01, steps=100
    )
    inputs = {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()}
    cotangent = np.random.default_rng(0).standard_normal(inputs["v0"].shape)
    cotangent = cotangent.astype(np.float32)

    def vjp(t, scale):
        out = t.vector_jacobian_product(
            inputs=inputs,
            vjp_inputs=["v0"],
            vjp_outputs=["result"],
            cotangent_vector={"result": scale * cotangent},
        )
        return np.asarray(out["v0"]) / scale

    with Tesseract.from_image(_IMAGE, gpus=["all"]) as t:
        unit, small = vjp(t, 1.0), vjp(t, 1e-5)
    error = np.linalg.norm(small - unit) / np.linalg.norm(unit)
    assert error < 1e-3
