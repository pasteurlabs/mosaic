# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""jax-cfd's input schema accepts periodic flows and rejects obstacle flows.

jax-cfd and equinox are not installed on the host, so both are stubbed out:
only the pydantic schema is exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

_API = (
    Path(__file__).resolve().parents[1]
    / "mosaic"
    / "tesseracts"
    / "navier-stokes-grid"
    / "jax-cfd"
    / "tesseract_api.py"
)


@pytest.fixture
def input_schema(monkeypatch):
    jax_cfd = MagicMock()
    monkeypatch.setitem(sys.modules, "jax_cfd", jax_cfd)
    monkeypatch.setitem(sys.modules, "jax_cfd.base", jax_cfd.base)
    monkeypatch.setitem(sys.modules, "equinox", MagicMock())
    spec = importlib.util.spec_from_file_location("jax_cfd_tesseract_api", _API)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.InputSchema


def _channel_inputs(**overrides) -> dict:
    inputs = cylinder.channel_inputs(cylinder.inflow_profile(10), 10)
    inputs = {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()}
    return {**inputs, **overrides}


def test_periodic_inputs_validate(input_schema):
    input_schema()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        # Periodic BCs, but an obstacle: there is no periodic obstacle mode either.
        {
            "v0": np.zeros((41, 41, 1, 2), dtype=np.float32),
            "boundary_conditions": {},
            "inflow_profile": None,
            "domain_extent": 0.41,
        },
    ],
)
def test_obstacle_inputs_raise(input_schema, overrides):
    with pytest.raises(ValidationError, match="fully periodic"):
        input_schema(**_channel_inputs(**overrides))
