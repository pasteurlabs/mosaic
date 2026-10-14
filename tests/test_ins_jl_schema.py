# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""INS.jl's input schema accepts periodic flows and rejects obstacle flows.

The tesseract loads Julia at import, so ``juliacall`` is stubbed out: only the
pydantic schema is exercised here.
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
    / "ins-jl"
    / "tesseract_api.py"
)


@pytest.fixture
def input_schema(monkeypatch):
    monkeypatch.setitem(sys.modules, "juliacall", MagicMock())
    spec = importlib.util.spec_from_file_location("ins_jl_tesseract_api", _API)
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
            "boundary_conditions": {},
            "inflow_profile": None,
            "domain_extent": [2.2, 0.41],
        },
    ],
)
def test_obstacle_inputs_raise(input_schema, overrides):
    with pytest.raises(ValidationError, match="fully periodic"):
        input_schema(**_channel_inputs(**overrides))
