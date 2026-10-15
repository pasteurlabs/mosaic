# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""INS.jl's input schema accepts periodic flows and rejects obstacle flows.

The tesseract loads Julia at import, so ``juliacall`` is stubbed out. Only the
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

from .helpers import channel_inputs

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


def test_periodic_inputs_validate(input_schema):
    input_schema()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        # Periodic BCs with an obstacle are rejected too.
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
        input_schema(**{**channel_inputs(10), **overrides})
