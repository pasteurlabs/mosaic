# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""INS.jl's input schema accepts periodic boxes and the DFG channel, and rejects the rest.

The tesseract loads Julia at import, so ``juliacall`` is stubbed out. Only the
pydantic schema is exercised here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from .helpers import channel_inputs, unsupported_bcs

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


def test_channel_inputs_validate(input_schema):
    input_schema(**channel_inputs(10))


@unsupported_bcs
def test_unsupported_inputs_raise(input_schema, change):
    with pytest.raises(ValidationError, match=r"INS\.jl supports"):
        input_schema(**{**channel_inputs(10), **change})
