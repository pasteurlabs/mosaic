# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 — sys.path bootstrap must precede workspace imports
"""Identity Navier--Stokes Tesseract for recurrent corrector smoke tests.

Unlike the corpus-wide constant-output dummy, this fixture exposes a nonzero,
exact VJP so a solver-in-the-loop test can verify gradient flow through more
than one recurrent ``apply_tesseract`` call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_TESSERACTS_DIR = Path(__file__).resolve().parents[3] / "mosaic" / "tesseracts"
if str(_TESSERACTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESSERACTS_DIR))

import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import ConfigDict, model_validator
from tesseract_core.runtime import ShapeDType


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["velocity", "viscosity", "dt", "inflow_profile"]
    )
):
    # The smoke test presents this fixture as different real solvers, whose
    # solver-specific tuning inputs must be accepted and ignored.
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, value):
        if isinstance(value, dict) and "v0" in value:
            raise ValueError("'v0' was removed; pass 'velocity'")
        return value


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result", "drag"])):
    pass


def apply(inputs: InputSchema) -> OutputSchema:
    """Return the input velocity unchanged."""
    return OutputSchema(
        result=np.asarray(inputs.velocity, dtype=np.float32),
        drag=np.asarray([0.0], dtype=np.float32),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return the exact identity VJP for ``result = velocity``."""
    del vjp_outputs
    out: dict[str, np.ndarray] = {}
    if "velocity" in vjp_inputs:
        out["velocity"] = np.asarray(
            cotangent_vector.get("result", np.zeros_like(inputs.velocity)),
            dtype=np.float32,
        )
    if "viscosity" in vjp_inputs:
        out["viscosity"] = np.zeros((1,), dtype=np.float32)
    if "dt" in vjp_inputs:
        out["dt"] = np.zeros((1,), dtype=np.float32)
    if "inflow_profile" in vjp_inputs and inputs.inflow_profile is not None:
        out["inflow_profile"] = np.zeros_like(
            np.asarray(inputs.inflow_profile), dtype=np.float32
        )
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    """Declare that ``result`` has the same shape and dtype as ``velocity``."""
    velocity = abstract_inputs.model_dump()["velocity"]
    if isinstance(velocity, dict) and "shape" in velocity:
        shape = tuple(velocity["shape"])
        dtype = velocity.get("dtype", "float32")
    else:
        array = np.asarray(velocity)
        shape = array.shape
        dtype = str(array.dtype)
    return {
        "result": ShapeDType(shape=shape, dtype=dtype),
        "drag": ShapeDType(shape=(1,), dtype="float32"),
    }
