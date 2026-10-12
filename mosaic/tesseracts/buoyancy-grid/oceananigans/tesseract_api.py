# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Oceananigans coupled scalar/velocity adapter (CPU)."""

import os
from pathlib import Path
from typing import Any

import juliacall
import numpy as np
from mosaic_shared.problems.buoyancy_grid import InputSchema as BaseInput
from mosaic_shared.problems.buoyancy_grid import OutputSchema as BaseOutput
from pydantic import model_validator
from tesseract_core.runtime import ShapeDType


class InputSchema(BaseInput):
    """Closed-wall inputs with conservative explicit-step stability limits."""

    @model_validator(mode="after")
    def _check_initial_cfl(self) -> "InputSchema":
        if isinstance(self.state0, (dict, ShapeDType)):
            return self
        state = np.asarray(self.state0)
        dt = float(np.asarray(self.dt)[0])
        inverse_dx = state.shape[0] / self.domain_extent
        diffusion = max(
            float(np.asarray(self.viscosity)[0]), float(np.asarray(self.diffusivity)[0])
        )
        if dt * diffusion * inverse_dx**2 > 0.25:
            raise ValueError(
                "Explicit diffusion requires dt * max(viscosity, diffusivity) * N² / L² <= 0.25."
            )
        speed = np.max(np.abs(state[:-1, :, 0])) + np.max(np.abs(state[:, :-1, 1]))
        if dt * inverse_dx * speed > 0.5:
            raise ValueError("Initial advective CFL must be <= 0.5; reduce dt.")
        return self


class OutputSchema(BaseOutput):
    """Final native staggered state."""


jl = juliacall.newmodule("oceananigans_buoyancy")
jl.include(
    os.environ.get(
        "OCEANANIGANS_SOLVER_PATH", str(Path(__file__).with_name("solver.jl"))
    )
)


def _args(inputs: InputSchema):
    return (
        juliacall.convert(jl.Array, np.asarray(inputs.state0, dtype=np.float64)),
        *(
            float(np.asarray(getattr(inputs, key))[0])
            for key in ("viscosity", "diffusivity", "buoyancy", "dt")
        ),
        inputs.steps,
        inputs.domain_extent,
    )


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Advance the upstream Boussinesq model without resampling faces."""
    return {"result": np.asarray(jl.solve(*_args(inputs)), dtype=np.float32).copy()}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, dict[str, Any]]:
    """Return the shape and dtype of the recurrent state."""
    state = abstract_inputs.state0
    if isinstance(state, dict):
        shape, dtype = tuple(state["shape"]), state["dtype"]
    else:
        shape, dtype = state.shape, str(state.dtype)
    return {"result": {"shape": shape, "dtype": dtype}}
