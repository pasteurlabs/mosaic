# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-box WaterLily buoyancy flow with an Enzyme coupled-state VJP."""

import os
from pathlib import Path
from typing import Any

import juliacall
import numpy as np
from mosaic_shared.problems.buoyancy_grid import InputSchema as BaseInput
from mosaic_shared.problems.buoyancy_grid import OutputSchema as BaseOutput
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator
from tesseract_core.runtime import ShapeDType


class InputSchema(
    make_differentiable(
        BaseInput, ["state0", "viscosity", "diffusivity", "buoyancy", "dt"]
    )
):
    """Differentiable inputs with the explicit diffusion stability bound."""

    @model_validator(mode="after")
    def _check_diffusion_cfl(self) -> "InputSchema":
        if isinstance(self.state0, (dict, ShapeDType)):
            return self
        n = np.asarray(self.state0).shape[0]
        dt = float(np.asarray(self.dt)[0])
        coefficient = max(
            float(np.asarray(self.viscosity)[0]), float(np.asarray(self.diffusivity)[0])
        )
        if dt * coefficient * (n / self.domain_extent) ** 2 > 0.25:
            raise ValueError(
                "Explicit diffusion requires dt * max(viscosity, diffusivity) * N² / L² <= 0.25."
            )
        return self


class OutputSchema(make_differentiable(BaseOutput, ["result"])):
    """Final state in the canonical staggered packing."""


jl = juliacall.newmodule("waterlily_buoyancy")
source_dir = Path(os.environ.get("MOSAIC_WATERLILY_SOURCE", Path(__file__).parent))
jl.include(str(source_dir / "solver.jl"))
jl.include(str(source_dir / "vjp.jl"))


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Advance the native momentum and conservative scalar together."""
    result = jl.MosaicWaterLily.simulate(
        juliacall.convert(jl.Array, np.asarray(inputs.state0, dtype=np.float64)),
        *[
            float(np.asarray(getattr(inputs, name))[0])
            for name in ("viscosity", "diffusivity", "buoyancy", "dt")
        ],
        inputs.steps,
        inputs.domain_extent,
    )
    return {"result": np.asarray(result, dtype=np.float32)}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Infer result shape without running Julia numerics."""
    state = abstract_inputs.state0
    shape = tuple(state["shape"]) if isinstance(state, dict) else state.shape
    return {"result": {"shape": shape, "dtype": "float32"}}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Differentiate through every coupled timestep using reverse-mode AD."""
    if vjp_outputs != {"result"}:
        raise ValueError("Only the result output is differentiable.")
    names = ("viscosity", "diffusivity", "buoyancy", "dt")
    state_grad, param_grad = jl.coupled_vjp(
        juliacall.convert(jl.Array, np.asarray(inputs.state0, dtype=np.float64)),
        *[float(np.asarray(getattr(inputs, name))[0]) for name in names],
        inputs.steps,
        inputs.domain_extent,
        juliacall.convert(
            jl.Array, np.asarray(cotangent_vector["result"], dtype=np.float64)
        ),
    )
    gradients = {"state0": np.asarray(state_grad, dtype=np.float32)}
    gradients.update(
        {
            name: np.array([value], dtype=np.float32)
            for name, value in zip(names, np.asarray(param_grad), strict=True)
        }
    )
    return {name: gradients[name] for name in vjp_inputs}
