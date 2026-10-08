# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioned periodic 3D operator with full-field and scalar-input VJPs."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import InputSchema as CanonicalInput
from mosaic_shared.problems.navier_stokes_grid import OutputSchema as CanonicalOutput
from mosaic_shared.schema_types import GridVectorField, make_differentiable
from pydantic import Field
from tesseract_core.runtime import Differentiable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_model as model


class InputSchema(make_differentiable(CanonicalInput, ["v0", "viscosity", "dt"])):
    """Cubic periodic initial velocity, positive viscosity and timestep."""

    v0: Differentiable[GridVectorField] = Field(
        default_factory=lambda: np.zeros((16, 16, 16, 3), np.float32),
        description="Periodic cubic velocity field, shape (N, N, N, 3), N >= 8.",
    )
    steps: int = 100


class OutputSchema(make_differentiable(CanonicalOutput, ["result"])):
    """Final velocity at steps × dt, with zero drag for periodic flow."""


_WEIGHTS: tuple[dict, dict] | None = None


def _weights_path() -> Path:
    packaged = Path("/tesseract/operator_weights.npz")
    return (
        packaged
        if packaged.exists()
        else Path(__file__).with_name("operator_weights.npz")
    )


def _load_weights() -> tuple[dict, dict]:
    """Load parameters once, rejecting incompatible or nonfinite artifacts."""
    global _WEIGHTS
    if _WEIGHTS is None:
        with np.load(_weights_path(), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["model_version"] != model.VERSION:
                raise ValueError("operator checkpoint version mismatch")
            source_hash = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
            if metadata["model_sha256"] != source_hash:
                raise ValueError("operator checkpoint model source mismatch")
            settings = metadata["settings"]
            expected = model.init_params(
                settings["width"], settings["modes"], settings["layers"], seed=0
            )
            actual = {k: np.asarray(data[k]) for k in data.files if k != "metadata"}
            if set(actual) != set(expected):
                raise ValueError("operator checkpoint parameter keys mismatch")
            for key, value in actual.items():
                if value.shape != expected[key].shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid operator parameter: {key}")
            if settings["eval_stride"] < 1:
                raise ValueError("invalid operator stride")
            _WEIGHTS = (
                {k: jnp.asarray(v, dtype=jnp.float32) for k, v in actual.items()},
                settings,
            )
    return _WEIGHTS


def _validate_contract(inputs: InputSchema) -> None:
    velocity = np.asarray(inputs.v0)
    if (
        velocity.ndim != 4
        or velocity.shape[-1] != 3
        or len(set(velocity.shape[:3])) != 1
        or velocity.shape[0] < 8
    ):
        raise ValueError("operator requires a cubic (N, N, N, 3) field with N >= 8")
    if not np.isfinite(velocity).all():
        raise ValueError("operator requires finite initial velocity")
    for key in ("viscosity", "dt", "domain_extent"):
        value = np.asarray(getattr(inputs, key))
        if not np.isfinite(value).all() or not (value > 0).all():
            raise ValueError(f"operator requires positive finite {key}")
    if inputs.steps < 0:
        raise ValueError("operator requires nonnegative steps")
    if (
        not inputs.boundary_conditions.is_fully_periodic
        or inputs.obstacle is not None
        or inputs.inflow_profile is not None
    ):
        raise ValueError(
            "operator supports periodic 3D flow without obstacles or inflow"
        )


@partial(
    jax.jit,
    static_argnames=("steps", "stride", "layers", "project_output", "conserve_energy"),
)
def _forward(
    params: dict,
    velocity: jax.Array,
    viscosity: jax.Array,
    dt: jax.Array,
    extent: jax.Array,
    *,
    steps: int,
    stride: int,
    layers: int,
    project_output: bool,
    conserve_energy: bool,
) -> jax.Array:
    return model.predict(
        params,
        velocity[None],
        viscosity[0],
        dt[0],
        extent,
        steps=steps,
        stride=stride,
        layers=layers,
        project_output=project_output,
        conserve_energy=conserve_energy,
    )[0]


def _arguments(inputs: InputSchema) -> tuple:
    params, settings = _load_weights()
    arguments = (
        params,
        jnp.asarray(inputs.v0, dtype=jnp.float32),
        jnp.asarray(inputs.viscosity, dtype=jnp.float32),
        jnp.asarray(inputs.dt, dtype=jnp.float32),
        jnp.float32(inputs.domain_extent),
    )
    options = {
        "steps": inputs.steps,
        "stride": settings["eval_stride"],
        "layers": settings["layers"],
        "project_output": settings["project_output"],
        "conserve_energy": settings.get("conserve_energy", False),
    }
    return arguments, options


def apply(inputs: InputSchema) -> dict:
    """Advance the requested physical horizon using shared resolution-independent weights."""
    _validate_contract(inputs)
    arguments, options = _arguments(inputs)
    # Other solvers can enable global x64 in a shared host process. Keep this
    # operator's FFT/projection arithmetic identical to its float32 training.
    with jax.enable_x64(False):
        return {
            "result": _forward(*arguments, **options),
            "drag": jnp.zeros((1,), jnp.float32),
        }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """Differentiate the identical forward map with respect to requested inputs."""
    _validate_contract(inputs)
    keys = ("v0", "viscosity", "dt")
    if set(vjp_inputs) - set(keys) or set(vjp_outputs) - {"result", "drag"}:
        raise ValueError("unsupported operator derivative input/output")
    arguments, options = _arguments(inputs)
    params, velocity, nu, dt, extent = arguments
    if "result" not in vjp_outputs:
        return {
            k: jnp.zeros_like(v)
            for k, v in zip(keys, (velocity, nu, dt), strict=True)
            if k in vjp_inputs
        }
    cotangent = jnp.asarray(cotangent_vector["result"], dtype=jnp.float32)
    if cotangent.shape != velocity.shape or not np.isfinite(cotangent).all():
        raise ValueError("result cotangent must be finite and match the velocity shape")
    with jax.enable_x64(False):
        _, pullback = jax.vjp(
            lambda u, n, t: _forward(params, u, n, t, extent, **options),
            velocity,
            nu,
            dt,
        )
        return {
            k: v
            for k, v in zip(keys, pullback(cotangent), strict=True)
            if k in vjp_inputs
        }


def abstract_eval(abstract_inputs: Any) -> dict:
    """Preserve the caller's spatial shape while exposing float32 outputs."""
    value = abstract_inputs.v0
    shape = tuple(value["shape"] if isinstance(value, dict) else value.shape)
    return {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
