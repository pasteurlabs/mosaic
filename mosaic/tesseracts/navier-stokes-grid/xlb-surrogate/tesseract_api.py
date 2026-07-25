# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-field XLB surrogate for the fixed cylinder drag-optimization task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.drag import drag_jax
from mosaic_shared.schema_types import make_differentiable


class InputSchema(make_differentiable(_CanonicalInputSchema, ["inflow_profile"])):
    """Task-specific surrogate input with differentiable inflow control."""


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result", "drag"])):
    """Full velocity field and surface-integral drag."""


_N = 32
_VISCOSITY = 0.0025
_DT = 0.02
_STEPS = 200
_DOMAIN_EXTENT = 1.0
_OBSTACLE_CENTER = (0.5, 0.5)
_OBSTACLE_RADIUS = 0.05
_WEIGHTS: dict[str, jax.Array] | None = None


def _weights_path() -> Path:
    """Return the packaged weight path in a container or source checkout."""
    packaged = Path("/tesseract/weights.npz")
    return packaged if packaged.exists() else Path(__file__).with_name("weights.npz")


def _load_weights() -> dict[str, jax.Array]:
    """Load and validate the exported surrogate parameter bundle once."""
    global _WEIGHTS
    if _WEIGHTS is None:
        path = _weights_path()
        if not path.exists():
            raise RuntimeError(f"XLB surrogate weights are missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "profile_mean",
                "profile_scale",
                "field_mean",
                "field_scale",
                "field_basis",
                "w_in",
                "b_in",
                "w_res_0",
                "b_res_0",
                "w_res_1",
                "b_res_1",
                "w_res_2",
                "b_res_2",
                "w_out",
                "b_out",
            }
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(
                    f"XLB surrogate weight bundle is incomplete: {sorted(missing)}"
                )
            _WEIGHTS = {key: jnp.asarray(data[key]) for key in required}
    return _WEIGHTS


def _solid_mask() -> jax.Array:
    """Rasterize the fixed cylinder using the canonical Mosaic convention."""
    center_x = _OBSTACLE_CENTER[0] * _N
    center_y = _OBSTACLE_CENTER[1] * _N
    radius = _OBSTACLE_RADIUS * _N
    x = jnp.arange(_N, dtype=jnp.float32)
    y = jnp.arange(_N, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 < radius**2


def _decode_fields(
    profile: jax.Array,
    weights: dict[str, jax.Array],
) -> jax.Array:
    """Map one inflow profile to final velocity and RANS traction fields.

    The four output channels are final ``u_x``, final ``u_y``, RANS ``u_x``,
    and RANS pressure. The final velocity is the public full-field result;
    the RANS channels feed the same drag functional used by the projection
    solvers.
    """
    # The teacher applies the inlet and then overwrites y=0 and y=N-1 with
    # no-slip walls, so those two profile entries are physically inactive.
    # Preserve the trained forward function exactly while preventing the MLP
    # from inventing sensitivities along the teacher's two null directions.
    profile = profile.at[0].set(jax.lax.stop_gradient(profile[0]))
    profile = profile.at[-1].set(jax.lax.stop_gradient(profile[-1]))
    x = (profile - weights["profile_mean"]) / weights["profile_scale"]
    hidden = jax.nn.gelu(x @ weights["w_in"] + weights["b_in"])
    for idx in range(3):
        residual = jax.nn.gelu(
            hidden @ weights[f"w_res_{idx}"] + weights[f"b_res_{idx}"]
        )
        hidden = (hidden + residual) * jnp.asarray(2.0**-0.5, dtype=hidden.dtype)
    coefficients = hidden @ weights["w_out"] + weights["b_out"]
    normalized = weights["field_mean"] + coefficients @ weights["field_basis"]
    fields = normalized.reshape(_N, _N, 4) * weights["field_scale"]

    # These values are exact properties of the fixed task and keep the decoded
    # field on the same boundary/obstacle manifold as the teacher.
    solid = _solid_mask()
    velocity = fields[..., :2]
    velocity = velocity.at[0, :, 0].set(profile)
    velocity = velocity.at[0, :, 1].set(0.0)
    velocity = velocity.at[:, 0, :].set(0.0)
    velocity = velocity.at[:, -1, :].set(0.0)
    velocity = jnp.where(solid[..., None], 0.0, velocity)

    rans_ux = fields[..., 2]
    rans_ux = rans_ux.at[0, :].set(profile)
    rans_ux = rans_ux.at[:, 0].set(0.0)
    rans_ux = rans_ux.at[:, -1].set(0.0)
    rans_ux = jnp.where(solid, 0.0, rans_ux)
    return jnp.concatenate(
        [velocity, rans_ux[..., None], fields[..., 3:4]],
        axis=-1,
    )


def _surrogate_forward(
    profile: jax.Array,
    weights: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Return the canonical velocity result and differentiable surface drag."""
    fields = _decode_fields(profile, weights)
    result = fields[..., :2, None]
    result = jnp.transpose(result, (0, 1, 3, 2))
    drag = drag_jax(
        fields[..., 2],
        fields[..., 3],
        _solid_mask(),
        jnp.asarray(_VISCOSITY, dtype=jnp.float32),
        jnp.asarray(_DOMAIN_EXTENT / _N, dtype=jnp.float32),
    )
    return {"result": result.astype(jnp.float32), "drag": drag}


@jax.jit
def _apply_jit(
    profile: jax.Array,
    weights: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    return _surrogate_forward(profile, weights)


@jax.jit
def _vjp_jit(
    profile: jax.Array,
    result_cotangent: jax.Array,
    drag_cotangent: jax.Array,
    weights: dict[str, jax.Array],
) -> jax.Array:
    _, pullback = jax.vjp(lambda p: _surrogate_forward(p, weights), profile)
    return pullback({"result": result_cotangent, "drag": drag_cotangent})[0].astype(
        jnp.float32
    )


def _as_float(value: Any) -> float:
    """Convert a scalar or one-element array to a Python float."""
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _validate_contract(inputs: InputSchema) -> None:
    """Reject inputs outside the training distribution's fixed physics task."""
    data = inputs.model_dump()
    v0 = np.asarray(data["v0"])
    profile = data.get("inflow_profile")
    if v0.shape != (_N, _N, 1, 2):
        raise ValueError(f"XLB surrogate requires v0 shape ({_N}, {_N}, 1, 2)")
    if profile is None or np.asarray(profile).shape != (_N,):
        raise ValueError(f"XLB surrogate requires inflow_profile shape ({_N},)")
    expected_v0 = np.zeros_like(v0)
    expected_v0[..., 0] = 0.5
    if not np.allclose(v0, expected_v0, rtol=0.0, atol=1e-6):
        raise ValueError("XLB surrogate requires the fixed uniform initial velocity")
    profile_array = np.asarray(profile)
    if np.any(profile_array < 0.0) or np.any(profile_array > 1.5):
        raise ValueError("XLB surrogate requires inflow_profile values in [0, 1.5]")
    fixed_scalars = {
        "viscosity": (_as_float(data["viscosity"]), _VISCOSITY),
        "dt": (_as_float(data["dt"]), _DT),
        "steps": (float(data["steps"]), float(_STEPS)),
        "domain_extent": (float(data["domain_extent"]), _DOMAIN_EXTENT),
    }
    for name, (actual, expected) in fixed_scalars.items():
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"XLB surrogate requires {name}={expected}, received {actual}"
            )
    obstacle = data.get("obstacle")
    if hasattr(obstacle, "model_dump"):
        obstacle = obstacle.model_dump()
    if not isinstance(obstacle, dict):
        raise ValueError("XLB surrogate requires the fixed cylinder obstacle")
    center = tuple(float(value) for value in obstacle.get("center", ()))
    shape_value = obstacle.get("shape", "")
    shape = str(getattr(shape_value, "value", shape_value)).lower()
    radius = float(obstacle.get("radius", -1.0))
    if (
        shape != "cylinder"
        or center != _OBSTACLE_CENTER
        or not np.isclose(radius, _OBSTACLE_RADIUS)
    ):
        raise ValueError("XLB surrogate requires the fixed cylinder obstacle")


def apply(inputs: InputSchema) -> dict[str, jax.Array]:
    """Evaluate the trained full-field surrogate."""
    _validate_contract(inputs)
    profile = jnp.asarray(inputs.inflow_profile, dtype=jnp.float32)
    return _apply_jit(profile, _load_weights())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, jax.Array]:
    """Differentiate the decoded field and physical drag w.r.t. inflow."""
    _validate_contract(inputs)
    unsupported = set(vjp_inputs) - {"inflow_profile"}
    if unsupported:
        raise ValueError(
            f"XLB surrogate only differentiates inflow_profile, not {unsupported}"
        )
    profile = jnp.asarray(inputs.inflow_profile, dtype=jnp.float32)
    result_cotangent = (
        jnp.asarray(cotangent_vector["result"], dtype=jnp.float32)
        if "result" in vjp_outputs
        else jnp.zeros((_N, _N, 1, 2), dtype=jnp.float32)
    )
    drag_cotangent = (
        jnp.asarray(cotangent_vector["drag"], dtype=jnp.float32)
        if "drag" in vjp_outputs
        else jnp.zeros((1,), dtype=jnp.float32)
    )
    gradient = _vjp_jit(
        profile,
        result_cotangent,
        drag_cotangent,
        _load_weights(),
    )
    return {"inflow_profile": gradient}


def abstract_eval(abstract_inputs: Any) -> dict[str, dict[str, Any]]:
    """Return fixed output metadata for the N=32 task."""
    v0_info = abstract_inputs.v0
    if isinstance(v0_info, dict):
        shape = tuple(v0_info["shape"])
        dtype = v0_info.get("dtype", "float32")
    else:
        shape = tuple(v0_info.shape)
        dtype = "float32"
    return {
        "result": {"shape": shape, "dtype": dtype},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
