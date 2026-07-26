# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct full-field XLB surrogate for the fixed 3D IC-recovery task."""

from __future__ import annotations

import math
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
from mosaic_shared.schema_types import make_differentiable


class InputSchema(make_differentiable(_CanonicalInputSchema, ["v0"])):
    """Fixed-task surrogate with differentiable full 3D initial velocity."""


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    """Fixed-horizon full 3D velocity field."""


_N = 16
_VISCOSITY = 0.01
_DT = 0.02
_STEPS = 100
_DOMAIN_EXTENT = 2.0 * math.pi
_WIDTH = 32
_MODES = 6
_LAYERS = 6
_WEIGHTS: dict[str, jax.Array] | None = None


def _weights_path() -> Path:
    packaged = Path("/tesseract/weights.npz")
    return packaged if packaged.exists() else Path(__file__).with_name("weights.npz")


def _load_weights() -> dict[str, jax.Array]:
    """Load and validate the packaged Fourier-neural-operator parameters."""
    global _WEIGHTS
    if _WEIGHTS is None:
        path = _weights_path()
        if not path.exists():
            raise RuntimeError(f"XLB 3D surrogate weights are missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            metadata = {
                "width": int(data["width"]),
                "modes": int(data["modes"]),
                "layers": int(data["layers"]),
            }
            expected = {
                "width": _WIDTH,
                "modes": _MODES,
                "layers": _LAYERS,
            }
            if metadata != expected:
                raise RuntimeError(
                    f"XLB 3D surrogate architecture mismatch: {metadata} != {expected}"
                )
            required = {
                "input_scale",
                "correction_scale",
                "w_lift",
                "w_out",
            }
            for layer in range(_LAYERS):
                required.add(f"w_local_{layer}")
                for quadrant in range(4):
                    required.add(f"spec_{layer}_{quadrant}_real")
                    required.add(f"spec_{layer}_{quadrant}_imag")
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(
                    f"XLB 3D surrogate weights are incomplete: {sorted(missing)}"
                )
            _WEIGHTS = {key: jnp.asarray(data[key]) for key in required}
    return _WEIGHTS


def _helmholtz_project(velocity: jax.Array) -> jax.Array:
    """Project a batched periodic velocity field to the divergence-free subspace."""
    velocity_hat = jnp.fft.rfftn(velocity, axes=(1, 2, 3))
    k = jnp.fft.fftfreq(_N, d=1.0 / _N)
    kz = jnp.fft.rfftfreq(_N, d=1.0 / _N)
    kx, ky, kz_grid = jnp.meshgrid(k, k, kz, indexing="ij")
    wave = jnp.stack([kx, ky, kz_grid], axis=-1)
    k2 = jnp.sum(wave**2, axis=-1)
    safe_k2 = jnp.where(k2 == 0, 1.0, k2)
    dot = jnp.sum(velocity_hat * wave, axis=-1)
    projected_hat = velocity_hat - wave * (dot / safe_k2)[..., None]
    projected_hat = projected_hat.at[:, 0, 0, 0, :].set(velocity_hat[:, 0, 0, 0, :])
    return jnp.fft.irfftn(
        projected_hat,
        s=(_N, _N, _N),
        axes=(1, 2, 3),
    ).real.astype(jnp.float32)


def _diffuse(velocity: jax.Array) -> jax.Array:
    """Exact linear viscous evolution over the fixed recovery horizon."""
    velocity_hat = jnp.fft.rfftn(
        _helmholtz_project(velocity),
        axes=(1, 2, 3),
    )
    k = jnp.fft.fftfreq(_N, d=1.0 / _N)
    kz = jnp.fft.rfftfreq(_N, d=1.0 / _N)
    kx, ky, kz_grid = jnp.meshgrid(k, k, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz_grid**2
    decay = jnp.exp(-_VISCOSITY * (_DT * _STEPS) * k2)
    return jnp.fft.irfftn(
        velocity_hat * decay[..., None],
        s=(_N, _N, _N),
        axes=(1, 2, 3),
    ).real.astype(jnp.float32)


def _spectral_convolution(
    hidden: jax.Array,
    weights: dict[str, jax.Array],
    layer: int,
) -> jax.Array:
    hidden_hat = jnp.fft.rfftn(hidden, axes=(1, 2, 3))
    output_hat = jnp.zeros_like(hidden_hat)
    selections = (
        (slice(0, _MODES), slice(0, _MODES)),
        (slice(-_MODES, None), slice(0, _MODES)),
        (slice(0, _MODES), slice(-_MODES, None)),
        (slice(-_MODES, None), slice(-_MODES, None)),
    )
    for quadrant, (sx, sy) in enumerate(selections):
        spectral_weight = (
            weights[f"spec_{layer}_{quadrant}_real"]
            + 1j * weights[f"spec_{layer}_{quadrant}_imag"]
        )
        transformed = jnp.einsum(
            "bxyzi,xyzio->bxyzo",
            hidden_hat[:, sx, sy, :_MODES, :],
            spectral_weight,
        )
        output_hat = output_hat.at[:, sx, sy, :_MODES, :].set(transformed)
    return jnp.fft.irfftn(
        output_hat,
        s=(_N, _N, _N),
        axes=(1, 2, 3),
    ).real.astype(jnp.float32)


def _surrogate_forward(
    initial_velocity: jax.Array,
    weights: dict[str, jax.Array],
) -> jax.Array:
    """Map one full initial field directly to the fixed-horizon final field."""
    velocity = _helmholtz_project(initial_velocity[None])
    normalized = velocity / weights["input_scale"]
    hidden = jnp.einsum("bxyzi,io->bxyzo", normalized, weights["w_lift"])
    for layer in range(_LAYERS):
        spectral = _spectral_convolution(hidden, weights, layer)
        local = jnp.einsum(
            "bxyzi,io->bxyzo",
            hidden,
            weights[f"w_local_{layer}"],
        )
        update = jax.nn.gelu(spectral + local)
        hidden = (hidden + update) * jnp.asarray(2.0**-0.5, jnp.float32)
    raw_correction = jnp.einsum(
        "bxyzi,io->bxyzo",
        hidden,
        weights["w_out"],
    )
    amplitude = jnp.sqrt(jnp.mean(velocity**2, keepdims=True) + 1e-12)
    correction = (
        raw_correction
        * weights["correction_scale"].reshape(1, 1, 1, 1, 3)
        * (amplitude / weights["input_scale"])
    )
    return _helmholtz_project(_diffuse(velocity) + correction)[0]


@jax.jit
def _apply_jit(
    initial_velocity: jax.Array,
    weights: dict[str, jax.Array],
) -> jax.Array:
    return _surrogate_forward(initial_velocity, weights)


@jax.jit
def _vjp_jit(
    initial_velocity: jax.Array,
    result_cotangent: jax.Array,
    weights: dict[str, jax.Array],
) -> jax.Array:
    _, pullback = jax.vjp(
        lambda velocity: _surrogate_forward(velocity, weights),
        initial_velocity,
    )
    return pullback(result_cotangent)[0].astype(jnp.float32)


def _as_float(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _validate_contract(inputs: InputSchema) -> None:
    data = inputs.model_dump()
    v0 = np.asarray(data["v0"])
    if v0.shape != (_N, _N, _N, 3):
        raise ValueError(f"XLB 3D surrogate requires v0 shape ({_N}, {_N}, {_N}, 3)")
    if not np.all(np.isfinite(v0)):
        raise ValueError("XLB 3D surrogate requires finite v0 values")
    fixed_scalars = {
        "viscosity": (_as_float(data["viscosity"]), _VISCOSITY),
        "dt": (_as_float(data["dt"]), _DT),
        "steps": (float(data["steps"]), float(_STEPS)),
        "domain_extent": (float(data["domain_extent"]), _DOMAIN_EXTENT),
    }
    for name, (actual, expected) in fixed_scalars.items():
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"XLB 3D surrogate requires {name}={expected}, received {actual}"
            )
    if not inputs.boundary_conditions.is_fully_periodic:
        raise ValueError("XLB 3D surrogate requires fully periodic boundaries")
    if inputs.obstacle is not None or inputs.inflow_profile is not None:
        raise ValueError(
            "XLB 3D surrogate does not support obstacles or inflow profiles"
        )


def apply(inputs: InputSchema) -> dict[str, jax.Array | None]:
    """Evaluate the direct IC-to-final-state full-field surrogate."""
    _validate_contract(inputs)
    initial_velocity = jnp.asarray(inputs.v0, dtype=jnp.float32)
    return {
        "result": _apply_jit(initial_velocity, _load_weights()),
        # The canonical grid schema always exposes drag.  This task has no
        # obstacle, so match the other periodic solvers with a zero scalar.
        "drag": jnp.zeros((1,), dtype=jnp.float32),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, jax.Array]:
    """Differentiate the full final field with respect to the full initial field."""
    _validate_contract(inputs)
    unsupported = set(vjp_inputs) - {"v0"}
    if unsupported:
        raise ValueError(f"XLB 3D surrogate only differentiates v0, not {unsupported}")
    initial_velocity = jnp.asarray(inputs.v0, dtype=jnp.float32)
    result_cotangent = (
        jnp.asarray(cotangent_vector["result"], dtype=jnp.float32)
        if "result" in vjp_outputs
        else jnp.zeros_like(initial_velocity)
    )
    return {
        "v0": _vjp_jit(
            initial_velocity,
            result_cotangent,
            _load_weights(),
        )
    }


def abstract_eval(abstract_inputs: Any) -> dict[str, dict[str, Any]]:
    """Return output metadata matching the full input velocity field."""
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
