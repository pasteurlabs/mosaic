# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoregressive 3D Fourier neural operator for the XLB recovery task.

This module is shared by the packaged inference API and the reproducible
training program. Keep architecture or checkpoint-format changes here so the
two paths cannot drift.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

N = 16
DOMAIN_EXTENT = 2.0 * math.pi
VISCOSITY = 0.01
SOLVER_DT = 0.02
SOLVER_STEPS = 100
STRIDE = 5
MACRO_DT = SOLVER_DT * STRIDE
ROLLOUT_STEPS = SOLVER_STEPS // STRIDE


def helmholtz_project(velocity: jax.Array) -> jax.Array:
    """Spectrally project channels-last periodic velocity to divergence-free."""
    axes = tuple(range(velocity.ndim - 4, velocity.ndim - 1))
    velocity_hat = jnp.fft.rfftn(velocity, axes=axes)
    k = jnp.fft.fftfreq(N, d=1.0 / N)
    kz = jnp.fft.rfftfreq(N, d=1.0 / N)
    kx, ky, kz_grid = jnp.meshgrid(k, k, kz, indexing="ij")
    wave = jnp.stack([kx, ky, kz_grid], axis=-1)
    k2 = jnp.sum(wave**2, axis=-1)
    safe_k2 = jnp.where(k2 == 0, 1.0, k2)
    dot = jnp.sum(velocity_hat * wave, axis=-1)
    projected_hat = velocity_hat - wave * (dot / safe_k2)[..., None]
    projected_hat = projected_hat.at[..., 0, 0, 0, :].set(velocity_hat[..., 0, 0, 0, :])
    return jnp.fft.irfftn(
        projected_hat,
        s=(N, N, N),
        axes=axes,
    ).real.astype(jnp.float32)


def diffuse_macro(velocity: jax.Array) -> jax.Array:
    """Exact linear viscous evolution for one five-solver-step interval."""
    axes = tuple(range(velocity.ndim - 4, velocity.ndim - 1))
    velocity_hat = jnp.fft.rfftn(helmholtz_project(velocity), axes=axes)
    k = jnp.fft.fftfreq(N, d=1.0 / N)
    kz = jnp.fft.rfftfreq(N, d=1.0 / N)
    kx, ky, kz_grid = jnp.meshgrid(k, k, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz_grid**2
    decay = jnp.exp(-VISCOSITY * MACRO_DT * k2)
    return jnp.fft.irfftn(
        velocity_hat * decay[..., None],
        s=(N, N, N),
        axes=axes,
    ).real.astype(jnp.float32)


def _spectral_convolution(
    x: jax.Array,
    params: dict[str, jax.Array],
    layer: int,
    modes: int,
) -> jax.Array:
    x_hat = jnp.fft.rfftn(x, axes=(1, 2, 3))
    output_hat = jnp.zeros_like(x_hat)
    selections = (
        (slice(0, modes), slice(0, modes)),
        (slice(-modes, None), slice(0, modes)),
        (slice(0, modes), slice(-modes, None)),
        (slice(-modes, None), slice(-modes, None)),
    )
    for quadrant, (sx, sy) in enumerate(selections):
        weight = (
            params[f"spec_{layer}_{quadrant}_real"]
            + 1j * params[f"spec_{layer}_{quadrant}_imag"]
        )
        transformed = jnp.einsum(
            "bxyzi,xyzio->bxyzo",
            x_hat[:, sx, sy, :modes, :],
            weight,
        )
        output_hat = output_hat.at[:, sx, sy, :modes, :].set(transformed)
    return jnp.fft.irfftn(
        output_hat,
        s=(N, N, N),
        axes=(1, 2, 3),
    ).real.astype(jnp.float32)


def one_step(
    params: dict[str, jax.Array],
    velocity: jax.Array,
    *,
    input_scale: jax.Array,
    correction_scale: jax.Array,
    modes: int,
    layers: int,
) -> jax.Array:
    """Apply the shared full-field neural operator for one macro time step."""
    velocity = helmholtz_project(velocity)
    normalized = velocity / input_scale
    hidden = jnp.einsum("bxyzi,io->bxyzo", normalized, params["w_lift"])
    for layer in range(layers):
        spectral = _spectral_convolution(hidden, params, layer, modes)
        local = jnp.einsum(
            "bxyzi,io->bxyzo",
            hidden,
            params[f"w_local_{layer}"],
        )
        update = jax.nn.gelu(spectral + local)
        hidden = (hidden + update) * jnp.asarray(2.0**-0.5, jnp.float32)
    raw_correction = jnp.einsum(
        "bxyzi,io->bxyzo",
        hidden,
        params["w_out"],
    )
    amplitude = jnp.sqrt(
        jnp.mean(velocity**2, axis=(1, 2, 3, 4), keepdims=True) + 1e-12
    )
    correction = (
        raw_correction
        * correction_scale.reshape(1, 1, 1, 1, 3)
        * (amplitude / input_scale)
    )
    return helmholtz_project(diffuse_macro(velocity) + correction)


def rollout(
    params: dict[str, jax.Array],
    velocity: jax.Array,
    *,
    steps: int,
    input_scale: jax.Array,
    correction_scale: jax.Array,
    modes: int,
    layers: int,
) -> jax.Array:
    """Autoregressively return ``steps`` predicted fields after ``velocity``."""

    def body(current: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        predicted = one_step(
            params,
            current,
            input_scale=input_scale,
            correction_scale=correction_scale,
            modes=modes,
            layers=layers,
        )
        return predicted, predicted

    _, predictions = jax.lax.scan(body, velocity, None, length=steps)
    return jnp.moveaxis(predictions, 0, 1)


def init_params(
    *,
    width: int,
    modes: int,
    layers: int,
    seed: int,
) -> dict[str, jax.Array]:
    """Initialize one FNO parameter tree deterministically."""
    rng = np.random.default_rng(seed)

    def weight(fan_in: int, fan_out: int) -> jax.Array:
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        return jnp.asarray(
            rng.normal(scale=scale, size=(fan_in, fan_out)),
            dtype=jnp.float32,
        )

    params: dict[str, jax.Array] = {
        "w_lift": weight(3, width),
        "w_out": weight(width, 3) * 0.03,
    }
    spectral_scale = 0.15 / math.sqrt(width)
    for layer in range(layers):
        local = np.eye(width, dtype=np.float32) * 0.2
        local += rng.normal(scale=0.015, size=(width, width)).astype(np.float32)
        params[f"w_local_{layer}"] = jnp.asarray(local)
        for quadrant in range(4):
            shape = (modes, modes, modes, width, width)
            params[f"spec_{layer}_{quadrant}_real"] = jnp.asarray(
                rng.normal(scale=spectral_scale, size=shape),
                dtype=jnp.float32,
            )
            params[f"spec_{layer}_{quadrant}_imag"] = jnp.asarray(
                rng.normal(scale=spectral_scale, size=shape),
                dtype=jnp.float32,
            )
    return params


def tree_l2(params: dict[str, jax.Array]) -> jax.Array:
    """Return the unnormalized squared L2 norm of a parameter tree."""
    return sum(jnp.sum(value**2) for value in params.values())


def serialize_params(
    params: dict[str, Any],
    *,
    input_scale: float,
    correction_scale: np.ndarray,
    width: int,
    modes: int,
    layers: int,
) -> dict[str, np.ndarray]:
    """Serialize parameters and fixed-task architecture metadata."""
    return {
        **{key: np.asarray(value) for key, value in params.items()},
        "input_scale": np.asarray(input_scale, dtype=np.float32),
        "correction_scale": np.asarray(correction_scale, dtype=np.float32),
        "width": np.asarray(width, dtype=np.int32),
        "modes": np.asarray(modes, dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int32),
        "solver_dt": np.asarray(SOLVER_DT, dtype=np.float32),
        "solver_steps": np.asarray(SOLVER_STEPS, dtype=np.int32),
        "stride": np.asarray(STRIDE, dtype=np.int32),
        "macro_dt": np.asarray(MACRO_DT, dtype=np.float32),
        "rollout_steps": np.asarray(ROLLOUT_STEPS, dtype=np.int32),
        "autoregressive": np.asarray(1, dtype=np.int32),
    }
