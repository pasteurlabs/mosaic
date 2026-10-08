# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioned periodic 3D FNO with weights indexed by signed Fourier mode.

Experimental operator checkpoints have their own schema. The packaged fixed-task
model remains available in surrogate_model.py until benchmark validation admits
its replacement. All differentiable training and inference share this module.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

VERSION = 2


def physical_waves(shape: tuple[int, int, int], extent: jax.Array) -> jax.Array:
    """Physical wavevectors; zero even-grid Nyquist components for real projection."""
    axes = []
    for axis, n in enumerate(shape):
        k = jnp.fft.rfftfreq(n) * n if axis == 2 else jnp.fft.fftfreq(n) * n
        if n % 2 == 0:
            k = k.at[n // 2].set(0)
        axes.append(k * (2 * jnp.pi / extent))
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


def project(velocity: jax.Array, extent: jax.Array = 2 * jnp.pi) -> jax.Array:
    """Project a real field using a Hermitian-compatible derivative convention."""
    shape = velocity.shape[1:4]
    wave = physical_waves(shape, extent)
    k2 = jnp.sum(wave**2, axis=-1)
    hat = jnp.fft.rfftn(velocity, axes=(1, 2, 3), norm="forward")
    dot = jnp.sum(hat * wave, axis=-1)
    projected = hat - wave * (dot / jnp.where(k2 == 0, 1, k2))[..., None]
    return jnp.fft.irfftn(projected, s=shape, axes=(1, 2, 3), norm="forward").astype(
        velocity.dtype
    )


def diffuse(
    velocity: jax.Array, nu: jax.Array, elapsed: jax.Array, extent: jax.Array
) -> jax.Array:
    """Resolution-independent exact vector diffusion, including Nyquist modes."""
    shape = velocity.shape[1:4]
    frequencies = [jnp.fft.fftfreq(n) * n for n in shape[:2]]
    frequencies.append(jnp.fft.rfftfreq(shape[2]) * shape[2])
    wave = jnp.stack(jnp.meshgrid(*frequencies, indexing="ij"), axis=-1) * (
        2 * jnp.pi / extent
    )
    k2 = jnp.sum(wave**2, axis=-1)
    hat = jnp.fft.rfftn(velocity, axes=(1, 2, 3), norm="forward")
    result = jnp.fft.irfftn(
        hat * jnp.exp(-nu * elapsed * k2)[..., None],
        s=shape,
        axes=(1, 2, 3),
        norm="forward",
    )
    return result.astype(velocity.dtype)


def spectral_convolution(x: jax.Array, real: jax.Array, imag: jax.Array) -> jax.Array:
    """Apply the same signed-mode bank on any grid without overlapping slices."""
    modes = real.shape[2] - 1
    shape = x.shape[1:4]
    kx = np.arange(
        -min(modes, (shape[0] - 1) // 2), min(modes, (shape[0] - 1) // 2) + 1
    )
    ky = np.arange(
        -min(modes, (shape[1] - 1) // 2), min(modes, (shape[1] - 1) // 2) + 1
    )
    kz = np.arange(min(modes, (shape[2] - 1) // 2) + 1)
    ix, iy, iz = np.meshgrid(kx % shape[0], ky % shape[1], kz, indexing="ij")
    wx, wy, wz = np.meshgrid(kx + modes, ky + modes, kz, indexing="ij")
    weights = real + 1j * imag
    # z=0 must satisfy W(-kx,-ky)=conj(W(kx,ky)); other z planes have
    # implicit conjugate partners in the omitted half of the real FFT.
    plane = weights[:, :, 0]
    weights = weights.at[:, :, 0].set(0.5 * (plane + jnp.conj(plane[::-1, ::-1])))
    selected = weights[wx, wy, wz]
    hat = jnp.fft.rfftn(x, axes=(1, 2, 3), norm="forward")
    transformed = jnp.einsum("bxyzi,xyzio->bxyzo", hat[:, ix, iy, iz], selected)
    out = jnp.zeros_like(hat).at[:, ix, iy, iz].set(transformed)
    return jnp.fft.irfftn(out, s=shape, axes=(1, 2, 3), norm="forward").astype(x.dtype)


def init_params(
    width: int = 16, modes: int = 4, layers: int = 3, seed: int = 0
) -> dict[str, jax.Array]:
    """Initialize one bank of weights shared by all spatial resolutions."""
    if min(width, modes, layers) < 1:
        raise ValueError("positive width, modes and layers required")
    rng = np.random.default_rng(seed)

    def normal(shape: tuple, scale: float) -> jax.Array:
        return jnp.asarray(rng.normal(0, scale, shape), dtype=jnp.float32)

    params = {
        "lift": normal((3, width), 1 / math.sqrt(3)),
        "out": normal((width, 3), 0.01 / math.sqrt(width)),
    }
    for layer in range(layers):
        params[f"local_{layer}"] = normal((width, width), 0.2 / math.sqrt(width))
        params[f"condition_{layer}"] = normal((5, width), 0.02)
        shape = (2 * modes + 1, 2 * modes + 1, modes + 1, width, width)
        params[f"real_{layer}"] = normal(shape, 0.1 / math.sqrt(width))
        params[f"imag_{layer}"] = normal(shape, 0.1 / math.sqrt(width))
    return params


def one_step(
    params: dict,
    velocity: jax.Array,
    nu: jax.Array,
    dt: jax.Array,
    extent: jax.Array,
    *,
    span: int = 1,
    layers: int = 3,
    input_scale: float = 0.3,
    project_output: bool = False,
    conserve_energy: bool = False,
) -> jax.Array:
    """Advance by span teacher steps with physics and discretization conditioning.

    The vector-diffusion skip preserves longitudinal input directions. Projection
    is optional: XLB targets are weakly compressible, including off-manifold
    perturbations used by derivative benchmarks. The zero field remains fixed.
    """
    if span < 1:
        raise ValueError("span must be positive")
    elapsed = dt * span
    features = jnp.stack(
        [
            jnp.log(nu / 0.01),
            jnp.log(dt / 0.02),
            jnp.log(elapsed / 0.02),
            jnp.log((extent / velocity.shape[1]) / (2 * jnp.pi / 16)),
            jnp.log(extent / (2 * jnp.pi)),
        ]
    ).astype(velocity.dtype)
    hidden = jnp.einsum("bxyzi,io->bxyzo", velocity / input_scale, params["lift"])
    for layer in range(layers):
        spectrum = spectral_convolution(
            hidden, params[f"real_{layer}"], params[f"imag_{layer}"]
        )
        local = jnp.einsum("bxyzi,io->bxyzo", hidden, params[f"local_{layer}"])
        gain = 1 + 0.25 * jnp.tanh(features @ params[f"condition_{layer}"])
        hidden = (hidden + jax.nn.gelu((local + spectrum) * gain)) * jnp.float32(
            2**-0.5
        )
    correction = jnp.einsum("bxyzi,io->bxyzo", hidden, params["out"])
    result = diffuse(velocity, nu, elapsed, extent) + correction * input_scale * (
        elapsed / 0.02
    )
    if project_output:
        result = project(result, extent)
    if conserve_energy:
        # Unforced periodic flow conserves mean momentum and dissipates energy.
        # Limit fluctuating energy while preserving the input mean exactly up to
        # reduction roundoff. Epsilon keeps the zero-state derivative finite.
        mean = jnp.mean(velocity, axis=(1, 2, 3), keepdims=True)
        fluctuation = result - jnp.mean(result, axis=(1, 2, 3), keepdims=True)
        previous_energy = jnp.mean(
            (velocity - mean) ** 2, axis=(1, 2, 3, 4), keepdims=True
        )
        next_energy = jnp.mean(fluctuation**2, axis=(1, 2, 3, 4), keepdims=True)
        scale = jnp.minimum(
            1.0, jnp.sqrt((previous_energy + 1e-12) / (next_energy + 1e-12))
        )
        result = mean + scale * fluctuation
    return result


def rollout(
    params: dict,
    initial: jax.Array,
    nu: jax.Array,
    dt: jax.Array,
    extent: jax.Array,
    *,
    updates: int,
    span: int = 1,
    layers: int = 3,
    input_scale: float = 0.3,
    project_output: bool = False,
    conserve_energy: bool = False,
) -> jax.Array:
    """Return autoregressive fields after each update, checkpointing activations."""

    def body(current: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        predicted = one_step(
            params,
            current,
            nu,
            dt,
            extent,
            span=span,
            layers=layers,
            input_scale=input_scale,
            project_output=project_output,
            conserve_energy=conserve_energy,
        )
        return predicted, predicted

    _, predictions = jax.lax.scan(jax.checkpoint(body), initial, None, length=updates)
    return jnp.moveaxis(predictions, 0, 1)


def predict(
    params: dict,
    initial: jax.Array,
    nu: jax.Array,
    dt: jax.Array,
    extent: jax.Array,
    *,
    steps: int,
    stride: int = 1,
    layers: int = 3,
    input_scale: float = 0.3,
    project_output: bool = False,
    conserve_energy: bool = False,
) -> jax.Array:
    """Predict an arbitrary external horizon without returning every field."""
    if steps < 0 or stride < 1:
        raise ValueError("nonnegative steps and positive stride required")

    def body(current: jax.Array, _: None) -> tuple[jax.Array, None]:
        predicted = one_step(
            params,
            current,
            nu,
            dt,
            extent,
            span=stride,
            layers=layers,
            input_scale=input_scale,
            project_output=project_output,
            conserve_energy=conserve_energy,
        )
        return predicted, None

    current, _ = jax.lax.scan(
        jax.checkpoint(body), initial, None, length=steps // stride
    )
    if steps % stride:
        current = one_step(
            params,
            current,
            nu,
            dt,
            extent,
            span=steps % stride,
            layers=layers,
            input_scale=input_scale,
            project_output=project_output,
            conserve_energy=conserve_energy,
        )
    return current
