# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions and the analytic 3D TGV reference."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _tgv3d(N: int, L: float = 2 * jnp.pi, seed: int = 0, **_: Any) -> jax.Array:
    """3D Taylor-Green vortex: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0.

    Divergence-free initial condition for the canonical 3D TGV benchmark.
    Develops turbulent vortex structures at moderate Re; kinetic energy
    dissipation rate peaks around t≈9/ν.

    Returns shape (N, N, N, 3), float32.
    """
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    vx = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    vy = -jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    vz = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, vz], axis=-1).astype(jnp.float32)


def _abc_flow(
    N: int,
    L: float = 2 * jnp.pi,
    A: float = 1.0,
    B: float = 0.8165,  # sqrt(2/3)
    C: float = 0.5774,  # sqrt(1/3)
    seed: int = 0,
    **_: Any,
) -> jax.Array:
    """Arnold-Beltrami-Childress (ABC) flow — a 3D Beltrami field.

    Exact steady solution to the Euler equations with helicity.  Particle
    trajectories are chaotic for A≈B≈C≈1, making this a demanding test for
    gradient signal retention at long horizons.

    u = A*sin(z) + C*cos(y)
    v = B*sin(x) + A*cos(z)
    w = C*sin(y) + B*cos(x)

    Returns shape (N, N, N, 3), float32, normalised to unit max speed.
    """
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    vx = A * jnp.sin(Z) + C * jnp.cos(Y)
    vy = B * jnp.sin(X) + A * jnp.cos(Z)
    vz = C * jnp.sin(Y) + B * jnp.cos(X)
    v = jnp.stack([vx, vy, vz], axis=-1)
    u_max = float(jnp.sqrt(jnp.sum(v**2, axis=-1)).max()) or 1.0
    return (v / u_max).astype(jnp.float32)


def _rand_div_free_3d(
    N: int,
    L: float = 2 * jnp.pi,
    seed: int = 0,
    k_peak: float = 2.0,
    k_width: float = 1.0,
    **_: Any,
) -> jax.Array:
    """Random divergence-free 3D velocity field via curl of a spectral vector potential.

    Generates three random Fourier vector-potential components with energy
    concentrated in a shell at |k| = k_peak (width k_width), then computes
    u = curl(A) in Fourier space (exactly divergence-free: div(curl(A)) = 0).
    Normalised to unit max speed.
    """
    key = jax.random.PRNGKey(seed)
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = jnp.meshgrid(kn, kn, kn, indexing="ij")
    K_abs = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    envelope = jnp.exp(-0.5 * ((K_abs - k_peak) / k_width) ** 2)
    keys = jax.random.split(key, 6)
    kfac = 2.0 * jnp.pi / L
    Ax = (
        jax.random.normal(keys[0], (N, N, N))
        + 1j * jax.random.normal(keys[1], (N, N, N))
    ) * envelope
    Ay = (
        jax.random.normal(keys[2], (N, N, N))
        + 1j * jax.random.normal(keys[3], (N, N, N))
    ) * envelope
    Az = (
        jax.random.normal(keys[4], (N, N, N))
        + 1j * jax.random.normal(keys[5], (N, N, N))
    ) * envelope
    ux_hat = 1j * kfac * (KY * Az - KZ * Ay)
    uy_hat = 1j * kfac * (KZ * Ax - KX * Az)
    uz_hat = 1j * kfac * (KX * Ay - KY * Ax)
    vx = jnp.real(jnp.fft.ifftn(ux_hat))
    vy = jnp.real(jnp.fft.ifftn(uy_hat))
    vz = jnp.real(jnp.fft.ifftn(uz_hat))
    u_max = float(jnp.sqrt(vx**2 + vy**2 + vz**2).max()) or 1.0
    return jnp.stack([vx / u_max, vy / u_max, vz / u_max], axis=-1).astype(jnp.float32)


def _tgv3d_analytic(
    ic: jax.Array, nu: float, t: float, L: float = 2 * jnp.pi
) -> jax.Array:
    """Exact 3D TGV viscous decay: u(t) = u(0) * exp(-2*nu*t)."""
    N = ic.shape[0]
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    z = jnp.linspace(0, L, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    decay = jnp.exp(-2.0 * nu * t)
    vx = jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z) * decay
    vy = -jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z) * decay
    vz = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, vz], axis=-1).astype(jnp.float32)
