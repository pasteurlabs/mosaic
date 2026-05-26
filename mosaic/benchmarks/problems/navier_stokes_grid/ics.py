# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions and the analytic TGV reference."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _multimode(N: int, L: float = 2 * jnp.pi, seed: int = 42, **_: Any) -> jax.Array:
    """Energy ring at k=2, σ=0.5, max speed normalised to 0.3."""
    key = jax.random.PRNGKey(seed)
    kn = jnp.fft.fftfreq(N, d=1.0 / N)
    KX, KY = jnp.meshgrid(kn, kn, indexing="ij")
    K_abs = jnp.sqrt(KX**2 + KY**2)
    envelope = jnp.exp(-0.5 * ((K_abs - 2.0) / 0.5) ** 2)
    phases = jax.random.uniform(key, (N, N), minval=0.0, maxval=2.0 * jnp.pi)
    psi_hat = envelope * jnp.exp(1j * phases)
    psi_hat = 0.5 * (psi_hat + jnp.conj(psi_hat[::-1, ::-1]))
    psi_hat = psi_hat.at[0, 0].set(0.0)
    kfac = 2.0 * jnp.pi / L
    vx = jnp.real(jnp.fft.ifft2(1j * KY * kfac * psi_hat))
    vy = jnp.real(jnp.fft.ifft2(-1j * KX * kfac * psi_hat))
    u_max = float(jnp.sqrt(vx**2 + vy**2).max()) or 1.0
    vx, vy = vx / u_max * 0.3, vy / u_max * 0.3
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)


def _tgv(N: int, L: float = 2 * jnp.pi, seed: int = 0, **_: Any) -> jax.Array:
    """Taylor-Green vortex: u=sin(x)cos(y), v=-cos(x)sin(y)."""
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    vx = jnp.sin(X) * jnp.cos(Y)
    vy = -jnp.cos(X) * jnp.sin(Y)
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)


def _uniform_flow(N: int, U: float = 1.0, **_: Any) -> jax.Array:
    """Uniform rightward flow u=(U, 0) — canonical IC for cylinder-wake experiments."""
    vx = jnp.full((N, N), U, dtype=jnp.float32)
    vy = jnp.zeros((N, N), dtype=jnp.float32)
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :]


def _flat_inflow(N: int = 64, U: float = 0.5, **_: Any) -> jax.Array:
    """Flat inlet profile u_x(y) = U, shape (N,). Starting point for drag_opt."""
    return jnp.full((N,), U, dtype=jnp.float32)


def _tgv_analytic(
    ic: jax.Array, nu: float, t: float, L: float = 2 * jnp.pi
) -> jax.Array:
    """Exact TGV solution at time t: decays as exp(-2*nu*t)."""
    N = ic.shape[0]
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    decay = jnp.exp(-2.0 * nu * t)
    vx = jnp.sin(X) * jnp.cos(Y) * decay
    vy = -jnp.cos(X) * jnp.sin(Y) * decay
    return jnp.stack([vx, vy], axis=-1)[:, :, None, :].astype(jnp.float32)
