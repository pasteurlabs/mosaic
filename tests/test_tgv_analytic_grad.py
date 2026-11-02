# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exact TGV gradient behind ``gradient/analytic_check``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv, _tgv_analytic_grad
from mosaic.benchmarks.problems.shared.diagnostics import project_divergence_free

L = 2 * np.pi


def _spectral_solver(N: int, nu: float, dt: float, steps: int):
    """Dealiased pseudo-spectral 2D Navier–Stokes with RK4, ``(N, N, 1, 2)`` in and out."""
    k = jnp.fft.fftfreq(N, d=L / N) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2s = jnp.where(K2 == 0, 1.0, K2)
    kmax = (2 / 3) * jnp.abs(k).max()
    dealias = (jnp.abs(KX) < kmax) & (jnp.abs(KY) < kmax)

    def proj(uh, vh):
        d = (KX * uh + KY * vh) / K2s
        return uh - KX * d, vh - KY * d

    def rhs(s):
        uh, vh = s
        u, v = jnp.fft.ifft2(uh).real, jnp.fft.ifft2(vh).real

        def adv(fh):
            fx = jnp.fft.ifft2(1j * KX * fh).real
            fy = jnp.fft.ifft2(1j * KY * fh).real
            return -jnp.fft.fft2(u * fx + v * fy) * dealias

        nu_h, nv_h = proj(adv(uh), adv(vh))
        return nu_h - nu * K2 * uh, nv_h - nu * K2 * vh

    def step(s, _):
        def add(a, b, c):
            return (a[0] + c * b[0], a[1] + c * b[1])

        k1 = rhs(s)
        k2 = rhs(add(s, k1, dt / 2))
        k3 = rhs(add(s, k2, dt / 2))
        k4 = rhs(add(s, k3, dt))
        return tuple(
            s[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(2)
        ), None

    def solve(v0):
        s = proj(jnp.fft.fft2(v0[:, :, 0, 0]), jnp.fft.fft2(v0[:, :, 0, 1]))
        (uh, vh), _ = jax.lax.scan(step, s, None, length=steps)
        u, v = jnp.fft.ifft2(uh).real, jnp.fft.ifft2(vh).real
        return jnp.stack([u, v], axis=-1)[:, :, None, :]

    return solve


@pytest.mark.parametrize("amplitude", [1.0, 0.3])
def test_closed_form_matches_a_spectral_solver(amplitude):
    """A solver that keeps TGV exact reproduces the closed form over the whole field.

    Away from the TGV mode the spectral solver's gradient is nonzero only if
    the advection terms fail to drop out, so this also checks the derivation.
    """
    N, nu, dt, steps = 16, 0.01, 0.05, 20
    solve = _spectral_solver(N, nu, dt, steps)
    u0 = amplitude * _tgv(N)
    g = np.asarray(jax.grad(lambda v0: jnp.sum(solve(v0) ** 2))(u0), dtype=np.float64)
    ref = _tgv_analytic_grad(np.asarray(u0), nu=nu, t=dt * steps)
    assert np.linalg.norm(g - ref) / np.linalg.norm(ref) < 1e-5


def test_projection_keeps_tgv_and_drops_gradients():
    N = 16
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    tgv = np.stack([np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)], -1)[:, :, None, :]
    # ∇(sin x sin 2y): curl-free, so the projection removes it entirely.
    grad = np.stack([np.cos(X) * np.sin(2 * Y), 2 * np.sin(X) * np.cos(2 * Y)], -1)
    grad = grad[:, :, None, :]
    np.testing.assert_allclose(project_divergence_free(tgv + grad, L), tgv, atol=1e-12)
