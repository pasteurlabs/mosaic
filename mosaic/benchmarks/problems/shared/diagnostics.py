"""Shared diagnostic functions for fluid-velocity ``DIAGNOSTICS`` dicts.

Used by both ``navier_stokes_grid`` (2D, shape ``(N, N, 1, 2)``) and
``navier_stokes_3d_grid`` (3D, shape ``(N, N, N, 3)``); the functions
read ``arr.shape[-1]`` and switch internally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def divergence_rms(arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_) -> float:
    """RMS divergence ∇·u for periodic velocity fields.

    Handles 2D ``(N, N, 1, 2)`` and 3D ``(N, N, N, 3)``.
    """
    ndim = arr.shape[-1]
    dx = domain_extent / arr.shape[0]
    div = sum(
        (jnp.roll(arr[..., i], -1, i) - jnp.roll(arr[..., i], 1, i))
        for i in range(ndim)
    ) / (2 * dx)
    return float(jnp.sqrt(jnp.mean(div**2)))


def kinetic_energy(arr: jax.Array, **_) -> float:
    """Mean kinetic energy ½ ⟨|u|²⟩."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def energy_spectrum(arr: jax.Array, **_) -> dict:
    """Isotropic 1-D energy spectrum E(k).

    Handles 2D ``(N, N, 1, 2)`` and 3D ``(N, N, N, 3)``.
    """
    ndim = arr.shape[-1]
    N = arr.shape[0]
    kn = jnp.fft.fftfreq(N, d=1.0 / N)

    if ndim == 2:
        v_hat = [jnp.fft.fft2(arr[:, :, 0, d]) / N**2 for d in range(2)]
        axes = jnp.meshgrid(kn, kn, indexing="ij")
    else:
        v_hat = [jnp.fft.fftn(arr[..., d]) / N**3 for d in range(3)]
        axes = jnp.meshgrid(kn, kn, kn, indexing="ij")

    E_hat = 0.5 * sum(jnp.abs(vh) ** 2 for vh in v_hat)
    K = jnp.sqrt(sum(ax**2 for ax in axes))
    k_bins = np.arange(1, N // 2)
    E_k = jnp.array(
        [float(E_hat[(k - 0.5 <= K) & (k + 0.5 > K)].sum()) for k in k_bins]
    )
    return {"k": k_bins.tolist(), "E_k": E_k.tolist()}


#: Default DIAGNOSTICS dict for fluid grid problems. Both 2D and 3D.
FLUID_DIAGNOSTICS = {
    "divergence_rms": divergence_rms,
    "kinetic_energy": kinetic_energy,
    "energy_spectrum": energy_spectrum,
}
