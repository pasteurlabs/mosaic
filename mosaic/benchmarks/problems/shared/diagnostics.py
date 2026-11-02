# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fluid-velocity helpers for ``DIAGNOSTICS`` dicts and divergence-free projection.

Used by both ``navier_stokes_grid`` (2D, shape ``(N, N, 1, 2)``) and
``navier_stokes_3d_grid`` (3D, shape ``(N, N, N, 3)``); the functions
read ``arr.shape[-1]`` and switch internally.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def divergence_rms(
    arr: jax.Array, domain_extent: float = 2 * jnp.pi, **_: Any
) -> float:
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


def kinetic_energy(arr: jax.Array, **_: Any) -> float:
    """Mean kinetic energy ½ ⟨|u|²⟩."""
    return float(0.5 * jnp.mean(jnp.sum(arr**2, axis=-1)))


def energy_spectrum(arr: jax.Array, **_: Any) -> dict:
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


def project_divergence_free(u: np.ndarray, domain_extent: float) -> np.ndarray:
    """Helmholtz-project a periodic velocity field onto ∇·u = 0.

    Spectral projection: û_df_i = û_i − k_i (k·û) / |k|²
    Handles both 2D (N, N, 1, 2) and 3D (N, N, N, 3) shapes.
    """
    nd = u.shape[-1]
    squeeze = u.ndim > nd + 1  # True for (N,N,1,2); False for (N,N,N,3)
    v = u.reshape((*u.shape[:nd], nd)) if squeeze else u

    N = v.shape[0]
    # Physical wavenumbers for a periodic box of size domain_extent.
    k1d = np.fft.fftfreq(N) * N * (2.0 * np.pi / domain_extent)
    grids = np.meshgrid(*([k1d] * nd), indexing="ij")  # nd arrays, each (N,...,N)
    k2 = sum(k**2 for k in grids)
    k2_safe = np.where(k2 == 0.0, 1.0, k2)  # avoid ÷0 at k=0

    spatial_axes = tuple(range(nd))
    v_hat = np.fft.fftn(v, axes=spatial_axes)  # (..., nd), complex

    k_dot_u = sum(grids[i] * v_hat[..., i] for i in range(nd))  # scalar field

    v_hat_df = np.stack(
        [v_hat[..., i] - grids[i] * k_dot_u / k2_safe for i in range(nd)],
        axis=-1,
    )

    v_df = np.fft.ifftn(v_hat_df, axes=spatial_axes).real
    v_df = v_df.reshape(u.shape) if squeeze else v_df
    return v_df.astype(u.dtype)


#: Default DIAGNOSTICS dict for fluid grid problems. Both 2D and 3D.
FLUID_DIAGNOSTICS = {
    "divergence_rms": divergence_rms,
    "kinetic_energy": kinetic_energy,
    "energy_spectrum": energy_spectrum,
}
