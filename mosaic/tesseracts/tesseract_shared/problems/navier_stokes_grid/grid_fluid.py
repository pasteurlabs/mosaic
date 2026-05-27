# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numpy utilities for 2D grid-based fluid dynamics problems.

All functions operate on raw numpy arrays with shape convention (N, N, 1, 2)
for vector fields and (N, N) for scalar fields, matching GridVectorField /
GridScalarField from tesseract_shared.types.
"""

import numpy as np


def speed(v: np.ndarray) -> np.ndarray:
    """Velocity magnitude |u|. Input (N, N, 1, 2) -> output (N, N)."""
    return np.sqrt(v[:, :, 0, 0] ** 2 + v[:, :, 0, 1] ** 2)


def vorticity_2d(v: np.ndarray) -> np.ndarray:
    """2D vorticity ω = ∂vy/∂x − ∂vx/∂y via central finite differences.

    Input (N, N, 1, 2) -> output (N, N).
    """
    vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    return np.gradient(vy, axis=0) - np.gradient(vx, axis=1)


def divergence_2d(v: np.ndarray, dx: float) -> np.ndarray:
    """2D divergence ∇·u via periodic central finite differences.

    Input (N, N, 1, 2) -> output (N, N).
    """
    vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    return (
        (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0))
        + (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))
    ) / (2 * dx)


def energy_spectrum(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic 1D kinetic energy spectrum E(k).

    Computed via 2D FFT of the velocity field.

    Args:
        v: Velocity field, shape (N, N, 1, 2).

    Returns:
        k_bins: Integer wavenumber bins, shape (N//2 - 1,).
        E_k:    Energy per bin, shape (N//2 - 1,).
    """
    n = v.shape[0]
    vx_hat = np.fft.fft2(v[:, :, 0, 0]) / n**2
    vy_hat = np.fft.fft2(v[:, :, 0, 1]) / n**2
    E_hat = 0.5 * (np.abs(vx_hat) ** 2 + np.abs(vy_hat) ** 2)
    kn = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    k_bins = np.arange(1, n // 2)
    E_k = np.array([E_hat[(k - 0.5 <= K) & (k + 0.5 > K)].sum() for k in k_bins])
    return k_bins, E_k


def spectrum_slope(
    k_bins: np.ndarray, E_k: np.ndarray, k_lo: float = 4.0, k_hi: float = 16.0
) -> float:
    """Power-law slope of E(k) in [k_lo, k_hi] via log-log linear fit.

    Returns nan if fewer than 2 valid points are available.
    Kolmogorov inertial-range theory predicts −5/3 ≈ −1.667.
    """
    mask = (k_bins >= k_lo) & (k_bins <= k_hi) & (E_k > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(k_bins[mask]), np.log(E_k[mask]), 1)[0])
