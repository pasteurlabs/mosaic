"""Explicit experimental projection matching the centered divergence/gradient.

The four joint DC/Nyquist null modes have zero inverse symbol. These modes are
already divergence-free under centered differences; they are preserved, not
filtered. Only the activated SSPRK3 variant uses this projection.
"""

import functools

import numpy as np
import warp as wp

_original_continuous_inverse = globals()["_inv_lambda_2d"]


@functools.cache
def _centered_inverse(n: int, domain_extent: float, device: str):
    symbol = np.sin(2 * np.pi * np.fft.fftfreq(n)) * n / domain_extent
    symbol[0] = 0.0
    if n % 2 == 0:
        symbol[n // 2] = 0.0
    lam = -(symbol[:, None] ** 2 + symbol[None, :] ** 2)
    inverse = np.zeros_like(lam)
    np.divide(1.0, lam, out=inverse, where=lam != 0)
    return wp.array2d(inverse.astype(np.float32), dtype=wp.float32, device=device)


def _inv_lambda_2d(n: int, domain_extent: float, device: str):
    if globals()["_rk3_enabled"]():
        return _centered_inverse(n, domain_extent, device)
    return _original_continuous_inverse(n, domain_extent, device)
