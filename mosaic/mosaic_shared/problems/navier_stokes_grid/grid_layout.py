# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared periodic velocity-layout conversions for grid-based NS solvers.

The public Mosaic velocity layout is collocated and component-last:
``(*spatial, ndim)``.  The staggered layout used here has the same dense shape,
with component ``i`` stored on the high face of each cell along spatial axis
``i``.  Two-dimensional canonical arrays may retain their singleton z axis.
"""

from typing import Any

import numpy as np


def _array_module(array: Any, xp: Any | None) -> Any:
    if xp is not None:
        return xp
    namespace = getattr(array, "__array_namespace__", None)
    return namespace() if namespace is not None else np


def _velocity_ndim(velocity: Any) -> int:
    if not hasattr(velocity, "shape") or len(velocity.shape) < 2:
        raise ValueError("velocity must have shape (*spatial, ndim)")
    ndim = int(velocity.shape[-1])
    if ndim not in (2, 3) or len(velocity.shape) - 1 < ndim:
        raise ValueError(
            "velocity must have two or three components and at least one "
            "spatial axis per component"
        )
    return ndim


def collocated_to_staggered_periodic(
    velocity: Any,
    *,
    xp: Any | None = None,
) -> Any:
    """Interpolate a collocated velocity onto periodic high faces."""
    xp = _array_module(velocity, xp)
    ndim = _velocity_ndim(velocity)
    return xp.stack(
        [
            0.5
            * (
                velocity[..., component]
                + xp.roll(velocity[..., component], -1, axis=component)
            )
            for component in range(ndim)
        ],
        axis=-1,
    )


def staggered_to_collocated_periodic(
    velocity: Any,
    *,
    xp: Any | None = None,
) -> Any:
    """Interpolate periodic high-face velocities back to cell centres."""
    xp = _array_module(velocity, xp)
    ndim = _velocity_ndim(velocity)
    return xp.stack(
        [
            0.5
            * (
                velocity[..., component]
                + xp.roll(velocity[..., component], 1, axis=component)
            )
            for component in range(ndim)
        ],
        axis=-1,
    )


def lift_collocated_to_staggered_periodic(
    correction: Any,
    *,
    max_gain: float = 32.0,
    xp: Any | None = None,
) -> Any:
    """Right-invert face-to-centre averaging on supported Fourier modes.

    Modes that would require amplification above ``max_gain`` are set to zero.
    This always excludes the unrepresentable Nyquist mode of an even grid and
    keeps the inverse's operator norm independent of grid resolution.
    """
    if max_gain < 1.0:
        raise ValueError("max_gain must be at least 1")
    xp = _array_module(correction, xp)
    ndim = _velocity_ndim(correction)
    lifted = []
    for component in range(ndim):
        axis = component
        n_axis = correction.shape[axis]
        frequencies = xp.fft.fftfreq(n_axis)
        transfer = 0.5 * (1.0 + xp.exp(-2j * xp.pi * frequencies))
        transfer = transfer.reshape(
            tuple(n_axis if dim == axis else 1 for dim in range(correction.ndim - 1))
        )
        spectrum = xp.fft.fft(correction[..., component], axis=axis)
        supported = xp.abs(transfer) * max_gain >= 1.0
        denominator = xp.where(supported, transfer, 1.0)
        face_spectrum = xp.where(supported, spectrum / denominator, 0.0)
        lifted.append(xp.fft.ifft(face_spectrum, axis=axis).real)
    return xp.stack(lifted, axis=-1).astype(correction.dtype)
