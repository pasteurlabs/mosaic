# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic maps between the canonical cell-centred velocity and staggered faces.

The canonical layout is collocated and component-last, ``(*spatial, ndim)``.
The staggered layout has the same shape, with component ``i`` stored on the
high face of each cell along axis ``i``. The solvers on staggered grids map
between the two by two-point averages; these helpers are those averages and
the inverse that solvers continuing from their own state use to take on a
canonical correction. They work on NumPy and JAX arrays alike (``xp``).
"""

from typing import Any

import numpy as np

# Largest amplification the lift may apply to a Fourier mode. The face-to-centre
# average damps mode k by cos(πk/N), which vanishes at the Nyquist mode of an
# even grid, so its inverse is unbounded there. Capping the gain drops the
# Nyquist mode (and only it, on grids up to N = 100) and keeps the lift's norm
# independent of the resolution.
MAX_LIFT_GAIN = 32.0


def _xp(array: Any, xp: Any | None) -> Any:
    if xp is not None:
        return xp
    namespace = getattr(array, "__array_namespace__", None)
    return namespace() if namespace is not None else np


def collocated_to_staggered_periodic(velocity: Any, *, xp: Any = None) -> Any:
    """Average a collocated velocity onto the high faces of a periodic grid."""
    xp = _xp(velocity, xp)
    return xp.stack(
        [
            0.5 * (velocity[..., i] + xp.roll(velocity[..., i], -1, axis=i))
            for i in range(velocity.shape[-1])
        ],
        axis=-1,
    )


def staggered_to_collocated_periodic(velocity: Any, *, xp: Any = None) -> Any:
    """Average high-face velocities back to the cell centres of a periodic grid."""
    xp = _xp(velocity, xp)
    return xp.stack(
        [
            0.5 * (velocity[..., i] + xp.roll(velocity[..., i], 1, axis=i))
            for i in range(velocity.shape[-1])
        ],
        axis=-1,
    )


def lift_collocated_to_staggered_periodic(
    correction: Any, *, max_gain: float = MAX_LIFT_GAIN, xp: Any = None
) -> Any:
    """High-face field whose face-to-centre average is ``correction``.

    The right inverse of :func:`staggered_to_collocated_periodic`, mode by
    mode. Modes it would have to amplify by more than ``max_gain`` are set to
    zero (see :data:`MAX_LIFT_GAIN`).
    """
    xp = _xp(correction, xp)
    lifted = []
    for i in range(correction.shape[-1]):
        n = correction.shape[i]
        transfer = 0.5 * (1.0 + xp.exp(-2j * np.pi * np.fft.fftfreq(n)))
        transfer = transfer.reshape(
            tuple(n if axis == i else 1 for axis in range(correction.ndim - 1))
        )
        supported = xp.abs(transfer) * max_gain >= 1.0
        spectrum = xp.fft.fft(correction[..., i], axis=i)
        spectrum = xp.where(supported, spectrum / xp.where(supported, transfer, 1.0), 0)
        lifted.append(xp.real(xp.fft.ifft(spectrum, axis=i)))
    return xp.stack(lifted, axis=-1).astype(correction.dtype)
