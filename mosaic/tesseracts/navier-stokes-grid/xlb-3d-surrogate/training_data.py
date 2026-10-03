# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial-condition distribution used for XLB trajectory generation."""

from __future__ import annotations

import numpy as np

N = 16


def _rand_div_free(
    rng: np.random.Generator,
    *,
    k_peak: float,
    k_width: float,
) -> np.ndarray:
    """Sample one smooth periodic divergence-free velocity field."""
    kn = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky, kz = np.meshgrid(kn, kn, kn, indexing="ij")
    k_abs = np.sqrt(kx**2 + ky**2 + kz**2)
    envelope = np.exp(-0.5 * ((k_abs - k_peak) / k_width) ** 2)
    potentials = [
        (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
        * envelope
        for _ in range(3)
    ]
    ax, ay, az = potentials
    velocity = np.stack(
        [
            np.fft.ifftn(1j * (ky * az - kz * ay)).real,
            np.fft.ifftn(1j * (kz * ax - kx * az)).real,
            np.fft.ifftn(1j * (kx * ay - ky * ax)).real,
        ],
        axis=-1,
    )
    velocity -= np.mean(velocity, axis=(0, 1, 2), keepdims=True)
    max_speed = np.sqrt(np.sum(velocity**2, axis=-1)).max()
    return (velocity / max(max_speed, 1e-12)).astype(np.float32)


def make_inputs(
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the deterministic training IC distribution and metadata arrays."""
    rng = np.random.default_rng(seed)
    fields = np.empty((count, N, N, N, 3), dtype=np.float32)
    amplitudes = np.empty((count,), dtype=np.float32)
    families = np.empty((count,), dtype=np.int8)
    for index in range(count):
        family = index % 5
        # Four fifths match the exact recovery spectrum. The last fifth
        # broadens coverage of states visited by optimization.
        if family < 4:
            k_peak = 2.0
            k_width = 1.0
        else:
            k_peak = rng.uniform(1.25, 4.5)
            k_width = rng.uniform(0.5, 1.6)
        field = _rand_div_free(rng, k_peak=k_peak, k_width=k_width)
        bucket = index % 10
        if bucket == 0:
            amplitude = rng.uniform(0.0, 0.15)
        elif bucket < 3:
            amplitude = rng.uniform(0.15, 0.5)
        else:
            amplitude = rng.uniform(0.5, 1.25)
        fields[index] = amplitude * field
        amplitudes[index] = amplitude
        families[index] = family

    # Exact anchors make the zero-to-unit-amplitude recovery path explicit.
    anchor_rng = np.random.default_rng(seed + 1_000_000)
    anchor = _rand_div_free(anchor_rng, k_peak=2.0, k_width=1.0)
    for index, amplitude in enumerate((0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25)):
        if index >= count:
            break
        fields[index] = amplitude * anchor
        amplitudes[index] = amplitude
        families[index] = 0
    return fields, amplitudes, families
