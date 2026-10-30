# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-dimensional Boussinesq flow in a closed, no-slip square.

The recurrent state has shape (N, N, 3). Channel 0 stores interior x-face
velocity at ((i+1)dx, (j+1/2)dx); its final x row is ignored on input and zero on output.
Channel 1 stores interior y-face velocity at ((i+1/2)dx, (j+1)dx); its
final y column is ignored on input and zero on output. Channel 2 is cell-centered buoyancy scalar.
Both velocity components are zero at the walls, and the scalar has zero
normal derivative. Positive scalar accelerates upward.
This packing preserves the solver state exactly across separate calls.
"""

import math

import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Float32, ShapeDType


def make_plume_ic(N: int = 32, seed: int = 0) -> np.ndarray:
    """Smooth warm plume with seed-dependent position and width, zero velocity."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(
        (np.arange(N) + 0.5) / N, (np.arange(N) + 0.5) / N, indexing="ij"
    )
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.18, 0.3)
    width = rng.uniform(0.07, 0.11)
    state = np.zeros((N, N, 3), dtype=np.float32)
    state[..., 2] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))
    return state


def make_interacting_plumes_ic(
    N: int = 64,
    seed: int = 0,
    position_jitter: float = 0.015,
    width_jitter: float = 0.05,
    amplitude_jitter: float = 0.0,
) -> np.ndarray:
    """Two unequal warm plumes at rest, averaged over each physical grid cell.

    Normalized centers are near (0.36, 0.18) and (0.64, 0.25), with Gaussian
    standard deviations 0.06 and 0.075 and peak amplitudes 1.0 and 0.8.
    Position jitter is absolute in normalized coordinates; width/amplitude
    jitters are relative fractions. Analytic cell averages preserve initial mass
    under refinement, using the same seeded geometry at every resolution.
    """
    for name, value in (
        ("position_jitter", position_jitter),
        ("width_jitter", width_jitter),
        ("amplitude_jitter", amplitude_jitter),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if width_jitter >= 1 or amplitude_jitter >= 1:
        raise ValueError("width_jitter and amplitude_jitter must be less than one.")
    rng = np.random.default_rng(seed)
    centers = np.array([[0.36, 0.18], [0.64, 0.25]]) + rng.uniform(
        -position_jitter, position_jitter, size=(2, 2)
    )
    if np.any((centers <= 0) | (centers >= 1)):
        raise ValueError("Sampled plume centers must remain inside the unit box.")
    widths = np.array([0.06, 0.075]) * rng.uniform(
        1 - width_jitter, 1 + width_jitter, size=2
    )
    amplitudes = np.array([1.0, 0.8])
    # Keep the original draw sequence and defaults bitwise unchanged.
    if amplitude_jitter > 0:
        amplitudes *= rng.uniform(1 - amplitude_jitter, 1 + amplitude_jitter, size=2)
    edges = np.arange(N + 1, dtype=np.float64) / N
    scalar = np.zeros((N, N), dtype=np.float64)
    for center, width, amplitude in zip(centers, widths, amplitudes, strict=True):
        averages = []
        for coordinate in center:
            scaled_edges = (edges - coordinate) / (math.sqrt(2) * width)
            primitive = np.fromiter(
                (math.erf(value) for value in scaled_edges), dtype=np.float64
            )
            averages.append(N * width * math.sqrt(math.pi / 2) * np.diff(primitive))
        scalar += amplitude * averages[0][:, None] * averages[1][None, :]
    state = np.zeros((N, N, 3), dtype=np.float32)
    state[..., 2] = scalar
    return state


def restrict_state(state: np.ndarray, factor: int) -> np.ndarray:
    """Restrict face velocities by face averaging and scalar by volume averaging."""
    state = np.asarray(state)
    if factor < 1 or state.ndim != 3 or state.shape[-1] != 3:
        raise ValueError(
            "Expected a (N, N, 3) state and a positive restriction factor."
        )
    n, m, _ = state.shape
    if n != m or n % factor:
        raise ValueError("Square grid resolution must be divisible by factor.")
    coarse = n // factor
    result = np.zeros((coarse, coarse, 3), dtype=state.dtype)
    result[:-1, :, 0] = (
        state[factor - 1 : n - 1 : factor, :, 0]
        .reshape(coarse - 1, coarse, factor)
        .mean(axis=2)
    )
    result[:, :-1, 1] = (
        state[:, factor - 1 : n - 1 : factor, 1]
        .reshape(coarse, factor, coarse - 1)
        .mean(axis=1)
    )
    result[..., 2] = (
        state[..., 2].reshape(coarse, factor, coarse, factor).mean(axis=(1, 3))
    )
    return result


class InputSchema(BaseModel):
    """Shared buoyancy-grid inputs; all solvers use the same face layout."""

    state0: Array[(None, None, 3), Float32] = Field(
        default_factory=make_plume_ic,
        description=(
            "Packed (N,N,3) staggered velocity ux,uy and cell-centered scalar. "
            "ux[:-1,:] stores interior x faces, uy[:,:-1] interior y faces. "
            "Final ux row and uy column are ignored. Closed no-slip walls; "
            "zero normal scalar derivative."
        ),
    )
    viscosity: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.001], np.float32),
        description="Nonnegative kinematic viscosity.",
    )
    diffusivity: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.0], np.float32),
        description="Nonnegative scalar diffusivity; zero gives an advected marker.",
    )
    buoyancy: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([1.0], np.float32),
        description="Vertical acceleration per unit scalar; positive points upward.",
    )
    dt: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.02], np.float32),
        description="Positive timestep size.",
    )
    steps: int = Field(default=1, ge=1)
    domain_extent: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_grid(self) -> "InputSchema":
        # Abstract evaluation may supply shape descriptors instead of arrays.
        if isinstance(self.state0, (dict, ShapeDType)):
            return self
        state = np.asarray(self.state0)
        if state.shape[0] != state.shape[1] or state.shape[0] < 4:
            raise ValueError("state0 must be square with at least four cells per axis.")
        if not np.isfinite(state).all():
            raise ValueError("state0 must be finite.")
        for name in ("viscosity", "diffusivity", "buoyancy", "dt"):
            value = getattr(self, name)
            if isinstance(value, (dict, ShapeDType)):
                continue
            scalar = float(np.asarray(value)[0])
            if (
                not np.isfinite(scalar)
                or (name in ("viscosity", "diffusivity") and scalar < 0)
                or (name == "dt" and scalar <= 0)
            ):
                raise ValueError(f"Invalid {name}.")
        return self


class OutputSchema(BaseModel):
    """Final state in the identical, exactly resumable staggered packing."""

    result: Array[(None, None, 3), Float32]
