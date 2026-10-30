# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical InputSchema / OutputSchema for navier-stokes-grid tesseracts.

The base schemas carry plain (non-`Differentiable`) array types.  Each solver
wraps the fields it actually supports gradients on via
``mosaic_shared.schema_types.make_differentiable``::

    from mosaic_shared.problems.navier_stokes_grid import (
        InputSchema as _Base,
        OutputSchema as _BaseOut,
    )
    from mosaic_shared.schema_types import make_differentiable

    InputSchema = make_differentiable(
        _Base, ["v0", "viscosity", "dt", "inflow_profile"]
    )
    OutputSchema = make_differentiable(_BaseOut, ["result", "body_force"])

Solvers with extra inputs (e.g. ``inner_steps``) should subclass the result of
``make_differentiable`` and add their additional fields.
"""

import numpy as np
from mosaic_shared.schema_types import (
    BCType,
    GridBC,
    GridObstacle,
    GridScalarField,
    GridVectorField,
)
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Float32


def make_vortex_ic(N: int = 64, L: float = 2 * np.pi, seed: int = 42) -> np.ndarray:
    """Random divergence-free velocity field from a Gaussian-ring stream function.

    Energy concentrated at wavenumber k₀=4 with half-width σ=1.5.

    Returns:
        shape (N, N, 1, 2), float32, normalised to unit max speed.
    """
    rng = np.random.RandomState(seed)
    kn = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    K_abs = np.sqrt(KX**2 + KY**2)

    k0, sigma_k = 4.0, 1.5
    envelope = np.exp(-0.5 * ((K_abs - k0) / sigma_k) ** 2)
    phases = rng.uniform(0, 2 * np.pi, (N, N))
    psi_hat = envelope * np.exp(1j * phases)
    psi_hat = 0.5 * (psi_hat + np.conj(psi_hat[::-1, ::-1]))
    psi_hat[0, 0] = 0.0

    kfac = 2.0 * np.pi / L
    vx = np.real(np.fft.ifft2(1j * KY * kfac * psi_hat))
    vy = np.real(np.fft.ifft2(-1j * KX * kfac * psi_hat))

    u_max = float(np.sqrt(vx**2 + vy**2).max()) or 1.0
    vx, vy = vx / u_max, vy / u_max

    v = np.stack([vx, vy], axis=-1)[:, :, None, :]  # (N, N, 1, 2)
    return v.astype(np.float32)


class InputSchema(BaseModel):
    """Canonical inputs for navier-stokes-grid tesseracts.

    Units are the solver's: lengths in the units of ``domain_extent``, time in
    those of ``dt``, density 1 (``viscosity`` is kinematic, pressure is p/ρ).

    Boundary conditions: every solver must implement the conditions it is
    given or raise. See :class:`~mosaic_shared.schema_types.GridBC` for the
    contract and :class:`~mosaic_shared.schema_types.BCType` for what each
    face type means.
    """

    v0: GridVectorField = Field(
        default_factory=make_vortex_ic,
        description=(
            "Initial velocity at the cell centers, shape (nx, ny, 1, 2) in 2-D "
            "or (nx, ny, nz, 3) in 3-D. Cell (i, j) is centered at "
            "((i + 1/2) dx, (j + 1/2) dx). "
            "Default: 64×64 divergence-free random vortex field (seed=42)."
        ),
    )
    viscosity: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.05], dtype=np.float32),
        description="Kinematic viscosity ν.",
    )
    dt: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.01], dtype=np.float32),
        description="Timestep size.",
    )
    steps: int = Field(
        default=300,
        description="Number of timesteps. Total simulated time = steps × dt.",
    )
    domain_extent: float | list[float] = Field(
        default=2 * np.pi,
        description=(
            "Physical size of the domain. A scalar L is a square (cube) of "
            "side L, and a list gives the length per axis, [Lx, Ly] or "
            "[Lx, Ly, Lz]. Periodic boxes take a scalar. The grid spacing must "
            "be the same along every axis."
        ),
    )
    boundary_conditions: GridBC = Field(
        default_factory=GridBC,
        description="Per-face boundary conditions. Default: all periodic.",
    )
    obstacle: GridObstacle | None = Field(
        default=None,
        description=(
            "Optional solid obstacle embedded in the domain (no-slip walls), "
            "in physical coordinates."
        ),
    )
    inflow_profile: Array[(None,), Float32] | None = Field(
        default=None,
        description=(
            "Inlet velocity u_x(y) at the x_lo face, sampled at the cell-center "
            "heights y_j = (j + 1/2) dx, shape (ny,). u_y is zero on the inlet. "
            "Requires x_lo to be 'dirichlet' with no value."
        ),
    )

    @property
    def extent(self) -> tuple[float, ...]:
        """Domain length along each spatial axis."""
        ndim = self.v0.shape[-1]
        if isinstance(self.domain_extent, list):
            return tuple(float(length) for length in self.domain_extent)
        return (float(self.domain_extent),) * ndim

    @property
    def dx(self) -> float:
        """Grid spacing, the same along every axis."""
        return self.extent[0] / self.v0.shape[0]

    @model_validator(mode="after")
    def _check_grid(self) -> "InputSchema":
        shape = tuple(self.v0.shape)
        ndim = shape[-1]
        if isinstance(self.domain_extent, list) and len(self.domain_extent) != ndim:
            raise ValueError(
                f"domain_extent has {len(self.domain_extent)} entries for a "
                f"{ndim}-D velocity field"
            )
        spacing = [length / n for length, n in zip(self.extent, shape, strict=False)]
        if max(spacing) - min(spacing) > 1e-6 * max(spacing):
            raise ValueError(
                f"grid spacing differs between axes ({spacing}) for v0 of shape "
                f"{shape} on a domain of {list(self.extent)}; the grid must be "
                "isotropic"
            )
        return self

    @model_validator(mode="after")
    def _check_bcs(self) -> "InputSchema":
        bc = self.boundary_conditions
        if bc.is_fully_periodic and isinstance(self.domain_extent, list):
            raise ValueError("A periodic box takes a scalar domain_extent.")
        if not bc.is_fully_periodic and self.obstacle is None:
            raise ValueError(
                "Non-periodic BCs require an obstacle. "
                "Use periodic BCs (the default) or add an obstacle."
            )
        if self.inflow_profile is not None:
            if bc.x_lo.type != BCType.DIRICHLET or bc.x_lo.value is not None:
                raise ValueError(
                    "inflow_profile requires x_lo to be 'dirichlet' with no value"
                )
            ny = self.v0.shape[1]
            if tuple(self.inflow_profile.shape) != (ny,):
                raise ValueError(
                    f"inflow_profile has shape {tuple(self.inflow_profile.shape)}, "
                    f"expected ({ny},) to match v0"
                )
        return self


class OutputSchema(BaseModel):
    """Canonical outputs for navier-stokes-grid tesseracts.

    ``pressure`` and ``body_force`` are required whenever the inputs carry an
    obstacle and may be omitted otherwise.
    """

    result: GridVectorField = Field(
        description="Final velocity at the cell centers, same shape as v0."
    )
    pressure: GridScalarField | None = Field(
        default=None,
        description=(
            "Final kinematic pressure p/ρ at the cell centers, shape "
            "(nx, ny, nz). Values inside the obstacle are ignored."
        ),
    )
    body_force: Array[(None, None), Float32] | None = Field(
        default=None,
        description=(
            "Force of the fluid on the obstacle per unit depth and density, "
            "shape (steps, ndim): row k is (F_x, F_y) at t = (k + 1) dt. "
            "Positive along +x and +y, so a flow in +x gives F_x > 0."
        ),
    )
