"""Canonical InputSchema / OutputSchema for navier-stokes-grid tesseracts.

Tesseracts that match this interface exactly can import and use these schemas
directly.  Tesseracts with solver-specific extras should subclass InputSchema
and add their additional fields::

    from mosaic_shared.problems.navier_stokes_grid import InputSchema as _Base

    class InputSchema(_Base):
        inner_steps: int = Field(1, description="Pressure solve sub-iterations")
"""

import numpy as np
from mosaic_shared.types import GridBC, GridObstacle, GridVectorField
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32


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
    v0: Differentiable[GridVectorField] = Field(
        default_factory=make_vortex_ic,
        description=(
            "Initial velocity field, shape (N, N, 1, 2). "
            "Default: 64×64 divergence-free random vortex field (seed=42)."
        ),
    )
    viscosity: Differentiable[Array[(1,), Float32]] = Field(
        default_factory=lambda: np.array([0.05], dtype=np.float32),
        description="Kinematic viscosity ν.",
    )
    dt: Differentiable[Array[(1,), Float32]] = Field(
        default_factory=lambda: np.array([0.01], dtype=np.float32),
        description="Timestep size.",
    )
    steps: int = Field(
        default=300,
        description="Number of timesteps. Total simulated time = steps × dt.",
    )
    domain_extent: float = Field(
        default=2 * np.pi,
        description="Side length of the square periodic domain.",
    )
    boundary_conditions: GridBC = Field(
        default_factory=GridBC,
        description="Per-face boundary conditions. Default: all periodic.",
    )
    obstacle: GridObstacle | None = Field(
        default=None,
        description="Optional solid obstacle embedded in the domain (no-slip walls).",
    )
    inflow_profile: Differentiable[Array[(None,), Float32]] | None = Field(
        default=None,
        description=(
            "1-D inlet velocity profile u_x(y), shape (N,). When provided, overrides "
            "the x_lo Dirichlet BC with a spatially-varying inlet. u_y is set to zero. "
            "Used for inflow-profile optimisation (e.g. cylinder drag minimisation)."
        ),
    )
    lid_velocity: Differentiable[Array[(None, None, 2), Float32]] | None = Field(
        default=None,
        description=(
            "Lid velocity field for 3-D lid-driven cavity experiments, shape (N, N, 2). "
            "The two components are the x- and y-tangential velocities on the moving top lid. "
            "When provided the solver runs in non-periodic cavity mode with no-slip walls. "
            "Used as the optimisation variable in lid_cavity experiments."
        ),
    )

    @model_validator(mode="after")
    def _check_bcs(self) -> "InputSchema":
        if not self.boundary_conditions.is_fully_periodic:
            if self.obstacle is None and self.lid_velocity is None:
                raise ValueError(
                    "Non-periodic BCs require either an obstacle or a lid_velocity. "
                    "Use periodic BCs (the default), add an obstacle, or provide lid_velocity."
                )
        return self


class OutputSchema(BaseModel):
    result: Differentiable[GridVectorField] = Field(
        description="Final velocity field, same shape as v0."
    )
    drag: Differentiable[Array[(1,), Float32]] | None = Field(
        default=None,
        description=(
            "x-direction drag force on the embedded obstacle, shape (1,). "
            "Computed as the surface integral of pressure and viscous stress: "
            "F_x = ∮_S (p n_x − μ (∂u/∂n)_x) dS. "
            "None when no obstacle is present."
        ),
    )
    velocity_mean: Differentiable[GridVectorField] | None = Field(
        default=None,
        description=(
            "Time-averaged (RANS) velocity field over the last steps // 2 steps of the "
            "rollout, same shape as result. Computed on solvers feasible for drag "
            "optimisation (xlb, phiflow, pict) when an obstacle and inflow profile are "
            "present. None on periodic/lid-cavity paths."
        ),
    )
