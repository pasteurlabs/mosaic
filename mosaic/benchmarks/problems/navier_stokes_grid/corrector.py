# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-owned neural corrector and periodic reference dynamics.

The corrector deliberately lives in the benchmark harness instead of any
solver Tesseract.  Every candidate solver therefore sees the same model,
initial weights, projection, optimiser, and reference trajectories; only the
``velocity -> result`` transition and its VJP vary between benchmark cells.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class PeriodicResidualCNN(eqx.Module):
    """Small translation-equivariant corrector built from periodic convolutions."""

    layers: tuple[eqx.nn.Conv2d, ...]
    architecture: str = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        *,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        channels = (2, hidden_channels, hidden_channels * 2, 2)
        keys = jax.random.split(key, len(channels) - 1)
        layers: list[eqx.nn.Conv2d] = []
        for idx, (cin, cout, layer_key) in enumerate(
            zip(channels[:-1], channels[1:], keys, strict=True)
        ):
            layer = eqx.nn.Conv2d(
                cin,
                cout,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="CIRCULAR",
                key=layer_key,
            )
            if idx == len(channels) - 2:
                # Start from the underlying solver exactly. This matters when
                # its forward residual is already near numerical precision.
                weight = jnp.zeros_like(layer.weight)
            else:
                fan_in = kernel_size * kernel_size * cin
                scale = np.sqrt(2.0 / fan_in)
                weight = (
                    jax.random.normal(layer_key, layer.weight.shape, dtype=jnp.float32)
                    * scale
                )
            layer = eqx.tree_at(
                lambda value: (value.weight, value.bias),
                layer,
                (
                    weight,
                    jnp.zeros((cout, 1, 1), dtype=jnp.float32),
                ),
            )
            layers.append(layer)
        self.layers = tuple(layers)
        self.architecture = "periodic_residual_cnn"
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

    def __call__(self, velocity: jax.Array) -> jax.Array:
        """Map a channel-first velocity field to an additive correction."""
        x = velocity
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


def init_corrector(
    key: jax.Array,
    *,
    hidden_channels: int = 32,
    kernel_size: int = 5,
    architecture: str = "periodic_residual_cnn",
) -> PeriodicResidualCNN:
    """Initialise the benchmark-owned Equinox corrector.

    The final layer starts at zero, so the initial corrected rollout is exactly
    the underlying solver. The first update trains the readout; subsequent
    updates propagate through the feature layers.
    """
    if architecture != "periodic_residual_cnn":
        raise ValueError(f"unknown corrector architecture: {architecture!r}")
    return PeriodicResidualCNN(
        key,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    )


def apply_corrector(
    model: PeriodicResidualCNN,
    velocity: jax.Array,
    *,
    velocity_scale: float | jax.Array,
) -> jax.Array:
    """Predict an additive velocity correction on ``(N, N, 1, 2)`` fields."""
    x = velocity[:, :, 0, :] / jnp.asarray(velocity_scale, dtype=velocity.dtype)
    delta = jnp.moveaxis(model(jnp.moveaxis(x, -1, 0)), 0, -1)
    return delta[:, :, jnp.newaxis, :] * jnp.asarray(
        velocity_scale, dtype=velocity.dtype
    )


def project_periodic_correction(delta: jax.Array, domain_extent: float) -> jax.Array:
    """Project a velocity correction onto zero-mean divergence-free fields."""
    field = delta[:, :, 0, :]
    nx, ny = field.shape[:2]
    dx, dy = domain_extent / nx, domain_extent / ny
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2

    u_hat = jnp.fft.fft2(field[..., 0])
    v_hat = jnp.fft.fft2(field[..., 1])
    k_dot_u = kx_grid * u_hat + ky_grid * v_hat
    safe_k2 = jnp.where(k2 == 0, 1.0, k2)
    u_hat = u_hat - kx_grid * k_dot_u / safe_k2
    v_hat = v_hat - ky_grid * k_dot_u / safe_k2
    # The signed Nyquist wave number has no conjugate partner on an even
    # real-valued grid.  Removing those two lines keeps the projected spectrum
    # Hermitian, rather than silently discarding an imaginary component below.
    if nx % 2 == 0:
        u_hat = u_hat.at[nx // 2, :].set(0.0)
        v_hat = v_hat.at[nx // 2, :].set(0.0)
    if ny % 2 == 0:
        u_hat = u_hat.at[:, ny // 2].set(0.0)
        v_hat = v_hat.at[:, ny // 2].set(0.0)
    # A corrector should not change the conserved mean flow.
    u_hat = u_hat.at[0, 0].set(0.0)
    v_hat = v_hat.at[0, 0].set(0.0)

    projected = jnp.stack(
        [jnp.fft.ifft2(u_hat).real, jnp.fft.ifft2(v_hat).real],
        axis=-1,
    )
    return projected[:, :, jnp.newaxis, :].astype(delta.dtype)


def corrected_velocity(
    model: PeriodicResidualCNN,
    provisional: jax.Array,
    *,
    velocity_scale: float | jax.Array,
    domain_extent: float,
) -> jax.Array:
    """Apply the learned residual and enforce a physical correction subspace."""
    delta = apply_corrector(model, provisional, velocity_scale=velocity_scale)
    return provisional + project_periodic_correction(delta, domain_extent)


def relative_l2(predicted: Any, target: Any) -> float:
    """Relative L2 error with a stable zero-target denominator."""
    p = np.asarray(predicted, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(p - q) / (np.linalg.norm(q) + 1e-12))


def divergence_rms(velocity: Any, domain_extent: float) -> float:
    """Spectral RMS divergence for one canonical 2D velocity field."""
    field = np.asarray(velocity)
    if field.ndim == 4:
        field = field[:, :, 0, :]
    n = field.shape[0]
    dx = domain_extent / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    div_hat = 1j * (kx * np.fft.fft2(field[..., 0]) + ky * np.fft.fft2(field[..., 1]))
    return float(np.sqrt(np.mean(np.fft.ifft2(div_hat).real ** 2)))


def centered_divergence_rms(velocity: Any, domain_extent: float) -> float:
    """Centered-difference RMS divergence for one canonical velocity field."""
    field = np.asarray(velocity)
    if field.ndim == 4:
        field = field[:, :, 0, :]
    dx = domain_extent / field.shape[0]
    dy = domain_extent / field.shape[1]
    du_dx = (np.roll(field[..., 0], -1, axis=0) - np.roll(field[..., 0], 1, axis=0)) / (
        2.0 * dx
    )
    dv_dy = (np.roll(field[..., 1], -1, axis=1) - np.roll(field[..., 1], 1, axis=1)) / (
        2.0 * dy
    )
    return float(np.sqrt(np.mean((du_dx + dv_dy) ** 2)))


def kinetic_energy(velocity: Any) -> float:
    """Mean kinetic energy density for one canonical velocity field."""
    field = np.asarray(velocity)
    if field.ndim == 4:
        field = field[:, :, 0, :]
    return float(0.5 * np.mean(np.sum(field**2, axis=-1)))


def enstrophy(velocity: Any, domain_extent: float) -> float:
    """Mean enstrophy density using centered periodic differences."""
    field = np.asarray(velocity)
    if field.ndim == 4:
        field = field[:, :, 0, :]
    dx = domain_extent / field.shape[0]
    dy = domain_extent / field.shape[1]
    dv_dx = (np.roll(field[..., 1], -1, axis=0) - np.roll(field[..., 1], 1, axis=0)) / (
        2.0 * dx
    )
    du_dy = (np.roll(field[..., 0], -1, axis=1) - np.roll(field[..., 0], 1, axis=1)) / (
        2.0 * dy
    )
    return float(0.5 * np.mean((dv_dx - du_dy) ** 2))


def spectral_restrict(velocity: Any, target_n: int) -> np.ndarray:
    """Fourier-restrict a periodic velocity field to ``target_n`` cells."""
    field = np.asarray(velocity)
    had_z_axis = field.ndim == 4
    if had_z_axis:
        field = field[:, :, 0, :]
    source_n = field.shape[0]
    if source_n == target_n:
        result = field.astype(np.float32, copy=True)
    else:
        if source_n < target_n or source_n % target_n != 0:
            raise ValueError("source grid must be an integer multiple of target_n")
        spectrum = np.fft.fftshift(np.fft.fft2(field, axes=(0, 1)), axes=(0, 1))
        start = (source_n - target_n) // 2
        stop = start + target_n
        cropped = spectrum[start:stop, start:stop, :]
        result = np.fft.ifft2(np.fft.ifftshift(cropped, axes=(0, 1)), axes=(0, 1)).real
        result *= (target_n / source_n) ** 2
        result = result.astype(np.float32)
    return result[:, :, np.newaxis, :] if had_z_axis else result


def spectral_prolong(velocity: Any, target_n: int) -> np.ndarray:
    """Fourier-prolong a periodic velocity field to ``target_n`` cells."""
    field = np.asarray(velocity)
    had_z_axis = field.ndim == 4
    if had_z_axis:
        field = field[:, :, 0, :]
    source_n = field.shape[0]
    if source_n == target_n:
        result = field.astype(np.float32, copy=True)
    else:
        if source_n > target_n or target_n % source_n != 0:
            raise ValueError("target grid must be an integer multiple of source_n")
        spectrum = np.fft.fftshift(np.fft.fft2(field, axes=(0, 1)), axes=(0, 1))
        padded = np.zeros((target_n, target_n, field.shape[-1]), dtype=np.complex128)
        start = (target_n - source_n) // 2
        stop = start + source_n
        padded[start:stop, start:stop, :] = spectrum
        result = np.fft.ifft2(
            np.fft.ifftshift(padded, axes=(0, 1)),
            axes=(0, 1),
        ).real
        result *= (target_n / source_n) ** 2
        result = result.astype(np.float32)
    return result[:, :, np.newaxis, :] if had_z_axis else result


def _velocity_to_vorticity_hat(
    velocity: np.ndarray, domain_extent: float
) -> np.ndarray:
    """Return the spectral scalar vorticity of a 2D velocity field."""
    field = velocity[:, :, 0, :] if velocity.ndim == 4 else velocity
    n = field.shape[0]
    dx = domain_extent / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    u_hat = np.fft.fft2(field[..., 0])
    v_hat = np.fft.fft2(field[..., 1])
    return 1j * kx * v_hat - 1j * ky * u_hat


def _vorticity_hat_to_velocity(
    omega_hat: np.ndarray, domain_extent: float
) -> np.ndarray:
    """Recover canonical velocity from spectral vorticity."""
    n = omega_hat.shape[0]
    dx = domain_extent / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    k2 = kx**2 + ky**2
    safe_k2 = np.where(k2 == 0, 1.0, k2)
    psi_hat = omega_hat / safe_k2
    psi_hat[0, 0] = 0.0
    u = np.fft.ifft2(1j * ky * psi_hat).real
    v = np.fft.ifft2(-1j * kx * psi_hat).real
    return np.stack([u, v], axis=-1)[:, :, np.newaxis, :].astype(np.float32)


def _vorticity_rhs(
    omega_hat: np.ndarray,
    *,
    viscosity: float,
    domain_extent: float,
    operators: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Dealiased pseudo-spectral RHS for 2D vorticity dynamics."""
    n = omega_hat.shape[0]
    if operators is None:
        dx = domain_extent / n
        wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        mode = np.fft.fftfreq(n) * n
        kx, ky = np.meshgrid(wave, wave, indexing="ij")
        mx, my = np.meshgrid(mode, mode, indexing="ij")
        k2 = kx**2 + ky**2
        cutoff = n // 3
        dealias = (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff)
    else:
        kx, ky, k2, dealias = operators
    safe_k2 = np.where(k2 == 0, 1.0, k2)
    psi_hat = omega_hat / safe_k2
    psi_hat[0, 0] = 0.0

    u = np.fft.ifft2(1j * ky * psi_hat).real
    v = np.fft.ifft2(-1j * kx * psi_hat).real
    grad_x = np.fft.ifft2(1j * kx * omega_hat).real
    grad_y = np.fft.ifft2(1j * ky * omega_hat).real
    advection_hat = np.fft.fft2(u * grad_x + v * grad_y)
    advection_hat *= dealias
    return -advection_hat - viscosity * k2 * omega_hat


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Three-argument minmod limiter for periodic finite-volume slopes."""
    same_sign = (np.signbit(a) == np.signbit(b)) & (np.signbit(b) == np.signbit(c))
    magnitude = np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    return np.where(same_sign, np.copysign(magnitude, a), 0.0)


def _finite_volume_vorticity_to_velocity(
    omega: np.ndarray,
    *,
    dx: float,
    safe_k2: np.ndarray,
) -> np.ndarray:
    """Recover cell-centred velocity with the finite-volume discrete operators."""
    psi_hat = np.fft.fft2(omega) / safe_k2
    psi_hat[0, 0] = 0.0
    psi = np.fft.ifft2(psi_hat).real
    u = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    v = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dx)
    return np.stack([u, v], axis=-1)[:, :, np.newaxis, :]


def _finite_volume_vorticity_rhs(
    omega: np.ndarray,
    *,
    viscosity: float,
    domain_extent: float,
    operators: tuple[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Conservative MUSCL finite-volume RHS for periodic 2-D vorticity.

    The advective update uses monotonized-central reconstruction and a local
    Rusanov flux. Velocity is recovered from a discrete streamfunction whose
    Fourier symbol exactly inverts the cell-centred five-point Laplacian. The
    evolution is therefore independently discretized from
    :func:`_vorticity_rhs`: only the periodic elliptic inversion uses FFTs,
    matching the pressure solve used by several candidate grid solvers.
    """
    n = omega.shape[0]
    if operators is None:
        dx = domain_extent / n
        mode = np.fft.fftfreq(n) * n
        mx, my = np.meshgrid(mode, mode, indexing="ij")
        discrete_k2 = (
            4.0 * (np.sin(np.pi * mx / n) ** 2 + np.sin(np.pi * my / n) ** 2) / dx**2
        )
        safe_k2 = np.where(discrete_k2 == 0.0, 1.0, discrete_k2)
    else:
        dx, safe_k2 = operators

    velocity = _finite_volume_vorticity_to_velocity(
        omega,
        dx=dx,
        safe_k2=safe_k2,
    )
    u = velocity[:, :, 0, 0]
    v = velocity[:, :, 0, 1]
    u_face = 0.5 * (u + np.roll(u, -1, axis=0))
    v_face = 0.5 * (v + np.roll(v, -1, axis=1))

    backward_x = omega - np.roll(omega, 1, axis=0)
    forward_x = np.roll(omega, -1, axis=0) - omega
    centered_x = 0.5 * (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0))
    slope_x = _minmod3(2.0 * backward_x, centered_x, 2.0 * forward_x)
    omega_x_left = omega + 0.5 * slope_x
    omega_x_right = np.roll(omega, -1, axis=0) - 0.5 * np.roll(slope_x, -1, axis=0)
    flux_x = 0.5 * u_face * (omega_x_left + omega_x_right) - 0.5 * np.abs(u_face) * (
        omega_x_right - omega_x_left
    )

    backward_y = omega - np.roll(omega, 1, axis=1)
    forward_y = np.roll(omega, -1, axis=1) - omega
    centered_y = 0.5 * (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1))
    slope_y = _minmod3(2.0 * backward_y, centered_y, 2.0 * forward_y)
    omega_y_left = omega + 0.5 * slope_y
    omega_y_right = np.roll(omega, -1, axis=1) - 0.5 * np.roll(slope_y, -1, axis=1)
    flux_y = 0.5 * v_face * (omega_y_left + omega_y_right) - 0.5 * np.abs(v_face) * (
        omega_y_right - omega_y_left
    )

    advection = (
        -((flux_x - np.roll(flux_x, 1, axis=0)) + (flux_y - np.roll(flux_y, 1, axis=1)))
        / dx
    )
    laplacian = (
        np.roll(omega, -1, axis=0)
        + np.roll(omega, 1, axis=0)
        + np.roll(omega, -1, axis=1)
        + np.roll(omega, 1, axis=1)
        - 4.0 * omega
    ) / dx**2
    return advection + viscosity * laplacian


def reference_trajectory(
    initial_velocity: Any,
    *,
    viscosity: float,
    dt: float,
    frame_steps: int,
    n_frames: int,
    substeps: int,
    domain_extent: float,
) -> np.ndarray:
    """Integrate a high-resolution periodic reference with spectral RK4."""
    velocity = np.asarray(initial_velocity, dtype=np.float64)
    omega_hat = _velocity_to_vorticity_hat(velocity, domain_extent)
    n = omega_hat.shape[0]
    dx = domain_extent / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    mode = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    mx, my = np.meshgrid(mode, mode, indexing="ij")
    k2_operator = kx**2 + ky**2
    cutoff = n // 3
    spectral_operators = (
        kx,
        ky,
        k2_operator,
        (np.abs(mx) <= cutoff) & (np.abs(my) <= cutoff),
    )
    step_dt = dt / substeps
    trajectory = [velocity.astype(np.float32)]
    for _ in range(n_frames):
        for _ in range(frame_steps * substeps):
            k1 = _vorticity_rhs(
                omega_hat,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=spectral_operators,
            )
            k2 = _vorticity_rhs(
                omega_hat + 0.5 * step_dt * k1,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=spectral_operators,
            )
            k3 = _vorticity_rhs(
                omega_hat + 0.5 * step_dt * k2,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=spectral_operators,
            )
            k4 = _vorticity_rhs(
                omega_hat + step_dt * k3,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=spectral_operators,
            )
            omega_hat = omega_hat + step_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        trajectory.append(_vorticity_hat_to_velocity(omega_hat, domain_extent))
    return np.stack(trajectory)


def finite_volume_reference_trajectory(
    initial_velocity: Any,
    *,
    viscosity: float,
    dt: float,
    frame_steps: int,
    n_frames: int,
    substeps: int,
    domain_extent: float,
) -> np.ndarray:
    """Integrate an independent conservative finite-volume reference with RK4."""
    velocity = np.asarray(initial_velocity, dtype=np.float64)
    omega = np.fft.ifft2(_velocity_to_vorticity_hat(velocity, domain_extent)).real
    n = omega.shape[0]
    dx = domain_extent / n
    mode = np.fft.fftfreq(n) * n
    mx, my = np.meshgrid(mode, mode, indexing="ij")
    discrete_k2 = (
        4.0 * (np.sin(np.pi * mx / n) ** 2 + np.sin(np.pi * my / n) ** 2) / dx**2
    )
    finite_volume_operators = (
        dx,
        np.where(discrete_k2 == 0.0, 1.0, discrete_k2),
    )
    step_dt = dt / substeps
    trajectory = [velocity.astype(np.float32)]
    for _ in range(n_frames):
        for _ in range(frame_steps * substeps):
            k1 = _finite_volume_vorticity_rhs(
                omega,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=finite_volume_operators,
            )
            k2 = _finite_volume_vorticity_rhs(
                omega + 0.5 * step_dt * k1,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=finite_volume_operators,
            )
            k3 = _finite_volume_vorticity_rhs(
                omega + 0.5 * step_dt * k2,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=finite_volume_operators,
            )
            k4 = _finite_volume_vorticity_rhs(
                omega + step_dt * k3,
                viscosity=viscosity,
                domain_extent=domain_extent,
                operators=finite_volume_operators,
            )
            omega = omega + step_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        trajectory.append(
            _finite_volume_vorticity_to_velocity(
                omega,
                dx=dx,
                safe_k2=finite_volume_operators[1],
            ).astype(np.float32)
        )
    return np.stack(trajectory)


__all__ = [
    "PeriodicResidualCNN",
    "apply_corrector",
    "centered_divergence_rms",
    "corrected_velocity",
    "divergence_rms",
    "enstrophy",
    "finite_volume_reference_trajectory",
    "init_corrector",
    "kinetic_energy",
    "project_periodic_correction",
    "reference_trajectory",
    "relative_l2",
    "spectral_prolong",
    "spectral_restrict",
]
