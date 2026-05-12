"""Initial conditions and IC registry for the thermal-mesh problem."""

from __future__ import annotations

import numpy as np

from mosaic.benchmarks.core.config import IcSpec


def _zero_source(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Zero volumetric source field, shape (nx·ny·nz,). Starting point for source recovery."""
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    return np.zeros((nx * ny * nz,), dtype=np.float32)


def _uniform(
    rho_0: float = 0.5,
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Uniform density ρ₀ over the mesh, shape (nx·ny·nz,).

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    return np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)


def _random(
    rho_0: float = 0.5,
    noise: float = 0.3,
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    seed: int = 0,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Random density field centred at ρ₀ with Gaussian noise, clipped to [0.05, 0.95].

    Breaks spatial symmetry so that gradient visualisations show non-trivial
    per-cell sensitivity rather than a flat field.

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    rng = np.random.default_rng(seed)
    rho = rho_0 + noise * rng.standard_normal(nx * ny * nz).astype(np.float32)
    return np.clip(rho, 0.05, 0.95).astype(np.float32)


# Canonical flat layout for per-cell fields (source, density, cell temperature):
#   shape (nz, ny, nx) with x innermost — i.e. ravel order is iz, iy, ix.
# Per-node fields (nodal temperature) follow (nz+1, ny+1, nx+1) with x innermost.
# Plot helpers (`benchmarks/plots/recovery.py::_reshape_canonical_2d`) reshape
# directly under this convention and imshow rows→y, cols→x with NO transpose.
def _gaussian_source(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    amplitude: float = 1.0,
    cx: float = 0.5,
    cy: float = 0.5,
    sigma: float = 0.2,
    Lx: float = 2.0,
    Ly: float = 1.0,
    **_,
) -> np.ndarray:
    """Gaussian heat source centred at (cx*Lx, cy*Ly) with width sigma*min(Lx,Ly).

    Returns per-element source field, shape (nx*ny*nz,).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)

    dx = Lx / nx
    dy = Ly / ny
    # Element centres
    xs = (np.arange(nx) + 0.5) * dx  # (nx,)
    ys = (np.arange(ny) + 0.5) * dy  # (ny,)
    # Meshgrid: (nz, ny, nx) → ravel to (nx*ny*nz,) in z-y-x order
    # (matches _hex_mesh_arrays loop order: iz, iy, ix)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    x0 = cx * Lx
    y0 = cy * Ly
    width = sigma * min(Lx, Ly)
    G = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * width**2))
    # Tile across z-layers
    G3d = np.tile(G[np.newaxis, :, :], (nz, 1, 1))  # (nz, ny, nx)
    return G3d.ravel().astype(np.float32)  # (nx*ny*nz,)


def _two_gaussians(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    amplitude: float = 1.0,
    sigma: float = 0.15,
    Lx: float = 2.0,
    Ly: float = 1.0,
    **_,
) -> np.ndarray:
    """Two Gaussian sources at (0.3*Lx, 0.5*Ly) and (0.7*Lx, 0.5*Ly).

    Returns per-element source field, shape (nx*ny*nz,).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)

    s1 = _gaussian_source(
        nx=nx,
        ny=ny,
        nz=nz,
        amplitude=amplitude,
        cx=0.3,
        cy=0.5,
        sigma=sigma,
        Lx=Lx,
        Ly=Ly,
    )
    s2 = _gaussian_source(
        nx=nx,
        ny=ny,
        nz=nz,
        amplitude=amplitude,
        cx=0.7,
        cy=0.5,
        sigma=sigma,
        Lx=Lx,
        Ly=Ly,
    )
    return (s1 + s2).astype(np.float32)


MAKE_IC: dict[str, IcSpec] = {
    "uniform": IcSpec(
        fn=_uniform,
        description=(
            "Uniform SIMP thermal conductivity density ρ₀ over all hex mesh elements; "
            "standard homogeneous starting point for heat-conduction topology optimisation."
        ),
        plot_params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
    ),
    "random": IcSpec(
        fn=_random,
        description=(
            "Gaussian-noise density field centred at ρ₀=0.5 (σ=0.3, clipped to [0.05, 0.95]); "
            "breaks spatial symmetry to produce non-trivial per-cell gradient sensitivity maps."
        ),
        plot_params={
            "rho_0": 0.5,
            "noise": 0.3,
            "nx": 16,
            "ny": 8,
            "nz": 1,
            "seed": 0,
        },
    ),
    "gaussian_source": IcSpec(
        fn=_gaussian_source,
        description=(
            "Gaussian heat source centred at (cx·Lx, cy·Ly) = (0.5·Lx, 0.5·Ly) with "
            "width σ·min(Lx,Ly). Used as the control field for source-identification experiments "
            "(ic_field='source' in physics dict)."
        ),
        plot_params={
            "nx": 16,
            "ny": 8,
            "nz": 1,
            "amplitude": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "sigma": 0.2,
        },
    ),
    "zero_source": IcSpec(
        fn=_zero_source,
        description=(
            "Zero volumetric heat source; standard zero-initialisation for source-recovery experiments."
        ),
        plot_params={"nx": 16, "ny": 8, "nz": 1},
    ),
    "two_gaussians": IcSpec(
        fn=_two_gaussians,
        description=(
            "Two-Gaussian volumetric heat source at (0.3·Lx, 0.5·Ly) and (0.7·Lx, 0.5·Ly). "
            "Ground-truth source for source-recovery experiments."
        ),
        plot_params={"nx": 16, "ny": 8, "nz": 1},
    ),
}
