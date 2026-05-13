"""Initial conditions and IC registry for structural-mesh."""

from __future__ import annotations

import numpy as np


def _uniform(
    rho_0: float = 0.5,
    nx: int = 8,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Uniform density ρ₀ over the mesh, shape (nx·ny·nz,).

    Default geometry: ny=2 (thin slab), nz=nx//2.  The thin-y default gives an
    almost-2D cantilever — better for plotting and topology visualisation.

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)
    return np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)


def _random(
    rho_0: float = 0.5,
    noise: float = 0.3,
    nx: int = 8,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    seed: int = 0,
    **_,
) -> np.ndarray:
    """Gaussian-noise density field centred at ρ₀, clipped to [0.05, 0.95].

    Breaks spatial symmetry so fd_check and jacobian_svd see non-trivial
    per-cell gradients rather than a flat field identical across solvers.
    N, if provided, overrides nx (resolution_sweep convention).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)
    rng = np.random.default_rng(seed)
    rho = rho_0 + noise * rng.standard_normal(nx * ny * nz).astype(np.float32)
    return np.clip(rho, 0.05, 0.95).astype(np.float32)


def _two_density_bumps(
    nx: int = 16,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    rho_bg: float = 0.1,
    rho_peak: float = 0.95,
    sigma: float = 0.12,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    **_,
) -> np.ndarray:
    """Ground-truth density with two stiff Gaussian bumps on a soft background.

    Two load-bearing ``rho_peak`` pillars of width σ·min(Lx,Lz) centred at
    (0.35·Lx, 0.5·Ly, 0.5·Lz) and (0.75·Lx, 0.5·Ly, 0.5·Lz); the background is
    ``rho_bg`` everywhere else. Clipped to [0.05, 1.0].

    Direct analog of thermal-mesh ``_two_gaussians`` — a spatially concentrated
    ground-truth field whose effect on the observed output (displacement here,
    temperature there) can be probed via gradient-based recovery. Recovering
    this density pattern from boundary-load displacement observations is the
    structural version of source-field recovery.

    Returned shape: (nx·ny·nz,), canonical z-y-x ravel (same as ``_hex_mesh_arrays``).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)

    dx = Lx / nx
    dz = Lz / nz
    xs = (np.arange(nx) + 0.5) * dx  # (nx,)
    zs = (np.arange(nz) + 0.5) * dz  # (nz,)

    # The canonical structural-mesh geometry is a thin slab (ny=2 by default),
    # so the two bumps are y-invariant (extruded across y): the Gaussian lives
    # on the (x, z) cross-section, tiled along y.  This mirrors thermal-mesh
    # ``_two_gaussians`` which is z-invariant on the quasi-2D nz=1 slab.
    Z2, X2 = np.meshgrid(zs, xs, indexing="ij")  # (nz, nx)
    width = sigma * min(Lx, Lz)
    inv2w2 = 1.0 / (2.0 * width * width)

    x1, z1 = 0.35 * Lx, 0.5 * Lz
    x2, z2 = 0.75 * Lx, 0.5 * Lz
    g1 = np.exp(-((X2 - x1) ** 2 + (Z2 - z1) ** 2) * inv2w2)
    g2 = np.exp(-((X2 - x2) ** 2 + (Z2 - z2) ** 2) * inv2w2)
    peak2d = (rho_peak - rho_bg) * np.maximum(g1, g2) + rho_bg  # (nz, nx)
    # Tile across y: (nz, ny, nx)
    peak_field = np.broadcast_to(peak2d[:, None, :], (nz, ny, nx)).copy()
    return np.clip(peak_field.ravel(), 0.05, 1.0).astype(np.float32)
