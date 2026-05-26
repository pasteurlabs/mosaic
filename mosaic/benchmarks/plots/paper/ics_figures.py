# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Figure: Initial conditions overview across all four benchmark domains.

Fields are computed directly in NumPy — no JAX / solver dependencies.
Produces one PDF per domain; labels come from the panel titles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH

# Full linewidth; per-panel height keeps panels near-square
_PH2 = (TEXTWIDTH / 2) * 0.85  # height for 2-panel row
_PH3 = (TEXTWIDTH / 3) * 0.85  # height for 3-panel row

RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# ---------------------------------------------------------------------------
# IC field generators
# ---------------------------------------------------------------------------


def _vorticity(
    u: np.ndarray, v: np.ndarray, N: int, L: float = 2 * np.pi
) -> np.ndarray:
    kn = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)))


def _ic_tgv(N: int = 64, L: float = 2 * np.pi) -> np.ndarray:
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return _vorticity(np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), N, L)


def _ic_multimode(N: int = 64, L: float = 2 * np.pi, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kn = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    envelope = np.exp(-0.5 * ((K - 2.0) / 0.5) ** 2)
    phases = rng.uniform(0, 2 * np.pi, (N, N))
    psi_hat = envelope * np.exp(1j * phases)
    psi_hat = 0.5 * (psi_hat + np.conj(psi_hat[::-1, ::-1]))
    psi_hat[0, 0] = 0.0
    kf = 2 * np.pi / L
    u = np.real(np.fft.ifft2(1j * KY * kf * psi_hat))
    v = np.real(np.fft.ifft2(-1j * KX * kf * psi_hat))
    return _vorticity(u, v, N, L)


def _ic_tgv3d_slice(N: int = 32, L: float = 2 * np.pi) -> np.ndarray:
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    return _vorticity(u, v, N, L)


def _ic_abc_slice(
    N: int = 32, L: float = 2 * np.pi, A: float = 1.0, B: float = 1.0, C: float = 1.0
) -> np.ndarray:
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = A * np.sin(0) + C * np.cos(Y)
    v = B * np.sin(X) + A * np.cos(0)
    return _vorticity(u, v, N, L)


def _ic_struct_uniform(nx: int = 48, ny: int = 24, rho_0: float = 0.5) -> np.ndarray:
    return np.full((nx, ny), rho_0)


def _ic_struct_random(nx: int = 48, ny: int = 24, seed: int = 0) -> np.ndarray:
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_struct_two_bumps(nx: int = 48, ny: int = 24) -> np.ndarray:
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sigma = 0.12 * min(Lx, Ly)
    rho = np.full((nx, ny), 0.1)
    for cx in [0.35, 0.75]:
        rho += 0.85 * np.exp(
            -((X - cx * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / (2 * sigma**2)
        )
    return np.clip(rho, 0.05, 0.95)


def _ic_thermal_uniform(nx: int = 48, ny: int = 24, rho_0: float = 0.5) -> np.ndarray:
    return np.full((nx, ny), rho_0)


def _ic_thermal_random(nx: int = 48, ny: int = 24, seed: int = 0) -> np.ndarray:
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_thermal_gaussian(nx: int = 48, ny: int = 24, sigma: float = 0.2) -> np.ndarray:
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = np.exp(
        -((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / (2 * (sigma * min(Lx, Ly)) ** 2)
    )
    return s / s.max()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _imshow_sym(ax: plt.Axes, field: np.ndarray, cmap: str = "RdBu_r") -> None:
    vmax = float(np.abs(field).max()) or 1.0
    ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.axis("off")


def _imshow_pos(ax: plt.Axes, field: np.ndarray, cmap: str = "viridis") -> None:
    ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=field.min(),
        vmax=field.max(),
        aspect="equal",
        interpolation="bilinear",
    )
    ax.axis("off")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def generate(out_dir: Path) -> None:
    """Generate initial conditions overview figure across all benchmark domains."""
    plt.rcParams.update(RCPARAMS)

    # Combined NS (2D + 3D) — 4 columns
    _PH4 = (TEXTWIDTH / 4) * 0.85
    labels_ns = ["TGV (2D)", "Multimode (2D)", "TGV (3D)", "ABC (3D)"]
    fields_ns = [_ic_tgv(), _ic_multimode(), _ic_tgv3d_slice(), _ic_abc_slice()]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, _PH4))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_ns, labels_ns, strict=False):
        _imshow_sym(ax, field)
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_ns_combined.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_ns_combined.pdf'}")

    # Structural
    labels_struct = ["Uniform density", "Random density", "Two density bumps"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, _PH3))
    fig.subplots_adjust(wspace=0.05)
    _imshow_pos(axes[0], _ic_struct_uniform())
    _imshow_pos(axes[1], _ic_struct_random())
    _imshow_pos(axes[2], _ic_struct_two_bumps())
    for ax, lbl in zip(axes, labels_struct, strict=False):
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_structural_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_structural_mesh.pdf'}")

    # Thermal — 1 row × 3 cols
    labels_thermal = ["Uniform conductivity", "Random conductivity", "Gaussian source"]
    fields_thermal = [
        _ic_thermal_uniform(),
        _ic_thermal_random(),
        _ic_thermal_gaussian(),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, _PH3))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_thermal, labels_thermal, strict=False):
        _imshow_pos(ax, field, cmap="hot")
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_thermal_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_thermal_mesh.pdf'}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
