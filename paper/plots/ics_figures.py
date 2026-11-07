"""Generate Figure: Initial conditions of the two Navier--Stokes domains.

Fields are computed directly in NumPy — no JAX / solver dependencies.
Labels come from the panel titles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import TEXTWIDTH

# Full linewidth; per-panel height keeps panels near-square
_PH4 = (TEXTWIDTH / 4) * 0.85  # height for 4-panel row

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


def _vorticity(u, v, N, L=2 * np.pi):
    kn = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)))


def _ic_tgv(N=64, L=2 * np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return _vorticity(np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), N, L)


def _ic_multimode(N=64, L=2 * np.pi, seed=42):
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


def _ic_tgv3d_slice(N=32, L=2 * np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    return _vorticity(u, v, N, L)


def _ic_rand_div_free_3d_slice(N=32, L=2 * np.pi, seed=0, k_peak=2.0, k_width=1.0):
    """z=0 vorticity slice of the curl of a random spectral vector potential."""
    rng = np.random.default_rng(seed)
    kn = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = np.meshgrid(kn, kn, kn, indexing="ij")
    envelope = np.exp(-0.5 * ((np.sqrt(KX**2 + KY**2 + KZ**2) - k_peak) / k_width) ** 2)
    Ax, Ay, Az = (
        (rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N))) * envelope
        for _ in range(3)
    )
    kf = 2 * np.pi / L
    u = np.real(np.fft.ifftn(1j * kf * (KY * Az - KZ * Ay)))
    v = np.real(np.fft.ifftn(1j * kf * (KZ * Ax - KX * Az)))
    return _vorticity(u[:, :, 0], v[:, :, 0], N, L)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _imshow_sym(ax, field, cmap="RdBu_r"):
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


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    # Combined NS (2D + 3D) — 4 columns
    labels_ns = ["TGV (2D)", "Multimode (2D)", "TGV (3D)", "Random div.-free (3D)"]
    fields_ns = [
        _ic_tgv(),
        _ic_multimode(),
        _ic_tgv3d_slice(),
        _ic_rand_div_free_3d_slice(),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, _PH4))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_ns, labels_ns, strict=True):
        _imshow_sym(ax, field)
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_ns_combined.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_ns_combined.pdf'}")
