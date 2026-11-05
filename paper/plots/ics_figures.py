"""Generate Figure: Initial conditions overview across all four benchmark domains.

Fields are computed directly in NumPy — no JAX / solver dependencies.
Produces one PDF per domain; labels come from the panel titles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import TEXTWIDTH

# Full linewidth; per-panel height keeps panels near-square
_PH2 = (TEXTWIDTH / 2) * 0.85  # height for 2-panel row
_PH3 = (TEXTWIDTH / 3) * 0.85  # height for 3-panel row
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


def _ic_abc_slice(N=32, L=2 * np.pi, A=1.0, B=0.8165, C=0.5774):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = A * np.sin(0) + C * np.cos(Y)
    v = B * np.sin(X) + A * np.cos(0)
    return _vorticity(u, v, N, L)


def _ic_rand_div_free_slice(N=32, L=2 * np.pi, seed=0, k_peak=2.0, k_width=1.0):
    """z=0 slice of the curl of a random vector potential with energy near |k| = k_peak."""
    rng = np.random.default_rng(seed)
    kn = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY, KZ = np.meshgrid(kn, kn, kn, indexing="ij")
    envelope = np.exp(-0.5 * ((np.sqrt(KX**2 + KY**2 + KZ**2) - k_peak) / k_width) ** 2)
    Ax, Ay, Az = (
        (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
        * envelope
        for _ in range(3)
    )
    kf = 2 * np.pi / L
    u = np.real(np.fft.ifftn(1j * kf * (KY * Az - KZ * Ay)))[:, :, 0]
    v = np.real(np.fft.ifftn(1j * kf * (KZ * Ax - KX * Az)))[:, :, 0]
    return _vorticity(u, v, N, L)


def _ic_dfg_inflow(H=0.41, U_m=0.3, n=100):
    """DFG 2D-1 parabolic inlet profile u_x(y) = 4 U_m y (H - y) / H^2."""
    y = np.linspace(0, H, n)
    return y, 4 * U_m * y * (H - y) / H**2


def _ic_struct_uniform(nx=48, ny=24, rho_0=0.5):
    return np.full((nx, ny), rho_0)


def _ic_struct_random(nx=48, ny=24, seed=0):
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_struct_two_bumps(nx=48, ny=24):
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sigma = 0.12 * min(Lx, Ly)
    bumps = [
        np.exp(-((X - cx * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / (2 * sigma**2))
        for cx in [0.35, 0.75]
    ]
    return np.clip(0.1 + 0.85 * np.maximum(*bumps), 0.05, 1.0)


def _ic_thermal_uniform(nx=48, ny=24, rho_0=0.5):
    return np.full((nx, ny), rho_0)


def _ic_thermal_random(nx=48, ny=24, seed=0):
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_thermal_gaussian(nx=48, ny=24, sigma=0.2, centers=((0.5, 0.5),)):
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = sum(
        np.exp(
            -((X - cx * Lx) ** 2 + (Y - cy * Ly) ** 2)
            / (2 * (sigma * min(Lx, Ly)) ** 2)
        )
        for cx, cy in centers
    )
    return s / s.max()


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


def _imshow_unit(ax, field, cmap="viridis"):
    """Show a field with values in [0, 1] on a fixed [0, 1] color scale."""
    ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.axis("off")


def _label_uniform(ax, text):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
    )


def _plot_profile(ax, y, u):
    ax.plot(u, y, color="#2471a3", lw=1.2)
    ax.fill_betweenx(y, 0, u, color="#2471a3", alpha=0.15, lw=0)
    ax.set_xlim(0, 0.33)
    ax.set_ylim(0, y[-1])
    ax.set_xticks([0, 0.3])
    ax.set_yticks([0, y[-1]])
    ax.set_xlabel("$u_x$", labelpad=1)
    ax.set_ylabel("$y$", labelpad=1)
    ax.tick_params(labelsize=7, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_box_aspect(1)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    # NS: F2 (2D) in the top row, F3 (3D) in the bottom row
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 2 * _PH3))
    fig.subplots_adjust(wspace=0.3, hspace=0.45)
    panels = [
        ("TGV (2D)", _ic_tgv()),
        ("Multimode (2D)", _ic_multimode()),
        ("DFG inlet profile (2D)", None),
        ("TGV (3D)", _ic_tgv3d_slice()),
        ("ABC (3D)", _ic_abc_slice()),
        ("Random div.-free (3D)", _ic_rand_div_free_slice()),
    ]
    for ax, (lbl, field) in zip(axes.flat, panels, strict=True):
        if field is None:
            _plot_profile(ax, *_ic_dfg_inflow())
        else:
            _imshow_sym(ax, field)
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_ns_combined.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_ns_combined.pdf'}")

    # Structural
    labels_struct = ["Uniform density", "Random density", "Two density bumps"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, _PH3))
    fig.subplots_adjust(wspace=0.05)
    _imshow_unit(axes[0], _ic_struct_uniform())
    _label_uniform(axes[0], r"$\rho_0 = 0.5$")
    _imshow_unit(axes[1], _ic_struct_random())
    _imshow_unit(axes[2], _ic_struct_two_bumps())
    for ax, lbl in zip(axes, labels_struct, strict=True):
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_structural_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_structural_mesh.pdf'}")

    # Thermal — 1 row × 4 cols
    labels_thermal = [
        "Uniform cond.",
        "Random cond.",
        "Gaussian source",
        "Two-Gaussian cond.",
    ]
    fields_thermal = [
        _ic_thermal_uniform(),
        _ic_thermal_random(),
        _ic_thermal_gaussian(),
        _ic_thermal_gaussian(sigma=0.15, centers=((0.3, 0.5), (0.7, 0.5))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, _PH4))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_thermal, labels_thermal, strict=True):
        _imshow_unit(ax, field, cmap="hot")
        ax.set_title(lbl, fontsize=8, pad=3)
    _label_uniform(axes[0], "0.5")
    fig.savefig(out_dir / "appendix_ics_thermal_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_thermal_mesh.pdf'}")
