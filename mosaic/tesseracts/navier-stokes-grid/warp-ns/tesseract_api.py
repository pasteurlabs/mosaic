# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001

"""GPU-accelerated differentiable 2-D/3-D Navier-Stokes via NVIDIA Warp.

Explicit-Euler Chorin projection (tentative velocity → pressure Poisson →
velocity correction) with wp.Tape VJP. Periodic boxes, 2-D and 3-D, use an
exact spectral FFT Poisson solve. The 2-D channel with a cylinder
(Schäfer–Turek DFG 2D-1) uses a staggered grid and a conjugate-gradient
Poisson solve (see :func:`channel_solve`).
"""

import contextlib
import functools
from typing import Any

import numpy as np
import warp as wp
import warp.optim.linear
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import (
    check_periodic_or_channel,
    solid_cells,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator

wp.init()

# ============================================================
# 2-D Chorin projection kernels (primitive variables)
# ============================================================


@wp.kernel
def tentative_vel_2d_kernel(
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    ux_star: wp.array2d(dtype=wp.float32),
    uy_star: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
    inv_h2: float,
    nu: wp.array(dtype=wp.float32),
) -> None:
    """2-D tentative velocity: u* = u + dt·(-u·∇u + ν∇²u).

    ``dt`` and ``nu`` are 1-element arrays (rather than plain floats) so that
    Warp's source-to-source autodiff tracks gradients w.r.t. them: Warp only
    differentiates through array-typed kernel arguments.

    [2D-only function]
    """
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n

    dt_ = dt[0]
    nu_ = nu[0]

    ui = ux[i, j]
    vi = uy[i, j]

    # ux component
    lap_ux = (ux[im1, j] + ux[ip1, j] + ux[i, jm1] + ux[i, jp1] - 4.0 * ui) * inv_h2
    adv_ux = (
        ui * (ux[ip1, j] - ux[im1, j]) * inv_2h
        + vi * (ux[i, jp1] - ux[i, jm1]) * inv_2h
    )
    ux_star[i, j] = ui + dt_ * (-adv_ux + nu_ * lap_ux)

    # uy component
    lap_uy = (uy[im1, j] + uy[ip1, j] + uy[i, jm1] + uy[i, jp1] - 4.0 * vi) * inv_h2
    adv_uy = (
        ui * (uy[ip1, j] - uy[im1, j]) * inv_2h
        + vi * (uy[i, jp1] - uy[i, jm1]) * inv_2h
    )
    uy_star[i, j] = vi + dt_ * (-adv_uy + nu_ * lap_uy)


@wp.kernel
def divergence_2d_to_complex_kernel(
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    div_c: wp.array2d(dtype=wp.vec2f),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
) -> None:
    """Compute ∇·u*/dt for 2-D pressure Poisson RHS, packed directly as complex.

    Writes directly into a complex (vec2f) buffer with zero imaginary part.
    Fuses the divergence kernel with the "pack real->complex" step that would
    otherwise precede the spectral Poisson FFT.
    """
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    inv_2h_over_dt = inv_2h / dt[0]
    div = ((ux[ip1, j] - ux[im1, j]) + (uy[i, jp1] - uy[i, jm1])) * inv_2h_over_dt
    div_c[i, j] = wp.vec2f(div, 0.0)


@wp.kernel
def pressure_correct_2d_from_complex_kernel(
    ux_star: wp.array2d(dtype=wp.float32),
    uy_star: wp.array2d(dtype=wp.float32),
    p_c: wp.array2d(dtype=wp.vec2f),
    p_divisor: float,
    ux_new: wp.array2d(dtype=wp.float32),
    uy_new: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
) -> None:
    """u^(n+1) = u* - dt·∇p in 2-D, reading p directly from the complex IFFT output.

    Reads the real part (normalized) instead of a separate materialized
    pressure array. Fuses the "extract real + normalize" step with the
    pressure-correction kernel.
    """
    i, j = wp.tid()
    n = p_c.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    dt_ = dt[0]
    p_im1 = p_c[im1, j][0] / p_divisor
    p_ip1 = p_c[ip1, j][0] / p_divisor
    p_jm1 = p_c[i, jm1][0] / p_divisor
    p_jp1 = p_c[i, jp1][0] / p_divisor
    dpdx = (p_ip1 - p_im1) * inv_2h
    dpdy = (p_jp1 - p_jm1) * inv_2h
    ux_new[i, j] = ux_star[i, j] - dt_ * dpdx
    uy_new[i, j] = uy_star[i, j] - dt_ * dpdy


# ============================================================
# 3-D NS kernels (Chorin projection)
# ============================================================


@wp.kernel
def tentative_vel_3d_kernel(
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    ux_star: wp.array3d(dtype=wp.float32),
    uy_star: wp.array3d(dtype=wp.float32),
    uz_star: wp.array3d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
    inv_h2: float,
    nu: wp.array(dtype=wp.float32),
) -> None:
    """3-D tentative velocity: u* = u + dt·(-u·∇u + ν∇²u).

    ``dt``/``nu`` are 1-element arrays (like the 2-D kernel) so the tape
    tracks gradients w.r.t. them when called from :func:`ns3d_solve_tape`;
    :func:`ns3d_solve_forward` passes non-grad-tracked 1-element arrays.
    """
    i, j, k = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n

    dt_ = dt[0]
    nu_ = nu[0]

    ui = ux[i, j, k]
    vi = uy[i, j, k]
    wi = uz[i, j, k]

    # ux component
    lap_ux = (
        ux[im1, j, k]
        + ux[ip1, j, k]
        + ux[i, jm1, k]
        + ux[i, jp1, k]
        + ux[i, j, km1]
        + ux[i, j, kp1]
        - 6.0 * ui
    ) * inv_h2
    adv_ux = (
        ui * (ux[ip1, j, k] - ux[im1, j, k]) * inv_2h
        + vi * (ux[i, jp1, k] - ux[i, jm1, k]) * inv_2h
        + wi * (ux[i, j, kp1] - ux[i, j, km1]) * inv_2h
    )
    ux_star[i, j, k] = ui + dt_ * (-adv_ux + nu_ * lap_ux)

    # uy component
    lap_uy = (
        uy[im1, j, k]
        + uy[ip1, j, k]
        + uy[i, jm1, k]
        + uy[i, jp1, k]
        + uy[i, j, km1]
        + uy[i, j, kp1]
        - 6.0 * vi
    ) * inv_h2
    adv_uy = (
        ui * (uy[ip1, j, k] - uy[im1, j, k]) * inv_2h
        + vi * (uy[i, jp1, k] - uy[i, jm1, k]) * inv_2h
        + wi * (uy[i, j, kp1] - uy[i, j, km1]) * inv_2h
    )
    uy_star[i, j, k] = vi + dt_ * (-adv_uy + nu_ * lap_uy)

    # uz component
    lap_uz = (
        uz[im1, j, k]
        + uz[ip1, j, k]
        + uz[i, jm1, k]
        + uz[i, jp1, k]
        + uz[i, j, km1]
        + uz[i, j, kp1]
        - 6.0 * wi
    ) * inv_h2
    adv_uz = (
        ui * (uz[ip1, j, k] - uz[im1, j, k]) * inv_2h
        + vi * (uz[i, jp1, k] - uz[i, jm1, k]) * inv_2h
        + wi * (uz[i, j, kp1] - uz[i, j, km1]) * inv_2h
    )
    uz_star[i, j, k] = wi + dt_ * (-adv_uz + nu_ * lap_uz)


@wp.kernel
def divergence_3d_to_complex_kernel(
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    div_c: wp.array3d(dtype=wp.vec2f),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
) -> None:
    """Compute ∇·u*/dt for pressure Poisson RHS, packed directly as complex.

    Writes directly into a complex (vec2f) buffer with zero imaginary part.
    Fuses the divergence kernel with the "pack real->complex" step that would
    otherwise precede the spectral Poisson FFT. ``dt`` is a 1-element array
    (like the 2-D kernel) so the tape tracks gradients w.r.t. it.
    """
    i, j, k = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    inv_2h_over_dt = inv_2h / dt[0]
    div = (
        (ux[ip1, j, k] - ux[im1, j, k])
        + (uy[i, jp1, k] - uy[i, jm1, k])
        + (uz[i, j, kp1] - uz[i, j, km1])
    ) * inv_2h_over_dt
    div_c[i, j, k] = wp.vec2f(div, 0.0)


@wp.kernel
def pressure_correct_3d_from_complex_kernel(
    ux_star: wp.array3d(dtype=wp.float32),
    uy_star: wp.array3d(dtype=wp.float32),
    uz_star: wp.array3d(dtype=wp.float32),
    p_c: wp.array3d(dtype=wp.vec2f),
    p_divisor: float,
    ux_new: wp.array3d(dtype=wp.float32),
    uy_new: wp.array3d(dtype=wp.float32),
    uz_new: wp.array3d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_2h: float,
) -> None:
    """u^(n+1) = u* - dt·∇p, reading p directly from the complex IFFT output.

    Reads the real part (normalized) instead of a separate materialized
    pressure array. Fuses the "extract real + normalize" step with the
    pressure-correction kernel. ``dt`` is a 1-element array (like the 2-D
    kernel) so the tape tracks gradients w.r.t. it.
    """
    i, j, k = wp.tid()
    n = p_c.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    dt_ = dt[0]
    p_im1 = p_c[im1, j, k][0] / p_divisor
    p_ip1 = p_c[ip1, j, k][0] / p_divisor
    p_jm1 = p_c[i, jm1, k][0] / p_divisor
    p_jp1 = p_c[i, jp1, k][0] / p_divisor
    p_km1 = p_c[i, j, km1][0] / p_divisor
    p_kp1 = p_c[i, j, kp1][0] / p_divisor
    dpdx = (p_ip1 - p_im1) * inv_2h
    dpdy = (p_jp1 - p_jm1) * inv_2h
    dpdz = (p_kp1 - p_km1) * inv_2h
    ux_new[i, j, k] = ux_star[i, j, k] - dt_ * dpdx
    uy_new[i, j, k] = uy_star[i, j, k] - dt_ * dpdy
    uz_new[i, j, k] = uz_star[i, j, k] - dt_ * dpdz


def _wlaunch(kernel, dim, inputs, block_dim=256, device="cpu"):
    """wp.launch wrapper. block_dim must be an int (Warp 1.12 dropped tuple support)."""
    wp.launch(kernel, dim=dim, inputs=inputs, block_dim=block_dim, device=device)


####################################################################
# GPU-resident spectral Poisson solve via wp.tile_fft
#
# Keeping the FFT on the device avoids a CUDA sync and a device->host->device
# round trip on every timestep, which would dominate the cost of the 3-D
# solver. wp.tile_fft/wp.tile_ifft support native Warp reverse-mode AD
# (verified: a tile_fft->tile_ifft round trip differentiates correctly against
# a hand-derived expectation, cosine similarity 1.0 to float32 roundoff), so no
# record_func/self-adjoint bookkeeping is needed — tape.backward() propagates
# through the FFT kernels like any other kernel.
#
# A 2-D/3-D FFT separates into 1-D transforms along each axis; wp.tile_fft
# only transforms the last axis of a tile, so each additional axis is handled
# by transposing the array and re-running the same row-wise kernel (pattern
# taken from warp/examples/core/example_fft_poisson_navier_stokes_2d.py).
#
# TILE_N/BLOCK_DIM are compile-time constants baked into the kernel by Warp's
# codegen, so kernels are generated per grid size N via an lru_cache'd
# factory and registered into a uniquely-named module (module=f"...{n}") to
# avoid colliding with other N's kernels of the same name.
####################################################################


@wp.func
def _scale_complex(v: wp.vec2f, s: float) -> wp.vec2f:
    """Multiply a complex value (as vec2f) by a real scalar."""
    return wp.vec2f(v[0] * s, v[1] * s)


####################################################################
# Non-power-of-two support via Bluestein's algorithm
#
# wp.tile_fft (cuFFTDx) only compiles power-of-two FFT lengths on GPU — Warp's
# own codegen rejects everything else (its non-pow2 tile_fft tests are CPU
# only). The benchmark spatial sweeps include non-pow2 resolutions (N=192 in
# 2-D, N=48 in 3-D), which crashed the solver at LTO-compile time.
#
# Bluestein's algorithm expresses an arbitrary-length-N DFT exactly as a
# length-M convolution, where M = next_pow2(2N-1) is a power of two — so the
# convolution is evaluated with the same pow2 tile_fft primitives that already
# work. This is exact (matches np.fft to float32 roundoff, verified in NumPy
# against the reference solver's eigenvalue solve), so the discretization is
# unchanged: pow2 grids keep the fast, fully-fused native path untouched; only
# non-pow2 grids take this (slightly slower, on-GPU, still-differentiable)
# route.
#
#   X[k] = conj(a[k]) * sum_j (b[j] * conj(a[j]) * x[j])   as a convolution
#   with the chirp a[j] = exp(-i·pi·j²/N). See _bluestein_setup for the packed
#   form actually launched.
####################################################################


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@functools.cache
def _bluestein_setup(n: int, device: str) -> dict:
    """Precompute the Bluestein chirp and the (pre-FFT'd) kernel for length ``n``.

    Returns the chirp ``w`` (length ``n``, as a complex vec2f array broadcast
    to a row) and ``B = fft(b) / m`` (length ``m = next_pow2(2n-1)``), both
    device arrays. ``b`` is the zero-padded, wrapped conjugate-chirp
    convolution kernel; its FFT is precomputed on the host once per ``n`` since
    it is constant across timesteps and carries no gradient. The ``1/m``
    cancels the unnormalized ``wp.tile_ifft`` that closes the convolution.
    """
    m = _next_pow2(2 * n - 1)
    j = np.arange(n)
    # chirp a[j] = exp(-i pi j^2 / n); w stored as the forward chirp
    w = np.exp(-1j * np.pi * (j * j % (2 * n)) / n)
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(w)
    b[m - n + 1 :] = np.conj(w[1:])[::-1]
    B = np.fft.fft(b) / m

    def _to_row_vec2f(arr: np.ndarray) -> wp.array:
        # shape (len, 2) -> (1, len, 2) so Warp reads it as a (1, len) array of
        # vec2f, matching the ``w[0, j]`` / ``B[0, k]`` row indexing in kernels.
        row = np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)[None, ...]
        return wp.array2d(row, dtype=wp.vec2f, device=device)

    return {
        "m": m,
        "w": _to_row_vec2f(w),
        "B": _to_row_vec2f(B),
    }


@wp.func
def _cmul(a: wp.vec2f, b: wp.vec2f) -> wp.vec2f:
    """Complex multiply (a, b interpreted as complex, stored as vec2f)."""
    return wp.vec2f(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@functools.cache
def _bluestein_kernels(n: int, m: int):
    """Build the pre/post chirp + pointwise-multiply kernels for length ``n``.

    The two pow2 transforms themselves reuse the length-``m`` pow2 tile-FFT
    kernels from :func:`_fft_kernels_2d`; these kernels only handle the
    chirp-multiply/pad (before) and crop/chirp-multiply (after) around them,
    plus the frequency-domain multiply by the precomputed ``B``. All are plain
    elementwise ``wp.tile_map``-free kernels so they differentiate natively
    through ``wp.Tape``.
    """

    @wp.kernel(module=f"bluestein_{n}_{m}")
    def premul(
        x: wp.array2d(dtype=wp.vec2f),
        w: wp.array2d(dtype=wp.vec2f),
        a: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """Pre-multiply by the chirp.

        ``a[row, j] = x[row, j] * w[0, j]``. Launched over ``j<n`` only; ``a``
        is a fresh zero buffer of width ``m``, so columns ``n..m-1`` stay zero
        (the pad).
        """
        row, j = wp.tid()
        a[row, j] = _cmul(x[row, j], w[0, j])

    @wp.kernel(module=f"bluestein_{n}_{m}")
    def freqmul(
        A: wp.array2d(dtype=wp.vec2f),
        B: wp.array2d(dtype=wp.vec2f),
        out: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """out[row, k] = A[row, k] * B[0, k] (length-m frequency-domain multiply)."""
        row, k = wp.tid()
        out[row, k] = _cmul(A[row, k], B[0, k])

    @wp.kernel(module=f"bluestein_{n}_{m}")
    def postmul(
        c: wp.array2d(dtype=wp.vec2f),
        w: wp.array2d(dtype=wp.vec2f),
        y: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """y[row, k] = c[row, k] * w[0, k] for the length-n crop of the conv."""
        row, k = wp.tid()
        y[row, k] = _cmul(c[row, k], w[0, k])

    return {"premul": premul, "freqmul": freqmul, "postmul": postmul}


def _bluestein_fft_rows(x: wp.array, n: int, direction: str, device: str) -> wp.array:
    """Length-``n`` DFT along the last axis of a ``(batch, n)`` complex array.

    ``direction`` is ``"fwd"`` or ``"bwd"`` (inverse). Both are unnormalized,
    like ``wp.tile_fft``/``wp.tile_ifft`` on the power-of-two path, so callers
    apply the same ``1/n`` per axis either way. The inverse reuses the forward
    transform via ``n * ifft(x) = conj(fft(conj(x)))``. Uses only power-of-two
    (length-``m``) tile FFTs, so it compiles on GPU for any ``n``.

    Every launch writes fresh ``requires_grad`` buffers so ``wp.Tape`` tracks
    the whole chain natively (no ``record_func`` needed) — the chirp/kernel
    multiplies and the pow2 FFTs are all differentiable Warp kernels.
    """
    batch = x.shape[0]
    setup = _bluestein_setup(n, device)
    m = setup["m"]
    bk = _bluestein_kernels(n, m)
    pow2 = _fft_kernels_2d(m)

    # For the inverse, conjugate on the way in and out.
    conj_in = direction == "bwd"

    src = x
    if conj_in:
        src = _conj_rows(x, device)

    a = wp.zeros((batch, m), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        bk["premul"],
        dim=(batch, n),
        inputs=[src, setup["w"]],
        outputs=[a],
        device=device,
    )
    A = wp.zeros((batch, m), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch_tiled(
        pow2["fft_tiled"],
        dim=[batch, 1],
        inputs=[a],
        outputs=[A],
        block_dim=pow2["block_dim"],
        device=device,
    )
    AB = wp.zeros((batch, m), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        bk["freqmul"],
        dim=(batch, m),
        inputs=[A, setup["B"]],
        outputs=[AB],
        device=device,
    )
    c = wp.zeros((batch, m), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch_tiled(
        pow2["ifft_tiled"],
        dim=[batch, 1],
        inputs=[AB],
        outputs=[c],
        block_dim=pow2["block_dim"],
        device=device,
    )
    # crop to length n (launch only n columns; c stays full-width m) and apply
    # the trailing chirp
    y = wp.zeros((batch, n), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        bk["postmul"],
        dim=(batch, n),
        inputs=[c, setup["w"]],
        outputs=[y],
        device=device,
    )
    if conj_in:
        y = _conj_rows(y, device)
    return y


@functools.cache
def _conj_kernels(_tag: str = "bluestein_conj"):
    """Conjugate kernel shared across all Bluestein sizes."""

    @wp.kernel(module="bluestein_conj")
    def conj_k(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)) -> None:
        i, j = wp.tid()
        y[i, j] = wp.vec2f(x[i, j][0], -x[i, j][1])

    return {"conj": conj_k}


def _conj_rows(x: wp.array, device: str) -> wp.array:
    y = wp.zeros(x.shape, dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        _conj_kernels()["conj"], dim=x.shape, inputs=[x], outputs=[y], device=device
    )
    return y


@functools.cache
def _generic_transpose_kernel(_tag: str = "bluestein_T"):
    """Elementwise (non-tiled) square transpose, valid for any ``n``.

    The tiled ``transpose_kernel`` in :func:`_fft_kernels_2d` needs
    ``n``-divisible tiles; this plain per-element copy has no such constraint
    and is only used on the (rare) non-pow2 path, so the loss of tiling is
    immaterial. Differentiates natively.
    """

    @wp.kernel(module="bluestein_T")
    def transpose_k(
        x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)
    ) -> None:
        i, j = wp.tid()
        y[j, i] = x[i, j]

    return transpose_k


def _transpose_rows(x: wp.array, device: str) -> wp.array:
    n0, n1 = x.shape
    y = wp.zeros((n1, n0), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        _generic_transpose_kernel(),
        dim=(n0, n1),
        inputs=[x],
        outputs=[y],
        device=device,
    )
    return y


@functools.cache
def _scale_by_field_kernel(_tag: str = "bluestein_scale"):
    """out[i,j] = z[i,j] * s[i,j] with z complex (vec2f) and s a real field."""

    @wp.kernel(module="bluestein_scale")
    def scale_k(
        z: wp.array2d(dtype=wp.vec2f),
        s: wp.array2d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        i, j = wp.tid()
        out[i, j] = _scale_complex(z[i, j], s[i, j])

    return scale_k


def _spectral_poisson_2d_bluestein(
    rhs_c: wp.array, domain_extent: float, device: str
) -> wp.array:
    """Non-pow2 2-D spectral Poisson solve (Bluestein FFTs, on-GPU, differentiable).

    Numerically identical to :func:`_spectral_poisson_2d_core` (same continuous
    eigenvalues, same DFT), but each 1-D transform goes through
    :func:`_bluestein_fft_rows` so it compiles for non-power-of-two ``n``. The
    spectral multiply is a separate elementwise kernel here rather than fused
    into the FFT — the non-pow2 path trades a little fusion for correctness.
    """
    n = rhs_c.shape[0]
    inv_lambda = _inv_lambda_2d(n, domain_extent, device)

    # forward 2-D FFT: rows (axis 1), then transpose, rows (axis 0), transpose back
    hat = _bluestein_fft_rows(rhs_c, n, "fwd", device)
    hat = _transpose_rows(hat, device)
    hat = _bluestein_fft_rows(hat, n, "fwd", device)
    hat = _transpose_rows(hat, device)

    scaled = wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        _scale_by_field_kernel(),
        dim=(n, n),
        inputs=[hat, inv_lambda],
        outputs=[scaled],
        device=device,
    )

    # inverse 2-D FFT, same two-pass structure
    out = _bluestein_fft_rows(scaled, n, "bwd", device)
    out = _transpose_rows(out, device)
    out = _bluestein_fft_rows(out, n, "bwd", device)
    out = _transpose_rows(out, device)
    return out


@functools.cache
def _fft_kernels_2d(n: int):
    """Build (and cache) the tiled FFT/transpose kernels for an N×N grid."""
    tile_transpose_dim = 16 if n % 16 == 0 else n
    block_dim = max(2, n // 2)  # cuFFTDx requires >=2 elements/thread

    @wp.kernel(module=f"dft2d_{n}")
    def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)) -> None:
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        wp.tile_fft(row)
        wp.tile_store(y, row, offset=(i, 0))

    @wp.kernel(module=f"dft2d_{n}")
    def ifft_tiled(
        x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)
    ) -> None:
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        wp.tile_ifft(row)
        wp.tile_store(y, row, offset=(i, 0))

    @wp.kernel(module=f"dft2d_{n}")
    def fft_multiply_ifft_tiled(
        x: wp.array2d(dtype=wp.vec2f),
        inv_lambda_row: wp.array2d(dtype=wp.float32),
        y: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """Fused middle stage: last forward FFT pass, spectral multiply, first inverse FFT pass.

        These three operations act on the same (already-transposed) row
        layout with no intervening transpose, so the row stays resident in
        registers/shared memory across all three.
        """
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        scale_row = wp.tile_load(inv_lambda_row, shape=(1, n), offset=(i, 0))
        wp.tile_fft(row)
        scaled = wp.tile_map(_scale_complex, row, scale_row)
        wp.tile_ifft(scaled)
        wp.tile_store(y, scaled, offset=(i, 0))

    @wp.kernel(module=f"transpose2d_{n}")
    def transpose_kernel(
        x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)
    ) -> None:
        i, j = wp.tid()
        tile = wp.tile_load(
            x,
            shape=(tile_transpose_dim, tile_transpose_dim),
            offset=(i * tile_transpose_dim, j * tile_transpose_dim),
            storage="shared",
        )
        out = wp.tile_transpose(tile)
        wp.tile_store(y, out, offset=(j * tile_transpose_dim, i * tile_transpose_dim))

    return {
        "fft_tiled": fft_tiled,
        "ifft_tiled": ifft_tiled,
        "fft_multiply_ifft_tiled": fft_multiply_ifft_tiled,
        "transpose_kernel": transpose_kernel,
        "block_dim": block_dim,
        "tile_transpose_dim": tile_transpose_dim,
    }


@functools.cache
def _inv_lambda_2d(n: int, domain_extent: float, device: str) -> wp.array:
    """Precompute 1/λ for the 2-D continuous-eigenvalue spectral Poisson solve.

    Uses the continuous eigenvalues -(2π/L)²k², unlike the discrete FD
    eigenvalues of the 3-D solve.
    ``kx``/``ky`` enter symmetrically, so this array is safe to index either
    as-is or transposed (the fused FFT pipeline below reads it in
    transposed/row-major order relative to how it is built here).
    """
    kfreq = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    lam = -((2.0 * np.pi / domain_extent) ** 2) * (kx**2 + ky**2)
    inv_lambda = np.zeros_like(lam)
    nonzero = lam != 0
    inv_lambda[nonzero] = 1.0 / lam[nonzero]
    return wp.array2d(inv_lambda.astype(np.float32), dtype=wp.float32, device=device)


def _spectral_poisson_2d_core(
    rhs_c: wp.array, domain_extent: float, device: str, workspace: dict | None = None
) -> wp.array:
    """Core 2-D spectral Poisson solve on an already-packed complex RHS.

    Returns the complex (unnormalized) pressure field. Call sites that
    produce/consume complex buffers directly (e.g. a divergence kernel fused
    with the real->complex pack, or a pressure-correction kernel fused with
    the extract+normalize step) can skip the redundant pack/unpack kernels.

    The middle stage (last forward FFT pass, spectral multiply, first inverse
    FFT pass) runs as a single fused tiled kernel instead of three separate
    launches with two intermediate global-memory round trips. Pass
    ``workspace`` (from :func:`_poisson_workspace_2d`) to reuse pre-allocated
    scratch buffers across steps instead of allocating fresh ones on every
    call. The forward-only (non-taped) path requires it, and it is a minor win
    for the taped path too.
    """
    n = rhs_c.shape[0]
    if not _is_pow2(n):
        # cuFFTDx tile_fft only compiles pow2 lengths; use the Bluestein path
        # (workspace fusion doesn't apply — always allocates fresh buffers).
        return _spectral_poisson_2d_bluestein(rhs_c, domain_extent, device)
    kernels = _fft_kernels_2d(n)
    inv_lambda = _inv_lambda_2d(n, domain_extent, device)

    tmp1 = (
        workspace["tmp1"]
        if workspace
        else wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    )
    rhs_hat = (
        workspace["rhs_hat"]
        if workspace
        else wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    )
    wp.launch_tiled(
        kernels["fft_tiled"],
        dim=[n, 1],
        inputs=[rhs_c],
        outputs=[tmp1],
        block_dim=kernels["block_dim"],
        device=device,
    )
    td = kernels["tile_transpose_dim"]
    wp.launch_tiled(
        kernels["transpose_kernel"],
        dim=(n // td, n // td),
        inputs=[tmp1],
        outputs=[rhs_hat],
        block_dim=td * td,
        device=device,
    )

    p_hat = (
        workspace["p_hat"]
        if workspace
        else wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    )
    wp.launch_tiled(
        kernels["fft_multiply_ifft_tiled"],
        dim=[n, 1],
        inputs=[rhs_hat, inv_lambda],
        outputs=[p_hat],
        block_dim=kernels["block_dim"],
        device=device,
    )

    tmp2 = (
        workspace["tmp2"]
        if workspace
        else wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    )
    p_c = (
        workspace["p_c"]
        if workspace
        else wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    )
    wp.launch_tiled(
        kernels["transpose_kernel"],
        dim=(n // td, n // td),
        inputs=[p_hat],
        outputs=[tmp2],
        block_dim=td * td,
        device=device,
    )
    wp.launch_tiled(
        kernels["ifft_tiled"],
        dim=[n, 1],
        inputs=[tmp2],
        outputs=[p_c],
        block_dim=kernels["block_dim"],
        device=device,
    )
    return p_c


def _poisson_workspace_2d(n: int, device: str) -> dict:
    """Pre-allocate (non-differentiable) scratch buffers for the forward-only path.

    Reused across every timestep instead of allocating fresh buffers per
    step, since the forward-only path has no tape to track per-step
    dependencies for.
    """
    return {
        "tmp1": wp.zeros((n, n), dtype=wp.vec2f, device=device),
        "rhs_hat": wp.zeros((n, n), dtype=wp.vec2f, device=device),
        "p_hat": wp.zeros((n, n), dtype=wp.vec2f, device=device),
        "tmp2": wp.zeros((n, n), dtype=wp.vec2f, device=device),
        "p_c": wp.zeros((n, n), dtype=wp.vec2f, device=device),
    }


@functools.cache
def _inv_lambda_3d(
    n: int, domain_extent: float, device: str, permute: tuple | None = None
) -> wp.array:
    """Precompute 1/λ for the 3-D spectral Poisson solve.

    Uses the eigenvalues of the compact 7-point Laplacian:
        λ(kx, ky, kz) = -(4/h²)(sin²(πkx/N) + sin²(πky/N) + sin²(πkz/N))
    where h = L/N and kx,ky,kz are integer wavenumbers. This is the stencil
    tentative_vel_3d_kernel uses for viscous diffusion. The projection itself
    takes central differences (inv_2h) for both the divergence and the
    pressure gradient, which compose to the wide-stencil Laplacian with
    eigenvalues -(1/h²)Σ sin²(2πk/N). The solve is therefore not its exact
    inverse, and the projected velocity is only approximately divergence-free
    under the central-difference divergence.

    The tape differentiates whichever operator is used, so this choice does
    not bias the gradient against finite differences of ``apply``.

    ``permute`` optionally applies an axis permutation to the returned array:
    the eigenvalue formula is symmetric in kx/ky/kz, so a permuted copy of
    this array can be indexed against an intermediate FFT buffer that is
    itself axis-permuted (e.g. the pre-axis-swap layout inside the fused
    middle FFT/multiply/IFFT kernel) without changing the underlying values.
    """
    h = domain_extent / n
    kfreq = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    lam = -(4.0 / h**2) * (
        np.sin(np.pi * kx / n) ** 2
        + np.sin(np.pi * ky / n) ** 2
        + np.sin(np.pi * kz / n) ** 2
    )
    inv_lambda = np.zeros_like(lam)
    nonzero = lam != 0
    inv_lambda[nonzero] = 1.0 / lam[nonzero]
    if permute is not None:
        inv_lambda = np.transpose(inv_lambda, permute)
    return wp.array3d(
        np.ascontiguousarray(inv_lambda.astype(np.float32)),
        dtype=wp.float32,
        device=device,
    )


@functools.cache
def _fft_kernels_3d(n: int):
    """Build (and cache) the tiled FFT/transpose kernels for an N×N×N grid.

    A 3-D FFT is three passes of the 2-D row-wise-FFT-then-transpose scheme,
    each pass transforming one axis by first swapping it into the last
    (contiguous, tile_fft-transformable) position via a 3-D permutation.
    We implement this by flattening the leading two axes into a single batch
    dimension for the FFT pass and using a dedicated axis-swap kernel between
    passes (cheaper than 3 full generalized transposes).
    """
    block_dim = max(2, n // 2)

    @wp.kernel(module=f"dft3d_{n}")
    def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)) -> None:
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        wp.tile_fft(row)
        wp.tile_store(y, row, offset=(i, 0))

    @wp.kernel(module=f"dft3d_{n}")
    def ifft_tiled(
        x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)
    ) -> None:
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        wp.tile_ifft(row)
        wp.tile_store(y, row, offset=(i, 0))

    @wp.kernel(module=f"dft3d_{n}")
    def fft_multiply_tiled(
        x: wp.array2d(dtype=wp.vec2f),
        inv_lambda_row: wp.array2d(dtype=wp.float32),
        y: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """Fused: last forward-transform row-pass + spectral multiply.

        Keeps the row resident across both operations instead of a separate
        multiply kernel reading/writing it back to global memory. The swap into
        the next pass's layout still happens as a separate kernel (see
        :func:`_fft3_pass`).
        """
        i, _, _ = wp.tid()
        row = wp.tile_load(x, shape=(1, n), offset=(i, 0))
        scale_row = wp.tile_load(inv_lambda_row, shape=(1, n), offset=(i, 0))
        wp.tile_fft(row)
        scaled = wp.tile_map(_scale_complex, row, scale_row)
        wp.tile_store(y, scaled, offset=(i, 0))

    @wp.kernel(module=f"axisswap3d_{n}")
    def swap_axes_012_to_120(
        x: wp.array3d(dtype=wp.vec2f), y: wp.array3d(dtype=wp.vec2f)
    ) -> None:
        """y[k, i, j] = x[i, j, k]: cyclic axis permutation for the next FFT pass."""
        i, j, k = wp.tid()
        y[k, i, j] = x[i, j, k]

    return {
        "fft_tiled": fft_tiled,
        "ifft_tiled": ifft_tiled,
        "fft_multiply_tiled": fft_multiply_tiled,
        "swap_axes": swap_axes_012_to_120,
        "block_dim": block_dim,
    }


def _fft3_pass(
    kernels: dict,
    direction: str,
    src: wp.array,
    n: int,
    device: str,
    inv_lambda_row: wp.array | None = None,
    workspace: tuple | None = None,
) -> wp.array:
    """Run one axis's row-wise FFT/IFFT over a (n,n,n) complex array.

    Cyclically permutes axes (0,1,2)->(2,0,1) afterward so the next pass's
    target axis becomes the new contiguous last axis. Three passes = full
    3-D transform, ending back at the original axis order.

    When ``inv_lambda_row`` is given (only for the last forward pass), the
    spectral multiply is fused into this pass's row-transform kernel instead
    of a separate elementwise launch.

    ``workspace``, when given, is a ``(flat_dst, swapped)`` pair of
    pre-allocated (non-differentiable) buffers to write into instead of
    allocating fresh ones — used by the forward-only path.
    """
    flat_src = src.reshape((n * n, n))
    grad = workspace is None
    flat_dst = (
        workspace[0]
        if workspace
        else wp.zeros((n * n, n), dtype=wp.vec2f, requires_grad=grad, device=device)
    )
    if inv_lambda_row is not None:
        wp.launch_tiled(
            kernels["fft_multiply_tiled"],
            dim=[n * n, 1],
            inputs=[flat_src, inv_lambda_row.reshape((n * n, n))],
            outputs=[flat_dst],
            block_dim=kernels["block_dim"],
            device=device,
        )
    else:
        kernel = kernels["fft_tiled"] if direction == "fwd" else kernels["ifft_tiled"]
        wp.launch_tiled(
            kernel,
            dim=[n * n, 1],
            inputs=[flat_src],
            outputs=[flat_dst],
            block_dim=kernels["block_dim"],
            device=device,
        )
    dst_3d = flat_dst.reshape((n, n, n))
    swapped = (
        workspace[1]
        if workspace
        else wp.zeros((n, n, n), dtype=wp.vec2f, requires_grad=grad, device=device)
    )
    wp.launch(
        kernels["swap_axes"],
        dim=(n, n, n),
        inputs=[dst_3d],
        outputs=[swapped],
        device=device,
    )
    return swapped


def _fft_3d(
    kernels: dict,
    direction: str,
    src: wp.array,
    n: int,
    device: str,
    inv_lambda_row: wp.array | None = None,
    workspaces: list | None = None,
) -> wp.array:
    """Full 3-D FFT/IFFT via three row-wise-transform + cyclic-axis-swap passes.

    ``inv_lambda_row`` (forward direction only) fuses the spectral multiply
    into the final pass — see :func:`_fft3_pass`. ``workspaces``, when given,
    is a 3-element list of per-pass ``(flat_dst, swapped)`` buffer pairs from
    :func:`_poisson_workspace_3d` (forward-only path).
    """
    x = src
    for pass_i in range(3):
        is_last_fwd = direction == "fwd" and pass_i == 2
        x = _fft3_pass(
            kernels,
            direction,
            x,
            n,
            device,
            inv_lambda_row=inv_lambda_row if is_last_fwd else None,
            workspace=workspaces[pass_i] if workspaces else None,
        )
    return x


@functools.cache
def _swap_axes_kernel(_tag: str = "bluestein_swap3d"):
    """y[k, i, j] = x[i, j, k]: cyclic axis permutation, valid for any ``n``."""

    @wp.kernel(module="bluestein_swap3d")
    def swap_k(x: wp.array3d(dtype=wp.vec2f), y: wp.array3d(dtype=wp.vec2f)) -> None:
        i, j, k = wp.tid()
        y[k, i, j] = x[i, j, k]

    return swap_k


@functools.cache
def _scale_by_field_kernel_3d(_tag: str = "bluestein_scale3d"):
    """out[i,j,k] = z[i,j,k] * s[i,j,k] with z complex and s a real field."""

    @wp.kernel(module="bluestein_scale3d")
    def scale_k(
        z: wp.array3d(dtype=wp.vec2f),
        s: wp.array3d(dtype=wp.float32),
        out: wp.array3d(dtype=wp.vec2f),
    ) -> None:
        i, j, k = wp.tid()
        out[i, j, k] = _scale_complex(z[i, j, k], s[i, j, k])

    return scale_k


def _bluestein_fft_3d(src: wp.array, n: int, direction: str, device: str) -> wp.array:
    """Full 3-D DFT via three (Bluestein-rows + cyclic-axis-swap) passes.

    Mirrors :func:`_fft_3d` but uses the non-pow2-capable Bluestein row
    transform. Three cyclic swaps return the array to its original axis order.
    """
    x = src
    for _ in range(3):
        flat = _bluestein_fft_rows(x.reshape((n * n, n)), n, direction, device).reshape(
            (n, n, n)
        )
        swapped = wp.zeros((n, n, n), dtype=wp.vec2f, requires_grad=True, device=device)
        wp.launch(
            _swap_axes_kernel(),
            dim=(n, n, n),
            inputs=[flat],
            outputs=[swapped],
            device=device,
        )
        x = swapped
    return x


def _spectral_poisson_3d_bluestein(
    rhs_c: wp.array, domain_extent: float, device: str
) -> wp.array:
    """Non-pow2 3-D spectral Poisson solve (Bluestein FFTs, on-GPU, differentiable).

    Numerically identical to :func:`_spectral_poisson_3d_core` (same discrete
    FD-Laplacian eigenvalues). The spectral multiply is a separate elementwise
    kernel in natural axis order — after the forward transform's three cyclic
    swaps the array is back in ``(kx, ky, kz)`` order, so the un-permuted
    ``_inv_lambda_3d`` applies directly (no fused-pass permutation needed).
    """
    n = rhs_c.shape[0]
    inv_lambda = _inv_lambda_3d(n, domain_extent, device)

    hat = _bluestein_fft_3d(rhs_c, n, "fwd", device)
    scaled = wp.zeros((n, n, n), dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        _scale_by_field_kernel_3d(),
        dim=(n, n, n),
        inputs=[hat, inv_lambda],
        outputs=[scaled],
        device=device,
    )
    return _bluestein_fft_3d(scaled, n, "bwd", device)


def _spectral_poisson_3d_core(
    rhs_c: wp.array, domain_extent: float, device: str, workspace: dict | None = None
) -> wp.array:
    """Core 3-D spectral Poisson solve on an already-packed complex RHS.

    Returns the complex (unnormalized) pressure field, so call sites that
    produce/consume complex buffers directly can skip the redundant
    pack/unpack kernels.

    The spectral multiply is fused into the last forward-FFT pass instead of
    running as a separate elementwise kernel. Right before that pass's
    axis-swap, the array is permuted (1,2,0) relative to the original
    (kx,ky,kz) axis order, so we index a matching
    (1,2,0)-permuted copy of 1/λ (cheap to precompute once, cached) rather
    than the natural-order array used elsewhere.

    Pass ``workspace`` (from :func:`_poisson_workspace_3d`) to reuse
    pre-allocated scratch buffers across steps instead of allocating fresh
    ones on every call — used by the forward-only (non-taped) path.
    """
    n = rhs_c.shape[0]
    if not _is_pow2(n):
        # cuFFTDx tile_fft only compiles pow2 lengths; use the Bluestein path.
        return _spectral_poisson_3d_bluestein(rhs_c, domain_extent, device)
    kernels = _fft_kernels_3d(n)
    inv_lambda_permuted = _inv_lambda_3d(n, domain_extent, device, permute=(1, 2, 0))

    fwd_ws = workspace["fwd"] if workspace else None
    bwd_ws = workspace["bwd"] if workspace else None
    p_c = _fft_3d(
        kernels,
        "fwd",
        rhs_c,
        n,
        device,
        inv_lambda_row=inv_lambda_permuted,
        workspaces=fwd_ws,
    )
    return _fft_3d(kernels, "bwd", p_c, n, device, workspaces=bwd_ws)


def _poisson_workspace_3d(n: int, device: str) -> dict:
    """Pre-allocate (non-differentiable) scratch buffers for the forward-only 3-D path.

    Reused across every timestep instead of allocating fresh buffers per
    step and per FFT pass (3 passes x 2 buffers x 2 directions = 12 fresh
    allocations per Poisson solve otherwise).
    """

    def _pass_bufs():
        return (
            wp.zeros((n * n, n), dtype=wp.vec2f, device=device),
            wp.zeros((n, n, n), dtype=wp.vec2f, device=device),
        )

    return {
        "fwd": [_pass_bufs() for _ in range(3)],
        "bwd": [_pass_bufs() for _ in range(3)],
    }


def ns2d_solve_tape(
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    device: str = "cpu",
    track_scalar_grads: bool = False,
) -> tuple[wp.Tape, wp.array, wp.array, wp.array, wp.array, wp.array, wp.array]:
    """Run periodic 2-D incompressible NS by Chorin projection under a tape.

    [2D-only function]

    Used only by :func:`vector_jacobian_product` — see :func:`ns2d_solve_forward`
    for the plain-forward path used by :func:`apply`. Steps per time-step:
        1. Tentative velocity: u* = u + dt·(-u·∇u + ν∇²u)
        2. Pressure Poisson: ∇²p = (1/dt)·∇·u*  (spectral FFT, periodic)
        3. Velocity correction: u^(n+1) = u* - dt·∇p

    Uses Warp's native source-to-source reverse-mode AD for every kernel
    (verified to agree with a from-scratch analytical adjoint and a central-FD
    ground truth to float32 roundoff — see repo history for the check). ``nu``
    and ``dt`` are passed to kernels as 1-element arrays rather than plain
    floats so that the tape can track gradients w.r.t. them, with no manual
    ``record_func`` bookkeeping required.

    ``track_scalar_grads`` gates ``requires_grad`` on the ``nu``/``dt``
    arrays: leave it False unless the caller actually requested a viscosity
    or dt gradient. Every ``requires_grad=True`` array adds bookkeeping to
    every tape node that reads it, and for a 3-D solve over many timesteps
    that overhead is measurable (~50% higher VJP wall-clock in testing) even
    though these are 1-element arrays — so this must not be unconditional
    when the caller only wants d(loss)/d(v0).

    Each step allocates fresh output buffers (rather than ping-ponging between
    two reused buffers) so the tape's per-step data dependencies are
    unambiguous — this also removes the need to manually zero stale adjoints
    between steps.

    Unlike a plain forward solve, this does NOT materialize the final result
    as a NumPy array: the caller (VJP) only needs the tape and the Warp
    arrays to seed cotangents on and read .grad from, so skipping the
    device->host copy of the (unused) forward result avoids a wasted sync.

    Returns:
        (tape, ux_final_wp, uy_final_wp, ux_ic_wp, uy_ic_wp, nu_wp, dt_wp)
        The final velocity Warp arrays have requires_grad=True so
        tape.backward() fills their .grad attributes.  nu_wp / dt_wp are
        (1,) scalar leaves; their grads are filled directly by tape.backward()
        only when track_scalar_grads=True (otherwise .grad stays None).
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / (h * h)

    # Warp 1.12+ requires block_dim as int (256 = 16×16 for 2D, 128 for 1D).
    # On CPU, Warp ignores block_dim and uses 1; these ints are safe on both.
    _bd_2d = 256

    ux_np = v0_np[:, :, 0, 0]
    uy_np = v0_np[:, :, 0, 1]

    ux_wp = wp.array(ux_np, dtype=wp.float32, requires_grad=True, device=device)
    uy_wp = wp.array(uy_np, dtype=wp.float32, requires_grad=True, device=device)

    nu_wp = wp.array(
        np.array([viscosity], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=track_scalar_grads,
        device=device,
    )
    dt_wp = wp.array(
        np.array([dt], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=track_scalar_grads,
        device=device,
    )

    tape = wp.Tape()
    with tape:
        cur_ux, cur_uy = ux_wp, uy_wp

        for _step_i in range(steps):
            ux_star = wp.zeros(
                (n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            uy_star = wp.zeros(
                (n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            wp.launch(
                tentative_vel_2d_kernel,
                dim=(n, n),
                inputs=[cur_ux, cur_uy, ux_star, uy_star, dt_wp, inv_2h, inv_h2, nu_wp],
                block_dim=_bd_2d,
                device=device,
            )

            # Step 2: pressure Poisson ∇²p = ∇·u*/dt (periodic, spectral FFT).
            # divergence_2d_to_complex_kernel fuses the divergence computation
            # with the real->complex pack that would otherwise precede the
            # first FFT stage.
            div_star_c = wp.zeros(
                (n, n), dtype=wp.vec2f, requires_grad=True, device=device
            )
            wp.launch(
                divergence_2d_to_complex_kernel,
                dim=(n, n),
                inputs=[ux_star, uy_star, div_star_c, dt_wp, inv_2h],
                block_dim=_bd_2d,
                device=device,
            )

            p_c = _spectral_poisson_2d_core(div_star_c, domain_extent, device)

            # Step 3: velocity correction u^(n+1) = u* - dt·∇p.
            # pressure_correct_2d_from_complex_kernel fuses the "extract real +
            # normalize" step with the pressure-correction kernel, reading p
            # directly out of the complex IFFT output.
            next_ux = wp.zeros(
                (n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            next_uy = wp.zeros(
                (n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            wp.launch(
                pressure_correct_2d_from_complex_kernel,
                dim=(n, n),
                inputs=[
                    ux_star,
                    uy_star,
                    p_c,
                    float(n * n),
                    next_ux,
                    next_uy,
                    dt_wp,
                    inv_2h,
                ],
                block_dim=_bd_2d,
                device=device,
            )

            cur_ux, cur_uy = next_ux, next_uy

    return (
        tape,
        cur_ux,
        cur_uy,
        ux_wp,
        uy_wp,
        nu_wp,
        dt_wp,
    )


def ns2d_solve_forward(
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    device: str = "cpu",
) -> np.ndarray:
    """Forward-only periodic 2-D incompressible NS by Chorin projection, no gradients.

    [2D-only function]

    Same physics as :func:`ns2d_solve_tape`, but structurally distinct: no
    ``wp.Tape``, no ``requires_grad`` arrays, two ping-pong velocity buffers
    reused every step instead of fresh per-step allocations
    (mirrors Warp's own ``example_wave.py`` grid-swap pattern), and one
    pre-allocated FFT workspace reused across all steps instead of allocating
    fresh scratch buffers on every Poisson solve.
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / (h * h)
    _bd_2d = 256

    ux_np = v0_np[:, :, 0, 0]
    uy_np = v0_np[:, :, 0, 1]

    vel_bufs_x = [
        wp.array(ux_np, dtype=wp.float32, device=device),
        wp.zeros((n, n), dtype=wp.float32, device=device),
    ]
    vel_bufs_y = [
        wp.array(uy_np, dtype=wp.float32, device=device),
        wp.zeros((n, n), dtype=wp.float32, device=device),
    ]
    nu_wp = wp.array(
        np.array([viscosity], dtype=np.float32), dtype=wp.float32, device=device
    )
    dt_wp = wp.array(np.array([dt], dtype=np.float32), dtype=wp.float32, device=device)

    ux_star = wp.zeros((n, n), dtype=wp.float32, device=device)
    uy_star = wp.zeros((n, n), dtype=wp.float32, device=device)
    div_star_c = wp.zeros((n, n), dtype=wp.vec2f, device=device)
    poisson_ws = _poisson_workspace_2d(n, device)

    src, dst = 0, 1
    for _step_i in range(steps):
        wp.launch(
            tentative_vel_2d_kernel,
            dim=(n, n),
            inputs=[
                vel_bufs_x[src],
                vel_bufs_y[src],
                ux_star,
                uy_star,
                dt_wp,
                inv_2h,
                inv_h2,
                nu_wp,
            ],
            block_dim=_bd_2d,
            device=device,
        )
        wp.launch(
            divergence_2d_to_complex_kernel,
            dim=(n, n),
            inputs=[ux_star, uy_star, div_star_c, dt_wp, inv_2h],
            block_dim=_bd_2d,
            device=device,
        )
        p_c = _spectral_poisson_2d_core(
            div_star_c, domain_extent, device, workspace=poisson_ws
        )
        wp.launch(
            pressure_correct_2d_from_complex_kernel,
            dim=(n, n),
            inputs=[
                ux_star,
                uy_star,
                p_c,
                float(n * n),
                vel_bufs_x[dst],
                vel_bufs_y[dst],
                dt_wp,
                inv_2h,
            ],
            block_dim=_bd_2d,
            device=device,
        )
        src, dst = dst, src

    ux_out = vel_bufs_x[src].numpy()
    uy_out = vel_bufs_y[src].numpy()
    return np.stack([ux_out, uy_out], axis=-1)[:, :, np.newaxis, :]  # (N,N,1,2)


def ns2d_vjp(
    tape: wp.Tape,
    ux_final: wp.array,
    uy_final: wp.array,
    ux_ic: wp.array,
    uy_ic: wp.array,
    cotangent_np: np.ndarray,
    device: str,
    nu_wp: "wp.array | None" = None,
    dt_wp: "wp.array | None" = None,
) -> dict[str, np.ndarray]:
    """Propagate cotangents through the 2-D projection tape.

    [2D-only function]

    v0, viscosity, and dt are all ordinary tape leaves (nu_wp/dt_wp are the
    1-element arrays passed into tentative_vel_2d_kernel, divergence_2d_kernel,
    and pressure_correct_2d_kernel in ns2d_solve_tape), so tape.backward() fills
    their .grad directly via Warp's native reverse-mode AD — no manual
    per-step gradient accumulation needed.
    """
    ux_final.grad = wp.array(
        cotangent_np[:, :, 0, 0].astype(np.float32), dtype=wp.float32, device=device
    )
    uy_final.grad = wp.array(
        cotangent_np[:, :, 0, 1].astype(np.float32), dtype=wp.float32, device=device
    )

    tape.backward()

    grad_v0 = np.zeros_like(cotangent_np)
    if ux_ic.grad is not None:
        grad_v0[:, :, 0, 0] = ux_ic.grad.numpy()
    if uy_ic.grad is not None:
        grad_v0[:, :, 0, 1] = uy_ic.grad.numpy()

    grad_nu = np.zeros(1, dtype=np.float32)
    if nu_wp is not None and nu_wp.grad is not None:
        grad_nu[0] = float(nu_wp.grad.numpy()[0])

    grad_dt = np.zeros(1, dtype=np.float32)
    if dt_wp is not None and dt_wp.grad is not None:
        grad_dt[0] = float(dt_wp.grad.numpy()[0])

    return {
        "v0": grad_v0.astype(np.float32),
        "viscosity": grad_nu,
        "dt": grad_dt,
    }


# ============================================================
# 3-D NS forward solve (Chorin projection)
# ============================================================


def ns3d_solve_tape(
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    device: str = "cpu",
    track_scalar_grads: bool = False,
) -> tuple[
    wp.Tape,
    wp.array,
    wp.array,
    wp.array,
    wp.array,
    wp.array,
    wp.array,
    wp.array,
    wp.array,
]:
    """Run 3-D incompressible NS by Chorin projection under a tape.

    Used only by :func:`vector_jacobian_product` — see :func:`ns3d_solve_forward`
    for the plain-forward path used by :func:`apply`. Uses Warp's native
    reverse-mode AD for every kernel (verified to agree with a from-scratch
    analytical adjoint and a central-FD ground truth to float32 roundoff).
    Each step allocates fresh output buffers instead of ping-ponging between
    two reused buffers, so the tape's per-step data dependencies are
    unambiguous and no manual adjoint-zeroing is needed.

    ``nu``/``dt`` are passed to kernels as 1-element arrays (like the 2-D
    solver) so the tape can track gradients w.r.t. them.

    ``track_scalar_grads`` gates ``requires_grad`` on the ``nu``/``dt``
    arrays: leave it False unless the caller actually requested a viscosity
    or dt gradient. Measured ~50% higher 3-D VJP wall-clock when these are
    unconditionally grad-tracked, even though they are 1-element arrays —
    each additional requires_grad=True array adds bookkeeping to every tape
    node that reads it, across every timestep. Must not be unconditional
    when the caller only wants d(loss)/d(v0).

    Unlike a plain forward solve, this does NOT materialize the final result
    as a NumPy array — the caller (VJP) only needs the tape and Warp arrays,
    so skipping that device->host copy avoids a wasted sync.

    Returns:
        (tape, ux_final_wp, uy_final_wp, uz_final_wp,
         ux_ic_wp, uy_ic_wp, uz_ic_wp, nu_wp, dt_wp)
        The final velocity Warp arrays have requires_grad=True so
        tape.backward() fills their .grad attributes.  nu_wp / dt_wp are
        (1,) scalar leaves; their grads are filled directly by tape.backward()
        only when track_scalar_grads=True (otherwise .grad stays None).
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    h2 = h * h
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2

    # Warp 1.12+ requires block_dim as int (256 = 16×16 or 8×8×4).
    _bd_3d = 256

    ux_wp = wp.array(
        v0_np[:, :, :, 0], dtype=wp.float32, requires_grad=True, device=device
    )
    uy_wp = wp.array(
        v0_np[:, :, :, 1], dtype=wp.float32, requires_grad=True, device=device
    )
    uz_wp = wp.array(
        v0_np[:, :, :, 2], dtype=wp.float32, requires_grad=True, device=device
    )
    nu_wp = wp.array(
        np.array([viscosity], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=track_scalar_grads,
        device=device,
    )
    dt_wp = wp.array(
        np.array([dt], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=track_scalar_grads,
        device=device,
    )

    tape = wp.Tape()
    with tape:
        cur_ux, cur_uy, cur_uz = ux_wp, uy_wp, uz_wp

        for _step_i in range(steps):
            # Step 1: tentative velocity u* = u + dt·(-u·∇u + ν∇²u)
            ux_star = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            uy_star = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            uz_star = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            wp.launch(
                tentative_vel_3d_kernel,
                dim=(n, n, n),
                inputs=[
                    cur_ux,
                    cur_uy,
                    cur_uz,
                    ux_star,
                    uy_star,
                    uz_star,
                    dt_wp,
                    inv_2h,
                    inv_h2,
                    nu_wp,
                ],
                block_dim=_bd_3d,
                device=device,
            )

            # Step 2: pressure Poisson ∇²p = ∇·u*/dt (periodic, spectral FFT).
            # divergence_3d_to_complex_kernel fuses the divergence computation
            # with the real->complex pack that would otherwise precede the
            # first FFT stage.
            div_star_c = wp.zeros(
                (n, n, n), dtype=wp.vec2f, requires_grad=True, device=device
            )
            _wlaunch(
                divergence_3d_to_complex_kernel,
                dim=(n, n, n),
                inputs=[ux_star, uy_star, uz_star, div_star_c, dt_wp, inv_2h],
                block_dim=_bd_3d,
                device=device,
            )
            p_c = _spectral_poisson_3d_core(div_star_c, domain_extent, device)

            # Step 3: velocity correction u^(n+1) = u* - dt·∇p.
            # pressure_correct_3d_from_complex_kernel fuses the "extract real +
            # normalize" step with the pressure-correction kernel, reading p
            # directly out of the complex IFFT output.
            next_ux = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            next_uy = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            next_uz = wp.zeros(
                (n, n, n), dtype=wp.float32, requires_grad=True, device=device
            )
            _wlaunch(
                pressure_correct_3d_from_complex_kernel,
                dim=(n, n, n),
                inputs=[
                    ux_star,
                    uy_star,
                    uz_star,
                    p_c,
                    float(n * n * n),
                    next_ux,
                    next_uy,
                    next_uz,
                    dt_wp,
                    inv_2h,
                ],
                block_dim=_bd_3d,
                device=device,
            )

            cur_ux, cur_uy, cur_uz = next_ux, next_uy, next_uz

    return (
        tape,
        cur_ux,
        cur_uy,
        cur_uz,
        ux_wp,
        uy_wp,
        uz_wp,
        nu_wp,
        dt_wp,
    )


def ns3d_solve_forward(
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    device: str = "cpu",
) -> np.ndarray:
    """Forward-only 3-D incompressible NS by Chorin projection, no gradients.

    Same physics as :func:`ns3d_solve_tape`, but structurally distinct: no
    ``wp.Tape``, no ``requires_grad`` arrays, two ping-pong velocity buffers
    per component reused every step instead of fresh per-step allocations,
    and one pre-allocated FFT workspace reused across all steps.
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / (h * h)
    _bd_3d = 256

    vel_bufs_x = [
        wp.array(v0_np[:, :, :, 0], dtype=wp.float32, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, device=device),
    ]
    vel_bufs_y = [
        wp.array(v0_np[:, :, :, 1], dtype=wp.float32, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, device=device),
    ]
    vel_bufs_z = [
        wp.array(v0_np[:, :, :, 2], dtype=wp.float32, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, device=device),
    ]

    ux_star = wp.zeros((n, n, n), dtype=wp.float32, device=device)
    uy_star = wp.zeros((n, n, n), dtype=wp.float32, device=device)
    uz_star = wp.zeros((n, n, n), dtype=wp.float32, device=device)
    div_star_c = wp.zeros((n, n, n), dtype=wp.vec2f, device=device)
    poisson_ws = _poisson_workspace_3d(n, device)
    nu_wp = wp.array(
        np.array([viscosity], dtype=np.float32), dtype=wp.float32, device=device
    )
    dt_wp = wp.array(np.array([dt], dtype=np.float32), dtype=wp.float32, device=device)

    src, dst = 0, 1
    for _step_i in range(steps):
        wp.launch(
            tentative_vel_3d_kernel,
            dim=(n, n, n),
            inputs=[
                vel_bufs_x[src],
                vel_bufs_y[src],
                vel_bufs_z[src],
                ux_star,
                uy_star,
                uz_star,
                dt_wp,
                inv_2h,
                inv_h2,
                nu_wp,
            ],
            block_dim=_bd_3d,
            device=device,
        )
        wp.launch(
            divergence_3d_to_complex_kernel,
            dim=(n, n, n),
            inputs=[ux_star, uy_star, uz_star, div_star_c, dt_wp, inv_2h],
            block_dim=_bd_3d,
            device=device,
        )
        p_c = _spectral_poisson_3d_core(
            div_star_c, domain_extent, device, workspace=poisson_ws
        )
        wp.launch(
            pressure_correct_3d_from_complex_kernel,
            dim=(n, n, n),
            inputs=[
                ux_star,
                uy_star,
                uz_star,
                p_c,
                float(n * n * n),
                vel_bufs_x[dst],
                vel_bufs_y[dst],
                vel_bufs_z[dst],
                dt_wp,
                inv_2h,
            ],
            block_dim=_bd_3d,
            device=device,
        )
        src, dst = dst, src

    ux_out = vel_bufs_x[src].numpy()
    uy_out = vel_bufs_y[src].numpy()
    uz_out = vel_bufs_z[src].numpy()
    return np.stack([ux_out, uy_out, uz_out], axis=-1)  # (N,N,N,3)


def ns3d_vjp(
    tape: wp.Tape,
    ux_final: wp.array,
    uy_final: wp.array,
    uz_final: wp.array,
    ux_ic: wp.array,
    uy_ic: wp.array,
    uz_ic: wp.array,
    cotangent_np: np.ndarray,
    device: str,
    nu_wp: "wp.array | None" = None,
    dt_wp: "wp.array | None" = None,
) -> dict[str, np.ndarray]:
    """Propagate cotangents through the 3-D projection tape.

    v0, viscosity, and dt are all ordinary tape leaves (nu_wp/dt_wp are the
    1-element arrays passed into tentative_vel_3d_kernel,
    divergence_3d_to_complex_kernel, and pressure_correct_3d_from_complex_kernel
    in ns3d_solve_tape), so tape.backward() fills their .grad directly.
    """
    ux_final.grad = wp.array(
        cotangent_np[:, :, :, 0].astype(np.float32), dtype=wp.float32, device=device
    )
    uy_final.grad = wp.array(
        cotangent_np[:, :, :, 1].astype(np.float32), dtype=wp.float32, device=device
    )
    uz_final.grad = wp.array(
        cotangent_np[:, :, :, 2].astype(np.float32), dtype=wp.float32, device=device
    )

    tape.backward()

    grad_v0 = np.zeros_like(cotangent_np)
    if ux_ic.grad is not None:
        grad_v0[:, :, :, 0] = ux_ic.grad.numpy()
    if uy_ic.grad is not None:
        grad_v0[:, :, :, 1] = uy_ic.grad.numpy()
    if uz_ic.grad is not None:
        grad_v0[:, :, :, 2] = uz_ic.grad.numpy()

    grad_nu = np.zeros(1, dtype=np.float32)
    if nu_wp is not None and nu_wp.grad is not None:
        grad_nu[0] = float(nu_wp.grad.numpy()[0])

    grad_dt = np.zeros(1, dtype=np.float32)
    if dt_wp is not None and dt_wp.grad is not None:
        grad_dt[0] = float(dt_wp.grad.numpy()[0])

    return {
        "v0": grad_v0.astype(np.float32),
        "viscosity": grad_nu,
        "dt": grad_dt,
    }


# ============================================================
# 2-D channel with a cylinder (Schäfer–Turek DFG 2D-1)
#
# Staggered (MAC) grid on the nx × ny cells: u on the x faces, shape
# (nx + 1, ny), face i at x = i dx (face 0 is the inlet, face nx the outlet);
# v on the y faces, shape (nx, ny + 1), face j at y = j dx (faces 0 and ny
# are the walls); p at the cell centers. The cylinder is the staircase of
# cells whose center lies strictly inside it. A face is "open" when it
# carries flow: between two fluid cells, or the inlet / outlet face of a
# fluid cell. Faces touching a solid cell are held at zero velocity.
#
# The step is the periodic path's explicit-Euler projection with central
# differences: u* = u + dt (ν ∇²u − u·∇u) on every face, then the pressure
# Poisson equation over the fluid cells (Neumann on walls, inlet and cylinder,
# p = 0 on the outlet face) by conjugate gradients, then u = u* − dt ∇p on
# the open faces.
# ============================================================


@wp.kernel
def channel_tentative_u_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    inflow: wp.array(dtype=wp.float32),
    u_star: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    nu: wp.array(dtype=wp.float32),
    inv_dx: float,
) -> None:
    """u* on the x faces, with the inflow profile on the inlet face."""
    i, j = wp.tid()
    nx = u.shape[0] - 1
    ny = u.shape[1]
    if i == 0:
        u_star[i, j] = inflow[j]
    else:
        uc = u[i, j]
        uw = u[i - 1, j]
        # Zero gradient past the outlet. Mirrored ghosts make u = 0 on the walls.
        ue = uc
        if i < nx:
            ue = u[i + 1, j]
        us = -uc
        if j > 0:
            us = u[i, j - 1]
        un = -uc
        if j < ny - 1:
            un = u[i, j + 1]
        # v at the face from the four surrounding y faces. Past the outlet the
        # last cell's values continue.
        il = i - 1
        ir = wp.min(i, nx - 1)
        va = 0.25 * (v[il, j] + v[il, j + 1] + v[ir, j] + v[ir, j + 1])
        adv = 0.5 * inv_dx * (uc * (ue - uw) + va * (un - us))
        lap = inv_dx * inv_dx * (ue + uw + un + us - 4.0 * uc)
        u_star[i, j] = uc + dt[0] * (nu[0] * lap - adv)


@wp.kernel
def channel_tentative_v_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    nu: wp.array(dtype=wp.float32),
    inv_dx: float,
) -> None:
    """v* on the y faces, zero on the walls."""
    i, j = wp.tid()
    nx = v.shape[0]
    ny = v.shape[1] - 1
    if j == 0 or j == ny:
        v_star[i, j] = 0.0
    else:
        vc = v[i, j]
        vs = v[i, j - 1]
        vn = v[i, j + 1]
        # v = 0 on the inlet face (mirrored ghost), zero gradient past the outlet.
        vw = -vc
        if i > 0:
            vw = v[i - 1, j]
        ve = vc
        if i < nx - 1:
            ve = v[i + 1, j]
        ua = 0.25 * (u[i, j - 1] + u[i + 1, j - 1] + u[i, j] + u[i + 1, j])
        adv = 0.5 * inv_dx * (ua * (ve - vw) + vc * (vn - vs))
        lap = inv_dx * inv_dx * (ve + vw + vn + vs - 4.0 * vc)
        v_star[i, j] = vc + dt[0] * (nu[0] * lap - adv)


@wp.kernel
def channel_divergence_kernel(
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    open_u: wp.array2d(dtype=wp.int32),
    open_v: wp.array2d(dtype=wp.int32),
    rhs: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    dx: float,
) -> None:
    """Right-hand side −dx ∇·u* / dt of the Poisson system scaled by dx².

    Closed faces contribute no flux, and solid cells get zero.
    """
    i, j = wp.tid()
    flux = (
        u_star[i + 1, j] * float(open_u[i + 1, j])
        - u_star[i, j] * float(open_u[i, j])
        + v_star[i, j + 1] * float(open_v[i, j + 1])
        - v_star[i, j] * float(open_v[i, j])
    )
    rhs[i, j] = -flux * dx / dt[0]


@wp.kernel
def channel_laplacian_kernel(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    z: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    open_u: wp.array2d(dtype=wp.int32),
    open_v: wp.array2d(dtype=wp.int32),
    fluid: wp.array2d(dtype=wp.int32),
) -> None:
    """Compute z = alpha A x + beta y, with A = −dx² ∇² over the fluid cells.

    A couples two fluid cells across every open face and adds the p = 0
    outlet face as a ghost of opposite sign, so it is symmetric positive
    definite. Solid cells carry the identity, which keeps them at zero.
    Arrays are flattened row-major from (nx, ny).
    """
    i, j = wp.tid()
    nx = fluid.shape[0]
    ny = fluid.shape[1]
    c = i * ny + j
    pc = x[c]
    ax = pc
    if fluid[i, j] == 1:
        ax = 0.0
        if i > 0 and open_u[i, j] == 1:
            ax += pc - x[c - ny]
        if i == nx - 1:
            ax += 2.0 * pc
        elif open_u[i + 1, j] == 1:
            ax += pc - x[c + ny]
        if open_v[i, j] == 1:
            ax += pc - x[c - 1]
        if open_v[i, j + 1] == 1:
            ax += pc - x[c + 1]
    z[c] = alpha * ax + beta * y[c]


@wp.kernel
def channel_correct_u_kernel(
    u_star: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    open_u: wp.array2d(dtype=wp.int32),
    inflow: wp.array(dtype=wp.float32),
    u_new: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_dx: float,
) -> None:
    """Correct u = u* − dt ∂p/∂x on the open x faces, keeping the inlet profile."""
    i, j = wp.tid()
    nx = p.shape[0]
    val = 0.0
    if i == 0:
        val = inflow[j]
    elif open_u[i, j] == 1:
        pw = p[i - 1, j]
        pe = -pw  # p = 0 on the outlet face
        if i < nx:
            pe = p[i, j]
        val = u_star[i, j] - dt[0] * inv_dx * (pe - pw)
    u_new[i, j] = val


@wp.kernel
def channel_correct_v_kernel(
    v_star: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    open_v: wp.array2d(dtype=wp.int32),
    v_new: wp.array2d(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    inv_dx: float,
) -> None:
    """Correct v = v* − dt ∂p/∂y on the open y faces."""
    i, j = wp.tid()
    val = 0.0
    if open_v[i, j] == 1:
        val = v_star[i, j] - dt[0] * inv_dx * (p[i, j] - p[i, j - 1])
    v_new[i, j] = val


@wp.kernel
def channel_force_kernel(
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    open_u: wp.array2d(dtype=wp.int32),
    open_v: wp.array2d(dtype=wp.int32),
    fluid: wp.array2d(dtype=wp.int32),
    force: wp.array2d(dtype=wp.float32),
    row: int,
    dt: wp.array(dtype=wp.float32),
    dx: float,
) -> None:
    """Add this step's force on the cylinder to ``force[row]``.

    Launched over (nx + 1, ny + 1). Each interior face touching a solid cell
    contributes the momentum the fluid would have carried through it, u* dx²,
    which the projection removes (advection and viscous stress), plus the
    pressure of the fluid cells on either side of it.
    """
    i, j = wp.tid()
    nx = fluid.shape[0]
    ny = fluid.shape[1]
    if i > 0 and i < nx and j < ny and open_u[i, j] == 0:
        fx = u_star[i, j] * dx * dx / dt[0]
        if fluid[i - 1, j] == 1:
            fx += p[i - 1, j] * dx
        if fluid[i, j] == 1:
            fx -= p[i, j] * dx
        wp.atomic_add(force, row, 0, fx)
    if j > 0 and j < ny and i < nx and open_v[i, j] == 0:
        fy = v_star[i, j] * dx * dx / dt[0]
        if fluid[i, j - 1] == 1:
            fy += p[i, j - 1] * dx
        if fluid[i, j] == 1:
            fy -= p[i, j] * dx
        wp.atomic_add(force, row, 1, fy)


@wp.kernel
def channel_init_u_kernel(
    ux: wp.array2d(dtype=wp.float32),
    open_u: wp.array2d(dtype=wp.int32),
    inflow: wp.array(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
) -> None:
    """x-face velocity from the cell-centered initial field."""
    i, j = wp.tid()
    nx = ux.shape[0]
    val = 0.0
    if i == 0:
        val = inflow[j]
    elif open_u[i, j] == 1:
        val = ux[i - 1, j]
        if i < nx:
            val = 0.5 * (ux[i - 1, j] + ux[i, j])
    u[i, j] = val


@wp.kernel
def channel_init_v_kernel(
    uy: wp.array2d(dtype=wp.float32),
    open_v: wp.array2d(dtype=wp.int32),
    v: wp.array2d(dtype=wp.float32),
) -> None:
    """y-face velocity from the cell-centered initial field."""
    i, j = wp.tid()
    val = 0.0
    if open_v[i, j] == 1:
        val = 0.5 * (uy[i, j - 1] + uy[i, j])
    v[i, j] = val


@wp.kernel
def channel_centers_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    fluid: wp.array2d(dtype=wp.int32),
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
) -> None:
    """Face velocities averaged to the cell centers, zero in solid cells."""
    i, j = wp.tid()
    f = float(fluid[i, j])
    ux[i, j] = 0.5 * (u[i, j] + u[i + 1, j]) * f
    uy[i, j] = 0.5 * (v[i, j] + v[i, j + 1]) * f


@wp.kernel
def channel_basis_rows_kernel(
    a: wp.array2d(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array2d(dtype=wp.float32),
) -> None:
    """Compute dst = a @ src, a change of basis along x."""
    i, j = wp.tid()
    s = float(0.0)  # noqa: UP018 (Warp needs a dynamic variable in the loop)
    for k in range(a.shape[1]):
        s += a[i, k] * src[k, j]
    dst[i, j] = s


@wp.kernel
def channel_basis_cols_kernel(
    src: wp.array2d(dtype=wp.float32),
    b: wp.array2d(dtype=wp.float32),
    scale: wp.array2d(dtype=wp.float32),
    y: wp.array2d(dtype=wp.float32),
    alpha: float,
    beta: float,
    dst: wp.array2d(dtype=wp.float32),
) -> None:
    """Compute dst = alpha scale ∘ (src @ b) + beta y, a change of basis along y."""
    i, j = wp.tid()
    s = float(0.0)  # noqa: UP018 (Warp needs a dynamic variable in the loop)
    for k in range(b.shape[0]):
        s += src[i, k] * b[k, j]
    dst[i, j] = alpha * scale[i, j] * s + beta * y[i, j]


@wp.kernel
def _flag_unconverged_kernel(
    r_norm_sq: wp.array(dtype=wp.float32),
    atol_sq: wp.array(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
) -> None:
    if r_norm_sq[0] > atol_sq[0]:
        flag[0] = 1


# Relative residual at which the pressure CG stops: the loosest at which Cd,
# Cl and the pressure drop stay within 0.3% of a 1e-7 solve at 10 and 20 cells
# per diameter, and drag_opt ends at the same drag. At 1e-2 Cl moves by 5%.
_CHANNEL_CG_TOL = 1e-3
_CHANNEL_CG_MAXITER = 5000


def _check_cg_converged(flag: wp.array) -> None:
    if flag.numpy()[0]:
        raise RuntimeError(
            f"pressure CG did not reach a relative residual of {_CHANNEL_CG_TOL} "
            f"in {_CHANNEL_CG_MAXITER} iterations"
        )


def _channel_masks(
    nx: int, ny: int, dx: float, obstacle: dict, device: str
) -> tuple[wp.array, wp.array, wp.array]:
    """``(fluid, open_u, open_v)`` int32 masks for the staircase cylinder."""
    fluid = ~solid_cells(nx, ny, dx, obstacle["center"], obstacle["radius"])
    open_u = np.zeros((nx + 1, ny), bool)
    open_u[0], open_u[nx] = fluid[0], fluid[nx - 1]
    open_u[1:nx] = fluid[:-1] & fluid[1:]
    open_v = np.zeros((nx, ny + 1), bool)
    open_v[:, 1:ny] = fluid[:, :-1] & fluid[:, 1:]
    return tuple(
        wp.array(m.astype(np.int32), dtype=wp.int32, device=device)
        for m in (fluid, open_u, open_v)
    )


def _channel_pressure_solver(
    fluid: wp.array, open_u: wp.array, open_v: wp.array, device: str
) -> tuple[Any, wp.array]:
    """``solve(rhs, x)`` for A x = rhs in place, and the flag it sets on failure.

    Conjugate gradients on :func:`channel_laplacian_kernel`, preconditioned by
    the exact inverse of the same operator without the cylinder, which
    separates into 1-D eigenbases along x (Neumann inlet, p = 0 outlet) and y
    (Neumann walls). The cylinder changes only a few hundred rows, so CG needs
    about 20 iterations instead of 150 at 10 cells per diameter. The dense x
    basis costs nx² ny per application, so the run time halves at 10 cells
    per diameter, breaks even at 20 and doubles at 40.

    On CUDA the iterations run in a conditional graph without a host sync
    per residual check, captured once per solution buffer ``x`` and replayed.
    Non-convergence is flagged on the device and raised by
    :func:`_check_cg_converged` once the run ends.
    """
    nx, ny = fluid.shape
    n = nx * ny

    def second_difference(m: int, last: float) -> np.ndarray:
        t = 2.0 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
        t[0, 0], t[-1, -1] = 1.0, last
        return t

    def to_wp(values: np.ndarray) -> wp.array:
        values = np.ascontiguousarray(values, dtype=np.float32)
        return wp.array(values, dtype=wp.float32, device=device)

    lam_x, vec_x = np.linalg.eigh(second_difference(nx, 3.0))
    lam_y, vec_y = np.linalg.eigh(second_difference(ny, 1.0))
    vx, vxt, vy, vyt = (to_wp(m) for m in (vec_x, vec_x.T, vec_y, vec_y.T))
    inv_lam, ones = to_wp(1.0 / np.add.outer(lam_x, lam_y)), to_wp(np.ones((nx, ny)))
    tmp_a, tmp_b = to_wp(np.zeros((nx, ny))), to_wp(np.zeros((nx, ny)))

    def matvec(x, y, z, alpha, beta):
        wp.launch(
            channel_laplacian_kernel,
            dim=(nx, ny),
            inputs=[x, y, z, float(alpha), float(beta), open_u, open_v, fluid],
            device=device,
        )

    def precondition(x, y, z, alpha, beta):
        def launch(kernel, inputs):
            wp.launch(kernel, dim=(nx, ny), inputs=inputs, device=device)

        x, y, z = (a.reshape((nx, ny)) for a in (x, y, z))
        launch(channel_basis_rows_kernel, [vxt, x, tmp_a])
        launch(channel_basis_cols_kernel, [tmp_a, vy, inv_lam, tmp_a, 1.0, 0.0, tmp_b])
        launch(channel_basis_rows_kernel, [vx, tmp_b, tmp_a])
        launch(
            channel_basis_cols_kernel,
            [tmp_a, vyt, ones, y, float(alpha), float(beta), z],
        )

    def operator(func):
        return warp.optim.linear.LinearOperator(
            (n, n), wp.float32, wp.get_device(device), func
        )

    use_graph = wp.get_device(device).is_cuda and wp.is_conditional_graph_supported()
    unconverged = wp.zeros(1, dtype=wp.int32, device=device)
    rhs_buffer = wp.zeros(n, dtype=wp.float32, device=device)
    cg = warp.optim.linear.cg(
        operator(matvec),
        rhs_buffer,
        wp.zeros(n, dtype=wp.float32, device=device),
        tol=_CHANNEL_CG_TOL,
        atol=0.0,
        maxiter=_CHANNEL_CG_MAXITER,
        M=operator(precondition),
        check_every=0 if use_graph else 10,
        run=False,
    )

    def run(x: wp.array) -> None:
        _, residual, atol = cg(b=rhs_buffer, x=x)
        if use_graph:
            wp.launch(
                _flag_unconverged_kernel,
                dim=1,
                inputs=[residual, atol, unconverged],
                device=device,
            )
        elif residual > atol:
            unconverged.fill_(1)

    graphs = {}

    def solve(rhs: wp.array, x: wp.array) -> None:
        wp.copy(rhs_buffer, rhs)
        if not use_graph:
            run(x)
            return
        if x.ptr not in graphs:
            with wp.ScopedCapture(device=device) as capture:
                run(x)
            graphs[x.ptr] = capture.graph
        wp.capture_launch(graphs[x.ptr])

    return solve, unconverged


def channel_solve(
    v0_np: np.ndarray,
    inflow_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    dx: float,
    obstacle: dict,
    device: str = "cpu",
    tape: wp.Tape | None = None,
    track_scalar_grads: bool = False,
) -> dict[str, wp.array]:
    """Run the 2-D DFG channel, recorded on ``tape`` when one is given.

    The pressure solve (:func:`_channel_pressure_solver`) is not a Warp
    kernel, so its backward is recorded with ``tape.record_func``. The
    Poisson operator is symmetric, so the adjoint of ``p = A⁻¹ b`` is another
    solve with A, ``b̄ = A⁻¹ p̄``. Each direction starts from its previous
    solution.

    :func:`channel_force_kernel` sums the force on the cylinder each step
    from the same u* and p the step uses, which makes it the scheme's
    discrete momentum exchange.

    Returns:
        Warp arrays: ``ux``, ``uy`` and ``pressure`` (nx, ny) at the cell
        centers, ``body_force`` (steps, 2), the leaves ``v0_x``, ``v0_y``,
        ``inflow``, ``nu`` and ``dt``, and ``cg_unconverged``, which the
        adjoint solves in ``tape.backward()`` set if they fail.
    """
    nx, ny = v0_np.shape[:2]
    inv_dx = 1.0 / dx
    grad = tape is not None
    recording = tape if grad else contextlib.nullcontext()

    def field(shape: tuple[int, int]) -> wp.array:
        return wp.zeros(shape, dtype=wp.float32, requires_grad=grad, device=device)

    def leaf(values: np.ndarray, requires_grad: bool = grad) -> wp.array:
        return wp.array(
            np.ascontiguousarray(values, dtype=np.float32),
            dtype=wp.float32,
            requires_grad=requires_grad,
            device=device,
        )

    fluid, open_u, open_v = _channel_masks(nx, ny, dx, obstacle, device)
    v0_x, v0_y = leaf(v0_np[:, :, 0, 0]), leaf(v0_np[:, :, 0, 1])
    inflow = leaf(inflow_np)
    nu_wp = leaf(np.array([viscosity]), grad and track_scalar_grads)
    dt_wp = leaf(np.array([dt]), grad and track_scalar_grads)
    force = field((steps, 2))

    p_guess = wp.zeros(nx * ny, dtype=wp.float32, device=device)
    adjoint_guess = wp.zeros(nx * ny, dtype=wp.float32, device=device)
    solve, unconverged = _channel_pressure_solver(fluid, open_u, open_v, device)

    with recording:
        u, v = field((nx + 1, ny)), field((nx, ny + 1))
        wp.launch(
            channel_init_u_kernel,
            dim=u.shape,
            inputs=[v0_x, open_u, inflow, u],
            device=device,
        )
        wp.launch(
            channel_init_v_kernel, dim=v.shape, inputs=[v0_y, open_v, v], device=device
        )

    for k in range(steps):
        u_star, v_star = field(u.shape), field(v.shape)
        rhs, p = field((nx, ny)), field((nx, ny))
        with recording:
            wp.launch(
                channel_tentative_u_kernel,
                dim=u.shape,
                inputs=[u, v, inflow, u_star, dt_wp, nu_wp, inv_dx],
                device=device,
            )
            wp.launch(
                channel_tentative_v_kernel,
                dim=v.shape,
                inputs=[u, v, v_star, dt_wp, nu_wp, inv_dx],
                device=device,
            )
            wp.launch(
                channel_divergence_kernel,
                dim=(nx, ny),
                inputs=[u_star, v_star, open_u, open_v, rhs, dt_wp, dx],
                device=device,
            )
        solve(rhs.flatten(), p_guess)
        wp.copy(p.flatten(), p_guess)
        if grad:

            def adjoint(rhs=rhs, p=p) -> None:
                solve(p.grad.flatten(), adjoint_guess)
                wp.copy(rhs.grad.flatten(), adjoint_guess)

            tape.record_func(backward=adjoint, arrays=[rhs, p])
        u_new, v_new = field(u.shape), field(v.shape)
        with recording:
            wp.launch(
                channel_correct_u_kernel,
                dim=u.shape,
                inputs=[u_star, p, open_u, inflow, u_new, dt_wp, inv_dx],
                device=device,
            )
            wp.launch(
                channel_correct_v_kernel,
                dim=v.shape,
                inputs=[v_star, p, open_v, v_new, dt_wp, inv_dx],
                device=device,
            )
            wp.launch(
                channel_force_kernel,
                dim=(nx + 1, ny + 1),
                inputs=[
                    u_star,
                    v_star,
                    p,
                    open_u,
                    open_v,
                    fluid,
                    force,
                    k,
                    dt_wp,
                    dx,
                ],
                device=device,
            )
        u, v = u_new, v_new

    _check_cg_converged(unconverged)
    ux, uy = field((nx, ny)), field((nx, ny))
    with recording:
        wp.launch(
            channel_centers_kernel,
            dim=(nx, ny),
            inputs=[u, v, fluid, ux, uy],
            device=device,
        )
    return {
        "ux": ux,
        "uy": uy,
        "pressure": p,
        "body_force": force,
        "v0_x": v0_x,
        "v0_y": v0_y,
        "inflow": inflow,
        "nu": nu_wp,
        "dt": dt_wp,
        "cg_unconverged": unconverged,
    }


# ============================================================
# Schema definitions
# ============================================================


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    """Warp NS solver input schema."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "warp-ns")
        return self


class OutputSchema(
    make_differentiable(_CanonicalOutputSchema, ["result", "body_force"])
):
    """Warp NS solver output schema."""


# ============================================================
# Utility
# ============================================================


def _warp_device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _is_3d(v0_np: np.ndarray) -> bool:
    """True if the velocity field is 3-D (shape N,N,N,3 with nz != 1)."""
    return v0_np.ndim == 4 and v0_np.shape[2] != 1 and v0_np.shape[3] == 3


# ============================================================
# Tesseract API endpoints
# ============================================================


def _channel_args(inputs: InputSchema) -> dict[str, Any]:
    return {
        "v0_np": np.asarray(inputs.v0, dtype=np.float32),
        "inflow_np": np.asarray(inputs.inflow_profile, dtype=np.float32),
        "viscosity": float(inputs.viscosity[0]),
        "dt": float(inputs.dt[0]),
        "steps": inputs.steps,
        "dx": inputs.dx,
        "obstacle": inputs.obstacle.model_dump(),
        "device": _warp_device(),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the Warp NS forward solver (2-D or 3-D) and return the result.

    Uses the forward-only solve path (no wp.Tape, no gradient-enabled
    arrays, ping-pong buffers), since apply() never needs gradients. The
    channel runs :func:`channel_solve` without a tape.
    """
    if inputs.obstacle is not None:
        out = channel_solve(**_channel_args(inputs))
        velocity = np.stack([out["ux"].numpy(), out["uy"].numpy()], axis=-1)
        return OutputSchema(
            result=velocity[:, :, None, :],
            pressure=out["pressure"].numpy()[:, :, None],
            body_force=out["body_force"].numpy(),
        )

    v0 = np.asarray(inputs.v0, dtype=np.float32)
    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    device = _warp_device()

    if _is_3d(v0):
        result = ns3d_solve_forward(
            v0, nu, dt, inputs.steps, inputs.domain_extent, device=device
        )
    else:
        result = ns2d_solve_forward(
            v0, nu, dt, inputs.steps, inputs.domain_extent, device=device
        )
    return OutputSchema(result=result)


def channel_vjp(
    inputs: InputSchema,
    vjp_inputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP of the channel's ``result`` and ``body_force`` via wp.Tape."""
    tape = wp.Tape()
    out = channel_solve(
        **_channel_args(inputs),
        tape=tape,
        track_scalar_grads=bool(vjp_inputs & {"viscosity", "dt"}),
    )
    device = out["ux"].device
    seeds = {}
    if cotangent_vector.get("result") is not None:
        cot = np.asarray(cotangent_vector["result"], dtype=np.float32)
        seeds[out["ux"]] = wp.array(cot[:, :, 0, 0], dtype=wp.float32, device=device)
        seeds[out["uy"]] = wp.array(cot[:, :, 0, 1], dtype=wp.float32, device=device)
    if cotangent_vector.get("body_force") is not None:
        cot = np.asarray(cotangent_vector["body_force"], dtype=np.float32)
        seeds[out["body_force"]] = wp.array(cot, dtype=wp.float32, device=device)
    tape.backward(grads=seeds)
    _check_cg_converged(out["cg_unconverged"])

    grad_v0 = np.stack([out["v0_x"].grad.numpy(), out["v0_y"].grad.numpy()], -1)
    grads = {
        "v0": grad_v0[:, :, None, :],
        "inflow_profile": out["inflow"].grad.numpy(),
    }
    for key, name in (("viscosity", "nu"), ("dt", "dt")):
        if key in vjp_inputs:
            grads[key] = out[name].grad.numpy()
    return {key: grads[key] for key in vjp_inputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute VJP via wp.Tape reverse-mode autodiff.

    Runs the forward pass under tape recording, then calls tape.backward()
    with the output cotangent to obtain gradients w.r.t. differentiable inputs.

    Differentiable inputs: v0, viscosity, dt (2D and 3D), and inflow_profile
    in the channel (see :func:`channel_vjp`).
    Periodic 2D and 3D use a spectral FFT Poisson solve for numerically exact VJPs.

    Gradient tracking for the ``viscosity``/``dt`` scalars is only turned on
    when the caller actually requests them via ``vjp_inputs``: each
    additional requires_grad=True array adds tape bookkeeping on every read
    across every timestep, and doing this unconditionally measured ~50%
    higher 3-D VJP wall-clock for the common v0-only case.
    """
    if inputs.obstacle is not None:
        return channel_vjp(inputs, vjp_inputs, cotangent_vector)

    v0 = np.asarray(inputs.v0, dtype=np.float32)
    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    device = _warp_device()
    track_scalar_grads = bool(vjp_inputs & {"viscosity", "dt"})

    if _is_3d(v0):
        tape, ux_f, uy_f, uz_f, ux_ic, uy_ic, uz_ic, nu_wp_3d, dt_wp_3d = (
            ns3d_solve_tape(
                v0,
                nu,
                dt,
                inputs.steps,
                inputs.domain_extent,
                device=device,
                track_scalar_grads=track_scalar_grads,
            )
        )
        cot_result = np.asarray(
            cotangent_vector.get("result", np.zeros_like(v0)), dtype=np.float32
        )
        grads = ns3d_vjp(
            tape,
            ux_f,
            uy_f,
            uz_f,
            ux_ic,
            uy_ic,
            uz_ic,
            cot_result,
            device,
            nu_wp=nu_wp_3d,
            dt_wp=dt_wp_3d,
        )
    else:
        (
            tape,
            ux_f,
            uy_f,
            ux_ic,
            uy_ic,
            nu_wp_2d,
            dt_wp_2d,
        ) = ns2d_solve_tape(
            v0,
            nu,
            dt,
            inputs.steps,
            inputs.domain_extent,
            device=device,
            track_scalar_grads=track_scalar_grads,
        )
        cot_result = np.asarray(
            cotangent_vector.get("result", np.zeros_like(v0)), dtype=np.float32
        )
        grads = ns2d_vjp(
            tape,
            ux_f,
            uy_f,
            ux_ic,
            uy_ic,
            cot_result,
            device,
            nu_wp=nu_wp_2d,
            dt_wp=dt_wp_2d,
        )

    result: dict[str, Any] = {}
    if "v0" in vjp_inputs:
        result["v0"] = grads["v0"]
    if "viscosity" in vjp_inputs:
        result["viscosity"] = grads["viscosity"]
    if "dt" in vjp_inputs:
        result["dt"] = grads["dt"]
    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Infer output shapes and dtypes without running the solver.

    Handles both concrete arrays and ShapeDtype dicts from the tesseract
    abstract evaluation protocol.
    """
    d = abstract_inputs.model_dump()
    v0 = d["v0"]

    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        shape = tuple(v0["shape"])
    else:
        shape = tuple(np.asarray(v0).shape)

    out = {"result": {"shape": shape, "dtype": "float32"}}
    if d["obstacle"] is not None:
        out["pressure"] = {"shape": (*shape[:2], 1), "dtype": "float32"}
        out["body_force"] = {"shape": (d["steps"], 2), "dtype": "float32"}
    return out
