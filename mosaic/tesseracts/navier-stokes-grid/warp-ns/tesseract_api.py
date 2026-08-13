# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001

"""GPU-accelerated differentiable 2-D/3-D Navier-Stokes via NVIDIA Warp.

Periodic-only IPCS (tentative velocity → pressure Poisson → velocity correction)
with exact spectral FFT Poisson and wp.Tape VJP.  Forward and reverse passes
work on triply-periodic boxes for both 2-D and 3-D.
"""

import functools
from typing import Any

import numpy as np
import warp as wp
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable

wp.init()

# ============================================================
# @wp.func helpers — inlined by the Warp JIT compiler into every
# kernel that calls them, so there is zero Python overhead.
# ============================================================


@wp.func
def sanitize_float(v: float, clip: float) -> float:
    """Replace NaN/Inf with 0 and clamp to [-clip, clip].

    Extracted as a @wp.func so it can be reused in multiple kernels without
    code duplication and benefits from Warp compiler inlining.
    """
    # NaN check: v != v is true only for NaN (wp.isnan unavailable in this context)
    if v != v:  # noqa: PLR0124
        v = wp.float32(0.0)
    if v > wp.float32(1.0e30):
        v = wp.float32(0.0)
    if v < wp.float32(-1.0e30):
        v = wp.float32(0.0)
    v = min(v, clip)
    v = max(v, -clip)
    return v


# ============================================================
# 2-D IPCS kernels (primitive-variable projection)
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
    otherwise precede the spectral Poisson FFT (discussion recommendation #2:
    fuse real-to-complex conversion with the first FFT stage).
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
    """u^(n+1) = u* - dt·∇p for 2-D IPCS, reading p directly from the complex IFFT output.

    Reads the real part (normalized) instead of a separate materialized
    pressure array. Fuses the "extract real + normalize" step with the
    pressure-correction kernel (discussion recommendation #2: fuse
    normalization with the IFFT-adjacent stage).
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
# 3-D NS kernels (IPCS / Chorin-Temam)
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
    otherwise precede the spectral Poisson FFT (discussion recommendation #2).
    ``dt`` is a 1-element array (like the 2-D kernel) so the tape tracks
    gradients w.r.t. it.
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
    pressure-correction kernel (discussion recommendation #2). ``dt`` is a
    1-element array (like the 2-D kernel) so the tape tracks gradients w.r.t. it.
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


@wp.kernel
def _clip_and_sanitize_3d_kernel(
    arr: wp.array3d(dtype=wp.float32),
    clip: float,
):
    """Element-wise clip of a 3-D float32 array into [-clip, clip].

    Also replaces NaN/Inf with 0.0 (hard safety).  Used inside the tape backward
    to bound velocity-adjoint magnitude per timestep and prevent float32
    overflow in the IPCS adjoint at turbulent high-Re regimes.
    Delegates to sanitize_float @wp.func which is inlined by the Warp compiler.
    """
    i, j, k = wp.tid()
    arr[i, j, k] = sanitize_float(arr[i, j, k], clip)


def _wlaunch(kernel, dim, inputs, block_dim=256, device="cpu"):
    """wp.launch wrapper. block_dim must be an int (Warp 1.12 dropped tuple support)."""
    wp.launch(kernel, dim=dim, inputs=inputs, block_dim=block_dim, device=device)


####################################################################
# GPU-resident spectral Poisson solve via wp.tile_fft
#
# Replaces the previous NumPy np.fft.fftn/ifftn implementation, which forced
# a CUDA sync plus a device->host->device round trip on every timestep (the
# dominant cost of the 3-D solver — see discussion #1636 recommendation #1).
# wp.tile_fft/wp.tile_ifft support native Warp reverse-mode AD (verified: a
# tile_fft->tile_ifft round trip differentiates correctly against a
# hand-derived expectation, cosine similarity 1.0 to float32 roundoff), so no
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
    to a row) and ``B = fft(b)`` (length ``m = next_pow2(2n-1)``), both device
    arrays. ``b`` is the zero-padded, wrapped conjugate-chirp convolution
    kernel; its FFT is precomputed on the host once per ``n`` since it is
    constant across timesteps and carries no gradient.
    """
    m = _next_pow2(2 * n - 1)
    j = np.arange(n)
    # chirp a[j] = exp(-i pi j^2 / n); w stored as the forward chirp
    w = np.exp(-1j * np.pi * (j * j % (2 * n)) / n)
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = np.conj(w)
    b[m - n + 1 :] = np.conj(w[1:])[::-1]
    B = np.fft.fft(b)

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

    ``direction`` is ``"fwd"`` or ``"bwd"`` (inverse). The inverse reuses the
    forward transform via ``ifft(x) = conj(fft(conj(x)))/n``; here that identity
    is folded into the chirp so a single code path serves both. Uses only
    power-of-two (length-``m``) tile FFTs, so it compiles on GPU for any ``n``.

    Every launch writes fresh ``requires_grad`` buffers so ``wp.Tape`` tracks
    the whole chain natively (no ``record_func`` needed) — the chirp/kernel
    multiplies and the pow2 FFTs are all differentiable Warp kernels.
    """
    batch = x.shape[0]
    setup = _bluestein_setup(n, device)
    m = setup["m"]
    bk = _bluestein_kernels(n, m)
    pow2 = _fft_kernels_2d(m)

    # For the inverse, conjugate on the way in and out and rescale by 1/n.
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
        y = _conj_scale_rows(y, 1.0 / float(n), device)
    return y


@functools.cache
def _conj_kernels(_tag: str = "bluestein_conj"):
    """Conjugate / conjugate-and-scale kernels shared across all Bluestein sizes."""

    @wp.kernel(module="bluestein_conj")
    def conj_k(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)) -> None:
        i, j = wp.tid()
        y[i, j] = wp.vec2f(x[i, j][0], -x[i, j][1])

    @wp.kernel(module="bluestein_conj")
    def conj_scale_k(
        x: wp.array2d(dtype=wp.vec2f), s: float, y: wp.array2d(dtype=wp.vec2f)
    ) -> None:
        i, j = wp.tid()
        y[i, j] = wp.vec2f(x[i, j][0] * s, -x[i, j][1] * s)

    return {"conj": conj_k, "conj_scale": conj_scale_k}


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


def _conj_scale_rows(x: wp.array, s: float, device: str) -> wp.array:
    y = wp.zeros(x.shape, dtype=wp.vec2f, requires_grad=True, device=device)
    wp.launch(
        _conj_kernels()["conj_scale"],
        dim=x.shape,
        inputs=[x, s],
        outputs=[y],
        device=device,
    )
    return y


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
        registers/shared memory across all three (discussion recommendation
        #2: fuse spectral multiplication with the middle FFT/IFFT stage).
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

    Matches the eigenvalues used by the previous NumPy implementation exactly
    (continuous -(2π/L)²k², not the discrete FD eigenvalues used in 3-D) —
    preserving the existing discretization per discussion recommendation #7.
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

    Returns the complex (unnormalized) pressure field. Split out from the
    original ``_spectral_poisson_2d_tape`` wrapper so call sites that
    produce/consume complex buffers directly (e.g. a divergence kernel fused
    with the real->complex pack, or a pressure-correction kernel fused with
    the extract+normalize step) can skip the redundant pack/unpack kernels —
    discussion recommendation #2.

    The middle stage (last forward FFT pass, spectral multiply, first inverse
    FFT pass) runs as a single fused tiled kernel instead of three separate
    launches with two intermediate global-memory round trips (recommendation
    #2, follow-up). Pass ``workspace`` (from :func:`_poisson_workspace_2d`) to
    reuse pre-allocated scratch buffers across steps instead of allocating
    fresh ones on every call — required for the forward-only (non-taped) path
    and a minor win for the taped path too.
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
    dependencies for (discussion recommendation: separate buffer policies for
    forward vs. taped execution).
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
    """Precompute 1/λ for the 3-D discrete-FD-eigenvalue spectral Poisson solve.

    Uses the discrete finite-difference Laplacian eigenvalues:
        λ_disc(kx, ky, kz) = -(4/h²)(sin²(πkx/N) + sin²(πky/N) + sin²(πkz/N))
    where h = L/N and kx,ky,kz are integer wavenumbers. This matches the
    stencil used by tentative_vel_3d_kernel and pressure_correct_3d_kernel
    (both use inv_h2 = 1/h² and inv_2h = 1/(2h) central differences), making
    the spectral solve the exact inverse of the discrete FD Laplacian.

    Using continuous eigenvalues -(2π/L)²k² instead introduces a ~(πk/N)²/3
    relative error per wavenumber (≈1.3% at k=1, N=16), which compounds across
    the VJP chain and causes a flat-plateau gradient magnitude bias in the
    fd_check — see discussion recommendation #7 (preserve discretization).

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
        multiply kernel reading/writing it back to global memory (discussion
        recommendation #2). The swap into the next pass's layout still
        happens as a separate kernel — see :func:`_fft3_pass`.
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
    spectral multiply is fused into this pass's row-transform kernel
    (discussion recommendation #2) instead of a separate elementwise launch.

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

    Returns the complex (unnormalized) pressure field. Split out from the
    original ``_spectral_poisson_3d_tape`` wrapper so call sites that
    produce/consume complex buffers directly can skip the redundant
    pack/unpack kernels — discussion recommendation #2.

    The spectral multiply is fused into the last forward-FFT pass instead of
    running as a separate elementwise kernel (recommendation #2, follow-up):
    right before that pass's axis-swap, the array is permuted (1,2,0) relative
    to the original (kx,ky,kz) axis order, so we index a matching
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
    """Run periodic 2-D incompressible NS via IPCS (Chorin-Temam) under a tape.

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
            # first FFT stage (discussion recommendation #2).
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
            # directly out of the complex IFFT output (recommendation #2).
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
    """Forward-only periodic 2-D incompressible NS via IPCS — no gradients.

    [2D-only function]

    Same physics as :func:`ns2d_solve_tape`, but structurally distinct (discussion
    recommendation: separate buffer policies for forward vs. taped
    execution): no ``wp.Tape``, no ``requires_grad`` arrays, two ping-pong
    velocity buffers reused every step instead of fresh per-step allocations
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
    """Propagate cotangents through the 2-D IPCS tape.

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
# 3-D NS forward solve (IPCS)
# ============================================================


def ns3d_solve_tape(
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    device: str = "cpu",
    adjoint_grad_clip: float | None = None,
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
    """Run 3-D incompressible NS via IPCS (Chorin-Temam) under a tape.

    Used only by :func:`vector_jacobian_product` — see :func:`ns3d_solve_forward`
    for the plain-forward path used by :func:`apply`. Uses Warp's native
    reverse-mode AD for every kernel (verified to agree with a from-scratch
    analytical adjoint and a central-FD ground truth to float32 roundoff).
    Each step allocates fresh output buffers instead of ping-ponging between
    two reused buffers, so the tape's per-step data dependencies are
    unambiguous and no manual adjoint-zeroing is needed.

    ``nu``/``dt`` are passed to kernels as 1-element arrays (like the 2-D
    solver) so the tape can track gradients w.r.t. them — previously this
    path always returned zero for viscosity/dt (schema advertised them as
    differentiable but the 3-D kernels only accepted plain floats).

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
            # ── Per-step adjoint gradient clipping (stability guard) ───────────
            # Registered before this step's kernels run; because record_func is
            # LIFO, this fires LAST in the step's backward — after
            # tentative_vel's auto-adjoint has written into cur_u{x,y,z}.grad.
            # Clipping the per-timestep adjoint prevents float32 overflow in the
            # IPCS adjoint at turbulent high-Re regimes without altering the
            # gradient direction when ||adj||_inf is already within bounds.
            # Only active when adjoint_grad_clip is set (>0).
            if adjoint_grad_clip is not None and adjoint_grad_clip > 0:
                _vx_cur, _vy_cur, _vz_cur = cur_ux, cur_uy, cur_uz
                _clip = float(adjoint_grad_clip)
                _d_clip = device
                _n_clip = n

                def _clip_cur_adj(
                    _vx=_vx_cur,
                    _vy=_vy_cur,
                    _vz=_vz_cur,
                    _c=_clip,
                    _d=_d_clip,
                    _n=_n_clip,
                ):
                    """Clip cur_u{x,y,z}.grad element-wise into [-c, c] via a Warp kernel.

                    Prevents float32 overflow in the IPCS adjoint at high-Re and
                    replaces NaN/Inf with 0 as a hard safety fallback.  GPU-native
                    (no numpy round-trip) so per-step cost is negligible.
                    """
                    for parent in (_vx, _vy, _vz):
                        if parent.grad is None:
                            continue
                        _wlaunch(
                            _clip_and_sanitize_3d_kernel,
                            dim=(_n, _n, _n),
                            inputs=[parent.grad, _c],
                            block_dim=_bd_3d,
                            device=_d,
                        )

                tape.record_func(
                    backward=_clip_cur_adj,
                    arrays=[cur_ux, cur_uy, cur_uz],
                )

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
            # first FFT stage (discussion recommendation #2).
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
            # directly out of the complex IFFT output (recommendation #2).
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
    """Forward-only 3-D incompressible NS via IPCS — no gradients.

    Same physics as :func:`ns3d_solve_tape`, but structurally distinct (discussion
    recommendation: separate buffer policies for forward vs. taped
    execution): no ``wp.Tape``, no ``requires_grad`` arrays, two ping-pong
    velocity buffers per component reused every step instead of fresh
    per-step allocations, and one pre-allocated FFT workspace reused across
    all steps.
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
    """Propagate cotangents through the 3-D IPCS tape.

    v0, viscosity, and dt are all ordinary tape leaves (nu_wp/dt_wp are the
    1-element arrays passed into tentative_vel_3d_kernel,
    divergence_3d_to_complex_kernel, and pressure_correct_3d_from_complex_kernel
    in ns3d_solve_tape), so tape.backward() fills their .grad directly —
    previously this function always returned zero for viscosity/dt.
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
# Schema definitions
# ============================================================


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["v0", "viscosity", "dt"])
):
    """Warp NS solver input schema."""


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
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


def _check_periodic(inputs: InputSchema) -> None:
    """warp-ns supports only fully-periodic flows."""
    if inputs.obstacle is not None:
        raise NotImplementedError(
            "warp-ns is periodic-only. Use phiflow / xlb / pict for "
            "cylinder/drag experiments."
        )
    if inputs.inflow_profile is not None:
        raise NotImplementedError(
            "warp-ns is periodic-only. Use phiflow / xlb / pict for "
            "inflow/channel experiments."
        )


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the Warp NS forward solver (2-D or 3-D) and return the result.

    Uses the forward-only solve path (no wp.Tape, no gradient-enabled
    arrays, ping-pong buffers) since apply() never needs gradients —
    discussion recommendation: give apply() a separate forward-only path.
    """
    _check_periodic(inputs)
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
    return OutputSchema(result=result, drag=None)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute VJP via wp.Tape reverse-mode autodiff.

    Runs the forward pass under tape recording, then calls tape.backward()
    with the output cotangent to obtain gradients w.r.t. differentiable inputs.

    Differentiable inputs: v0, viscosity, dt (2D and 3D).
    Both 2D and 3D use IPCS with spectral FFT Poisson for numerically exact VJPs.

    Gradient tracking for the ``viscosity``/``dt`` scalars is only turned on
    when the caller actually requests them via ``vjp_inputs``: each
    additional requires_grad=True array adds tape bookkeeping on every read
    across every timestep, and doing this unconditionally measured ~50%
    higher 3-D VJP wall-clock for the common v0-only case.
    """
    _check_periodic(inputs)
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

    return {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": None,
    }
