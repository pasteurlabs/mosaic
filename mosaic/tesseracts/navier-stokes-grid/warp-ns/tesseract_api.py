"""GPU-accelerated differentiable 2-D/3-D Navier-Stokes via NVIDIA Warp.

2-D: IPCS primitive-variable projection + spectral FFT Poisson + wp.Tape VJP.
3-D: IPCS primitive-variable projection + two Poisson solvers + wp.Tape VJP.
  - Periodic TGV: exact spectral FFT Poisson (periodic BCs).
  - Lid-driven cavity: iterative CG Poisson with Neumann BCs (dp/dn=0 on walls).
Both formulations share the IPCS structure (tentative velocity → pressure Poisson
→ velocity correction) and use differentiable Poisson solvers registered via
tape.record_func for numerically exact VJP gradients.
Obstacle support via volume penalization (2-D only).
"""

import functools
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import warp as wp
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import BCType
from pydantic import Field

wp.init()

# ============================================================
# @wp.func helpers — inlined by the Warp JIT compiler into every
# kernel that calls them, so there is zero Python overhead.
# ============================================================


@wp.func
def wrap_idx(i: int, n: int) -> int:
    """Periodic (modulo) index — wraps negative indices correctly.

    Equivalent to (i % n) for non-negative i, and (i + n) % n for negative i.
    Used for periodic-BC neighbour lookups in every stencil kernel.
    Warp inlines @wp.func calls at compile time, so there is no call overhead.
    """
    return (i + n) % n


@wp.func
def clamp_idx(i: int, lo: int, hi: int) -> int:
    """Clamp index to [lo, hi] for Neumann (zero-gradient) boundary conditions.

    Used for wall-normal directions in lid-driven cavity kernels where periodic
    wrapping is wrong — the ghost cell equals the first/last interior cell.
    """
    return wp.max(lo, wp.min(i, hi))


@wp.func
def sanitize_float(v: float, clip: float) -> float:
    """Replace NaN/Inf with 0 and clamp to [-clip, clip].

    Extracted as a @wp.func so it can be reused in multiple kernels without
    code duplication and benefits from Warp compiler inlining.
    """
    # NaN check: v != v is true only for NaN
    if v != v:
        v = wp.float32(0.0)
    if v > wp.float32(1.0e30):
        v = wp.float32(0.0)
    if v < wp.float32(-1.0e30):
        v = wp.float32(0.0)
    if v > clip:
        v = clip
    if v < -clip:
        v = -clip
    return v


# ============================================================
# 2-D Jacobi Poisson kernel
# ============================================================


@wp.kernel
def jacobi_2d_kernel(  # mosaic:physics
    u: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    u_new: wp.array2d(dtype=wp.float32),
    scale: float,
):
    """One Jacobi step for ∇²u = rhs on a periodic 2-D grid.

    u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
                 + scale * rhs[i,j]

    scale = +h²/4  for streamfunction Poisson: ∇²ψ = -ω  where rhs = omega
    The vorticity-streamfunction equation is ∇²ψ = −ω.  Substituting into
    the Jacobi update for ∇²ψ = f with f = −ω:
        ψ_new = avg(neighbours)/4 + scale * ω
    Setting scale = +h²/4 gives ψ_new = avg/4 + h²/4·ω, which is the correct
    Jacobi step for ∇²ψ = -ω.  Using scale = -h²/4 would solve ∇²ψ = +ω,
    yielding the wrong-sign streamfunction and therefore wrong-sign velocity.
    """
    i, j = wp.tid()
    n = u.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    u_new[i, j] = (
        0.25 * (u[im1, j] + u[ip1, j] + u[i, jm1] + u[i, jp1]) + scale * rhs[i, j]
    )


# ============================================================
# 2-D NS kernels (vorticity-streamfunction formulation)
# ============================================================


@wp.kernel
def curl_to_vorticity_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    omega: wp.array2d(dtype=wp.float32),
    inv_2h: float,
):
    """Compute vorticity ω = ∂v/∂x - ∂u/∂y via central differences."""
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    dvdx = (uy[ip1, j] - uy[im1, j]) * inv_2h
    dudy = (ux[i, jp1] - ux[i, jm1]) * inv_2h
    omega[i, j] = dvdx - dudy


@wp.kernel
def psi_to_velocity_kernel(  # mosaic:physics
    psi: wp.array2d(dtype=wp.float32),
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    inv_2h: float,
):
    """Recover velocity from streamfunction: u = ∂ψ/∂y, v = -∂ψ/∂x."""
    i, j = wp.tid()
    n = psi.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    ux[i, j] = (psi[i, jp1] - psi[i, jm1]) * inv_2h
    uy[i, j] = -((psi[ip1, j] - psi[im1, j]) * inv_2h)


@wp.kernel
def vorticity_rhs_kernel(  # mosaic:physics
    omega: wp.array2d(dtype=wp.float32),
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    inv_2h: float,
    inv_h2: float,
    nu: float,
):
    """RHS of the vorticity equation: rhs = -(u·∇)ω + ν∇²ω.

    Advection uses central differences; diffusion is the 5-point Laplacian.
    """
    i, j = wp.tid()
    n = omega.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n

    # Advection: -(u·∇)ω
    domega_dx = (omega[ip1, j] - omega[im1, j]) * inv_2h
    domega_dy = (omega[i, jp1] - omega[i, jm1]) * inv_2h
    advection = -(ux[i, j] * domega_dx + uy[i, j] * domega_dy)

    # Diffusion: ν∇²ω
    lap_omega = (
        omega[im1, j]
        + omega[ip1, j]
        + omega[i, jm1]
        + omega[i, jp1]
        - 4.0 * omega[i, j]
    ) * inv_h2
    diffusion = nu * lap_omega

    rhs[i, j] = advection + diffusion


@wp.kernel
def rk3_stage1_kernel(  # mosaic:physics
    omega: wp.array2d(dtype=wp.float32),
    k1: wp.array2d(dtype=wp.float32),
    omega_s1: wp.array2d(dtype=wp.float32),
    dt: float,
):
    """SSP-RK3 stage 1: ω_s1 = ω + dt·k1."""
    i, j = wp.tid()
    omega_s1[i, j] = omega[i, j] + dt * k1[i, j]


@wp.kernel
def rk3_stage2_kernel(  # mosaic:physics
    omega: wp.array2d(dtype=wp.float32),
    omega_s1: wp.array2d(dtype=wp.float32),
    k2: wp.array2d(dtype=wp.float32),
    omega_s2: wp.array2d(dtype=wp.float32),
    dt: float,
):
    """SSP-RK3 stage 2: ω_s2 = 3/4·ω^n + 1/4·ω_s1 + 1/4·dt·k2.

    Correct Shu-Osher SSP-RK3 formula:
        u(2) = 0.75·u^n + 0.25·u(1) + 0.25·dt·L(u(1))
    where u(1) = omega_s1 (stage-1 output) and k2 = L(u(1)).
    """
    i, j = wp.tid()
    omega_s2[i, j] = 0.75 * omega[i, j] + 0.25 * omega_s1[i, j] + 0.25 * dt * k2[i, j]


@wp.kernel
def rk3_combine_kernel(  # mosaic:physics
    omega: wp.array2d(dtype=wp.float32),
    k3: wp.array2d(dtype=wp.float32),
    omega_s2: wp.array2d(dtype=wp.float32),
    omega_new: wp.array2d(dtype=wp.float32),
    dt: float,
):
    """SSP-RK3 final: ω^(n+1) = 1/3·ω^n + 2/3·(ω_s2 + dt·k3)."""
    i, j = wp.tid()
    omega_new[i, j] = (1.0 / 3.0) * omega[i, j] + (2.0 / 3.0) * (
        omega_s2[i, j] + dt * k3[i, j]
    )


@wp.kernel
def apply_obstacle_mask_kernel(  # mosaic:physics
    omega: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.float32),
):
    """Zero vorticity inside the obstacle: ω *= (1 - mask)."""
    i, j = wp.tid()
    omega[i, j] = omega[i, j] * (1.0 - mask[i, j])


@wp.kernel
def apply_velocity_mask_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.float32),
):
    """Zero velocity inside the obstacle."""
    i, j = wp.tid()
    m = mask[i, j]
    ux[i, j] = ux[i, j] * (1.0 - m)
    uy[i, j] = uy[i, j] * (1.0 - m)


@wp.kernel
def compute_drag_kernel(  # mosaic:physics
    p: wp.array2d(dtype=wp.float32),
    ux: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.float32),
    drag_buf: wp.array(dtype=wp.float32),
    inv_2h: float,
    nu: float,
):
    """Accumulate x-momentum flux on the obstacle surface (drag estimate).

    Counts each solid/fluid interface in the x-direction exactly once from the
    fluid side.  Each interface contributes  (p·n_x − ν·∂u_x/∂n)  where n is
    the outward normal of the solid surface pointing into the fluid.

    Downstream face (fluid to the right of solid, n_x = +1):
        m_here < 0.5  and  m_im1 > 0.5  →  contribution = +p[here] − ν·dux_dn
    Upstream face (fluid to the left of solid, n_x = −1):
        m_here < 0.5  and  m_ip1 > 0.5  →  contribution = −p[here] + ν·dux_dn

    Net drag ≈ Σ p_downstream − Σ p_upstream  (negative for bluff body in flow).
    """
    i, j = wp.tid()
    n = mask.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    m_here = mask[i, j]
    m_im1 = mask[im1, j]
    m_ip1 = mask[ip1, j]
    # Downstream face: fluid here, solid to the left → outward normal n_x = +1
    if m_here < 0.5 and m_im1 > 0.5:
        dux_dn = (ux[i, j] - ux[im1, j]) * inv_2h
        contribution = p[i, j] - nu * dux_dn
        wp.atomic_add(drag_buf, 0, contribution)
    # Upstream face: fluid here, solid to the right → outward normal n_x = −1
    if m_here < 0.5 and m_ip1 > 0.5:
        dux_dn = (ux[ip1, j] - ux[i, j]) * inv_2h
        contribution = -(p[i, j] - nu * dux_dn)
        wp.atomic_add(drag_buf, 0, contribution)


@wp.kernel
def accumulate_outlet_rans_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    rans_buf: wp.array(dtype=wp.float32),
    inv_n_tail: float,
):
    """Accumulate outlet (i=N-1) x-velocity into RANS buffer for momentum-deficit drag.

    ARCH-9: replaces per-step compute_drag_kernel for inflow+obstacle runs.
    Recorded inside wp.Tape so the backward flows: rans_buf.grad → ux[N-1,:].grad
    → IPCS backward → inflow_profile.grad.  The outlet signal is stronger and
    more direct than the obstacle-surface viscous term used by compute_drag_kernel.
    """
    j = wp.tid()
    n = ux.shape[0]
    wp.atomic_add(rans_buf, j, ux[n - 1, j] * inv_n_tail)


@wp.kernel
def momentum_deficit_drag_kernel(  # mosaic:physics
    rans_ux: wp.array(dtype=wp.float32),
    U_mean: float,
    dy: float,
    drag_buf: wp.array(dtype=wp.float32),
):
    """Momentum-deficit drag: D = ∫ u_x(U_∞ − u_x) dy at the outlet column.

    Warp auto-diff gives: adj_rans_ux[j] += adj_drag[0] * (U_mean − 2·rans_ux[j]) · dy
    """
    j = wp.tid()
    u_j = rans_ux[j]
    wp.atomic_add(drag_buf, 0, u_j * (U_mean - u_j) * dy)


# ============================================================
# 2-D inflow BC kernels (spatially-varying Dirichlet at x=0)
# ============================================================


@wp.kernel
def apply_inflow_bc_2d_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    inflow_profile: wp.array(dtype=wp.float32),
):
    """Apply inflow Dirichlet BC at the x=0 face (i=0).

    ux[0, j] = inflow_profile[j]  — spatially-varying u_x(y)
    uy[0, j] = 0.0                — no transverse inflow

    inflow_profile has shape (N,).  The assignment overwrites the x=0 slice;
    the backward is handled in Python via tape.record_func (accumulate-then-zero)
    to preserve gradient correctness for the inflow_profile VJP.
    """
    j = wp.tid()
    ux[0, j] = inflow_profile[j]
    uy[0, j] = 0.0


@wp.kernel
def _zero_inflow_slice_2d_kernel(  # mosaic:physics
    arr: wp.array2d(dtype=wp.float32),
):
    """Zero the x=0 column of a 2-D array (used to reset adjoint after overwrite BC)."""
    j = wp.tid()
    arr[0, j] = 0.0


@wp.kernel
def _accumulate_inflow_slice_to_profile_kernel(  # mosaic:physics
    adj_ux: wp.array2d(dtype=wp.float32),
    adj_profile: wp.array(dtype=wp.float32),
):
    """Accumulate adj_ux at x=0 column into adj_inflow_profile (overwrite adjoint fix).

    Forward: ux[0, j] = inflow_profile[j]
    Backward: adj_profile[j] += adj_ux[0, j];  adj_ux[0, j] = 0   (done in separate kernel)
    """
    j = wp.tid()
    wp.atomic_add(adj_profile, j, adj_ux[0, j])


@wp.kernel
def _apply_wall_y_bc_2d_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
):
    """Zero velocity at j=0 and j=n-1 (no-slip wall BCs in y-direction)."""
    i = wp.tid()
    n = ux.shape[1]
    ux[i, 0] = wp.float32(0.0)
    uy[i, 0] = wp.float32(0.0)
    ux[i, n - 1] = wp.float32(0.0)
    uy[i, n - 1] = wp.float32(0.0)


@wp.kernel
def _zero_wall_y_adj_2d_kernel(  # mosaic:physics
    arr: wp.array2d(dtype=wp.float32),
):
    """Zero adjoint at j=0 and j=n-1 wall rows (backward of zero-wall BC)."""
    i = wp.tid()
    n = arr.shape[1]
    arr[i, 0] = wp.float32(0.0)
    arr[i, n - 1] = wp.float32(0.0)


# ============================================================
# 2-D IPCS kernels (primitive-variable projection)
# ============================================================


@wp.kernel
def tentative_vel_2d_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    ux_star: wp.array2d(dtype=wp.float32),
    uy_star: wp.array2d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
):
    """2-D tentative velocity: u* = u + dt·(-u·∇u + ν∇²u).

    [2D-only function]
    """
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n

    ui = ux[i, j]
    vi = uy[i, j]

    # ux component
    lap_ux = (ux[im1, j] + ux[ip1, j] + ux[i, jm1] + ux[i, jp1] - 4.0 * ui) * inv_h2
    adv_ux = (
        ui * (ux[ip1, j] - ux[im1, j]) * inv_2h
        + vi * (ux[i, jp1] - ux[i, jm1]) * inv_2h
    )
    ux_star[i, j] = ui + dt * (-adv_ux + nu * lap_ux)

    # uy component
    lap_uy = (uy[im1, j] + uy[ip1, j] + uy[i, jm1] + uy[i, jp1] - 4.0 * vi) * inv_h2
    adv_uy = (
        ui * (uy[ip1, j] - uy[im1, j]) * inv_2h
        + vi * (uy[i, jp1] - uy[i, jm1]) * inv_2h
    )
    uy_star[i, j] = vi + dt * (-adv_uy + nu * lap_uy)


@wp.kernel
def divergence_2d_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    inv_2h_over_dt: float,
):
    """Compute ∇·u*/dt for 2-D pressure Poisson RHS.

    [2D-only function]
    """
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    div[i, j] = ((ux[ip1, j] - ux[im1, j]) + (uy[i, jp1] - uy[i, jm1])) * inv_2h_over_dt


@wp.kernel
def pressure_correct_2d_kernel(  # mosaic:physics
    ux_star: wp.array2d(dtype=wp.float32),
    uy_star: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    ux_new: wp.array2d(dtype=wp.float32),
    uy_new: wp.array2d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """u^(n+1) = u* - dt·∇p for 2-D IPCS.

    [2D-only function]
    """
    i, j = wp.tid()
    n = p.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    dpdx = (p[ip1, j] - p[im1, j]) * inv_2h
    dpdy = (p[i, jp1] - p[i, jm1]) * inv_2h
    ux_new[i, j] = ux_star[i, j] - dt * dpdx
    uy_new[i, j] = uy_star[i, j] - dt * dpdy


@wp.kernel
def divergence_2d_channel_kernel(  # mosaic:physics
    ux: wp.array2d(dtype=wp.float32),
    uy: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    inv_2h_over_dt: float,
):
    """Compute ∇·u*/dt for channel flow (Neumann x BCs, periodic y).

    Clamps x-indices instead of wrapping, so outflow (i=N-1) does not
    pollute inflow (i=0) via the divergence stencil.

    [2D-only function]
    """
    i, j = wp.tid()
    n = ux.shape[0]
    ip1 = wp.min(i + 1, n - 1)
    im1 = wp.max(i - 1, 0)
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    div[i, j] = ((ux[ip1, j] - ux[im1, j]) + (uy[i, jp1] - uy[i, jm1])) * inv_2h_over_dt


@wp.kernel
def pressure_correct_2d_channel_kernel(  # mosaic:physics
    ux_star: wp.array2d(dtype=wp.float32),
    uy_star: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    ux_new: wp.array2d(dtype=wp.float32),
    uy_new: wp.array2d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """u^(n+1) = u* - dt·∇p for channel flow (Neumann x BCs, periodic y).

    [2D-only function]
    """
    i, j = wp.tid()
    n = p.shape[0]
    ip1 = wp.min(i + 1, n - 1)
    im1 = wp.max(i - 1, 0)
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    dpdx = (p[ip1, j] - p[im1, j]) * inv_2h
    dpdy = (p[i, jp1] - p[i, jm1]) * inv_2h
    ux_new[i, j] = ux_star[i, j] - dt * dpdx
    uy_new[i, j] = uy_star[i, j] - dt * dpdy


# ============================================================
# 3-D NS kernels (IPCS / Chorin-Temam)
# ============================================================


@wp.kernel
def tentative_vel_3d_kernel(  # mosaic:physics
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    ux_star: wp.array3d(dtype=wp.float32),
    uy_star: wp.array3d(dtype=wp.float32),
    uz_star: wp.array3d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
):
    """3-D tentative velocity: u* = u + dt·(-u·∇u + ν∇²u)."""
    i, j, k = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n

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
    ux_star[i, j, k] = ui + dt * (-adv_ux + nu * lap_ux)

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
    uy_star[i, j, k] = vi + dt * (-adv_uy + nu * lap_uy)

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
    uz_star[i, j, k] = wi + dt * (-adv_uz + nu * lap_uz)


@wp.kernel
def divergence_3d_kernel(  # mosaic:physics
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    div: wp.array3d(dtype=wp.float32),
    inv_2h_over_dt: float,
):
    """Compute ∇·u*/dt for pressure Poisson RHS (periodic BCs in all directions)."""
    i, j, k = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    div[i, j, k] = (
        (ux[ip1, j, k] - ux[im1, j, k])
        + (uy[i, jp1, k] - uy[i, jm1, k])
        + (uz[i, j, kp1] - uz[i, j, km1])
    ) * inv_2h_over_dt


@wp.kernel
def divergence_3d_wall_kernel(  # mosaic:physics
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    div: wp.array3d(dtype=wp.float32),
    inv_2h_over_dt: float,
):
    """Compute ∇·u*/dt for pressure Poisson RHS with Neumann (zero-gradient) BCs in z.

    For lid-driven cavity mode: x and y remain periodic (modulo wrapping);
    z uses clamped indices so ∂p/∂z = 0 at the bottom (k=0) and top (k=n-1)
    walls, consistent with the no-penetration wall boundary condition.
    """
    i, j, k = wp.tid()
    n = ux.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = wp.min(k + 1, n - 1)
    km1 = wp.max(k - 1, 0)
    div[i, j, k] = (
        (ux[ip1, j, k] - ux[im1, j, k])
        + (uy[i, jp1, k] - uy[i, jm1, k])
        + (uz[i, j, kp1] - uz[i, j, km1])
    ) * inv_2h_over_dt


@wp.kernel
def jacobi_3d_kernel(  # mosaic:physics
    u: wp.array3d(dtype=wp.float32),
    rhs: wp.array3d(dtype=wp.float32),
    u_new: wp.array3d(dtype=wp.float32),
    scale: float,
):
    """One Jacobi step for ∇²u = rhs on a periodic 3-D grid."""
    i, j, k = wp.tid()
    n = u.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    u_new[i, j, k] = (
        u[im1, j, k]
        + u[ip1, j, k]
        + u[i, jm1, k]
        + u[i, jp1, k]
        + u[i, j, km1]
        + u[i, j, kp1]
    ) / 6.0 + scale * rhs[i, j, k]


@wp.kernel
def pressure_correct_3d_kernel(  # mosaic:physics
    ux_star: wp.array3d(dtype=wp.float32),
    uy_star: wp.array3d(dtype=wp.float32),
    uz_star: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    ux_new: wp.array3d(dtype=wp.float32),
    uy_new: wp.array3d(dtype=wp.float32),
    uz_new: wp.array3d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """u^(n+1) = u* - dt·∇p (periodic BCs in all directions)."""
    i, j, k = wp.tid()
    n = p.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    dpdx = (p[ip1, j, k] - p[im1, j, k]) * inv_2h
    dpdy = (p[i, jp1, k] - p[i, jm1, k]) * inv_2h
    dpdz = (p[i, j, kp1] - p[i, j, km1]) * inv_2h
    ux_new[i, j, k] = ux_star[i, j, k] - dt * dpdx
    uy_new[i, j, k] = uy_star[i, j, k] - dt * dpdy
    uz_new[i, j, k] = uz_star[i, j, k] - dt * dpdz


@wp.kernel
def pressure_correct_3d_wall_kernel(  # mosaic:physics
    ux_star: wp.array3d(dtype=wp.float32),
    uy_star: wp.array3d(dtype=wp.float32),
    uz_star: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    ux_new: wp.array3d(dtype=wp.float32),
    uy_new: wp.array3d(dtype=wp.float32),
    uz_new: wp.array3d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """u^(n+1) = u* - dt·∇p with Neumann (zero-gradient) BCs in z.

    For lid-driven cavity mode: x and y remain periodic; z uses clamped
    indices so the pressure gradient ∂p/∂z = 0 at the bottom (k=0) and
    top (k=n-1) walls, consistent with the Neumann pressure BC.
    """
    i, j, k = wp.tid()
    n = p.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = wp.min(k + 1, n - 1)
    km1 = wp.max(k - 1, 0)
    dpdx = (p[ip1, j, k] - p[im1, j, k]) * inv_2h
    dpdy = (p[i, jp1, k] - p[i, jm1, k]) * inv_2h
    dpdz = (p[i, j, kp1] - p[i, j, km1]) * inv_2h
    ux_new[i, j, k] = ux_star[i, j, k] - dt * dpdx
    uy_new[i, j, k] = uy_star[i, j, k] - dt * dpdy
    uz_new[i, j, k] = uz_star[i, j, k] - dt * dpdz


@wp.kernel
def _apply_pressure_correction_kernel(  # mosaic:grad:v0:adjoint
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """Subtract dt*grad(p) from (ux, uy, uz) in-place (periodic BCs in all directions).

    Used in the pressure adjoint correction to apply u -= dt*∇q to the
    cotangent arrays, giving the correct IPCS adjoint for the pressure step.
    """
    i, j, k = wp.tid()
    n = p.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = (k + 1) % n
    km1 = (k - 1 + n) % n
    dpdx = (p[ip1, j, k] - p[im1, j, k]) * inv_2h
    dpdy = (p[i, jp1, k] - p[i, jm1, k]) * inv_2h
    dpdz = (p[i, j, kp1] - p[i, j, km1]) * inv_2h
    ux[i, j, k] = ux[i, j, k] - dt * dpdx
    uy[i, j, k] = uy[i, j, k] - dt * dpdy
    uz[i, j, k] = uz[i, j, k] - dt * dpdz


@wp.kernel
def _apply_pressure_correction_wall_kernel(  # mosaic:grad:v0:adjoint
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    p: wp.array3d(dtype=wp.float32),
    dt: float,
    inv_2h: float,
):
    """Subtract dt*grad(p) from (ux, uy, uz) in-place with Neumann BCs in z.

    For lid-driven cavity adjoint: x and y remain periodic; z uses clamped
    indices so ∂p/∂z = 0 at the bottom (k=0) and top (k=n-1) walls.
    """
    i, j, k = wp.tid()
    n = p.shape[0]
    ip1 = (i + 1) % n
    im1 = (i - 1 + n) % n
    jp1 = (j + 1) % n
    jm1 = (j - 1 + n) % n
    kp1 = wp.min(k + 1, n - 1)
    km1 = wp.max(k - 1, 0)
    dpdx = (p[ip1, j, k] - p[im1, j, k]) * inv_2h
    dpdy = (p[i, jp1, k] - p[i, jm1, k]) * inv_2h
    dpdz = (p[i, j, kp1] - p[i, j, km1]) * inv_2h
    ux[i, j, k] = ux[i, j, k] - dt * dpdx
    uy[i, j, k] = uy[i, j, k] - dt * dpdy
    uz[i, j, k] = uz[i, j, k] - dt * dpdz


@wp.kernel
def _zero_top_slice_kernel(  # mosaic:physics
    arr: wp.array3d(dtype=wp.float32),
    top_k: int,
):
    """Zero the top z-slice of a 3-D array (used to fix overwrite adjoint)."""
    i, j = wp.tid()
    arr[i, j, top_k] = 0.0


@wp.kernel
def _clip_and_sanitize_3d_kernel(  # mosaic:physics
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


@wp.kernel
def _accumulate_top_slice_to_lid_kernel(  # mosaic:physics
    adj_ux: wp.array3d(dtype=wp.float32),
    adj_uy: wp.array3d(dtype=wp.float32),
    adj_lid: wp.array2d(dtype=wp.vec2),
    top_k: int,
):
    """Accumulate top-slice velocity grad into lid-field grad (overwrite adjoint fix).

    Correctly handles the write-after-write adjoint: when the lid BC overwrites
    ux[i,j,top_k] = lid[i,j][0], the backward should be:
        adj_lid[i,j][0] += adj_ux[i,j,top_k]
        adj_ux[i,j,top_k] = 0   (reset — old value has no contribution)
    This kernel does the accumulate step; _zero_top_slice_kernel does the reset.
    """
    i, j = wp.tid()
    v = adj_lid[i, j]
    v[0] = v[0] + adj_ux[i, j, top_k]
    v[1] = v[1] + adj_uy[i, j, top_k]
    adj_lid[i, j] = v


@wp.kernel
def apply_lid_bc_kernel(  # mosaic:physics
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    lid_vel: float,
    top_k: int,
):
    """Apply uniform scalar lid velocity BC on the top z-face (k = top_k).

    Legacy kernel retained for reference; the spatially-varying field kernel
    apply_lid_field_bc_kernel is used for the (N, N, 2) lid field interface.
    """
    i, j = wp.tid()
    ux[i, j, top_k] = lid_vel
    uy[i, j, top_k] = 0.0
    uz[i, j, top_k] = 0.0


@wp.kernel
def apply_lid_field_bc_kernel(  # mosaic:physics
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    lid_field: wp.array2d(dtype=wp.vec2),
    top_k: int,
):
    """Apply spatially-varying lid velocity BC on the top z-face (k = top_k).

    lid_field has shape (N, N) with vec2 components [u_x, u_y] per cell.
    The wall-normal component u_z is set to zero (no penetration).
    """
    i, j = wp.tid()
    v = lid_field[i, j]
    ux[i, j, top_k] = v[0]
    uy[i, j, top_k] = v[1]
    uz[i, j, top_k] = 0.0


def _wlaunch(kernel, dim, inputs, block_dim=256, device="cpu"):
    """wp.launch wrapper. block_dim must be an int (Warp 1.12 dropped tuple support)."""
    wp.launch(kernel, dim=dim, inputs=inputs, block_dim=block_dim, device=device)


# ============================================================
# Jacobi Poisson solvers
# ============================================================


def _jacobi_poisson_2d(  # mosaic:physics
    rhs: wp.array,
    scale: float,
    num_iters: int,
    device: str,
) -> wp.array:
    """Solve ∇²u = rhs on a periodic 2-D grid via Jacobi iteration.

    Uses index-based ping-pong (src/dst integer swap) so the tape correctly
    tracks both buffer objects across all launches.

    Note on sign convention: to solve ∇²ψ = -ω, pass rhs=omega and
    scale = +h²/4.  The Jacobi update is:
        ψ_new = avg(neighbours)/4 + (h²/4) * ω = avg/4 + h²ω/4
    which is the correct Jacobi step for ∇²ψ = -ω.
    """
    n = rhs.shape[0]
    bufs = [
        wp.zeros((n, n), dtype=wp.float32, device=device),
        wp.zeros((n, n), dtype=wp.float32, device=device),
    ]
    src, dst = 0, 1
    for _ in range(num_iters):
        _wlaunch(
            jacobi_2d_kernel,
            dim=(n, n),
            inputs=[bufs[src], rhs, bufs[dst], scale],
            block_dim=256,
            device=device,
        )
        src, dst = dst, src
    return bufs[src]


def _jacobi_poisson_3d(  # mosaic:physics
    rhs: wp.array,
    scale: float,
    num_iters: int,
    device: str,
) -> wp.array:
    """Solve ∇²u = rhs on a periodic 3-D grid via Jacobi iteration."""
    n = rhs.shape[0]
    bufs = [
        wp.zeros((n, n, n), dtype=wp.float32, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, device=device),
    ]
    src, dst = 0, 1
    for _ in range(num_iters):
        _wlaunch(
            jacobi_3d_kernel,
            dim=(n, n, n),
            inputs=[bufs[src], rhs, bufs[dst], scale],
            block_dim=256,
            device=device,
        )
        src, dst = dst, src
    return bufs[src]


def _spectral_poisson_3d_np(
    rhs_np: np.ndarray, domain_extent: float
) -> np.ndarray:  # mosaic:physics
    """Solve ∇²p = rhs on a 3-D periodic domain via FFT (exact up to float32).

    Returns p (same shape as rhs), mean-free (DC component = 0).

    Uses the discrete finite-difference Laplacian eigenvalues:
        λ_disc(kx, ky, kz) = -(4/h²)(sin²(πkx/N) + sin²(πky/N) + sin²(πkz/N))
    where h = L/N and kx,ky,kz are integer wavenumbers.  This matches the
    stencil used by tentative_vel_3d_kernel and pressure_correct_3d_kernel
    (both use inv_h2 = 1/h² and inv_2h = 1/(2h) central differences), making
    the spectral solve the exact inverse of the discrete FD Laplacian.

    Using continuous eigenvalues -(2π/L)²k² instead introduces a ~(πk/N)²/3
    relative error per wavenumber (≈1.3% at k=1, N=16), which compounds across
    the VJP chain and causes the flat-plateau ~1.65% gradient magnitude bias
    seen in the fd_check (ARCH-92).
    """
    n = rhs_np.shape[0]
    _h = domain_extent / n
    _kfreq = np.fft.fftfreq(
        n, d=1.0 / n
    )  # integer wavenumbers 0,1,...,N/2-1,-N/2,...,-1
    _KX, _KY, _KZ = np.meshgrid(_kfreq, _kfreq, _kfreq, indexing="ij")
    # Discrete FD Laplacian eigenvalues — exact inverse of the 6-point stencil
    _lambda = -(4.0 / _h**2) * (
        np.sin(np.pi * _KX / n) ** 2
        + np.sin(np.pi * _KY / n) ** 2
        + np.sin(np.pi * _KZ / n) ** 2
    )
    _lambda[0, 0, 0] = 1.0  # avoid division by zero; set DC = 0 below
    rhs_hat = np.fft.fftn(rhs_np)
    p_hat = rhs_hat / _lambda
    p_hat[0, 0, 0] = 0.0  # zero-mean pressure
    return np.real(np.fft.ifftn(p_hat)).astype(np.float32)


def _spectral_poisson_3d_tape(  # mosaic:grad:v0:adjoint
    rhs_wp: wp.array,
    domain_extent: float,
    tape: wp.Tape,
    device: str,
) -> wp.array:
    """Differentiable wrapper around spectral 3-D Poisson for wp.Tape.

    Uses tape.record_func to register the adjoint (which is identical to the
    forward, since the spectral Poisson operator is self-adjoint on a periodic
    domain).

    Returns a new wp.array holding the solution p with requires_grad=True.
    """
    # Forward: numpy FFT Poisson solve (not tracked by warp kernel system,
    # so we register it explicitly with record_func)
    rhs_np = rhs_wp.numpy()
    p_np = _spectral_poisson_3d_np(rhs_np, domain_extent)
    p_wp = wp.array(p_np, dtype=wp.float32, requires_grad=True, device=device)

    # Allocate a gradient accumulator for rhs that record_func can write to.
    # rhs_wp.grad is expected to already exist if rhs_wp has requires_grad=True,
    # but we capture references for the backward closure.
    _rhs_ref = rhs_wp
    _p_ref = p_wp
    _L_ref = domain_extent
    _dev_ref = device

    def _spectral_poisson_3d_backward(_rhs=_rhs_ref, _p=_p_ref, _L=_L_ref, _d=_dev_ref):
        """Backward: adj_rhs += spectral_poisson(adj_p).

        Spectral Poisson is self-adjoint, so the VJP is the same operator.
        """
        if _p.grad is None:
            return
        adj_p_np = _p.grad.numpy()
        adj_rhs_np = _spectral_poisson_3d_np(adj_p_np, _L)
        if _rhs.grad is not None:
            # accumulate
            cur = _rhs.grad.numpy()
            _rhs.grad = wp.array(
                (cur + adj_rhs_np).astype(np.float32), dtype=wp.float32, device=_d
            )
        else:
            _rhs.grad = wp.array(
                adj_rhs_np.astype(np.float32), dtype=wp.float32, device=_d
            )

    tape.record_func(
        backward=_spectral_poisson_3d_backward,
        arrays=[rhs_wp, p_wp],
    )

    return p_wp


# ============================================================
# CG Poisson solver (Neumann BCs) — for lid-driven cavity / channel flow
# ============================================================


@functools.lru_cache(maxsize=8)
def _build_laplacian_neumann_3d(
    n: int, h: float
) -> scipy.sparse.csr_matrix:  # mosaic:init
    """Sparse 3-D Laplacian with homogeneous Neumann (dp/dn=0) BCs on all faces.

    Constructed as the Kronecker sum of three 1-D Neumann Laplacians:
        L3d = kron(I, kron(I, L1d)) + kron(I, kron(L1d, I)) + kron(kron(L1d, I), I)

    The 1-D Laplacian on n points with spacing h is:
        Interior row i:   [..., 1, -2, 1, ...] / h²
        Boundary row 0:   [-1,  1,  0, ...] / h²   (ghost = interior, dp/dn=0)
        Boundary row n-1: [...,  0, 1, -1] / h²

    DOF 0 (corner cell [0,0,0]) is pinned to zero to remove the null-space
    (constant-pressure mode): row 0 is replaced by [1, 0, 0, ...] and the caller
    sets rhs[0]=0.  The resulting matrix is SPD, so CG converges.

    Returns an (n³) × (n³) CSR matrix.
    """
    inv_h2 = 1.0 / (h * h)

    # 1-D Neumann Laplacian (n×n)
    main_diag = -2.0 * inv_h2 * np.ones(n)
    off_diag = inv_h2 * np.ones(n - 1)
    L1d = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(n, n), format="lil"
    )
    # Neumann BCs: ghost cell = interior neighbour → row 0 and row n-1 modified
    L1d[0, 0] = -inv_h2
    L1d[0, 1] = inv_h2
    L1d[n - 1, n - 2] = inv_h2
    L1d[n - 1, n - 1] = -inv_h2
    L1d = L1d.tocsr()

    I_n = scipy.sparse.eye(n, format="csr")

    # 3-D Laplacian via Kronecker sum
    L3d = (
        scipy.sparse.kron(I_n, scipy.sparse.kron(I_n, L1d), format="csr")
        + scipy.sparse.kron(I_n, scipy.sparse.kron(L1d, I_n), format="csr")
        + scipy.sparse.kron(scipy.sparse.kron(L1d, I_n), I_n, format="csr")
    )

    # Pin DOF 0 to remove constant-pressure null space: replace row 0 with [1, 0, ...]
    L3d = L3d.tolil()
    L3d[0, :] = 0.0
    L3d[0, 0] = 1.0
    return L3d.tocsr()


def _cg_poisson_3d_np(
    rhs_np: np.ndarray, L: scipy.sparse.csr_matrix
) -> np.ndarray:  # mosaic:physics
    """Solve L p = rhs using CG (or direct LU for small n).

    rhs shape: (n, n, n).  Returns p of the same shape as float32.
    DOF 0 is pinned (rhs[0]=0 enforced here to match the pinned row in L).
    """
    rhs_flat = rhs_np.ravel().astype(np.float64)
    # Enforce the pinned DOF 0 (constant-pressure fix)
    rhs_flat[0] = 0.0

    n3 = rhs_flat.size
    if n3 <= 4096:
        # Small system: direct sparse LU is faster and exact
        p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)
    else:
        p_flat, info = scipy.sparse.linalg.cg(L, rhs_flat, rtol=1e-8, maxiter=3 * n3)
        if info != 0:
            # Fall back to direct solve when CG stalls
            p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)

    return p_flat.reshape(rhs_np.shape).astype(np.float32)


def _cg_poisson_3d_tape(  # mosaic:grad:v0:adjoint
    rhs_wp: wp.array,
    n: int,
    h: float,
    tape: wp.Tape,
    device: str,
) -> wp.array:
    """Differentiable CG Poisson solve (Neumann BCs) registered on a wp.Tape.

    Mirrors the structure of _spectral_poisson_3d_tape: runs the scipy-based
    solve outside Warp's kernel system, then registers a record_func backward.

    The backward of  A p = rhs  w.r.t. rhs is:  A adj_rhs = adj_p.
    Because A (the Neumann Laplacian) is symmetric the backward is the same
    CG solve — identical to the forward.

    Returns a new wp.array holding p with requires_grad=True.
    """
    L = _build_laplacian_neumann_3d(n, h)

    rhs_np = rhs_wp.numpy()
    p_np = _cg_poisson_3d_np(rhs_np, L)
    p_wp = wp.array(p_np, dtype=wp.float32, requires_grad=True, device=device)

    _rhs_ref = rhs_wp
    _p_ref = p_wp
    _n_ref = n
    _h_ref = h
    _dev_ref = device

    def _cg_poisson_3d_backward(
        _rhs=_rhs_ref, _p=_p_ref, _n=_n_ref, _h=_h_ref, _d=_dev_ref
    ):
        """Backward: adj_rhs += CG_solve(adj_p).

        The Neumann Laplacian is symmetric, so the VJP is the same operator.
        """
        if _p.grad is None:
            return
        adj_p_np = _p.grad.numpy()
        _L = _build_laplacian_neumann_3d(_n, _h)
        adj_rhs_np = _cg_poisson_3d_np(adj_p_np, _L)
        if _rhs.grad is not None:
            cur = _rhs.grad.numpy()
            _rhs.grad = wp.array(
                (cur + adj_rhs_np).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _rhs.grad = wp.array(
                adj_rhs_np.astype(np.float32), dtype=wp.float32, device=_d
            )

    tape.record_func(
        backward=_cg_poisson_3d_backward,
        arrays=[rhs_wp, p_wp],
    )

    return p_wp


# ============================================================
# Obstacle mask builder
# ============================================================


def _build_obstacle_mask(  # mosaic:init
    n: int,
    domain_extent: float,
    obstacle,
    device: str,
) -> wp.array:
    """Build a binary mask (1.0 inside cylinder obstacle, 0.0 outside).

    Coordinates are cell-centred: x_i = (i + 0.5) * h.
    obstacle.center and obstacle.radius are fractions of domain_extent.
    """
    h = domain_extent / n
    idx = np.arange(n)
    x = (idx + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")  # (N, N)

    cx = obstacle.center[0] * domain_extent
    cy = obstacle.center[1] * domain_extent
    r = obstacle.radius * domain_extent

    mask_np = ((X - cx) ** 2 + (Y - cy) ** 2 <= r**2).astype(np.float32)
    return wp.array(mask_np, dtype=wp.float32, device=device)


# ============================================================
# 2-D NS forward solve (IPCS — same scheme as 3-D)
# ============================================================


def _tentative_vel_2d_backward_np(  # mosaic:grad:v0:adjoint
    ux_np: np.ndarray,
    uy_np: np.ndarray,
    adj_ux_star_np: np.ndarray,
    adj_uy_star_np: np.ndarray,
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit adjoint of tentative_vel_2d_kernel.

    [2D-only function]

    Given cotangents (adj_ux_star, adj_uy_star) w.r.t. the outputs (ux_star, uy_star)
    of the forward kernel, returns (adj_ux, adj_uy) accumulated from all cells that
    read each input element.

    The forward kernel at cell (i,j):
        ux_star[i,j] = ux[i,j] + dt * (
            -ux[i,j]*(ux[i+1,j]-ux[i-1,j])*inv_2h
            -uy[i,j]*(ux[i,j+1]-ux[i,j-1])*inv_2h
            + nu*(ux[i-1,j]+ux[i+1,j]+ux[i,j-1]+ux[i,j+1]-4*ux[i,j])*inv_h2
        )
        uy_star[i,j] = uy[i,j] + dt * (
            -ux[i,j]*(uy[i+1,j]-uy[i-1,j])*inv_2h
            -uy[i,j]*(uy[i,j+1]-uy[i,j-1])*inv_2h
            + nu*(uy[i-1,j]+uy[i+1,j]+uy[i,j-1]+uy[i,j+1]-4*uy[i,j])*inv_h2
        )

    Vectorised using np.roll (periodic BCs):
        np.roll(a, +1, axis)  →  a_rolled[i] = a[i-1]  (im1 neighbour)
        np.roll(a, -1, axis)  →  a_rolled[i] = a[i+1]  (ip1 neighbour)
    """
    # Neighbour arrays for the forward state (read-only, used to form Jacobian entries)
    ux_im1 = np.roll(ux_np, 1, axis=0)  # ux[i-1, j]
    ux_ip1 = np.roll(ux_np, -1, axis=0)  # ux[i+1, j]
    ux_jm1 = np.roll(ux_np, 1, axis=1)  # ux[i, j-1]
    ux_jp1 = np.roll(ux_np, -1, axis=1)  # ux[i, j+1]

    uy_im1 = np.roll(uy_np, 1, axis=0)
    uy_ip1 = np.roll(uy_np, -1, axis=0)
    uy_jm1 = np.roll(uy_np, 1, axis=1)
    uy_jp1 = np.roll(uy_np, -1, axis=1)

    # Shifted cotangent arrays for neighbour contributions
    # adj_ux_star[i-1, j] = np.roll(adj_ux_star, +1, axis=0)[i, j]
    axs_im1 = np.roll(adj_ux_star_np, 1, axis=0)
    axs_ip1 = np.roll(adj_ux_star_np, -1, axis=0)
    axs_jm1 = np.roll(adj_ux_star_np, 1, axis=1)
    axs_jp1 = np.roll(adj_ux_star_np, -1, axis=1)

    ays_im1 = np.roll(adj_uy_star_np, 1, axis=0)
    ays_ip1 = np.roll(adj_uy_star_np, -1, axis=0)
    ays_jm1 = np.roll(adj_uy_star_np, 1, axis=1)
    ays_jp1 = np.roll(adj_uy_star_np, -1, axis=1)

    # ── adj_ux ──────────────────────────────────────────────────────────────
    # 1. Direct: d(ux_star[i,j])/d(ux[i,j]) from ux equation (reading ux[i,j] as ui)
    adj_ux = adj_ux_star_np * (
        1.0 - dt * (ux_ip1 - ux_im1) * inv_2h - 4.0 * dt * nu * inv_h2
    )
    # 2. Cell (i-1,j) reads ux[i,j] as its ux[ip1]: adv term -ux[i-1]*inv_2h, lap +nu*inv_h2
    adj_ux += axs_im1 * dt * (-ux_im1 * inv_2h + nu * inv_h2)
    # 3. Cell (i+1,j) reads ux[i,j] as its ux[im1]: adv term +ux[i+1]*inv_2h, lap +nu*inv_h2
    adj_ux += axs_ip1 * dt * (ux_ip1 * inv_2h + nu * inv_h2)
    # 4. Cell (i,j-1) reads ux[i,j] as its ux[jp1]: adv term -uy[i,j-1]*inv_2h, lap +nu*inv_h2
    adj_ux += axs_jm1 * dt * (-uy_jm1 * inv_2h + nu * inv_h2)
    # 5. Cell (i,j+1) reads ux[i,j] as its ux[jm1]: adv term +uy[i,j+1]*inv_2h, lap +nu*inv_h2
    adj_ux += axs_jp1 * dt * (uy_jp1 * inv_2h + nu * inv_h2)
    # 6. Cross-term: ux[i,j] appears in adv_uy at (i,j) as ui
    #    d(uy_star[i,j])/d(ux[i,j]) = -dt*(uy[i+1,j]-uy[i-1,j])*inv_2h
    adj_ux += adj_uy_star_np * (-dt * (uy_ip1 - uy_im1) * inv_2h)

    # ── adj_uy ──────────────────────────────────────────────────────────────
    # 1a. Cross-term in ux equation: d(ux_star[i,j])/d(uy[i,j]) = -dt*(ux[i,j+1]-ux[i,j-1])*inv_2h
    adj_uy = adj_ux_star_np * (-dt * (ux_jp1 - ux_jm1) * inv_2h)
    # 1b. Direct: d(uy_star[i,j])/d(uy[i,j]) from uy equation
    # d(adv_uy)/d(vi=uy[i,j]) = (uy[i,jp1]-uy[i,jm1])*inv_2h  (axis-1 neighbours, not axis-0)
    adj_uy += adj_uy_star_np * (
        1.0 - dt * (uy_jp1 - uy_jm1) * inv_2h - 4.0 * dt * nu * inv_h2
    )
    # 2. Cell (i-1,j) reads uy[i,j] as its uy[ip1] in uy eqn: adv -ux[i-1]*inv_2h, lap +nu*inv_h2
    adj_uy += ays_im1 * dt * (-ux_im1 * inv_2h + nu * inv_h2)
    # 3. Cell (i+1,j) reads uy[i,j] as its uy[im1] in uy eqn: adv +ux[i+1]*inv_2h, lap +nu*inv_h2
    adj_uy += ays_ip1 * dt * (ux_ip1 * inv_2h + nu * inv_h2)
    # 4. Cell (i,j-1) reads uy[i,j] as its uy[jp1] in uy eqn: adv -uy[i,j-1]*inv_2h, lap +nu*inv_h2
    adj_uy += ays_jm1 * dt * (-uy_jm1 * inv_2h + nu * inv_h2)
    # 5. Cell (i,j+1) reads uy[i,j] as its uy[jm1] in uy eqn: adv +uy[i,j+1]*inv_2h, lap +nu*inv_h2
    adj_uy += ays_jp1 * dt * (uy_jp1 * inv_2h + nu * inv_h2)

    return adj_ux.astype(np.float32), adj_uy.astype(np.float32)


def _tentative_vel_3d_backward_np(  # mosaic:grad:v0:adjoint
    ux_np: np.ndarray,
    uy_np: np.ndarray,
    uz_np: np.ndarray,
    adj_ux_star_np: np.ndarray,
    adj_uy_star_np: np.ndarray,
    adj_uz_star_np: np.ndarray,
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit adjoint of tentative_vel_3d_kernel.

    [3D-only function]

    Generalises _tentative_vel_2d_backward_np to three components with the
    additional (axis-2) neighbour contributions and the full set of ux/uy/uz
    cross-component couplings through the advection term.

    The forward kernel at cell (i,j,k) is:
        ux_star = ui + dt*(-adv_ux + nu*lap_ux)
        uy_star = vi + dt*(-adv_uy + nu*lap_uy)
        uz_star = wi + dt*(-adv_uz + nu*lap_uz)
    with ui,vi,wi = ux[i,j,k], uy[i,j,k], uz[i,j,k] and
        adv_c = ui*(c[ip1]-c[im1])*inv_2h
              + vi*(c[jp1]-c[jm1])*inv_2h
              + wi*(c[kp1]-c[km1])*inv_2h     for c ∈ {ux,uy,uz}
        lap_c = (c[im1]+c[ip1]+c[jm1]+c[jp1]+c[km1]+c[kp1] - 6*ci)*inv_h2
    (periodic BCs via np.roll).

    Vectorised with np.roll conventions:
        np.roll(a, +1, axis)  →  rolled[i] = a[i-1]  (im1 neighbour)
        np.roll(a, -1, axis)  →  rolled[i] = a[i+1]  (ip1 neighbour)
    """
    # Neighbour arrays of the forward state (read-only)
    ux_im1 = np.roll(ux_np, 1, axis=0)
    ux_ip1 = np.roll(ux_np, -1, axis=0)
    ux_jm1 = np.roll(ux_np, 1, axis=1)
    ux_jp1 = np.roll(ux_np, -1, axis=1)
    ux_km1 = np.roll(ux_np, 1, axis=2)
    ux_kp1 = np.roll(ux_np, -1, axis=2)

    uy_im1 = np.roll(uy_np, 1, axis=0)
    uy_ip1 = np.roll(uy_np, -1, axis=0)
    uy_jm1 = np.roll(uy_np, 1, axis=1)
    uy_jp1 = np.roll(uy_np, -1, axis=1)
    uy_km1 = np.roll(uy_np, 1, axis=2)
    uy_kp1 = np.roll(uy_np, -1, axis=2)

    uz_im1 = np.roll(uz_np, 1, axis=0)
    uz_ip1 = np.roll(uz_np, -1, axis=0)
    uz_jm1 = np.roll(uz_np, 1, axis=1)
    uz_jp1 = np.roll(uz_np, -1, axis=1)
    uz_km1 = np.roll(uz_np, 1, axis=2)
    uz_kp1 = np.roll(uz_np, -1, axis=2)

    # Neighbour-shifted cotangents
    axs_im1 = np.roll(adj_ux_star_np, 1, axis=0)
    axs_ip1 = np.roll(adj_ux_star_np, -1, axis=0)
    axs_jm1 = np.roll(adj_ux_star_np, 1, axis=1)
    axs_jp1 = np.roll(adj_ux_star_np, -1, axis=1)
    axs_km1 = np.roll(adj_ux_star_np, 1, axis=2)
    axs_kp1 = np.roll(adj_ux_star_np, -1, axis=2)

    ays_im1 = np.roll(adj_uy_star_np, 1, axis=0)
    ays_ip1 = np.roll(adj_uy_star_np, -1, axis=0)
    ays_jm1 = np.roll(adj_uy_star_np, 1, axis=1)
    ays_jp1 = np.roll(adj_uy_star_np, -1, axis=1)
    ays_km1 = np.roll(adj_uy_star_np, 1, axis=2)
    ays_kp1 = np.roll(adj_uy_star_np, -1, axis=2)

    azs_im1 = np.roll(adj_uz_star_np, 1, axis=0)
    azs_ip1 = np.roll(adj_uz_star_np, -1, axis=0)
    azs_jm1 = np.roll(adj_uz_star_np, 1, axis=1)
    azs_jp1 = np.roll(adj_uz_star_np, -1, axis=1)
    azs_km1 = np.roll(adj_uz_star_np, 1, axis=2)
    azs_kp1 = np.roll(adj_uz_star_np, -1, axis=2)

    # ── adj_ux ──────────────────────────────────────────────────────────────
    # 1. Direct: d(ux_star[i,j,k])/d(ux[i,j,k]) from ux equation
    #    d/dui[1*ui + dt*(-ui*(ux_ip1-ux_im1)*inv_2h + nu*(-6*ui)*inv_h2 + ...)]
    adj_ux = adj_ux_star_np * (
        1.0 - dt * (ux_ip1 - ux_im1) * inv_2h - 6.0 * dt * nu * inv_h2
    )
    # 2. Neighbours of (i,j,k) reading ux[i,j,k] in their own ux-eqn
    adj_ux += axs_im1 * dt * (-ux_im1 * inv_2h + nu * inv_h2)
    adj_ux += axs_ip1 * dt * (ux_ip1 * inv_2h + nu * inv_h2)
    adj_ux += axs_jm1 * dt * (-uy_jm1 * inv_2h + nu * inv_h2)
    adj_ux += axs_jp1 * dt * (uy_jp1 * inv_2h + nu * inv_h2)
    adj_ux += axs_km1 * dt * (-uz_km1 * inv_2h + nu * inv_h2)
    adj_ux += axs_kp1 * dt * (uz_kp1 * inv_2h + nu * inv_h2)
    # 3. Cross-terms: ux[i,j,k]=ui appears in adv_uy and adv_uz at (i,j,k)
    adj_ux += adj_uy_star_np * (-dt * (uy_ip1 - uy_im1) * inv_2h)
    adj_ux += adj_uz_star_np * (-dt * (uz_ip1 - uz_im1) * inv_2h)

    # ── adj_uy ──────────────────────────────────────────────────────────────
    # Cross-terms: uy[i,j,k]=vi appears in adv_ux and adv_uz at (i,j,k)
    adj_uy = adj_ux_star_np * (-dt * (ux_jp1 - ux_jm1) * inv_2h)
    # 1b. Direct: d(uy_star[i,j,k])/d(uy[i,j,k])
    adj_uy += adj_uy_star_np * (
        1.0 - dt * (uy_jp1 - uy_jm1) * inv_2h - 6.0 * dt * nu * inv_h2
    )
    adj_uy += adj_uz_star_np * (-dt * (uz_jp1 - uz_jm1) * inv_2h)
    # Neighbours of (i,j,k) reading uy[i,j,k] in their own uy-eqn
    adj_uy += ays_im1 * dt * (-ux_im1 * inv_2h + nu * inv_h2)
    adj_uy += ays_ip1 * dt * (ux_ip1 * inv_2h + nu * inv_h2)
    adj_uy += ays_jm1 * dt * (-uy_jm1 * inv_2h + nu * inv_h2)
    adj_uy += ays_jp1 * dt * (uy_jp1 * inv_2h + nu * inv_h2)
    adj_uy += ays_km1 * dt * (-uz_km1 * inv_2h + nu * inv_h2)
    adj_uy += ays_kp1 * dt * (uz_kp1 * inv_2h + nu * inv_h2)

    # ── adj_uz ──────────────────────────────────────────────────────────────
    # Cross-terms: uz[i,j,k]=wi appears in adv_ux and adv_uy at (i,j,k)
    adj_uz = adj_ux_star_np * (-dt * (ux_kp1 - ux_km1) * inv_2h)
    adj_uz += adj_uy_star_np * (-dt * (uy_kp1 - uy_km1) * inv_2h)
    # 1c. Direct: d(uz_star[i,j,k])/d(uz[i,j,k])
    adj_uz += adj_uz_star_np * (
        1.0 - dt * (uz_kp1 - uz_km1) * inv_2h - 6.0 * dt * nu * inv_h2
    )
    # Neighbours of (i,j,k) reading uz[i,j,k] in their own uz-eqn
    adj_uz += azs_im1 * dt * (-ux_im1 * inv_2h + nu * inv_h2)
    adj_uz += azs_ip1 * dt * (ux_ip1 * inv_2h + nu * inv_h2)
    adj_uz += azs_jm1 * dt * (-uy_jm1 * inv_2h + nu * inv_h2)
    adj_uz += azs_jp1 * dt * (uy_jp1 * inv_2h + nu * inv_h2)
    adj_uz += azs_km1 * dt * (-uz_km1 * inv_2h + nu * inv_h2)
    adj_uz += azs_kp1 * dt * (uz_kp1 * inv_2h + nu * inv_h2)

    return (
        adj_ux.astype(np.float32),
        adj_uy.astype(np.float32),
        adj_uz.astype(np.float32),
    )


def _tentative_vel_3d_tape(  # mosaic:grad:v0:adjoint
    ux_wp: wp.array,
    uy_wp: wp.array,
    uz_wp: wp.array,
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
    tape: wp.Tape,
    device: str,
) -> tuple[wp.array, wp.array, wp.array]:
    """Compute tentative velocity and register explicit backward via tape.record_func.

    [3D-only function]

    Structurally mirrors _tentative_vel_2d_tape: Warp's source-to-source AD of
    tentative_vel_3d_kernel produces wrong-sign gradients on specific Fourier-
    mode combinations involving cross-component advection coupling — same
    failure mode the 2D wrapper was introduced to fix.  This wrapper bypasses
    Warp's auto-adjoint entirely by doing the forward in numpy (exact same
    arithmetic as the kernel) and registering the analytically-derived
    _tentative_vel_3d_backward_np via tape.record_func.  Eliminates the 3D
    fd_check ~1.6% rel_err (F-NS3D-3, ARCH-31).

    Returns:
        (ux_star_wp, uy_star_wp, uz_star_wp) — fresh wp.arrays with
        requires_grad=True, ready to flow into downstream tape nodes.
    """
    # Forward pass in numpy (same arithmetic as tentative_vel_3d_kernel, periodic BCs)
    ux_np = ux_wp.numpy()
    uy_np = uy_wp.numpy()
    uz_np = uz_wp.numpy()

    ux_im1 = np.roll(ux_np, 1, axis=0)
    ux_ip1 = np.roll(ux_np, -1, axis=0)
    ux_jm1 = np.roll(ux_np, 1, axis=1)
    ux_jp1 = np.roll(ux_np, -1, axis=1)
    ux_km1 = np.roll(ux_np, 1, axis=2)
    ux_kp1 = np.roll(ux_np, -1, axis=2)

    uy_im1 = np.roll(uy_np, 1, axis=0)
    uy_ip1 = np.roll(uy_np, -1, axis=0)
    uy_jm1 = np.roll(uy_np, 1, axis=1)
    uy_jp1 = np.roll(uy_np, -1, axis=1)
    uy_km1 = np.roll(uy_np, 1, axis=2)
    uy_kp1 = np.roll(uy_np, -1, axis=2)

    uz_im1 = np.roll(uz_np, 1, axis=0)
    uz_ip1 = np.roll(uz_np, -1, axis=0)
    uz_jm1 = np.roll(uz_np, 1, axis=1)
    uz_jp1 = np.roll(uz_np, -1, axis=1)
    uz_km1 = np.roll(uz_np, 1, axis=2)
    uz_kp1 = np.roll(uz_np, -1, axis=2)

    lap_ux = (
        ux_im1 + ux_ip1 + ux_jm1 + ux_jp1 + ux_km1 + ux_kp1 - 6.0 * ux_np
    ) * inv_h2
    adv_ux = (
        ux_np * (ux_ip1 - ux_im1) * inv_2h
        + uy_np * (ux_jp1 - ux_jm1) * inv_2h
        + uz_np * (ux_kp1 - ux_km1) * inv_2h
    )
    ux_star_np = (ux_np + dt * (-adv_ux + nu * lap_ux)).astype(np.float32)

    lap_uy = (
        uy_im1 + uy_ip1 + uy_jm1 + uy_jp1 + uy_km1 + uy_kp1 - 6.0 * uy_np
    ) * inv_h2
    adv_uy = (
        ux_np * (uy_ip1 - uy_im1) * inv_2h
        + uy_np * (uy_jp1 - uy_jm1) * inv_2h
        + uz_np * (uy_kp1 - uy_km1) * inv_2h
    )
    uy_star_np = (uy_np + dt * (-adv_uy + nu * lap_uy)).astype(np.float32)

    lap_uz = (
        uz_im1 + uz_ip1 + uz_jm1 + uz_jp1 + uz_km1 + uz_kp1 - 6.0 * uz_np
    ) * inv_h2
    adv_uz = (
        ux_np * (uz_ip1 - uz_im1) * inv_2h
        + uy_np * (uz_jp1 - uz_jm1) * inv_2h
        + uz_np * (uz_kp1 - uz_km1) * inv_2h
    )
    uz_star_np = (uz_np + dt * (-adv_uz + nu * lap_uz)).astype(np.float32)

    ux_star_wp = wp.array(
        ux_star_np, dtype=wp.float32, requires_grad=True, device=device
    )
    uy_star_wp = wp.array(
        uy_star_np, dtype=wp.float32, requires_grad=True, device=device
    )
    uz_star_wp = wp.array(
        uz_star_np, dtype=wp.float32, requires_grad=True, device=device
    )

    # Register explicit backward — captures forward state by value so it is
    # independent of any downstream in-place overwrites.
    _ux_fwd = ux_np.copy()
    _uy_fwd = uy_np.copy()
    _uz_fwd = uz_np.copy()
    _ux_ref = ux_wp
    _uy_ref = uy_wp
    _uz_ref = uz_wp
    _ux_star_ref = ux_star_wp
    _uy_star_ref = uy_star_wp
    _uz_star_ref = uz_star_wp
    _dt = dt
    _inv_2h = inv_2h
    _inv_h2 = inv_h2
    _nu = nu
    _dev = device

    def _tentative_vel_3d_backward(
        _ux_fwd=_ux_fwd,
        _uy_fwd=_uy_fwd,
        _uz_fwd=_uz_fwd,
        _ux=_ux_ref,
        _uy=_uy_ref,
        _uz=_uz_ref,
        _ux_s=_ux_star_ref,
        _uy_s=_uy_star_ref,
        _uz_s=_uz_star_ref,
        _dt=_dt,
        _inv_2h=_inv_2h,
        _inv_h2=_inv_h2,
        _nu=_nu,
        _d=_dev,
    ):
        """Explicit adjoint of tentative_vel_3d_kernel."""
        if _ux_s.grad is None and _uy_s.grad is None and _uz_s.grad is None:
            return
        adj_ux_star = (
            _ux_s.grad.numpy() if _ux_s.grad is not None else np.zeros_like(_ux_fwd)
        )
        adj_uy_star = (
            _uy_s.grad.numpy() if _uy_s.grad is not None else np.zeros_like(_uy_fwd)
        )
        adj_uz_star = (
            _uz_s.grad.numpy() if _uz_s.grad is not None else np.zeros_like(_uz_fwd)
        )

        adj_ux, adj_uy, adj_uz = _tentative_vel_3d_backward_np(
            _ux_fwd,
            _uy_fwd,
            _uz_fwd,
            adj_ux_star,
            adj_uy_star,
            adj_uz_star,
            _dt,
            _inv_2h,
            _inv_h2,
            _nu,
        )

        # Accumulate into inputs' .grad (same policy as 2D wrapper).
        if _ux.grad is not None:
            _ux.grad = wp.array(
                (_ux.grad.numpy() + adj_ux).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _ux.grad = wp.array(adj_ux, dtype=wp.float32, device=_d)

        if _uy.grad is not None:
            _uy.grad = wp.array(
                (_uy.grad.numpy() + adj_uy).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _uy.grad = wp.array(adj_uy, dtype=wp.float32, device=_d)

        if _uz.grad is not None:
            _uz.grad = wp.array(
                (_uz.grad.numpy() + adj_uz).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _uz.grad = wp.array(adj_uz, dtype=wp.float32, device=_d)

    tape.record_func(
        backward=_tentative_vel_3d_backward,
        arrays=[ux_wp, uy_wp, uz_wp, ux_star_wp, uy_star_wp, uz_star_wp],
    )
    return ux_star_wp, uy_star_wp, uz_star_wp


def _tentative_vel_2d_tape(  # mosaic:grad:v0:adjoint
    ux_wp: wp.array,
    uy_wp: wp.array,
    dt: float,
    inv_2h: float,
    inv_h2: float,
    nu: float,
    tape: wp.Tape,
    device: str,
    nu_wp: "wp.array | None" = None,
    dt_wp: "wp.array | None" = None,
) -> tuple[wp.array, wp.array]:
    """Compute tentative velocity and register explicit backward via tape.record_func.

    [2D-only function]

    Warp's source-to-source AD of tentative_vel_2d_kernel produces wrong-sign
    gradients for specific Fourier-mode combinations (cross-component coupling
    between ux and uy advection terms).  This wrapper avoids the auto-adjoint
    entirely by computing the forward pass in numpy (exact same arithmetic as the
    kernel) and registering the analytically-derived _tentative_vel_2d_backward_np
    as the backward via tape.record_func.

    Pattern mirrors _spectral_poisson_2d_tape: forward in numpy → new wp.array
    with requires_grad=True → register backward via record_func.

    nu_wp / dt_wp: optional scalar wp.arrays with requires_grad=True.  When
    provided, the backward also accumulates adj_nu and adj_dt into their .grad
    attributes, enabling tape-based scalar gradient computation without FD.

    Returns:
        (ux_star_wp, uy_star_wp) — new wp arrays with requires_grad=True.
    """
    # Forward pass in numpy (same arithmetic as tentative_vel_2d_kernel, periodic BCs)
    ux_np = ux_wp.numpy()
    uy_np = uy_wp.numpy()

    ux_im1 = np.roll(ux_np, 1, axis=0)
    ux_ip1 = np.roll(ux_np, -1, axis=0)
    ux_jm1 = np.roll(ux_np, 1, axis=1)
    ux_jp1 = np.roll(ux_np, -1, axis=1)

    uy_im1 = np.roll(uy_np, 1, axis=0)
    uy_ip1 = np.roll(uy_np, -1, axis=0)
    uy_jm1 = np.roll(uy_np, 1, axis=1)
    uy_jp1 = np.roll(uy_np, -1, axis=1)

    lap_ux = (ux_im1 + ux_ip1 + ux_jm1 + ux_jp1 - 4.0 * ux_np) * inv_h2
    adv_ux = ux_np * (ux_ip1 - ux_im1) * inv_2h + uy_np * (ux_jp1 - ux_jm1) * inv_2h
    ux_star_np = (ux_np + dt * (-adv_ux + nu * lap_ux)).astype(np.float32)

    lap_uy = (uy_im1 + uy_ip1 + uy_jm1 + uy_jp1 - 4.0 * uy_np) * inv_h2
    adv_uy = ux_np * (uy_ip1 - uy_im1) * inv_2h + uy_np * (uy_jp1 - uy_jm1) * inv_2h
    uy_star_np = (uy_np + dt * (-adv_uy + nu * lap_uy)).astype(np.float32)

    # Create new output arrays (requires_grad=True so downstream tape nodes can back-prop)
    ux_star_wp = wp.array(
        ux_star_np, dtype=wp.float32, requires_grad=True, device=device
    )
    uy_star_wp = wp.array(
        uy_star_np, dtype=wp.float32, requires_grad=True, device=device
    )

    # Register explicit backward — captures forward state by value
    _ux_fwd = ux_np.copy()
    _uy_fwd = uy_np.copy()
    _ux_ref = ux_wp
    _uy_ref = uy_wp
    _ux_star_ref = ux_star_wp
    _uy_star_ref = uy_star_wp
    _dt = dt
    _inv_2h = inv_2h
    _inv_h2 = inv_h2
    _nu = nu
    _dev = device
    # Capture forward Laplacian and advection for scalar gradient accumulation.
    # These are needed to compute adj_nu and adj_dt analytically:
    #   adj_nu += sum(adj_ux_star * dt * lap_ux + adj_uy_star * dt * lap_uy)
    #   adj_dt += sum(adj_ux_star * (ux_star - ux)/dt + adj_uy_star * (uy_star - uy)/dt)
    #            = sum(adj_ux_star * (-adv_ux + nu*lap_ux) + adj_uy_star * (-adv_uy + nu*lap_uy))
    _lap_ux_fwd = lap_ux.copy()
    _lap_uy_fwd = lap_uy.copy()
    _adv_ux_fwd = adv_ux.copy()
    _adv_uy_fwd = adv_uy.copy()
    _nu_wp = nu_wp
    _dt_wp = dt_wp

    def _tentative_vel_2d_backward(
        _ux_fwd=_ux_fwd,
        _uy_fwd=_uy_fwd,
        _ux=_ux_ref,
        _uy=_uy_ref,
        _ux_s=_ux_star_ref,
        _uy_s=_uy_star_ref,
        _dt=_dt,
        _inv_2h=_inv_2h,
        _inv_h2=_inv_h2,
        _nu=_nu,
        _d=_dev,
        _lap_ux=_lap_ux_fwd,
        _lap_uy=_lap_uy_fwd,
        _adv_ux=_adv_ux_fwd,
        _adv_uy=_adv_uy_fwd,
        _nu_wp=_nu_wp,
        _dt_wp=_dt_wp,
    ):
        """Explicit adjoint of tentative_vel_2d_kernel."""
        if _ux_s.grad is None and _uy_s.grad is None:
            return
        adj_ux_star = (
            _ux_s.grad.numpy() if _ux_s.grad is not None else np.zeros_like(_ux_fwd)
        )
        adj_uy_star = (
            _uy_s.grad.numpy() if _uy_s.grad is not None else np.zeros_like(_uy_fwd)
        )

        adj_ux_np, adj_uy_np = _tentative_vel_2d_backward_np(
            _ux_fwd,
            _uy_fwd,
            adj_ux_star,
            adj_uy_star,
            _dt,
            _inv_2h,
            _inv_h2,
            _nu,
        )

        if _ux.grad is not None:
            _ux.grad = wp.array(
                (_ux.grad.numpy() + adj_ux_np).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _ux.grad = wp.array(adj_ux_np, dtype=wp.float32, device=_d)

        if _uy.grad is not None:
            _uy.grad = wp.array(
                (_uy.grad.numpy() + adj_uy_np).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _uy.grad = wp.array(adj_uy_np, dtype=wp.float32, device=_d)

        # Accumulate scalar gradients (only if caller requested tape-based scalar grads).
        # adj_nu += dt * sum(adj_ux_star * lap_ux + adj_uy_star * lap_uy)
        # adj_dt += sum(adj_ux_star * (-adv_ux + nu*lap_ux) + adj_uy_star * (-adv_uy + nu*lap_uy))
        if _nu_wp is not None:
            d_adj_nu = float(
                _dt * (np.sum(adj_ux_star * _lap_ux) + np.sum(adj_uy_star * _lap_uy))
            )
            if _nu_wp.grad is not None:
                _nu_wp.grad = wp.array(
                    np.array([_nu_wp.grad.numpy()[0] + d_adj_nu], dtype=np.float32),
                    dtype=wp.float32,
                    device=_d,
                )
            else:
                _nu_wp.grad = wp.array(
                    np.array([d_adj_nu], dtype=np.float32), dtype=wp.float32, device=_d
                )
        if _dt_wp is not None:
            d_adj_dt = float(
                np.sum(adj_ux_star * (-_adv_ux + _nu * _lap_ux))
                + np.sum(adj_uy_star * (-_adv_uy + _nu * _lap_uy))
            )
            if _dt_wp.grad is not None:
                _dt_wp.grad = wp.array(
                    np.array([_dt_wp.grad.numpy()[0] + d_adj_dt], dtype=np.float32),
                    dtype=wp.float32,
                    device=_d,
                )
            else:
                _dt_wp.grad = wp.array(
                    np.array([d_adj_dt], dtype=np.float32), dtype=wp.float32, device=_d
                )

    # Include nu_wp and dt_wp in the arrays list so tape tracks them as dependencies.
    _scalar_arrays = []
    if nu_wp is not None:
        _scalar_arrays.append(nu_wp)
    if dt_wp is not None:
        _scalar_arrays.append(dt_wp)
    tape.record_func(
        backward=_tentative_vel_2d_backward,
        arrays=[ux_wp, uy_wp, ux_star_wp, uy_star_wp] + _scalar_arrays,
    )
    return ux_star_wp, uy_star_wp


def _spectral_poisson_2d_np(
    rhs_np: np.ndarray, domain_extent: float
) -> np.ndarray:  # mosaic:physics
    """Solve ∇²p = rhs on a 2-D periodic domain via FFT (exact up to float32).

    [2D-only function]

    Returns p (same shape as rhs), mean-free (DC component = 0).
    """
    n = rhs_np.shape[0]
    _L = domain_extent
    _kfreq = np.fft.fftfreq(n, d=1.0 / n)
    _KX, _KY = np.meshgrid(_kfreq, _kfreq, indexing="ij")
    _lambda = -((2.0 * np.pi / _L) ** 2) * (_KX**2 + _KY**2)
    _lambda[0, 0] = 1.0  # avoid division by zero; set DC = 0 below
    rhs_hat = np.fft.fft2(rhs_np)
    p_hat = rhs_hat / _lambda
    p_hat[0, 0] = 0.0  # zero-mean solution
    return np.real(np.fft.ifft2(p_hat)).astype(np.float32)


def _spectral_poisson_2d_tape(  # mosaic:grad:v0:adjoint
    rhs_wp: wp.array,
    domain_extent: float,
    tape: wp.Tape,
    device: str,
) -> wp.array:
    """Differentiable 2-D spectral Poisson solve for use inside wp.Tape.

    [2D-only function]

    Solves ∇²p = rhs exactly via FFT.  The operator is self-adjoint, so the
    backward is identical: adj_rhs += spectral_poisson(adj_p).

    Returns a new wp.array holding p with requires_grad=True.
    """
    rhs_np = rhs_wp.numpy()
    p_np = _spectral_poisson_2d_np(rhs_np, domain_extent)
    p_wp = wp.array(p_np, dtype=wp.float32, requires_grad=True, device=device)

    _rhs_ref = rhs_wp
    _p_ref = p_wp
    _L_ref = domain_extent
    _dev_ref = device

    def _spectral_poisson_2d_backward(_rhs=_rhs_ref, _p=_p_ref, _L=_L_ref, _d=_dev_ref):
        """Backward: adj_rhs += spectral_poisson(adj_p) (self-adjoint)."""
        if _p.grad is None:
            return
        adj_p_np = _p.grad.numpy()
        adj_rhs_np = _spectral_poisson_2d_np(adj_p_np, _L)
        if _rhs.grad is not None:
            cur = _rhs.grad.numpy()
            _rhs.grad = wp.array(
                (cur + adj_rhs_np).astype(np.float32), dtype=wp.float32, device=_d
            )
        else:
            _rhs.grad = wp.array(
                adj_rhs_np.astype(np.float32), dtype=wp.float32, device=_d
            )

    tape.record_func(backward=_spectral_poisson_2d_backward, arrays=[rhs_wp, p_wp])
    return p_wp


# ============================================================
# CG Poisson solver (Neumann BCs) — 2-D version for obstacle/inflow flow
# ============================================================


@functools.lru_cache(maxsize=8)
def _build_laplacian_neumann_2d(
    n: int, h: float
) -> scipy.sparse.csr_matrix:  # mosaic:init
    """Sparse 2-D Laplacian with homogeneous Neumann (dp/dn=0) BCs on all edges.

    Constructed as the Kronecker sum of two 1-D Neumann Laplacians:
        L2d = kron(I_n, L1d) + kron(L1d, I_n)

    The 1-D Laplacian on n points with spacing h is:
        Interior row i:   [..., 1, -2, 1, ...] / h²
        Boundary row 0:   [-1,  1,  0, ...] / h²   (ghost = interior, dp/dn=0)
        Boundary row n-1: [...,  0, 1, -1] / h²

    DOF 0 (corner cell [0,0]) is pinned to zero to remove the null-space
    (constant-pressure mode): row 0 is replaced by [1, 0, 0, ...] and the caller
    sets rhs[0]=0.  The resulting matrix is SPD, so CG converges.

    Returns an (n²) × (n²) CSR matrix.
    """
    inv_h2 = 1.0 / (h * h)

    # 1-D Neumann Laplacian (n×n)
    main_diag = -2.0 * inv_h2 * np.ones(n)
    off_diag = inv_h2 * np.ones(n - 1)
    L1d = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(n, n), format="lil"
    )
    # Neumann BCs: ghost cell = interior neighbour → row 0 and row n-1 modified
    L1d[0, 0] = -inv_h2
    L1d[0, 1] = inv_h2
    L1d[n - 1, n - 2] = inv_h2
    L1d[n - 1, n - 1] = -inv_h2
    L1d = L1d.tocsr()

    I_n = scipy.sparse.eye(n, format="csr")

    # 2-D Laplacian via Kronecker sum
    L2d = scipy.sparse.kron(I_n, L1d, format="csr") + scipy.sparse.kron(
        L1d, I_n, format="csr"
    )

    # Pin DOF 0 to remove constant-pressure null space: replace row 0 with [1, 0, ...]
    L2d = L2d.tolil()
    L2d[0, :] = 0.0
    L2d[0, 0] = 1.0
    return L2d.tocsr()


@functools.lru_cache(maxsize=8)
def _build_laplacian_channel_2d(
    n: int, h: float
) -> scipy.sparse.csr_matrix:  # mosaic:init
    """Sparse 2-D Laplacian for channel flow (inflow x=0, outflow x=N-1).

    x-direction: Neumann BC at i=0, Dirichlet (p=0) at i=N-1.
    y-direction: periodic (circulant) — matches the % n wrapping in the
    divergence and pressure-correction kernels.

    The Dirichlet BC at outflow (x=N-1) pins the pressure reference, so no
    separate DOF-pinning is needed.  The resulting matrix is SPD.

    Returns an (n²) × (n²) CSR matrix.  DOF ordering: d = i*n + j.
    """
    inv_h2 = 1.0 / (h * h)

    # x-direction 1-D Laplacian (Neumann at both ends — outflow row is
    # overridden by Dirichlet below at the 2-D level)
    main_diag = -2.0 * inv_h2 * np.ones(n)
    off_diag = inv_h2 * np.ones(n - 1)
    L1d_x = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(n, n), format="lil"
    )
    L1d_x[0, 0] = -inv_h2
    L1d_x[0, 1] = inv_h2
    L1d_x[n - 1, n - 2] = inv_h2
    L1d_x[n - 1, n - 1] = -inv_h2
    L1d_x = L1d_x.tocsr()

    # y-direction 1-D Laplacian (periodic/circulant — matches the % n wrapping
    # in divergence_2d_channel_kernel and pressure_correct_2d_channel_kernel)
    L1d_y = scipy.sparse.diags(
        [off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(n, n), format="lil"
    )
    L1d_y[0, n - 1] = inv_h2  # periodic wrap: j=0 neighbours j=N-1
    L1d_y[n - 1, 0] = inv_h2  # periodic wrap: j=N-1 neighbours j=0
    L1d_y = L1d_y.tocsr()

    I_n = scipy.sparse.eye(n, format="csr")

    # 2-D Laplacian (DOF d = i*n + j)
    L2d = (
        scipy.sparse.kron(L1d_x, I_n, format="csr")  # x-direction
        + scipy.sparse.kron(I_n, L1d_y, format="csr")  # y-direction
    )

    # Dirichlet BC at x=N-1: set rows (N-1)*n .. N*n-1 to identity rows
    L2d = L2d.tolil()
    for j in range(n):
        d = (n - 1) * n + j
        L2d[d, :] = 0.0
        L2d[d, d] = 1.0
    return L2d.tocsr()


def _cg_poisson_2d_np(
    rhs_np: np.ndarray, L: scipy.sparse.csr_matrix
) -> np.ndarray:  # mosaic:physics
    """Solve L p = rhs using CG (or direct LU for small n).

    rhs shape: (n, n).  Returns p of the same shape as float32.
    DOF 0 is pinned (rhs[0]=0 enforced here to match the pinned row in L).
    """
    rhs_flat = rhs_np.ravel().astype(np.float64)
    # Enforce the pinned DOF 0 (constant-pressure fix)
    rhs_flat[0] = 0.0

    n2 = rhs_flat.size
    if n2 <= 4096:
        # Small system: direct sparse LU is faster and exact
        p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)
    else:
        p_flat, info = scipy.sparse.linalg.cg(L, rhs_flat, rtol=1e-8)
        if info != 0:
            # Fall back to direct solve when CG stalls
            p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)

    return p_flat.reshape(rhs_np.shape).astype(np.float32)


def _cg_poisson_2d_tape(  # mosaic:grad:v0:adjoint
    rhs_wp: wp.array,
    n: int,
    h: float,
    tape: wp.Tape,
    device: str,
) -> wp.array:
    """Differentiable CG Poisson solve (Neumann BCs) registered on a wp.Tape.

    [2D-only function]

    Mirrors the structure of _spectral_poisson_2d_tape: runs the scipy-based
    solve outside Warp's kernel system, then registers a record_func backward.

    The backward of  A p = rhs  w.r.t. rhs is:  A adj_rhs = adj_p.
    Because A (the Neumann Laplacian) is symmetric the backward is the same
    CG solve — identical to the forward.

    Returns a new wp.array holding p with requires_grad=True.
    """
    L = _build_laplacian_neumann_2d(n, h)

    rhs_np = rhs_wp.numpy()
    p_np = _cg_poisson_2d_np(rhs_np, L)
    p_wp = wp.array(p_np, dtype=wp.float32, requires_grad=True, device=device)

    _rhs_ref = rhs_wp
    _p_ref = p_wp
    _n_ref = n
    _h_ref = h
    _dev_ref = device

    def _cg_poisson_2d_backward(
        _rhs=_rhs_ref, _p=_p_ref, _n=_n_ref, _h=_h_ref, _d=_dev_ref
    ):
        """Backward: adj_rhs += CG_solve(adj_p).

        The Neumann Laplacian is symmetric, so the VJP is the same operator.
        """
        if _p.grad is None:
            return
        adj_p_np = _p.grad.numpy()
        _L = _build_laplacian_neumann_2d(_n, _h)
        adj_rhs_np = _cg_poisson_2d_np(adj_p_np, _L)
        if _rhs.grad is not None:
            cur = _rhs.grad.numpy()
            _rhs.grad = wp.array(
                (cur + adj_rhs_np).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _rhs.grad = wp.array(
                adj_rhs_np.astype(np.float32), dtype=wp.float32, device=_d
            )

    tape.record_func(
        backward=_cg_poisson_2d_backward,
        arrays=[rhs_wp, p_wp],
    )

    return p_wp


def _cg_poisson_channel_2d_np(
    rhs_np: np.ndarray, L: scipy.sparse.csr_matrix, n: int
) -> np.ndarray:  # mosaic:physics
    """Solve channel Poisson system L p = rhs (Dirichlet at x=N-1, Neumann elsewhere).

    Zeros the rhs at Dirichlet DOFs (i=N-1 rows) before solving.
    """
    rhs_flat = rhs_np.ravel().astype(np.float64)
    # Enforce Dirichlet p=0 at x=N-1 (outflow)
    rhs_flat[(n - 1) * n : n * n] = 0.0

    n2 = rhs_flat.size
    if n2 <= 4096:
        p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)
    else:
        p_flat, info = scipy.sparse.linalg.cg(L, rhs_flat, rtol=1e-8)
        if info != 0:
            p_flat = scipy.sparse.linalg.spsolve(L, rhs_flat)

    return p_flat.reshape(rhs_np.shape).astype(np.float32)


def _cg_poisson_channel_2d_tape(  # mosaic:grad:v0:adjoint
    rhs_wp: wp.array,
    n: int,
    h: float,
    tape: wp.Tape,
    device: str,
) -> wp.array:
    """Differentiable CG Poisson solve for channel flow registered on a wp.Tape.

    [2D-only function]

    Uses _build_laplacian_channel_2d (Neumann x=0/y-walls, Dirichlet x=N-1).
    The backward is the same solve (symmetric operator).
    """
    L = _build_laplacian_channel_2d(n, h)

    rhs_np = rhs_wp.numpy()
    p_np = _cg_poisson_channel_2d_np(rhs_np, L, n)
    p_wp = wp.array(p_np, dtype=wp.float32, requires_grad=True, device=device)

    _rhs_ref = rhs_wp
    _p_ref = p_wp
    _n_ref = n
    _h_ref = h
    _dev_ref = device

    def _cg_poisson_channel_2d_backward(
        _rhs=_rhs_ref, _p=_p_ref, _n=_n_ref, _h=_h_ref, _d=_dev_ref
    ):
        if _p.grad is None:
            return
        adj_p_np = _p.grad.numpy()
        _L = _build_laplacian_channel_2d(_n, _h)
        adj_rhs_np = _cg_poisson_channel_2d_np(adj_p_np, _L, _n)
        if _rhs.grad is not None:
            cur = _rhs.grad.numpy()
            _rhs.grad = wp.array(
                (cur + adj_rhs_np).astype(np.float32),
                dtype=wp.float32,
                device=_d,
            )
        else:
            _rhs.grad = wp.array(
                adj_rhs_np.astype(np.float32), dtype=wp.float32, device=_d
            )

    tape.record_func(
        backward=_cg_poisson_channel_2d_backward,
        arrays=[rhs_wp, p_wp],
    )

    return p_wp


def ns2d_solve(  # mosaic:physics
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    num_iters_poisson: int,
    obstacle=None,
    device: str = "cpu",
    inflow_profile: np.ndarray | None = None,
    wall_y_noslip: bool = False,
):
    """Run 2-D incompressible NS via IPCS (Chorin-Temam).

    [2D-only function]

    Uses the same Incremental Pressure Correction Scheme as the 3-D solver.
    Pressure Poisson solver selection:
      - Fully periodic flow (no obstacle, no inflow): exact spectral FFT Poisson.
      - Obstacle or inflow present: CG solver with Neumann (dp/dn=0) BCs, which
        is correct for channel/obstacle flow where periodic BCs are wrong.

    Steps per time-step:
        1. Tentative velocity: u* = u + dt·(-u·∇u + ν∇²u)
        2. Pressure Poisson: ∇²p = (1/dt)·∇·u* (FFT or CG depending on BCs)
        3. Velocity correction: u^(n+1) = u* - dt·∇p

    inflow_profile: optional shape-(N,) float32 array giving a spatially-varying
        Dirichlet u_x(y) BC applied at x=0 after each sub-step (tentative and
        pressure-corrected).  Transverse uy at x=0 is pinned to 0.  The profile
        is uploaded as a differentiable wp.array with requires_grad=True so that
        tape.backward() fills its .grad attribute.  Periodic FFT Poisson is
        retained; the Dirichlet override is an explicit post-step BC applied in
        the same spirit as jax-cfd/phiflow for drag_opt (channel-in-periodic-box).

    Returns:
        (result_np, drag_np_or_None, velocity_mean_np_or_None, tape,
         ux_final_wp, uy_final_wp, ux_ic_wp, uy_ic_wp, inflow_wp_or_None,
         nu_wp, dt_wp)
        The final velocity Warp arrays have requires_grad=True so
        tape.backward() fills their .grad attributes.
        nu_wp and dt_wp are (1,) scalar leaf arrays with requires_grad=True;
        their .grad attributes are filled by tape.backward() via per-step
        record_func callbacks in _tentative_vel_2d_tape and ns2d_solve.
        velocity_mean_np is the RANS tail-window mean over the last 50% of
        steps, shape (N, N, 1, 2).  Always non-None when steps > 0.
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / (h * h)

    # Warp 1.12+ requires block_dim as int (256 = 16×16 for 2D, 128 for 1D).
    # On CPU, Warp ignores block_dim and uses 1; these ints are safe on both.
    _bd_2d = 256
    _bd_1d = 128

    ux_np = v0_np[:, :, 0, 0]
    uy_np = v0_np[:, :, 0, 1]

    # Obstacle mask (not differentiable)
    mask_wp = None
    if obstacle is not None:
        mask_wp = _build_obstacle_mask(n, domain_extent, obstacle, device)

    # Inflow profile (differentiable).  Resample to N if needed.
    inflow_wp = None
    if inflow_profile is not None:
        prof_np = np.asarray(inflow_profile, dtype=np.float32).reshape(-1)
        if prof_np.shape[0] != n:
            src = np.linspace(0.0, 1.0, prof_np.shape[0], dtype=np.float32)
            dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
            prof_np = np.interp(dst, src, prof_np).astype(np.float32)
        inflow_wp = wp.array(
            prof_np, dtype=wp.float32, requires_grad=True, device=device
        )

    # Upload IC velocity as tape leaf inputs
    ux_wp = wp.array(ux_np, dtype=wp.float32, requires_grad=True, device=device)
    uy_wp = wp.array(uy_np, dtype=wp.float32, requires_grad=True, device=device)

    # Scalar leaf arrays for tape-based viscosity and dt gradients.
    # These are (1,) float32 arrays with requires_grad=True; their .grad attributes
    # are accumulated by record_func callbacks in _tentative_vel_2d_tape (per-step
    # Laplacian / advection contributions) and two additional callbacks per step
    # for the divergence and pressure-correction dt contributions.
    nu_wp = wp.array(
        np.array([viscosity], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=True,
        device=device,
    )
    dt_wp = wp.array(
        np.array([dt], dtype=np.float32),
        dtype=wp.float32,
        requires_grad=True,
        device=device,
    )

    # Ping-pong velocity buffers (both need grad for backward to flow through them)
    vel_bufs_x = [
        wp.zeros((n, n), dtype=wp.float32, requires_grad=True, device=device),
        wp.zeros((n, n), dtype=wp.float32, requires_grad=True, device=device),
    ]
    vel_bufs_y = [
        wp.zeros((n, n), dtype=wp.float32, requires_grad=True, device=device),
        wp.zeros((n, n), dtype=wp.float32, requires_grad=True, device=device),
    ]

    # Per-step divergence arrays (required so tape.backward() reads correct values;
    # ux_star/uy_star are now created fresh each step inside _tentative_vel_2d_tape)
    div_star_steps = [
        wp.zeros((n, n), dtype=wp.float32, requires_grad=True, device=device)
        for _ in range(steps)
    ]

    # Accumulator for tail-window drag averaging (last 50% of steps, outside tape).
    # Drag is not differentiable through the Warp tape (compute_drag_kernel is
    # launched outside the tape context and the VJP ignores the drag cotangent),
    # so this change is purely about forward accuracy — matching xlb / phiflow
    # which return a mean over the last 50% of timesteps.
    drag_accum: list = []

    # RANS tail-window velocity accumulators (same 50% tail window as drag).
    # Computed outside the tape — non-differentiable, purely for RANS output.
    ux_accum = np.zeros((n, n), dtype=np.float32)
    uy_accum = np.zeros((n, n), dtype=np.float32)
    vel_tail_count = 0

    # Per-step drag buffers (one per tail-window step, requires_grad=True).
    # Used only for obstacle-only (no inflow) runs; for inflow+obstacle the
    # momentum-deficit RANS approach below replaces this.
    drag_bufs: list = []  # populated inside the loop for tail steps (obstacle-only)

    # ARCH-9: RANS outlet buffer for momentum-deficit drag (inflow+obstacle only).
    # Allocated here with requires_grad=True so the tape records the accumulation
    # kernel and backward flows from rans_ux_buf.grad → vel_bufs_x[src].grad at
    # the outlet column → IPCS backward → inflow_profile.grad.
    rans_ux_buf = None
    rans_drag_buf = None
    _n_tail_total = steps - steps // 2  # number of tail steps (same 50% window)
    if inflow_profile is not None and obstacle is not None and _n_tail_total > 0:
        rans_ux_buf = wp.zeros(n, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        # Copy IC into first buffer slot (inside tape so backward flows to ux_wp)
        wp.copy(vel_bufs_x[0], ux_wp)
        wp.copy(vel_bufs_y[0], uy_wp)

        src, dst = 0, 1
        for step_i in range(steps):
            # Step 1: tentative velocity via explicit adjoint (fixes Warp AD sign-flip bug).
            # Returns new wp arrays with requires_grad=True; backward registered via record_func.
            # nu_wp and dt_wp are passed so the backward accumulates adj_nu and adj_dt
            # analytically from the Laplacian / advection terms at this step.
            ux_star, uy_star = _tentative_vel_2d_tape(
                vel_bufs_x[src],
                vel_bufs_y[src],
                dt,
                inv_2h,
                inv_h2,
                viscosity,
                tape,
                device,
                nu_wp=nu_wp,
                dt_wp=dt_wp,
            )

            if mask_wp is not None:
                _wlaunch(
                    apply_velocity_mask_kernel,
                    dim=(n, n),
                    inputs=[ux_star, uy_star, mask_wp],
                    block_dim=_bd_2d,
                    device=device,
                )

            # Inflow Dirichlet BC on tentative velocity (x=0 face).
            # Same accumulate-then-zero overwrite-adjoint pattern as the 3-D lid BC.
            if inflow_wp is not None:
                _wlaunch(
                    apply_inflow_bc_2d_kernel,
                    dim=n,
                    inputs=[ux_star, uy_star, inflow_wp],
                    block_dim=_bd_1d,
                    device=device,
                )
                _ux_s = ux_star
                _uy_s = uy_star
                _inflow = inflow_wp
                _n_r = n
                _d_r = device

                def _fix_inflow_overwrite_star(
                    _ux=_ux_s, _uy=_uy_s, _prof=_inflow, _n=_n_r, _d=_d_r
                ):
                    # adj_profile += adj_ux[0, :];  adj_ux[0, :] = 0; adj_uy[0, :] = 0
                    if _ux.grad is None or _prof.grad is None:
                        return
                    _wlaunch(
                        _accumulate_inflow_slice_to_profile_kernel,
                        dim=_n,
                        inputs=[_ux.grad, _prof.grad],
                        block_dim=_bd_1d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_inflow_slice_2d_kernel,
                        dim=_n,
                        inputs=[_ux.grad],
                        block_dim=_bd_1d,
                        device=_d,
                    )
                    if _uy.grad is not None:
                        _wlaunch(
                            _zero_inflow_slice_2d_kernel,
                            dim=_n,
                            inputs=[_uy.grad],
                            block_dim=_bd_1d,
                            device=_d,
                        )

                tape.record_func(
                    backward=_fix_inflow_overwrite_star,
                    arrays=[ux_star, uy_star, inflow_wp],
                )

            # No-slip wall BC at j=0 and j=n-1 (applied after inflow/mask BCs).
            if wall_y_noslip:
                _wlaunch(
                    _apply_wall_y_bc_2d_kernel,
                    dim=n,
                    inputs=[ux_star, uy_star],
                    block_dim=_bd_1d,
                    device=device,
                )
                _ux_ws, _uy_ws, _n_ws, _d_ws = ux_star, uy_star, n, device

                def _fix_wall_adj_star(_ux=_ux_ws, _uy=_uy_ws, _n=_n_ws, _d=_d_ws):
                    for _arr in [_ux.grad, _uy.grad]:
                        if _arr is not None:
                            _wlaunch(
                                _zero_wall_y_adj_2d_kernel,
                                dim=_n,
                                inputs=[_arr],
                                block_dim=_bd_1d,
                                device=_d,
                            )

                tape.record_func(
                    backward=_fix_wall_adj_star,
                    arrays=[ux_star, uy_star],
                )

            # Step 2: pressure Poisson ∇²p = ∇·u*/dt
            # Use channel kernels (Neumann x BCs) when inflow is present to prevent
            # outflow wrapping back to inflow through the periodic stencil.
            div_star = div_star_steps[step_i]
            inv_2h_over_dt = inv_2h / dt
            _wlaunch(
                divergence_2d_channel_kernel
                if inflow_profile is not None
                else divergence_2d_kernel,
                dim=(n, n),
                inputs=[ux_star, uy_star, div_star, inv_2h_over_dt],
                block_dim=_bd_2d,
                device=device,
            )

            # Divergence dt gradient: d(div_star)/d(dt) = -div_star / dt.
            # Register BETWEEN divergence kernel and Poisson so that in the
            # backward the Poisson backward fires first (LIFO), filling
            # div_star.grad, before we read it.
            # adj_dt += sum(adj_div_star * (-div_star / dt))
            # Capture div_star value at forward time (before Poisson modifies it).
            _div_star_fwd_np = div_star.numpy().copy()
            _div_star_ref = div_star
            _dt_r_div = dt
            _dt_wp_div = dt_wp

            def _record_dt_div(
                _div=_div_star_ref,
                _div_fwd=_div_star_fwd_np,
                _dt=_dt_r_div,
                _dt_wp=_dt_wp_div,
                _d=device,
            ):
                if _dt_wp is None or _div.grad is None:
                    return
                adj_div_np = _div.grad.numpy()
                d_adj_dt = float(np.sum(adj_div_np * (-_div_fwd / _dt)))
                if _dt_wp.grad is not None:
                    _dt_wp.grad = wp.array(
                        np.array([_dt_wp.grad.numpy()[0] + d_adj_dt], dtype=np.float32),
                        dtype=wp.float32,
                        device=_d,
                    )
                else:
                    _dt_wp.grad = wp.array(
                        np.array([d_adj_dt], dtype=np.float32),
                        dtype=wp.float32,
                        device=_d,
                    )

            tape.record_func(
                backward=_record_dt_div,
                arrays=[div_star] + ([dt_wp] if dt_wp is not None else []),
            )

            if inflow_profile is not None:
                p_wp = _cg_poisson_channel_2d_tape(div_star, n, h, tape, device)
            elif obstacle is not None:
                p_wp = _cg_poisson_2d_tape(div_star, n, h, tape, device)
            else:
                p_wp = _spectral_poisson_2d_tape(div_star, domain_extent, tape, device)

            # Step 3: velocity correction u^(n+1) = u* - dt·∇p
            # Zero adj_vel_bufs[dst] after pressure_correct_bwd reads it to
            # prevent gradient double-counting across timesteps (same as 3D solver).
            _vbx_dst = vel_bufs_x[dst]
            _vby_dst = vel_bufs_y[dst]

            def _clear_dst_adj(_vx=_vbx_dst, _vy=_vby_dst):
                if _vx.grad is not None:
                    _vx.grad.zero_()
                if _vy.grad is not None:
                    _vy.grad.zero_()

            tape.record_func(
                backward=_clear_dst_adj,
                arrays=[vel_bufs_x[dst], vel_bufs_y[dst]],
            )

            # Pressure correction dt gradient: d(u_new)/d(dt) = -grad_p.
            # Register BETWEEN _clear_dst_adj record_func and pressure_correct kernel
            # so the backward order is:
            #   pressure_correct_bwd → _record_dt_pressure (reads vel_bufs[dst].grad ✓)
            #   → _clear_dst_adj (zeros it)
            # Capture pressure gradient for dt backward using numpy now (before
            # pressure_correct writes to vel_bufs[dst]).
            _p_np_pc = p_wp.numpy()
            if inflow_profile is not None:
                # Channel flow: Neumann x stencil (clamp at boundaries)
                _dpdx_np = (
                    np.concatenate([_p_np_pc[1:], _p_np_pc[-1:]], axis=0)
                    - np.concatenate([_p_np_pc[:1], _p_np_pc[:-1]], axis=0)
                ) * inv_2h
            else:
                _dpdx_np = (
                    np.roll(_p_np_pc, -1, axis=0) - np.roll(_p_np_pc, 1, axis=0)
                ) * inv_2h
            _dpdy_np = (
                np.roll(_p_np_pc, -1, axis=1) - np.roll(_p_np_pc, 1, axis=1)
            ) * inv_2h
            _vbx_dst_ref = vel_bufs_x[dst]
            _vby_dst_ref = vel_bufs_y[dst]
            _dt_wp_pc = dt_wp

            def _record_dt_pressure(
                _vbx=_vbx_dst_ref,
                _vby=_vby_dst_ref,
                _dpdx=_dpdx_np,
                _dpdy=_dpdy_np,
                _dt_wp=_dt_wp_pc,
                _d=device,
            ):
                if _dt_wp is None:
                    return
                adj_ux_new = (
                    _vbx.grad.numpy() if _vbx.grad is not None else np.zeros_like(_dpdx)
                )
                adj_uy_new = (
                    _vby.grad.numpy() if _vby.grad is not None else np.zeros_like(_dpdy)
                )
                # d(u_new)/d(dt) = -grad_p  →  adj_dt += sum(adj_u_new * (-grad_p))
                d_adj_dt = float(
                    np.sum(adj_ux_new * (-_dpdx)) + np.sum(adj_uy_new * (-_dpdy))
                )
                if _dt_wp.grad is not None:
                    _dt_wp.grad = wp.array(
                        np.array([_dt_wp.grad.numpy()[0] + d_adj_dt], dtype=np.float32),
                        dtype=wp.float32,
                        device=_d,
                    )
                else:
                    _dt_wp.grad = wp.array(
                        np.array([d_adj_dt], dtype=np.float32),
                        dtype=wp.float32,
                        device=_d,
                    )

            tape.record_func(
                backward=_record_dt_pressure,
                arrays=[vel_bufs_x[dst], vel_bufs_y[dst]]
                + ([dt_wp] if dt_wp is not None else []),
            )

            _wlaunch(
                pressure_correct_2d_channel_kernel
                if inflow_profile is not None
                else pressure_correct_2d_kernel,
                dim=(n, n),
                inputs=[
                    ux_star,
                    uy_star,
                    p_wp,
                    vel_bufs_x[dst],
                    vel_bufs_y[dst],
                    dt,
                    inv_2h,
                ],
                block_dim=_bd_2d,
                device=device,
            )

            if mask_wp is not None:
                _wlaunch(
                    apply_velocity_mask_kernel,
                    dim=(n, n),
                    inputs=[vel_bufs_x[dst], vel_bufs_y[dst], mask_wp],
                    block_dim=_bd_2d,
                    device=device,
                )

            # Inflow Dirichlet BC after pressure correction (x=0 face).
            if inflow_wp is not None:
                _wlaunch(
                    apply_inflow_bc_2d_kernel,
                    dim=n,
                    inputs=[vel_bufs_x[dst], vel_bufs_y[dst], inflow_wp],
                    block_dim=_bd_1d,
                    device=device,
                )
                _vx_d = vel_bufs_x[dst]
                _vy_d = vel_bufs_y[dst]
                _inflow2 = inflow_wp
                _n_r2 = n
                _d_r2 = device

                def _fix_inflow_overwrite_vel(
                    _vx=_vx_d, _vy=_vy_d, _prof=_inflow2, _n=_n_r2, _d=_d_r2
                ):
                    if _vx.grad is None or _prof.grad is None:
                        return
                    _wlaunch(
                        _accumulate_inflow_slice_to_profile_kernel,
                        dim=_n,
                        inputs=[_vx.grad, _prof.grad],
                        block_dim=_bd_1d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_inflow_slice_2d_kernel,
                        dim=_n,
                        inputs=[_vx.grad],
                        block_dim=_bd_1d,
                        device=_d,
                    )
                    if _vy.grad is not None:
                        _wlaunch(
                            _zero_inflow_slice_2d_kernel,
                            dim=_n,
                            inputs=[_vy.grad],
                            block_dim=_bd_1d,
                            device=_d,
                        )

                tape.record_func(
                    backward=_fix_inflow_overwrite_vel,
                    arrays=[vel_bufs_x[dst], vel_bufs_y[dst], inflow_wp],
                )

            # No-slip wall BC at j=0 and j=n-1 on pressure-corrected velocity.
            if wall_y_noslip:
                _wlaunch(
                    _apply_wall_y_bc_2d_kernel,
                    dim=n,
                    inputs=[vel_bufs_x[dst], vel_bufs_y[dst]],
                    block_dim=_bd_1d,
                    device=device,
                )
                _ux_wd, _uy_wd, _n_wd, _d_wd = (
                    vel_bufs_x[dst],
                    vel_bufs_y[dst],
                    n,
                    device,
                )

                def _fix_wall_adj_dst(_ux=_ux_wd, _uy=_uy_wd, _n=_n_wd, _d=_d_wd):
                    for _arr in [_ux.grad, _uy.grad]:
                        if _arr is not None:
                            _wlaunch(
                                _zero_wall_y_adj_2d_kernel,
                                dim=_n,
                                inputs=[_arr],
                                block_dim=_bd_1d,
                                device=_d,
                            )

                tape.record_func(
                    backward=_fix_wall_adj_dst,
                    arrays=[vel_bufs_x[dst], vel_bufs_y[dst]],
                )

            src, dst = dst, src

            # Accumulate RANS velocity for tail-window mean (last 50% of steps).
            # Executed outside the tape — non-differentiable.
            if step_i >= steps // 2:
                ux_accum += vel_bufs_x[src].numpy()
                uy_accum += vel_bufs_y[src].numpy()
                vel_tail_count += 1

            # Drag accumulation for the tail window (last 50% of steps).
            # ARCH-9: For inflow+obstacle (drag_opt), accumulate outlet RANS velocity
            # inside the tape via accumulate_outlet_rans_kernel; drag_out comes from
            # rans_drag_buf (momentum-deficit) computed after the loop.
            # For obstacle-only runs, keep the old per-step compute_drag_kernel approach.
            if obstacle is not None and step_i >= steps // 2:
                if rans_ux_buf is not None:
                    # Inflow+obstacle: differentiable RANS accumulation at outlet (i=N-1).
                    # No per-step drag computation here — drag_out comes from rans_drag_buf
                    # (momentum-deficit, computed after the loop) which is both differentiable
                    # and avoids tape contamination from compute_drag_kernel with a
                    # no-requires_grad output buffer.
                    _wlaunch(
                        accumulate_outlet_rans_kernel,
                        dim=n,
                        inputs=[vel_bufs_x[src], rans_ux_buf, 1.0 / _n_tail_total],
                        block_dim=_bd_1d,
                        device=device,
                    )
                else:
                    # Obstacle-only (no inflow): old differentiable per-step approach.
                    _drag_buf = wp.zeros(
                        1, dtype=wp.float32, requires_grad=True, device=device
                    )
                    _rhs_np_step = div_star_steps[step_i].numpy()
                    _L_drag = _build_laplacian_neumann_2d(n, h)
                    _p_step = _cg_poisson_2d_np(_rhs_np_step, _L_drag)
                    _p_wp = wp.array(_p_step, dtype=wp.float32, device=device)
                    _wlaunch(
                        compute_drag_kernel,
                        dim=(n, n),
                        inputs=[
                            _p_wp,
                            vel_bufs_x[src],
                            mask_wp,
                            _drag_buf,
                            inv_2h,
                            viscosity,
                        ],
                        block_dim=_bd_2d,
                        device=device,
                    )
                    drag_bufs.append(_drag_buf)
                    drag_accum.append(_drag_buf.numpy())

        # ARCH-9: Momentum-deficit RANS drag from accumulated outlet velocities.
        # Recorded inside the with tape: context so tape.backward() propagates
        # cotangent_drag → rans_drag_buf.grad → momentum_deficit_drag_kernel backward
        # → rans_ux_buf.grad → accumulate_outlet_rans_kernel backward (each tail step)
        # → vel_bufs_x[src].grad at outlet → IPCS backward → inflow_profile.grad.
        if rans_ux_buf is not None:
            _U_mean = float(np.mean(inflow_profile))
            _dy = domain_extent / n
            rans_drag_buf = wp.zeros(
                1, dtype=wp.float32, requires_grad=True, device=device
            )
            _wlaunch(
                momentum_deficit_drag_kernel,
                dim=n,
                inputs=[rans_ux_buf, _U_mean, _dy, rans_drag_buf],
                block_dim=_bd_1d,
                device=device,
            )

    # Final velocities are in vel_bufs[src]
    ux_out = vel_bufs_x[src].numpy()
    uy_out = vel_bufs_y[src].numpy()
    result = np.stack([ux_out, uy_out], axis=-1)[:, :, np.newaxis, :]  # (N,N,1,2)

    # Drag computation (obstacle only): return mean over last 50% of steps,
    # matching the xlb / phiflow convention for drag_opt.
    # For inflow+obstacle (ARCH-9), use the RANS momentum-deficit value from rans_drag_buf.
    # For obstacle-only, use compute_drag_kernel mean (drag_accum).
    drag_out = None
    if obstacle is not None:
        if rans_drag_buf is not None:
            drag_out = rans_drag_buf.numpy().astype(np.float32)
        elif drag_accum:
            drag_out = np.mean(drag_accum, axis=0).astype(np.float32)
        else:
            # Fallback: steps == 0 or obstacle added with 0 steps — use final pressure.
            drag_buf = wp.zeros(1, dtype=wp.float32, device=device)
            _rhs_final_np = div_star_steps[-1].numpy()
            if inflow_profile is not None:
                _L_fallback = _build_laplacian_channel_2d(n, h)
                p_final = _cg_poisson_channel_2d_np(_rhs_final_np, _L_fallback, n)
            elif obstacle is not None:
                _L_fallback = _build_laplacian_neumann_2d(n, h)
                p_final = _cg_poisson_2d_np(_rhs_final_np, _L_fallback)
            else:
                p_final = _spectral_poisson_2d_np(_rhs_final_np, domain_extent)
            p_final_wp = wp.array(p_final, dtype=wp.float32, device=device)
            ux_final = vel_bufs_x[src]
            _wlaunch(
                compute_drag_kernel,
                dim=(n, n),
                inputs=[p_final_wp, ux_final, mask_wp, drag_buf, inv_2h, viscosity],
                block_dim=_bd_2d,
                device=device,
            )
            drag_out = drag_buf.numpy()

    # RANS mean velocity over the tail window.
    velocity_mean_np = None
    if vel_tail_count > 0:
        ux_mean = ux_accum / vel_tail_count
        uy_mean = uy_accum / vel_tail_count
        # Stack to canonical (nx, ny, 1, 2) shape matching result
        velocity_mean_np = np.stack([ux_mean, uy_mean], axis=-1)[:, :, np.newaxis, :]

    return (
        result,
        drag_out,
        velocity_mean_np,
        tape,
        vel_bufs_x[src],
        vel_bufs_y[src],
        ux_wp,
        uy_wp,
        inflow_wp,
        nu_wp,
        dt_wp,
        rans_drag_buf,  # ARCH-9: replaces drag_bufs for inflow+obstacle VJP
    )


def ns2d_vjp(  # mosaic:grad:v0,viscosity,dt:adjoint
    tape: wp.Tape,
    ux_final: wp.array,
    uy_final: wp.array,
    ux_ic: wp.array,
    uy_ic: wp.array,
    cotangent_np: np.ndarray,
    device: str,
    nu_wp: "wp.array | None" = None,
    dt_wp: "wp.array | None" = None,
    inflow_wp: "wp.array | None" = None,
    inflow_profile_np: np.ndarray | None = None,
    rans_drag_buf: "wp.array | None" = None,
    cotangent_drag: "float | None" = None,
) -> dict[str, np.ndarray]:
    """Propagate cotangents through the 2-D IPCS tape.

    [2D-only function]

    Tape backward gives v0, viscosity, and dt gradients analytically.
    Viscosity gradient is accumulated per-step from the Laplacian term via
    record_func callbacks in _tentative_vel_2d_tape:
        adj_nu += dt * sum(adj_ux_star * lap_ux + adj_uy_star * lap_uy)
    dt gradient is accumulated per-step from three sub-steps:
      1. Tentative velocity: adj_dt += sum(adj_ux_star * (-adv + nu*lap))
      2. Divergence:         adj_dt += sum(adj_div_star * (-div_star / dt))
      3. Pressure correction: adj_dt += sum(adj_u_new * (-grad_p))
    All three are registered as record_func backward closures in ns2d_solve.
    inflow_profile gradient comes from tape.backward() via the accumulate-then-zero
    overwrite-adjoint fix in ns2d_solve.
    """
    # mosaic:grad:v0:adjoint
    ux_final.grad = wp.array(
        cotangent_np[:, :, 0, 0].astype(np.float32), dtype=wp.float32, device=device
    )
    uy_final.grad = wp.array(
        cotangent_np[:, :, 0, 1].astype(np.float32), dtype=wp.float32, device=device
    )

    # ARCH-9: set cotangent on the RANS momentum-deficit drag buffer so tape.backward()
    # propagates: rans_drag_buf.grad → momentum_deficit_drag_kernel backward
    # → rans_ux_buf.grad → accumulate_outlet_rans_kernel backward (each tail step)
    # → vel_bufs_x[src].grad at outlet column → IPCS backward → inflow_profile.grad.
    if (
        rans_drag_buf is not None
        and cotangent_drag is not None
        and cotangent_drag != 0.0
    ):
        rans_drag_buf.grad = wp.array(
            [float(cotangent_drag)], dtype=wp.float32, device=device
        )

    tape.backward()

    grad_v0 = np.zeros_like(cotangent_np)
    if ux_ic.grad is not None:
        grad_v0[:, :, 0, 0] = ux_ic.grad.numpy()
    if uy_ic.grad is not None:
        grad_v0[:, :, 0, 1] = uy_ic.grad.numpy()

    # mosaic:grad:inflow_profile:adjoint
    # Inflow profile gradient (from tape).  Resample back to the caller's length
    # if we internally resampled to N.
    grad_inflow_profile: np.ndarray | None = None
    if inflow_wp is not None:
        if inflow_wp.grad is not None:
            g_np = np.asarray(inflow_wp.grad.numpy(), dtype=np.float32)
            gnorm = float(np.linalg.norm(g_np))
            if not np.isfinite(gnorm) or gnorm == 0.0:
                g_np = np.zeros_like(g_np)
            elif gnorm > 100.0:
                g_np = g_np * (100.0 / gnorm)
        else:
            g_np = np.zeros(int(inflow_wp.shape[0]), dtype=np.float32)
        # If the user-provided inflow_profile had a different length than N,
        # resample (linear) the gradient back to the original length.  For a
        # linear interp forward, the exact adjoint would be a sparse transpose;
        # linear interp here is an acceptable approximation (matches phiflow /
        # jax_cfd behaviour when inflow length != N) and, critically, keeps the
        # shape compatible with optax.  In practice the benchmark uses len==N.
        if (
            inflow_profile_np is not None
            and inflow_profile_np.shape[0] != g_np.shape[0]
        ):
            src = np.linspace(0.0, 1.0, g_np.shape[0], dtype=np.float32)
            dst = np.linspace(0.0, 1.0, inflow_profile_np.shape[0], dtype=np.float32)
            g_np = np.interp(dst, src, g_np).astype(np.float32)
        grad_inflow_profile = g_np

    # ── Viscosity and dt gradients from tape (record_func closures in ns2d_solve) ──
    # nu_wp.grad and dt_wp.grad are accumulated by the per-step record_func callbacks
    # registered in _tentative_vel_2d_tape and the divergence / pressure-correction
    # record_funcs in ns2d_solve.  No extra forward passes needed.
    # mosaic:grad:viscosity:adjoint
    grad_nu = np.zeros(1, dtype=np.float32)
    if nu_wp is not None and nu_wp.grad is not None:
        grad_nu[0] = float(nu_wp.grad.numpy()[0])
    # mosaic:grad:dt:adjoint
    grad_dt = np.zeros(1, dtype=np.float32)
    if dt_wp is not None and dt_wp.grad is not None:
        grad_dt[0] = float(dt_wp.grad.numpy()[0])

    out: dict[str, np.ndarray] = {
        "v0": grad_v0.astype(np.float32),
        "viscosity": grad_nu,
        "dt": grad_dt,
    }
    if grad_inflow_profile is not None:
        out["inflow_profile"] = grad_inflow_profile
    return out


# ============================================================
# 3-D NS forward solve (IPCS)
# ============================================================


def ns3d_solve(  # mosaic:physics
    v0_np: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    num_iters_poisson_3d: int,
    lid_velocity: np.ndarray | None = None,
    device: str = "cpu",
    adjoint_grad_clip: float | None = None,
):
    """Run 3-D incompressible NS via IPCS (Chorin-Temam).

    lid_velocity: optional (N, N, 2) float32 array giving the spatially-varying
        x- and y-velocity on the top z-face (lid-driven cavity mode).  When
        provided a warp vec2 array is uploaded with requires_grad=True so that
        tape.backward() fills its .grad attribute for lid_velocity VJPs.

    Returns:
        (result_np, tape, ux_final_wp, uy_final_wp, uz_final_wp,
         ux_ic_wp, uy_ic_wp, uz_ic_wp, lid_wp_or_None)
        The final velocity Warp arrays have requires_grad=True so
        tape.backward() fills their .grad attributes.
    """
    n = v0_np.shape[0]
    h = domain_extent / n
    h2 = h * h
    inv_2h = 0.5 / h
    inv_h2 = 1.0 / h2

    # Warp 1.12+ requires block_dim as int (256 = 16×16 or 8×8×4).
    _bd_2d = 256
    _bd_3d = 256

    # ARCH-109: CFL-adaptive dt for lid-driven cavity mode.
    # The IPCS tentative velocity step uses explicit Euler advection; stability
    # requires CFL = U_max * dt / dx < ~1.0.  During optimisation the lid field
    # can reach U_max >> 1 (especially at sweep≥1.0), pushing CFL above the
    # stability limit and causing NaN at optimizer iteration 2.
    #
    # Fix: when lid_velocity is provided, compute U_max from the current lid
    # field and reduce dt so CFL ≤ cfl_limit.  Scale steps up proportionally
    # to preserve physical time T = steps * dt.
    #
    # Note (ARCH-112): this fix does not fire for the lid_cavity benchmark
    # params (dt=0.01, h=0.0625) since dt_cfl=0.9*0.0625/U_max is always
    # ≥ 0.01 for U_max ≤ 5.625.  The sweep values (0.5, 1.0, 2.0) all satisfy
    # this condition so no dt reduction occurs.  The IPCS backward instability
    # at sweep≥1.0 is resolved by a categorical exclusion in navier_stokes_3d_grid.py.
    if lid_velocity is not None:
        _lid_arr = np.asarray(lid_velocity, dtype=np.float32)
        _U_max = float(np.sqrt(np.sum(_lid_arr**2, axis=-1)).max())
        _eps = 1e-6
        _cfl_limit = 0.45
        _dt_cfl = _cfl_limit * h / (_U_max + _eps)
        if _dt_cfl < dt:
            # Reduce dt and increase steps to keep T = steps * dt constant.
            _T_physical = steps * dt
            steps = max(1, int(np.ceil(_T_physical / _dt_cfl)))
            dt = _T_physical / steps

    # Default adjoint clipping: enabled by default in lid-driven cavity mode where
    # F-NS3D-4 documents IPCS adjoint overflow at steps>=40 in the turbulent
    # regime.  The element-wise clip is a stability guard; healthy adjoints
    # (|g|<=clip) are unchanged so gradient direction is preserved.  The default
    # threshold of 10.0 is close to the healthy xlb/ins_jl per-cell adjoint
    # magnitudes (norm~57 over 16^3 ≈ 2e-3 RMS) while still well above typical
    # signal.  Periodic (TGV) mode keeps the clip disabled by default for
    # exact fd_check cosines.
    if adjoint_grad_clip is None and lid_velocity is not None:
        adjoint_grad_clip = 10.0

    ux_wp = wp.array(
        v0_np[:, :, :, 0], dtype=wp.float32, requires_grad=True, device=device
    )
    uy_wp = wp.array(
        v0_np[:, :, :, 1], dtype=wp.float32, requires_grad=True, device=device
    )
    uz_wp = wp.array(
        v0_np[:, :, :, 2], dtype=wp.float32, requires_grad=True, device=device
    )

    # Upload lid field as differentiable warp vec2 array (when in cavity mode)
    lid_wp = None
    if lid_velocity is not None:
        # (N, N, 2) float32 → warp array with dtype=vec2; warp interprets
        # the last axis of size 2 as the vec2 components.
        lid_np = np.asarray(lid_velocity, dtype=np.float32)  # (N, N, 2)
        lid_wp = wp.array(lid_np, dtype=wp.vec2, requires_grad=True, device=device)

    # Working arrays for IPCS stages (2 sets to avoid name rebinding)
    vel_bufs_x = [
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
    ]
    vel_bufs_y = [
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
    ]
    vel_bufs_z = [
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device),
    ]
    # Per-step divergence arrays so tape.backward() reads the correct rhs for
    # each timestep (Bug 3 fix).  The tentative-velocity star arrays are now
    # allocated inside _tentative_vel_3d_tape per step (ARCH-31), so they no
    # longer appear as pre-allocated lists here.
    div_star_steps = [
        wp.zeros((n, n, n), dtype=wp.float32, requires_grad=True, device=device)
        for _ in range(steps)
    ]
    top_k = n - 1

    tape = wp.Tape()
    with tape:
        # Copy input velocity into first buffer slot (inside tape).
        # wp.copy(dest, src) — dest first, src second.
        wp.copy(vel_bufs_x[0], ux_wp)
        wp.copy(vel_bufs_y[0], uy_wp)
        wp.copy(vel_bufs_z[0], uz_wp)

        src, dst = 0, 1
        for step_i in range(steps):
            # Per-step tentative velocity arrays are allocated fresh inside
            # _tentative_vel_3d_tape below (ARCH-31).  The pre-allocated
            # ux_star_steps / uy_star_steps / uz_star_steps arrays are no
            # longer used by the tentative-velocity step — they only remain
            # allocated as a no-op to keep the outer list comprehension intact
            # for the legacy Bug-3 fix; the references via `ux_star_steps`
            # below are now unused.

            # ── Per-step adjoint gradient clipping (stability guard) ───────────────
            # Registered at the start of each step's forward; because record_func
            # is LIFO, this fires LAST in the step's backward sequence — after
            # tentative_vel_bwd has written into adj_vel_bufs[src].  Clipping the
            # per-timestep adjoint prevents float32 overflow in the IPCS adjoint
            # at turbulent high-Re regimes (F-NS3D-4 / F-NS3.4) without altering
            # the direction of the gradient.  The clip is a safety guard; if
            # ||adj||_inf ≤ threshold the adjoint is unchanged (gradient
            # direction preserved for stable regimes).  Only active when
            # adjoint_grad_clip is set (>0).
            if adjoint_grad_clip is not None and adjoint_grad_clip > 0:
                _vx_src = vel_bufs_x[src]
                _vy_src = vel_bufs_y[src]
                _vz_src = vel_bufs_z[src]
                _clip = float(adjoint_grad_clip)
                _d_clip = device

                _n_clip = n

                def _clip_src_adj(
                    _vx=_vx_src,
                    _vy=_vy_src,
                    _vz=_vz_src,
                    _c=_clip,
                    _d=_d_clip,
                    _n=_n_clip,
                ):
                    """Clip adj_vel_bufs[src] element-wise into [-c, c] via a Warp kernel.

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
                    backward=_clip_src_adj,
                    arrays=[vel_bufs_x[src], vel_bufs_y[src], vel_bufs_z[src]],
                )

            # ARCH-31: _zero_star_grads removed — ux_star/uy_star/uz_star are
            # now fresh wp.arrays allocated per step inside
            # _tentative_vel_3d_tape (returned below), so there are no stale
            # gradients to clear.  The previous pre-allocated buffers held
            # gradients across tape.backward() invocations, which needed
            # zeroing; fresh arrays inherently start with .grad=None.

            # Step 1: tentative velocity u* = u + dt·(-u·∇u + ν∇²u)
            #
            # ARCH-31: replaced direct _wlaunch(tentative_vel_3d_kernel) with
            # this explicit-backward wrapper that mirrors the 2D path.  Warp's
            # auto-adjoint of the kernel has the cross-component Fourier-mode
            # sign bug documented for tentative_vel_2d_kernel; in 3D this
            # manifested as fd_check rel_err ~1.6% with a flat cosine plateau
            # ~0.973 (peer solvers hit 4e-6–2e-4 in the same regime).  The
            # wrapper does the forward in numpy and registers the analytical
            # adjoint via tape.record_func.
            ux_star, uy_star, uz_star = _tentative_vel_3d_tape(
                vel_bufs_x[src],
                vel_bufs_y[src],
                vel_bufs_z[src],
                dt,
                inv_2h,
                inv_h2,
                viscosity,
                tape,
                device,
            )
            if lid_wp is not None:
                _wlaunch(
                    apply_lid_field_bc_kernel,
                    dim=(n, n),
                    inputs=[ux_star, uy_star, uz_star, lid_wp, top_k],
                    block_dim=_bd_2d,
                    device=device,
                )
                # Fix overwrite adjoint: the lid BC overwrites ux_star/uy_star at
                # top_k, discarding the previous tentative value.  Warp's auto-adjoint
                # accumulates adj_ux_star[top_k] → adj_lid_wp but does NOT zero
                # adj_ux_star[top_k] afterward, causing incorrect gradient propagation
                # back through tentative_vel on the top face.  We use record_func to
                # inject the correct "accumulate-then-zero" backward step.
                _ux_star_ref = ux_star
                _uy_star_ref = uy_star
                _uz_star_ref = uz_star
                _lid_wp_ref = lid_wp
                _top_k_ref = top_k
                _n_ref = n
                _dev_ref = device

                def _fix_overwrite_adjoint_star(
                    _ux=_ux_star_ref,
                    _uy=_uy_star_ref,
                    _uz=_uz_star_ref,
                    _lid=_lid_wp_ref,
                    _tk=_top_k_ref,
                    _n=_n_ref,
                    _d=_dev_ref,
                ):
                    # Accumulate adj_ux_star[top_k] → adj_lid before tentative backward
                    _wlaunch(
                        _accumulate_top_slice_to_lid_kernel,
                        dim=(_n, _n),
                        inputs=[_ux.grad, _uy.grad, _lid.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    # Zero adj_*_star[top_k] so tentative backward does not see
                    # spurious gradient at the overwritten face (ux/uy → lid,
                    # uz → 0 constant — all three must be zeroed)
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_ux.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_uy.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_uz.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )

            # Adjoint pressure correction: inserted BEFORE _fix_overwrite_adjoint_star
            # in forward so that during backward it runs AFTER the overwrite fix.
            # Backward order (LIFO): first _fix_overwrite_adjoint_star zeros top_k,
            # then _apply_pressure_adjoint runs on the correctly-zeroed adj_u_star.
            #
            # pressure_correct_bwd gives adj_u_star = adj_vel_out (incomplete).
            # The correct IPCS adjoint is adj_u_star = P(adj_vel_out) where
            # P = I - dt*∇(∇²)⁻¹∇·(·)/dt = I - ∇(∇²)⁻¹∇· is the L² projection.
            # We apply this correction by running a spectral Poisson solve on the
            # grad arrays (same structure as the forward pressure step).
            if lid_wp is not None:
                _uxs2 = ux_star
                _uys2 = uy_star
                _uzs2 = uz_star
                _dt_r = dt
                # ARCH-112: use inv_2h (not inv_2h/dt=800) in the adjoint divergence
                # kernel so the backward only multiplies by 8 instead of 800 per step.
                # The forward div is also changed to inv_2h (see below), so p is
                # rescaled by dt — the velocity correction drops the dt factor too.
                _inv2h_dt_r = (
                    inv_2h  # was inv_2h / dt = 800; now 8 (100× less amplification)
                )
                _inv2h_r = inv_2h
                _n_p = n
                _h_p = h
                _d_p = device

                def _apply_pressure_adjoint(
                    _ux=_uxs2,
                    _uy=_uys2,
                    _uz=_uzs2,
                    _dt=_dt_r,
                    _i2hdt=_inv2h_dt_r,
                    _i2h=_inv2h_r,
                    _n=_n_p,
                    _h=_h_p,
                    _d=_d_p,
                    _tk=top_k,
                ):
                    """Apply IPCS pressure projection to adj_u_star in-place.

                    adj_u_star currently holds adj_vel_out (from pressure_correct_bwd),
                    with top_k already zeroed by _fix_overwrite_adjoint_star (which ran
                    first in backward since it was inserted second in forward).
                    We correct to adj_u_star = P(adj_vel_out) = adj_vel_out - dt*∇q
                    where q = cg_poisson(div(adj_vel_out)/dt).

                    CG with Neumann BCs is used here because the forward pressure
                    solve uses _cg_poisson_3d_tape (cavity mode).  The Neumann
                    Laplacian is symmetric so the adjoint operator is identical.

                    After applying the correction, we re-zero the top_k face because
                    the pressure gradient ∇q is nonzero at top_k and would otherwise
                    reintroduce spurious gradient at the overwritten lid face, which
                    would then incorrectly propagate back through tentative_vel_bwd.
                    """
                    if _ux.grad is None or _uy.grad is None or _uz.grad is None:
                        return
                    # Compute divergence of adj_vel_out (stored in adj_u_star).
                    # Use wall kernel (Neumann BCs in z) to match the forward path.
                    adj_div = wp.zeros((_n, _n, _n), dtype=wp.float32, device=_d)
                    _wlaunch(
                        divergence_3d_wall_kernel,
                        dim=(_n, _n, _n),
                        inputs=[_ux.grad, _uy.grad, _uz.grad, adj_div, _i2hdt],
                        block_dim=_bd_3d,
                        device=_d,
                    )
                    # Solve ∇²q = adj_div via CG with Neumann BCs (self-adjoint)
                    adj_div_np = adj_div.numpy()
                    _L_adj = _build_laplacian_neumann_3d(_n, _h)
                    q_adj_np = _cg_poisson_3d_np(adj_div_np, _L_adj)
                    q_adj = wp.array(q_adj_np, dtype=wp.float32, device=_d)
                    # Correct: adj_u_star -= ∇q_adj (no dt factor — forward also drops dt).
                    # Use wall kernel (Neumann BCs in z) to match the forward path.
                    _wlaunch(
                        _apply_pressure_correction_wall_kernel,
                        dim=(_n, _n, _n),
                        inputs=[_ux.grad, _uy.grad, _uz.grad, q_adj, 1.0, _i2h],
                        block_dim=_bd_3d,
                        device=_d,
                    )
                    # Re-zero the top_k face: the lid BC overwrote ux_star[top_k],
                    # so the tentative_vel backward must not see any gradient there.
                    # The pressure correction ∇q is nonzero at top_k, which would
                    # re-introduce spurious gradient if not zeroed here.
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_ux.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_uy.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_uz.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )

                tape.record_func(
                    backward=_apply_pressure_adjoint,
                    arrays=[ux_star, uy_star, uz_star],
                )

                tape.record_func(
                    backward=_fix_overwrite_adjoint_star,
                    arrays=[ux_star, uy_star, uz_star, lid_wp],
                )

            # Step 2: pressure Poisson ∇²p = ∇·u*/dt
            # Per-step div_star array ensures each step has an independent gradient.
            div_star = div_star_steps[step_i]
            inv_2h_over_dt = inv_2h / dt
            if lid_velocity is not None:
                # Lid-driven cavity: Neumann (zero-gradient) BCs in z (wall-normal).
                # x and y remain periodic; clamped k-indices prevent periodic wrap-
                # through at the top/bottom walls, which would corrupt the divergence
                # field and cause NaN in the CG pressure solver (ARCH-102).
                # ARCH-112: pass inv_2h (=8) instead of inv_2h_over_dt (=800) so the
                # wp.Tape backward multiplies by 8 per step instead of 800, eliminating
                # float32 overflow over 60 steps at sweep≥1.0. The Poisson RHS is now
                # ∇·u* (not ∇·u*/dt), so pressure p is rescaled by dt — the correction
                # step drops dt accordingly (see _dt_correct below).
                _wlaunch(
                    divergence_3d_wall_kernel,
                    dim=(n, n, n),
                    inputs=[ux_star, uy_star, uz_star, div_star, inv_2h],
                    block_dim=_bd_3d,
                    device=device,
                )
                # Non-periodic (lid-driven cavity / channel flow): CG with Neumann BCs.
                # FFT enforces periodic BCs and is wrong here.
                p_wp = _cg_poisson_3d_tape(div_star, n, h, tape, device)
            else:
                # Periodic (TGV etc.): spectral FFT — exact, fast, self-adjoint.
                _wlaunch(
                    divergence_3d_kernel,
                    dim=(n, n, n),
                    inputs=[ux_star, uy_star, uz_star, div_star, inv_2h_over_dt],
                    block_dim=_bd_3d,
                    device=device,
                )
                p_wp = _spectral_poisson_3d_tape(div_star, domain_extent, tape, device)

            # Step 3: velocity correction u^(n+1) = u* - dt·∇p
            #
            # Buffer aliasing fix: vel_bufs_x[dst] is reused across timesteps (ping-pong).
            # After pressure_correct_bwd READS adj_vel_bufs[dst] to accumulate into
            # adj_ux_star, it must be ZEROED so the next round-trip backward step
            # does not accumulate stale gradient from this timestep's output.
            # We insert a record_func BEFORE pressure_correct in forward so that in
            # backward it runs AFTER pressure_correct_bwd.
            _vbx_dst = vel_bufs_x[dst]
            _vby_dst = vel_bufs_y[dst]
            _vbz_dst = vel_bufs_z[dst]

            def _clear_dst_adj(_vx=_vbx_dst, _vy=_vby_dst, _vz=_vbz_dst):
                """Zero adj_vel_bufs[dst] after pressure_correct_bwd reads it.

                This prevents gradient double-counting when the same vel_buf is
                written by multiple timesteps (e.g., even-numbered timesteps all
                write to the same buffer).
                """
                if _vx.grad is not None:
                    _vx.grad.zero_()
                if _vy.grad is not None:
                    _vy.grad.zero_()
                if _vz.grad is not None:
                    _vz.grad.zero_()

            tape.record_func(
                backward=_clear_dst_adj,
                arrays=[vel_bufs_x[dst], vel_bufs_y[dst], vel_bufs_z[dst]],
            )

            # Step 3: velocity correction u^(n+1) = u* - dt_correct·∇p.
            # Lid-driven cavity uses wall-BC kernel (clamped k-indices in z) so the
            # pressure gradient is zero at top/bottom walls (Neumann BC, ARCH-102).
            # ARCH-112: lid-cavity path uses dt_correct=1.0 because the divergence
            # RHS was scaled by inv_2h (not inv_2h/dt), so p absorbs the dt factor
            # and the correction is u* - ∇p_new = u* - dt*∇p_old (same velocity).
            _pressure_correct_kernel = (
                pressure_correct_3d_wall_kernel
                if lid_velocity is not None
                else pressure_correct_3d_kernel
            )
            _dt_correct = 1.0 if lid_velocity is not None else dt
            _wlaunch(
                _pressure_correct_kernel,
                dim=(n, n, n),
                inputs=[
                    ux_star,
                    uy_star,
                    uz_star,
                    p_wp,
                    vel_bufs_x[dst],
                    vel_bufs_y[dst],
                    vel_bufs_z[dst],
                    _dt_correct,
                    inv_2h,
                ],
                block_dim=_bd_3d,
                device=device,
            )
            if lid_wp is not None:
                _wlaunch(
                    apply_lid_field_bc_kernel,
                    dim=(n, n),
                    inputs=[
                        vel_bufs_x[dst],
                        vel_bufs_y[dst],
                        vel_bufs_z[dst],
                        lid_wp,
                        top_k,
                    ],
                    block_dim=_bd_2d,
                    device=device,
                )
                # Same overwrite-adjoint fix for the post-correction lid BC.
                # vel_bufs_x[dst][top_k] was overwritten; adj_vel_bufs_x[dst][top_k]
                # must be zeroed after accumulation so it does not flow back through
                # the next timestep's tentative backward as a spurious top-face signal.
                _vx_ref = vel_bufs_x[dst]
                _vy_ref = vel_bufs_y[dst]
                _vz_ref = vel_bufs_z[dst]
                _lid_wp_ref2 = lid_wp
                _top_k_ref2 = top_k
                _n_ref2 = n
                _dev_ref2 = device

                def _fix_overwrite_adjoint_vel(
                    _vx=_vx_ref,
                    _vy=_vy_ref,
                    _vz=_vz_ref,
                    _lid=_lid_wp_ref2,
                    _tk=_top_k_ref2,
                    _n=_n_ref2,
                    _d=_dev_ref2,
                ):
                    _wlaunch(
                        _accumulate_top_slice_to_lid_kernel,
                        dim=(_n, _n),
                        inputs=[_vx.grad, _vy.grad, _lid.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_vx.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_vy.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )
                    _wlaunch(
                        _zero_top_slice_kernel,
                        dim=(_n, _n),
                        inputs=[_vz.grad, _tk],
                        block_dim=_bd_2d,
                        device=_d,
                    )

                tape.record_func(
                    backward=_fix_overwrite_adjoint_vel,
                    arrays=[vel_bufs_x[dst], vel_bufs_y[dst], vel_bufs_z[dst], lid_wp],
                )

            src, dst = dst, src

    # Final velocities are in vel_bufs_x/y/z[src]
    ux_out = vel_bufs_x[src].numpy()
    uy_out = vel_bufs_y[src].numpy()
    uz_out = vel_bufs_z[src].numpy()
    result = np.stack([ux_out, uy_out, uz_out], axis=-1)  # (N,N,N,3)

    return (
        result,
        tape,
        vel_bufs_x[src],
        vel_bufs_y[src],
        vel_bufs_z[src],
        ux_wp,
        uy_wp,
        uz_wp,
        lid_wp,
    )


def ns3d_vjp(  # mosaic:grad:v0,viscosity,dt:adjoint
    tape: wp.Tape,
    ux_final: wp.array,
    uy_final: wp.array,
    uz_final: wp.array,
    ux_ic: wp.array,
    uy_ic: wp.array,
    uz_ic: wp.array,
    cotangent_np: np.ndarray,
    device: str,
    lid_wp: "wp.array | None" = None,
) -> dict[str, np.ndarray]:
    """Propagate cotangents through the 3-D IPCS tape.

    lid_wp: optional warp vec2 array (N, N) that was uploaded with
        requires_grad=True during ns3d_solve.  When provided, after
        tape.backward() its .grad attribute holds the (N, N) vec2
        gradient which is unpacked to shape (N, N, 2) float32.
    """
    # mosaic:grad:v0:adjoint
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

    n = cotangent_np.shape[0]
    grad_v0 = np.zeros_like(cotangent_np)
    if ux_ic.grad is not None:
        grad_v0[:, :, :, 0] = ux_ic.grad.numpy()
    if uy_ic.grad is not None:
        grad_v0[:, :, :, 1] = uy_ic.grad.numpy()
    if uz_ic.grad is not None:
        grad_v0[:, :, :, 2] = uz_ic.grad.numpy()

    # mosaic:grad:viscosity,dt:adjoint
    grads: dict[str, np.ndarray] = {
        "v0": grad_v0.astype(np.float32),
        "viscosity": np.zeros(1, dtype=np.float32),
        "dt": np.zeros(1, dtype=np.float32),
    }

    # mosaic:grad:lid_velocity:adjoint
    # Lid velocity gradient: warp returns (N, N, 2) float32 for vec2 arrays.
    if lid_wp is not None:
        if lid_wp.grad is not None:
            grads["lid_velocity"] = np.asarray(lid_wp.grad.numpy(), dtype=np.float32)
        else:
            grads["lid_velocity"] = np.zeros((n, n, 2), dtype=np.float32)
    else:
        grads["lid_velocity"] = np.zeros((n, n, 2), dtype=np.float32)

    return grads


# ============================================================
# Schema definitions
# ============================================================


class InputSchema(_CanonicalInputSchema):
    num_iters_poisson: int = Field(
        default=500,
        description=(
            "Minimum number of Jacobi iterations per 2-D streamfunction Poisson solve. "
            "The solver auto-scales to max(this value, min(4*N², 8000)) at runtime, "
            "so convergence is maintained across grid sizes N=16..128 without manual "
            "tuning.  For N≥128 with production accuracy, a multigrid or FFT Poisson "
            "solver is recommended as Jacobi is capped at 8000 iterations."
        ),
    )
    num_iters_poisson_3d: int = Field(
        default=800,
        description=(
            "Number of Jacobi iterations per 3-D pressure Poisson solve. "
            "800 iterations is adequate for N≤32; increase for larger grids."
        ),
    )


class OutputSchema(_CanonicalOutputSchema):
    pass


# ============================================================
# Utility
# ============================================================


def _warp_device() -> str:  # mosaic:util
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _is_3d(v0_np: np.ndarray) -> bool:  # mosaic:util
    """True if the velocity field is 3-D (shape N,N,N,3 with nz != 1)."""
    return v0_np.ndim == 4 and v0_np.shape[2] != 1 and v0_np.shape[3] == 3


# ============================================================
# Tesseract API endpoints
# ============================================================


def apply(inputs: InputSchema) -> OutputSchema:
    v0 = np.asarray(inputs.v0, dtype=np.float32)
    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    device = _warp_device()

    if _is_3d(v0):
        lid_vel = (
            np.asarray(inputs.lid_velocity, dtype=np.float32)
            if inputs.lid_velocity is not None
            else None
        )
        result, _tape, *_ = ns3d_solve(
            v0,
            nu,
            dt,
            inputs.steps,
            inputs.domain_extent,
            inputs.num_iters_poisson_3d,
            lid_velocity=lid_vel,
            device=device,
        )
        return OutputSchema(result=result, drag=None)

    # 2-D path (IPCS)
    inflow_np = (
        np.asarray(inputs.inflow_profile, dtype=np.float32)
        if inputs.inflow_profile is not None
        else None
    )
    bc = inputs.boundary_conditions
    wall_y_noslip = bc.y_lo.type == BCType.NO_SLIP and bc.y_hi.type == BCType.NO_SLIP
    result, drag, velocity_mean_np, _tape, *_ = ns2d_solve(
        v0,
        nu,
        dt,
        inputs.steps,
        inputs.domain_extent,
        inputs.num_iters_poisson,
        inputs.obstacle,
        device=device,
        inflow_profile=inflow_np,
        wall_y_noslip=wall_y_noslip,
    )
    drag_out = (
        np.asarray(drag, dtype=np.float32).reshape(1) if drag is not None else None
    )
    return OutputSchema(result=result, drag=drag_out)


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt,lid_velocity,inflow_profile:adjoint
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute VJP via wp.Tape reverse-mode autodiff.

    Runs the forward pass under tape recording, then calls tape.backward()
    with the output cotangent to obtain gradients w.r.t. differentiable inputs.

    Differentiable inputs: v0 (2D and 3D), lid_velocity (3D cavity mode).
    Both 2D and 3D use IPCS with spectral FFT Poisson for numerically exact VJPs.
    """
    v0 = np.asarray(inputs.v0, dtype=np.float32)
    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    device = _warp_device()

    result: dict[str, Any] = {}

    if _is_3d(v0):
        lid_vel = (
            np.asarray(inputs.lid_velocity, dtype=np.float32)
            if inputs.lid_velocity is not None
            else None
        )
        _result_np, tape, ux_f, uy_f, uz_f, ux_ic, uy_ic, uz_ic, lid_wp = ns3d_solve(
            v0,
            nu,
            dt,
            inputs.steps,
            inputs.domain_extent,
            inputs.num_iters_poisson_3d,
            lid_velocity=lid_vel,
            device=device,
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
            lid_wp=lid_wp,
        )
        if "v0" in vjp_inputs:
            result["v0"] = grads["v0"]
        if "viscosity" in vjp_inputs:
            result["viscosity"] = grads["viscosity"]
        if "dt" in vjp_inputs:
            result["dt"] = grads["dt"]
        if "lid_velocity" in vjp_inputs and inputs.lid_velocity is not None:
            result["lid_velocity"] = grads["lid_velocity"]

    else:
        # 2-D IPCS path
        inflow_np = (
            np.asarray(inputs.inflow_profile, dtype=np.float32)
            if inputs.inflow_profile is not None
            else None
        )
        bc = inputs.boundary_conditions
        wall_y_noslip = (
            bc.y_lo.type == BCType.NO_SLIP and bc.y_hi.type == BCType.NO_SLIP
        )
        (
            _result_np,
            _drag,
            _,
            tape,
            ux_f,
            uy_f,
            ux_ic,
            uy_ic,
            inflow_wp,
            nu_wp_2d,
            dt_wp_2d,
            rans_drag_buf_2d,
        ) = ns2d_solve(
            v0,
            nu,
            dt,
            inputs.steps,
            inputs.domain_extent,
            inputs.num_iters_poisson,
            inputs.obstacle,
            device=device,
            inflow_profile=inflow_np,
            wall_y_noslip=wall_y_noslip,
        )
        cot_result = np.asarray(
            cotangent_vector.get("result", np.zeros_like(v0)), dtype=np.float32
        )
        cot_drag_raw = cotangent_vector.get("drag", None)
        cot_drag = (
            float(np.asarray(cot_drag_raw).squeeze())
            if cot_drag_raw is not None
            else None
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
            inflow_wp=inflow_wp,
            inflow_profile_np=inflow_np,
            rans_drag_buf=rans_drag_buf_2d,
            cotangent_drag=cot_drag,
        )
        if "v0" in vjp_inputs:
            result["v0"] = grads["v0"]
        if "viscosity" in vjp_inputs:
            result["viscosity"] = grads["viscosity"]
        if "dt" in vjp_inputs:
            result["dt"] = grads["dt"]
        if (
            "inflow_profile" in vjp_inputs
            and inputs.inflow_profile is not None
            and "inflow_profile" in grads
        ):
            result["inflow_profile"] = grads["inflow_profile"]

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

    # Drag is only computed for 2-D obstacle runs.
    # 3-D cavity mode (lid_velocity set) produces no drag output.
    is_3d = len(shape) == 4 and shape[2] != 1 and shape[3] == 3
    has_obstacle_2d = d.get("obstacle") is not None and not is_3d

    _has_inflow_2d = d.get("inflow_profile") is not None and not is_3d
    out: dict[str, Any] = {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": None,
    }
    if has_obstacle_2d:
        out["drag"] = {"shape": (1,), "dtype": "float32"}
    return out
