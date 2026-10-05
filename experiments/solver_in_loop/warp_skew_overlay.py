"""Experimental energy-conserving skew form; used only by the RK3 overlay."""

import warp as wp


@wp.kernel
def skew_tentative_vel_2d_kernel(
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
    conservative_ux = (
        ux[ip1, j] * ux[ip1, j]
        - ux[im1, j] * ux[im1, j]
        + uy[i, jp1] * ux[i, jp1]
        - uy[i, jm1] * ux[i, jm1]
    ) * inv_2h
    adv_ux = 0.5 * (adv_ux + conservative_ux)
    ux_star[i, j] = ui + dt_ * (-adv_ux + nu_ * lap_ux)

    # uy component
    lap_uy = (uy[im1, j] + uy[ip1, j] + uy[i, jm1] + uy[i, jp1] - 4.0 * vi) * inv_h2
    adv_uy = (
        ui * (uy[ip1, j] - uy[im1, j]) * inv_2h
        + vi * (uy[i, jp1] - uy[i, jm1]) * inv_2h
    )
    conservative_uy = (
        ux[ip1, j] * uy[ip1, j]
        - ux[im1, j] * uy[im1, j]
        + uy[i, jp1] * uy[i, jp1]
        - uy[i, jm1] * uy[i, jm1]
    ) * inv_2h
    adv_uy = 0.5 * (adv_uy + conservative_uy)
    uy_star[i, j] = vi + dt_ * (-adv_uy + nu_ * lap_uy)
