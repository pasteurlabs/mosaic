# ruff: noqa: ANN001
"""Experimental two-dimensional SSPRK3 stage overlay for the Warp adapter.

A build script inserts this block before the adapter API. The existing Euler
entry points remain defaults. No projection, filtering or forcing changes are
included; the full forced solver is not claimed to be third-order accurate.
"""

import contextlib
import os

import numpy as np
import warp as wp


@wp.kernel
def _rk3_blend_kernel(
    original_x: wp.array2d(dtype=wp.float32),
    original_y: wp.array2d(dtype=wp.float32),
    advanced_x: wp.array2d(dtype=wp.float32),
    advanced_y: wp.array2d(dtype=wp.float32),
    result_x: wp.array2d(dtype=wp.float32),
    result_y: wp.array2d(dtype=wp.float32),
    original_weight: float,
    advanced_weight: float,
) -> None:
    i, j = wp.tid()
    result_x[i, j] = (
        original_weight * original_x[i, j] + advanced_weight * advanced_x[i, j]
    )
    result_y[i, j] = (
        original_weight * original_y[i, j] + advanced_weight * advanced_y[i, j]
    )


def _rk3_enabled() -> bool:
    method = os.environ.get("MOSAIC_WARP_INTEGRATOR", "euler")
    if method not in {"euler", "ssprk3"}:
        raise ValueError("unknown experimental Warp integrator")
    return method == "ssprk3"


def _rk3_solve(
    v0, viscosity, dt, steps, domain_extent, device, record, track_scalar_grads
):
    base = globals()
    n = v0.shape[0]
    inv_2h, inv_h2 = 0.5 * n / domain_extent, (n / domain_extent) ** 2

    def zeros(dtype=wp.float32):
        return wp.zeros((n, n), dtype=dtype, device=device, requires_grad=record)

    initial_x = wp.array(
        v0[:, :, 0, 0], dtype=wp.float32, device=device, requires_grad=record
    )
    initial_y = wp.array(
        v0[:, :, 0, 1], dtype=wp.float32, device=device, requires_grad=record
    )
    nu = wp.array(
        np.asarray([viscosity], dtype=np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=record and track_scalar_grads,
    )
    timestep = wp.array(
        np.asarray([dt], dtype=np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=record and track_scalar_grads,
    )
    workspace = None if record else base["_poisson_workspace_2d"](n, device)

    def advance(x, y):
        star_x, star_y = zeros(), zeros()
        wp.launch(
            base["tentative_vel_2d_kernel"],
            dim=(n, n),
            inputs=[x, y, star_x, star_y, timestep, inv_2h, inv_h2, nu],
            block_dim=256,
            device=device,
        )
        divergence = zeros(wp.vec2f)
        wp.launch(
            base["divergence_2d_to_complex_kernel"],
            dim=(n, n),
            inputs=[star_x, star_y, divergence, timestep, inv_2h],
            block_dim=256,
            device=device,
        )
        pressure = base["_spectral_poisson_2d_core"](
            divergence, domain_extent, device, workspace=workspace
        )
        next_x, next_y = zeros(), zeros()
        wp.launch(
            base["pressure_correct_2d_from_complex_kernel"],
            dim=(n, n),
            inputs=[
                star_x,
                star_y,
                pressure,
                float(n * n),
                next_x,
                next_y,
                timestep,
                inv_2h,
            ],
            block_dim=256,
            device=device,
        )
        return next_x, next_y

    def blend(original_x, original_y, advanced_x, advanced_y, a, b):
        x, y = zeros(), zeros()
        wp.launch(
            _rk3_blend_kernel,
            dim=(n, n),
            inputs=[original_x, original_y, advanced_x, advanced_y, x, y, a, b],
            block_dim=256,
            device=device,
        )
        return x, y

    tape = wp.Tape() if record else None
    with tape if record else contextlib.nullcontext():
        x, y = initial_x, initial_y
        for _ in range(steps):
            first_x, first_y = advance(x, y)
            second_x, second_y = advance(first_x, first_y)
            second_x, second_y = blend(x, y, second_x, second_y, 0.75, 0.25)
            third_x, third_y = advance(second_x, second_y)
            x, y = blend(x, y, third_x, third_y, 1 / 3, 2 / 3)
    return tape, x, y, initial_x, initial_y, nu, timestep


def ns2d_ssprk3_forward(
    v0, viscosity, dt, steps, domain_extent, device="cpu"
) -> np.ndarray:
    """Run the opt-in stage composition without recording a reverse tape."""
    _, x, y, *_ = _rk3_solve(
        v0, viscosity, dt, steps, domain_extent, device, False, False
    )
    return np.stack([x.numpy(), y.numpy()], axis=-1)[:, :, None, :]


def ns2d_ssprk3_tape(
    v0, viscosity, dt, steps, domain_extent, device="cpu", track_scalar_grads=False
) -> tuple:
    """Record exactly the same stage arithmetic for the existing VJP endpoint."""
    return _rk3_solve(
        v0, viscosity, dt, steps, domain_extent, device, True, track_scalar_grads
    )
