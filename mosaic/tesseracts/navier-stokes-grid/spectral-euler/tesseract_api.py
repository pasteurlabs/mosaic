"""Minimal spectral forward-Euler solver for 2D incompressible Navier-Stokes.

Solves the vorticity equation ∂ω/∂t + (u·∇)ω = ν∇²ω using FFT-based spectral
differentiation with explicit Euler time stepping.  Incompressibility is
enforced to machine precision via the stream-function formulation (Biot-Savart
inversion).

This solver is deliberately minimal and intended as a CI smoke-test to exercise
the full Tesseract apply / VJP / abstract_eval path.
"""

from typing import Any

import jax
import jax.numpy as jnp
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["v0", "viscosity", "dt"])
):
    pass


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    pass


# ---------------------------------------------------------------------------
# Spectral helpers (2D only)
# ---------------------------------------------------------------------------


def _velocity_to_vorticity(v: jnp.ndarray, domain_extent: float) -> jnp.ndarray:
    """(N, N, 2) velocity -> (N, N) vorticity via spectral curl."""
    N = v.shape[0]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    vx_hat = jnp.fft.fft2(v[..., 0])
    vy_hat = jnp.fft.fft2(v[..., 1])
    omega_hat = 1j * kx * vy_hat - 1j * ky * vx_hat
    return jnp.real(jnp.fft.ifft2(omega_hat))


def _vorticity_to_velocity(omega: jnp.ndarray, domain_extent: float) -> jnp.ndarray:
    """(N, N) vorticity -> (N, N, 2) velocity via spectral Biot-Savart."""
    N = omega.shape[0]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2 = k2.at[0, 0].set(1.0)
    omega_hat = jnp.fft.fft2(omega)
    vx = jnp.real(jnp.fft.ifft2(1j * ky / k2 * omega_hat))
    vy = jnp.real(jnp.fft.ifft2(-1j * kx / k2 * omega_hat))
    return jnp.stack([vx, vy], axis=-1)


def _advection_rhs(omega: jnp.ndarray, v: jnp.ndarray, domain_extent: float):
    """Compute -(u·∇)ω spectrally."""
    N = omega.shape[0]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    omega_hat = jnp.fft.fft2(omega)
    domega_dx = jnp.real(jnp.fft.ifft2(1j * kx * omega_hat))
    domega_dy = jnp.real(jnp.fft.ifft2(1j * ky * omega_hat))
    return -(v[..., 0] * domega_dx + v[..., 1] * domega_dy)


def _diffusion_rhs(omega: jnp.ndarray, viscosity: float, domain_extent: float):
    """Compute ν∇²ω spectrally."""
    N = omega.shape[0]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    omega_hat = jnp.fft.fft2(omega)
    return jnp.real(jnp.fft.ifft2(-viscosity * k2 * omega_hat))


# ---------------------------------------------------------------------------
# Core forward function
# ---------------------------------------------------------------------------


def spectral_euler_fwd(
    v0: jnp.ndarray,
    dt: float,
    steps: int,
    viscosity: float,
    domain_extent: float,
    **_kwargs,
) -> jnp.ndarray:
    """2D incompressible NS via spectral forward-Euler on vorticity.

    Args:
        v0: Velocity field, shape (N, N, 1, 2).
        dt: Timestep.
        steps: Number of steps.
        viscosity: Kinematic viscosity.
        domain_extent: Side length of periodic domain.

    Returns:
        Final velocity field, shape (N, N, 1, 2).
    """
    v = v0[:, :, 0, :]  # (N, N, 2)
    v_mean = jnp.mean(v, axis=(0, 1), keepdims=True)
    omega = _velocity_to_vorticity(v - v_mean, domain_extent)

    def step(omega, _):
        vel = _vorticity_to_velocity(omega, domain_extent)
        adv = _advection_rhs(omega, vel, domain_extent)
        diff = _diffusion_rhs(omega, viscosity, domain_extent)
        return omega + dt * (adv + diff), None

    omega_final, _ = jax.lax.scan(step, omega, None, length=steps)
    result = _vorticity_to_velocity(omega_final, domain_extent) + v_mean
    return result[:, :, None, :]


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------

_SCALAR_KEYS = ("dt", "viscosity")


def _unpack_scalars(d: dict) -> dict:
    for key in _SCALAR_KEYS:
        if key in d and not isinstance(d[key], (int, float)):
            d[key] = float(d[key][0])
    return d


@jax.jit
def _apply_jit(v0, dt, steps, viscosity, domain_extent):
    return spectral_euler_fwd(
        v0=v0, dt=dt, steps=steps, viscosity=viscosity, domain_extent=domain_extent
    )


def apply(inputs: InputSchema) -> OutputSchema:
    d = _unpack_scalars(inputs.model_dump())
    result = _apply_jit(
        jnp.asarray(d["v0"]),
        d["dt"],
        d["steps"],
        d["viscosity"],
        d["domain_extent"],
    )
    return {"result": result}


# ---------------------------------------------------------------------------
# Abstract eval
# ---------------------------------------------------------------------------


def abstract_eval(abstract_inputs):
    v0_info = abstract_inputs.v0
    if isinstance(v0_info, dict):
        shape = tuple(v0_info["shape"])
        dtype = v0_info.get("dtype", "float32")
    else:
        shape = v0_info.shape
        dtype = getattr(v0_info, "dtype", "float32")
    return {
        "result": {"shape": shape, "dtype": str(dtype)},
        "drag": {"shape": (1,), "dtype": "float32"},
    }


# ---------------------------------------------------------------------------
# VJP
# ---------------------------------------------------------------------------

_DIFF_INPUT_KEYS = ("v0", "viscosity", "dt")
_vjp_cache: dict = {}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    d = _unpack_scalars(inputs.model_dump())
    present = tuple(
        k for k in vjp_inputs if k in _DIFF_INPUT_KEYS and d.get(k) is not None
    )
    if not present:
        return {}

    diff_bundle = {}
    for k in present:
        v = d[k]
        if k in _SCALAR_KEYS:
            diff_bundle[k] = jnp.asarray(v, dtype=jnp.float32)
        else:
            diff_bundle[k] = jnp.asarray(v)

    v0_shape = tuple(jnp.asarray(d["v0"]).shape)
    cache_key = (v0_shape, d.get("steps"), present, tuple(sorted(vjp_outputs)))

    if cache_key not in _vjp_cache:
        frozen_d = d
        frozen_outputs = vjp_outputs

        def fwd(bundle):
            fwd_args = {
                "steps": frozen_d["steps"],
                "domain_extent": frozen_d["domain_extent"],
            }
            fwd_args["v0"] = jnp.asarray(bundle.get("v0", frozen_d["v0"]))
            for k in _SCALAR_KEYS:
                src = bundle.get(k, frozen_d.get(k))
                if src is not None:
                    fwd_args[k] = float(src) if jnp.ndim(src) == 0 else float(src)
            result = spectral_euler_fwd(**fwd_args)
            out = {}
            if "result" in frozen_outputs:
                out["result"] = result
            return out

        @jax.jit
        def vjp_compiled(bundle, cotan):
            _, vjp_fn = jax.vjp(fwd, bundle)
            return vjp_fn(cotan)[0]

        _vjp_cache[cache_key] = vjp_compiled

    grads = _vjp_cache[cache_key](diff_bundle, cotangent_vector)
    out = {}
    for k, g in grads.items():
        if k in _SCALAR_KEYS:
            out[k] = jnp.atleast_1d(g)
        else:
            out[k] = g
    return out
