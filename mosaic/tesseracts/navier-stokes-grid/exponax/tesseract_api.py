from typing import Any

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable
from pydantic import Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    drag: Differentiable[Array[(1,), Float32]] = Field(
        description="Linear drag coefficient",
        default_factory=lambda: np.array([0.0], dtype=np.float32),
    )
    order: int = Field(description="ETDRK time integration order (0-4)", default=2)
    kolmogorov_forcing: bool = Field(
        description="Enable Kolmogorov sinusoidal body forcing for sustained turbulence",
        default=False,
    )
    injection_mode: int = Field(
        description="Wavenumber at which energy is injected (Kolmogorov forcing)",
        default=4,
    )
    injection_scale: Differentiable[Array[(1,), Float32]] = Field(
        description="Amplitude of the Kolmogorov body forcing",
        default_factory=lambda: np.array([1.0], dtype=np.float32),
    )

    @model_validator(mode="after")
    def _check_periodic_bcs(self) -> "InputSchema":
        if not self.boundary_conditions.is_fully_periodic:
            raise ValueError(
                "exponax uses a spectral (FFT) discretisation and only supports "
                "periodic boundary conditions. Set all faces to 'periodic' "
                "(the default)."
            )
        return self


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    pass


# ---------------------------------------------------------------------------
# Spectral velocity <-> vorticity helpers (2D, differentiable via JAX FFT)
# ---------------------------------------------------------------------------


def _velocity_to_vorticity(
    v: jnp.ndarray, domain_extent: float
) -> jnp.ndarray:  # mosaic:io
    """(N, N, 2) velocity -> (1, N, N) vorticity via spectral curl.

    Uses physical wavenumbers k_phys = k_int * 2π/L so the vorticity values
    are in correct physical units regardless of domain_extent.
    """
    N = v.shape[0]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac  # physical wavenumbers
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    vx_hat = jnp.fft.fft2(v[..., 0])
    vy_hat = jnp.fft.fft2(v[..., 1])
    omega_hat = 1j * kx * vy_hat - 1j * ky * vx_hat
    omega = jnp.real(jnp.fft.ifft2(omega_hat))
    return omega[None, :, :]  # (1, N, N)


def _vorticity_to_velocity(
    omega: jnp.ndarray, domain_extent: float
) -> jnp.ndarray:  # mosaic:io
    """(1, N, N) vorticity -> (N, N, 2) velocity via spectral Biot-Savart.

    Uses physical wavenumbers. Recovers the divergence-free velocity with zero
    mean (standard for periodic NS).
    """
    N = omega.shape[-1]
    kfac = 2.0 * jnp.pi / domain_extent
    k = jnp.fft.fftfreq(N) * N * kfac  # physical wavenumbers
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2 = k2.at[0, 0].set(1.0)  # avoid divide-by-zero; mean velocity = 0
    omega_hat = jnp.fft.fft2(omega[0])
    vx = jnp.real(jnp.fft.ifft2(1j * ky / k2 * omega_hat))
    vy = jnp.real(jnp.fft.ifft2(-1j * kx / k2 * omega_hat))
    return jnp.stack([vx, vy], axis=-1)  # (N, N, 2)


# ---------------------------------------------------------------------------
# Core forward function
# ---------------------------------------------------------------------------


def exponax_fwd(  # mosaic:physics
    v0: jnp.ndarray,
    dt: float,
    steps: int,
    viscosity: float,
    domain_extent: float,
    drag: float,
    order: int,
    kolmogorov_forcing: bool,
    injection_mode: int,
    injection_scale: float,
    boundary_conditions: dict | None = None,
    **_kwargs,
) -> jnp.ndarray:
    """Run 2D or 3D incompressible Navier-Stokes using exponax.

    2D uses the pseudo-spectral streamfunction-vorticity stepper with an
    internal velocity<->vorticity conversion so the interface matches the
    other CFD tesseracts.

    3D uses the Leray-projected velocity stepper directly.

    Args:
        v0: Velocity field, shape (nx, ny, 1, 2) for 2D or (nx, ny, nz, 3) for 3D.
        dt: Timestep size.
        steps: Number of simulation steps.
        viscosity: Kinematic diffusivity (viscosity).
        domain_extent: Side length of the periodic domain.
        drag: Linear drag coefficient.
        order: ETDRK integration order (0-4).
        kolmogorov_forcing: Enable Kolmogorov body forcing.
        injection_mode: Forcing wavenumber.
        injection_scale: Forcing amplitude.

    Returns:
        Final velocity field, same shape as v0.
    """
    ndim = v0.shape[-1]  # 2 or 3
    num_points = v0.shape[0]  # isotropic: nx == ny (== nz)

    stepper_kwargs = dict(
        domain_extent=domain_extent,
        num_points=num_points,
        dt=dt,
        diffusivity=viscosity,
        drag=drag,
        order=order,
        dealiasing_fraction=1.0,  # disable 2/3-rule truncation to match other solvers
    )

    if ndim == 2:
        # v0: (N, N, 1, 2) -> squeeze z -> (N, N, 2)
        v = v0[:, :, 0, :]

        # Mean velocity is conserved by incompressible periodic NS (Galilean invariance).
        # The vorticity formulation strips it (∂const/∂x = 0), so we save and restore it.
        v_mean = jnp.mean(v, axis=(0, 1), keepdims=True)  # (1, 1, 2)

        # velocity -> vorticity: (1, N, N)  (operate on fluctuation only)
        vort = _velocity_to_vorticity(v - v_mean, domain_extent)

        if kolmogorov_forcing:
            stepper = ex.stepper.KolmogorovFlowVorticity(
                num_spatial_dims=2,
                injection_mode=injection_mode,
                injection_scale=injection_scale,
                **stepper_kwargs,
            )
        else:
            stepper = ex.stepper.NavierStokesVorticity(
                num_spatial_dims=2, **stepper_kwargs
            )

        def step_fn(carry, _):
            return stepper(carry).astype(carry.dtype), None

        result_vort, _ = jax.lax.scan(step_fn, vort, None, length=steps)

        # vorticity -> velocity: (N, N, 2) -> (N, N, 1, 2); restore conserved mean
        result = _vorticity_to_velocity(result_vort, domain_extent) + v_mean
        return result[:, :, None, :]

    else:  # 3D
        # v0: (N, N, N, 3) channels-last -> (3, N, N, N) channels-first for exponax
        v0_cf = jnp.moveaxis(v0, -1, 0)

        if kolmogorov_forcing:
            stepper = ex.stepper.KolmogorovFlowVelocity(
                num_spatial_dims=3,
                injection_mode=injection_mode,
                injection_scale=injection_scale,
                **stepper_kwargs,
            )
        else:
            stepper = ex.stepper.NavierStokesVelocity(
                num_spatial_dims=3, **stepper_kwargs
            )

        def step_fn(carry, _):
            return stepper(carry).astype(carry.dtype), None

        result_cf, _ = jax.lax.scan(step_fn, v0_cf, None, length=steps)

        # (3, N, N, N) -> (N, N, N, 3)
        return jnp.moveaxis(result_cf, 0, -1)


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:  # mosaic:physics
    return dict(result=exponax_fwd(**inputs))


_SCALAR_KEYS = ("dt", "viscosity", "drag", "injection_scale")


def _unpack_scalars(d: dict) -> dict:  # mosaic:io
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in _SCALAR_KEYS:
        if key in d:
            d[key] = float(d[key][0])
    return d


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(_unpack_scalars(inputs.model_dump()))


# ---------------------------------------------------------------------------
# Float64 gradient helpers (mirrors XLB pattern)
# ---------------------------------------------------------------------------

_DIFF_INPUT_KEYS: tuple[str, ...] = (
    "v0",
    "viscosity",
    "dt",
    "drag",
    "injection_scale",
)

# Module-level caches: static config -> jit-compiled fn. Avoids recompiling the
# XLA kernel on every HTTP request (optimizer iteration).
_vjp_compiled_cache: dict = {}
_jvp_compiled_cache: dict = {}


def _build_diff_bundle(
    inputs: dict, include: tuple[str, ...]
) -> dict:  # mosaic:grad:v0,viscosity,dt,drag,injection_scale:autodiff
    """Build a {path: value} dict for jax.vjp / jax.jvp over `include` keys."""
    bundle: dict = {}
    for k in include:
        if k not in _DIFF_INPUT_KEYS:
            continue
        v = inputs.get(k)
        if v is None:
            continue
        if k in _SCALAR_KEYS:
            bundle[k] = jnp.asarray(float(v[0]) if hasattr(v, "__len__") else float(v))
        else:
            bundle[k] = jnp.asarray(v)
    return bundle


def _run_forward(
    inputs: dict, diff_bundle: dict
) -> jnp.ndarray:  # mosaic:grad:v0,viscosity,dt,drag,injection_scale:autodiff
    """Run exponax_fwd with diff inputs overridden from diff_bundle."""
    fwd_kwargs = {}
    for k in (
        "steps",
        "domain_extent",
        "boundary_conditions",
        "order",
        "kolmogorov_forcing",
        "injection_mode",
    ):
        if k in inputs:
            fwd_kwargs[k] = inputs[k]

    fwd_kwargs["v0"] = jnp.asarray(diff_bundle.get("v0", inputs["v0"]))

    for k in ("viscosity", "dt", "drag", "injection_scale"):
        src = diff_bundle.get(k, inputs.get(k))
        if src is None:
            continue
        fwd_kwargs[k] = float(src)

    return exponax_fwd(**fwd_kwargs)


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt,drag,injection_scale:autodiff
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        _unpack_scalars(inputs.model_dump()),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape from input shapes."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    def to_jax(x):
        if not is_shapedtype_dict(x):
            return x
        s = jax.ShapeDtypeStruct(**x)
        # 1-element arrays are scalar physics params; use a concrete Python float
        # so they stay static inside jax.eval_shape (constructors can't accept tracers).
        if len(s.shape) <= 1:
            return 1.0
        return s

    jaxified_inputs = jax.tree.map(
        to_jax, abstract_inputs.model_dump(), is_leaf=is_shapedtype_dict
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    """Reverse-mode VJP over any subset of diff inputs."""
    present = tuple(
        k for k in vjp_inputs if k in _DIFF_INPUT_KEYS and inputs.get(k) is not None
    )
    if not present:
        return {}

    diff_bundle = _build_diff_bundle(inputs, present)

    v0_src = inputs.get("v0")
    v0_shape = tuple(v0_src.shape) if hasattr(v0_src, "shape") else ()
    cache_key = (v0_shape, inputs.get("steps"), present, tuple(sorted(vjp_outputs)))

    if cache_key not in _vjp_compiled_cache:
        _inputs_frozen = inputs
        _vjp_outputs_frozen = vjp_outputs

        def _fwd_static(bundle):
            result = _run_forward(_inputs_frozen, bundle)
            out = {}
            if "result" in _vjp_outputs_frozen:
                out["result"] = result
            return out

        @jax.jit
        def _vjp_compiled(bundle, cotan):
            _, vjp_func = jax.vjp(_fwd_static, bundle)
            return vjp_func(cotan)[0]

        _vjp_compiled_cache[cache_key] = _vjp_compiled

    grads = _vjp_compiled_cache[cache_key](diff_bundle, cotangent_vector)

    out: dict = {}
    for k, g in grads.items():
        if k in _SCALAR_KEYS:
            g = jnp.atleast_1d(g)
        elif g.ndim == 0:
            g = jnp.atleast_1d(g)
        out[k] = g
    return out
