# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xlb
import xlb.velocity_set
from tesseract_core.runtime.jax_recipes import jax_vjp

# Enable 64-bit floats in JAX.  Must be set before any JAX computation.
# This is required so that the VJP/JVP paths can run the LBM in float64 to
# avoid float32 cancellation errors that corrupt gradients near omega≈2.
jax.config.update("jax_enable_x64", True)

# Precision of the whole solver: the apply() forward and the VJP/JVP paths.
# They share it so that they compute the same function, which the fd_check
# requires; timings therefore compare like with like.
#
# The default is float64: near omega≈2 the LBM develops float32 cancellation
# errors that corrupt gradients, and float32 gradients fail gradient/fd_check
# there (see issue #125).  Set XLB_VJP_FP32=1 to run everything in float32
# instead; use it only where the fd_check stays acceptable.
_USE_F64_DIFF = os.environ.get("XLB_VJP_FP32", "0") == "0"
from mosaic_shared.problems.navier_stokes_grid import (  # noqa: E402
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (  # noqa: E402
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import (  # noqa: E402
    check_periodic_or_channel,
    solid_cells,
)
from mosaic_shared.schema_types import make_differentiable  # noqa: E402
from pydantic import model_validator  # noqa: E402


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    """XLB solver input schema with differentiable fields."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "xlb")
        return self


class OutputSchema(
    make_differentiable(_CanonicalOutputSchema, ["result", "body_force"])
):
    """XLB solver output schema with differentiable fields."""


from xlb.compute_backend import ComputeBackend  # noqa: E402

# XLB operator imports
from xlb.operator.collision import BGK  # noqa: E402
from xlb.operator.equilibrium import QuadraticEquilibrium  # noqa: E402
from xlb.operator.macroscopic import Macroscopic  # noqa: E402
from xlb.operator.stream import Stream  # noqa: E402
from xlb.precision_policy import PrecisionPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# XLB one-time initialisation (needed to instantiate velocity sets)
# ---------------------------------------------------------------------------


def _make_ops(vset: Any, pp: Any, cb: Any, fdtype: Any):
    """Build the XLB operator bundle for a given (ndim, precision).

    XLB keeps the velocity set and precision policy in global state, so
    re-initialise it with this bundle's before constructing its operators.
    """
    xlb.init(velocity_set=vset, default_backend=cb, default_precision_policy=pp)
    collide = BGK(velocity_set=vset, precision_policy=pp, compute_backend=cb)
    return {
        "C": jnp.array(vset.c, dtype=fdtype),
        "W": jnp.array(vset.w, dtype=fdtype),
        "eq": QuadraticEquilibrium(
            velocity_set=vset, precision_policy=pp, compute_backend=cb
        ),
        "stream": Stream(velocity_set=vset, precision_policy=pp, compute_backend=cb),
        "macro": Macroscopic(
            velocity_set=vset, precision_policy=pp, compute_backend=cb
        ),
        "bgk": collide,
        "fdtype": fdtype,
    }


_vsets = {
    (2, False): xlb.velocity_set.D2Q9(
        precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.JAX
    ),
    (2, True): xlb.velocity_set.D2Q9(
        precision_policy=PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX
    ),
    (3, False): xlb.velocity_set.D3Q27(
        precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.JAX
    ),
    (3, True): xlb.velocity_set.D3Q27(
        precision_policy=PrecisionPolicy.FP64FP64, compute_backend=ComputeBackend.JAX
    ),
}
xlb.init(
    velocity_set=_vsets[(2, False)],
    default_backend=ComputeBackend.JAX,
    default_precision_policy=PrecisionPolicy.FP32FP32,
)
_OPS: dict[tuple[int, bool], dict] = {
    (ndim, f64): _make_ops(
        _vsets[(ndim, f64)],
        PrecisionPolicy.FP64FP64 if f64 else PrecisionPolicy.FP32FP32,
        ComputeBackend.JAX,
        jnp.float64 if f64 else jnp.float32,
    )
    for ndim in (2, 3)
    for f64 in (False, True)
}

# Concrete D2Q9 lattice velocities (q, 2) and the index of each one's opposite,
# for building the channel's boundary masks at trace time.
_D2Q9_C = np.array(_vsets[(2, False)].c).T.astype(int)
_D2Q9_OPP = np.array([np.flatnonzero((_D2Q9_C == -c).all(axis=1))[0] for c in _D2Q9_C])


def _checkpointed_scan(body: Any, f0: jnp.ndarray, steps: int):
    """Run ``steps`` LBM steps with sqrt-checkpointing for memory-bounded AD.

    A plain ``jax.lax.scan`` stores every step's populations for the reverse
    pass, so reverse-mode VJP memory grows as ``steps × q × N^d`` — quartic in
    N here (steps scale ∝ N), which OOMs the trajectory long before the forward
    does (110 GB at N=64 vs a 0.3 GB forward).  We split the scan into
    ``outer × inner ≈ √steps`` blocks and wrap the inner block in
    ``jax.checkpoint``: only the ``√steps`` block-boundary carries are kept, and
    each block is recomputed once during the backward pass.  Peak AD memory
    drops from O(steps) to O(√steps) at the cost of one extra forward per block.

    The pure forward pass is unaffected — ``jax.checkpoint`` only recomputes
    when a backward pass actually differentiates through it.
    """
    if steps <= 1:
        return jax.lax.scan(body, f0, None, length=steps)

    inner = max(1, round(math.sqrt(steps)))
    outer, rem = divmod(steps, inner)

    ckpt_body = jax.checkpoint(body)

    def block(carry: jnp.ndarray, _: Any):
        c, ys = jax.lax.scan(ckpt_body, carry, None, length=inner)
        return c, ys

    f_cur, ys_blocks = jax.lax.scan(block, f0, None, length=outer)
    # ys_blocks: (outer, inner, *per_step_out) -> flatten to (outer*inner, ...)
    ys = jax.tree_util.tree_map(
        lambda a: a.reshape((outer * inner, *a.shape[2:])), ys_blocks
    )

    # Remainder steps that didn't fit into a full block (steps not a perfect
    # multiple of ``inner``); run them checkpointed too and concatenate.
    if rem:
        f_cur, ys_rem = jax.lax.scan(ckpt_body, f_cur, None, length=rem)
        ys = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0), ys, ys_rem
        )

    return f_cur, ys


# Lattice Mach number that runs sub-step down to. Periodic runs keep it up to
# N = _MA_TARGET_N and then scale it ∝ 1/N (diffusive scaling), so LBM's
# O(Ma²) compressibility error falls with the O(dx²) discretisation error
# instead of putting a floor under grid refinement.
_MA_TARGET = 0.1
_MA_TARGET_N = 64


def _sub_steps(inputs: dict) -> int:
    """LBM sub-steps per dt that bring a run to its target Mach number.

    The lattice Mach number is Ma = u_max · (dt/dx) / cs with cs = 1/√3,
    u_max = max|v0| on a periodic box and the peak inflow speed in the channel.
    k sub-steps of dt/k divide it by k and preserve the physical end time.
    Computed from the concrete primal inputs, since k fixes the traced graph.
    """
    if inputs.get("obstacle") is None:
        v0 = np.asarray(inputs["v0"], dtype=np.float64)
        u_max = float(np.sqrt((v0**2).sum(axis=-1)).max())
        ma_target = _MA_TARGET * min(1.0, _MA_TARGET_N / v0.shape[0])
    else:
        u_max = float(np.max(np.abs(np.asarray(inputs["inflow_profile"]))))
        ma_target = _MA_TARGET
    ma = u_max * float(inputs["dt"]) / float(inputs["dx"]) * math.sqrt(3.0)
    return max(1, math.ceil(ma / ma_target))


def _periodic_populations(u: jnp.ndarray, omega: Any, ops: dict) -> jnp.ndarray:
    """Populations consistent with a divergence-free periodic velocity field.

    Starting from feq(ρ=1, u) leaves out the pressure field and the viscous
    stress, which then appear as acoustic and start-up transients. Instead
    (Mei et al. 2006, J. Comput. Phys. 220):

    - ρ = 1 + p/cs² from the periodic pressure Poisson equation
      ∇²p = -∂_a∂_b(u_a u_b), solved spectrally;
    - plus the first-order Chapman–Enskog part f¹ = -w ρ τ/cs² Q:S with
      Q_ab = c_a c_b - cs² δ_ab and S the strain rate. The scan streams before
      it collides, so the populations are post-collision ones: (1 - ω) f¹.

    Args:
        u:     Lattice velocity, shape (d, *spatial).
        omega: BGK relaxation rate 1/τ.
        ops:   Operator bundle from ``_make_ops``.

    Returns:
        Populations, shape (q, *spatial).
    """
    d = u.shape[0]
    spatial = u.shape[1:]
    axes = tuple(range(1, d + 1))
    k = jnp.meshgrid(
        *(2.0 * jnp.pi * jnp.fft.fftfreq(n).astype(u.dtype) for n in spatial),
        indexing="ij",
    )
    k2 = sum(ka**2 for ka in k)
    k2 = jnp.where(k2 == 0, 1.0, k2)

    u_hat = jnp.fft.fftn(u, axes=axes)
    grad = jnp.stack(
        [jnp.real(jnp.fft.ifftn(1j * k[a] * u_hat, axes=axes)) for a in range(d)]
    )  # grad[a, b] = ∂_a u_b
    strain = 0.5 * (grad + jnp.swapaxes(grad, 0, 1))

    uu_hat = jnp.fft.fftn(u[:, None] * u[None, :], axes=tuple(a + 1 for a in axes))
    p_hat = -sum(k[a] * k[b] * uu_hat[a, b] for a in range(d) for b in range(d)) / k2
    p_hat = p_hat.at[(0,) * d].set(0.0)
    rho = 1.0 + 3.0 * jnp.real(jnp.fft.ifftn(p_hat))[None]

    C, W = ops["C"], ops["W"]
    q_ab = C.T[:, :, None] * C.T[:, None, :] - jnp.eye(d, dtype=C.dtype) / 3.0
    q_s = jnp.einsum("qab,ab...->q...", q_ab, strain)
    w = W.reshape((-1,) + (1,) * d)
    f_neq = -3.0 * (1.0 / omega - 1.0) * w * rho * q_s
    return ops["eq"](rho, u) + f_neq


def _channel_links(
    shape: tuple[int, int], dx: float, obstacle: dict
) -> tuple[np.ndarray, ...]:
    """Solid nodes and boundary links of the 2-D channel with a cylinder.

    Nodes sit at the cell centers, so every boundary lies halfway between a
    node and its missing upstream neighbor. For lattice direction i, each
    returned (q, nx, ny) mask marks the fluid nodes x whose neighbor x - c_i
    lies beyond a wall (y outside the channel), behind the inlet (x < 0), past
    the outlet (x ≥ nx) or inside the cylinder. A cell is solid when its
    center lies strictly inside the cylinder.

    Returns:
        (solid, wall, inlet, outlet, body): solid is (nx, ny), the rest
        (q, nx, ny), all bool.
    """
    nx, ny = shape
    solid = solid_cells(nx, ny, dx, obstacle["center"], obstacle["radius"])
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    masks = np.zeros((4, len(_D2Q9_C), nx, ny), dtype=bool)
    for q, (cx, cy) in enumerate(_D2Q9_C):
        si, sj = i - cx, j - cy
        beyond_wall = (sj < 0) | (sj >= ny)
        in_x = (si >= 0) & (si < nx)
        source_solid = solid[np.clip(si, 0, nx - 1), np.clip(sj, 0, ny - 1)]
        masks[0, q] = beyond_wall
        masks[1, q] = ~beyond_wall & (si < 0)
        masks[2, q] = ~beyond_wall & (si >= nx)
        masks[3, q] = ~beyond_wall & in_x & source_solid & ~solid
    return (solid, *masks)


def _channel_run(
    u0: jnp.ndarray,
    u_in: jnp.ndarray,
    omega: Any,
    steps: int,
    sub_k: int,
    dx: float,
    obstacle: dict,
    ops: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the DFG channel in lattice units.

    Boundaries, all halfway between nodes:

    - walls and cylinder: bounce-back, f_ī(x, t+1) = f*_i(x, t);
    - inlet x = 0: velocity bounce-back, adding 2 w ρ c·u_in / cs² with
      u_in = (inflow_profile(y), 0) and ρ = 1;
    - outlet x = L: anti-bounce-back with the face velocity extrapolated
      from the last two columns. LBM is weakly compressible, and a plain
      p = 0 face would reflect the start-up pressure waves, which then ring
      between inlet and outlet almost undamped at ν = 1e-3. The face density
      1 + (u_x - ū_x)/cs follows the outgoing characteristic of the last
      column's u_x relative to ū_x, the fully developed profile with the
      inlet's flux, so outgoing waves leave and p = 0 on the face once the
      flow is steady.

    The force on the cylinder is the momentum exchanged on its bounce-back
    links, 2 c_i f*_i per link and step, taken at the last sub-step of each
    outer step.

    Args:
        u0:     Initial lattice velocity, shape (2, nx, ny).
        u_in:   Lattice inflow speed at the node heights, shape (ny,).
        omega:  Relaxation rate.
        steps:  Outer steps.
        sub_k:  LBM steps per outer step.
        dx:     Grid spacing.
        obstacle: Cylinder ``center`` and ``radius`` in physical units.
        ops:    Operator bundle from ``_make_ops``.

    Returns:
        (f, force): final populations (q, nx, ny), and the lattice force on
        the cylinder after each outer step, shape (steps, 2).
    """
    solid, wall, inlet, outlet, body = _channel_links(u0.shape[1:], dx, obstacle)
    C, W = ops["C"], ops["W"]
    w = W[:, None]
    f_rest = W[:, None, None] * jnp.ones_like(u0[:1])
    bounce = wall | body
    inlet_term = 6.0 * w[:, :, None] * C[0][:, None, None] * u_in[None, None, :]
    # Fully developed outflow carrying the inlet's flux.
    shape = np.arange(u0.shape[2]) + 0.5
    shape = shape * (u0.shape[2] - shape)
    u_developed = jnp.sum(u_in) * shape / shape.sum()

    def sub_step(f: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        rho, u = ops["macro"](f)
        f_post = ops["bgk"](f, ops["eq"](rho, u), rho, u, omega)
        reflected = f_post[_D2Q9_OPP]
        force = -2.0 * C @ jnp.sum(jnp.where(body, reflected, 0.0), axis=(1, 2))

        f_new = ops["stream"](f_post)
        f_new = jnp.where(bounce, reflected, f_new)
        f_new = jnp.where(inlet, reflected + inlet_term, f_new)
        u_out = 1.5 * u[:, -1, :] - 0.5 * u[:, -2, :]
        rho_out = 1.0 + math.sqrt(3.0) * (u[0, -1, :] - u_developed)
        cu = C.T @ u_out
        f_out = -reflected[:, -1, :] + 2.0 * w * rho_out * (
            1.0 + 4.5 * cu**2 - 1.5 * jnp.sum(u_out**2, axis=0)
        )
        f_new = f_new.at[:, -1, :].set(
            jnp.where(outlet[:, -1, :], f_out, f_new[:, -1, :])
        )
        return jnp.where(solid, f_rest, f_new), force

    def outer_step(f: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        f, forces = jax.lax.scan(sub_step, f, None, length=sub_k)
        return f, forces[-1]

    f0 = jnp.where(solid, f_rest, ops["eq"](jnp.ones_like(u0[:1]), u0))
    return _checkpointed_scan(outer_step, f0, steps)


def xlb_fwd(
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    dx: float,
    obstacle: dict | None = None,
    inflow_profile: jnp.ndarray | None = None,
    _use_f64: bool = False,
    _sub_k: int = 1,
    **_: Any,
) -> dict:
    """Run a 2D or 3D LBM simulation from an initial velocity field.

    Accepts physical units; internally converts to lattice units via:
        scale = dt / dx          (u_lb = u_phys * scale)
        nu_lb = viscosity * dt / dx**2

    Each dt runs as ``_sub_k`` LBM steps of dt/_sub_k, which preserves the
    physical end time T = dt*steps while dividing the lattice Mach number,
    and so the O(Ma²) compressibility error, by _sub_k (see ``_sub_steps``).
    Periodic runs start from populations consistent with v0 (see
    ``_periodic_populations``). With an obstacle, the run is the 2-D channel
    of ``_channel_run``, started from the equilibrium at ρ=1 and v0.

    Args:
        v0:            Initial velocity in physical units, shape (nx, ny, 1, 2) or (nx, ny, nz, 3).
        viscosity:     Physical kinematic viscosity.
        dt:            Physical timestep size.
        steps:         Number of timesteps.
        dx:            Grid spacing.
        obstacle:      Cylinder (``center``, ``radius``, physical units) in the
                       channel, or None for a periodic box.
        inflow_profile: Channel inlet u_x at the cell-center heights, shape (ny,).
        _use_f64:      If True, run the entire LBM computation in float64 for accurate
                       gradients. Output is cast back to float32 before returning.
                       This prevents float32 cancellation errors in the VJP/JVP paths,
                       especially near omega≈2 where small perturbations amplify rapidly.
        _sub_k:        LBM sub-steps per dt (see ``_sub_steps``).

    Returns:
        ``result`` (same shape as v0), and in the channel also ``pressure``
        (nx, ny, 1), kinematic with p = 0 at the outlet, and ``body_force``
        (steps, 2), the force of the fluid on the cylinder per unit depth.
    """
    ndim = v0.shape[-1]  # 2 or 3

    # Apply sub-stepping: use effective dt and total step count
    dt_eff = dt / _sub_k
    steps_eff = steps * _sub_k
    scale_eff = dt_eff / dx  # u_lb = u_phys * scale_eff
    nu_lb = viscosity * dt_eff / dx**2
    omega = 1.0 / (3.0 * nu_lb + 0.5)

    ops = _OPS[(ndim, _use_f64)]
    fdtype = ops["fdtype"]
    xlb_eq, xlb_stream, xlb_macro, xlb_bgk = (
        ops["eq"],
        ops["stream"],
        ops["macro"],
        ops["bgk"],
    )

    if ndim == 2:
        # (nx, ny, 1, 2) → (2, nx, ny)
        u0 = jnp.moveaxis(v0[:, :, 0, :], -1, 0).astype(fdtype) * scale_eff
    else:
        # (nx, ny, nz, 3) → (3, nx, ny, nz)
        u0 = jnp.moveaxis(v0, -1, 0).astype(fdtype) * scale_eff

    out = {}
    if obstacle is None:
        f0 = _periodic_populations(u0, omega, ops)

        def body(f: jnp.ndarray, _: Any) -> tuple[jnp.ndarray, None]:
            f_s = xlb_stream(f)
            rho_s, u_s = xlb_macro(f_s)
            feq = xlb_eq(rho_s, u_s)
            return xlb_bgk(f_s, feq, rho_s, u_s, omega), None

        f_final, _ = _checkpointed_scan(body, f0, steps_eff)
        _rho_f, u_out = xlb_macro(f_final)
    else:
        u_in = inflow_profile.astype(fdtype) * scale_eff
        f_final, force = _channel_run(u0, u_in, omega, steps, _sub_k, dx, obstacle, ops)
        rho_f, u_out = xlb_macro(f_final)
        fluid = ~_channel_links(u0.shape[1:], dx, obstacle)[0]
        u_out = jnp.where(fluid, u_out, 0.0)
        # Lattice → physical: pressure p = cs² (ρ - 1) in (dx/dt)², force per
        # unit depth in dx³/dt² (ρ = 1).
        p = (rho_f[0] - 1.0) / 3.0 / scale_eff**2
        out["pressure"] = jnp.where(fluid, p, 0.0)[:, :, None].astype(jnp.float32)
        out["body_force"] = (force * dx**3 / dt_eff**2).astype(jnp.float32)

    if ndim == 2:
        # (2, nx, ny) → (nx, ny, 1, 2), lattice → physical units
        result = jnp.moveaxis(u_out, 0, -1)[:, :, None, :] / scale_eff
    else:
        # (3, nx, ny, nz) → (nx, ny, nz, 3), lattice → physical units
        result = jnp.moveaxis(u_out, 0, -1) / scale_eff

    # Cast back to float32 when running in float64 gradient mode so that outputs
    # always have a consistent dtype regardless of the computation path.
    if _use_f64:
        result = result.astype(jnp.float32)
    out["result"] = result
    return out


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """JIT-compiled forward pass for the XLB solver."""
    return xlb_fwd(**inputs)


def _concrete_inputs(inputs: InputSchema) -> dict:
    """Inputs as a dict with Python-float viscosity and dt, and the grid spacing."""
    d = inputs.model_dump()
    for key in ("viscosity", "dt"):
        d[key] = float(d[key][0])
    d["dx"] = inputs.dx
    return d


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the XLB forward solver on the given inputs."""
    d = _concrete_inputs(inputs)
    # Run the forward pass in the same precision as the VJP so apply() and
    # the VJP compute the same function. Without this the FD check calls
    # apply() in one precision while the VJP runs in another, and float32
    # quantisation noise swamps the FD numerator at fine ε (omega≈2 at low
    # viscosity). Precision follows _USE_F64_DIFF (see XLB_VJP_FP32). xlb_fwd
    # casts float64 output back to float32 before returning so the schema
    # contract holds either way.
    d["_use_f64"] = _USE_F64_DIFF
    d["_sub_k"] = _sub_steps(d)
    return apply_jit(d)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """Compute the vector-Jacobian product for the XLB solver."""
    d = _concrete_inputs(inputs)
    vjp_inputs = {k for k in vjp_inputs if d.get(k) is not None}
    if not vjp_inputs:
        return {}
    return jax_vjp(
        _diff_apply(_sub_steps(d), d["dx"]),
        inputs,
        vjp_inputs,
        vjp_outputs,
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: Any) -> dict:
    """Shapes of ``result`` (as v0) and, with an obstacle, pressure and body_force."""
    v0_info = abstract_inputs.v0
    shape = tuple(v0_info["shape"] if isinstance(v0_info, dict) else v0_info.shape)
    out = {"result": {"shape": shape, "dtype": "float32"}}
    if abstract_inputs.obstacle is not None:
        out["pressure"] = {"shape": (*shape[:2], 1), "dtype": "float32"}
        out["body_force"] = {"shape": (abstract_inputs.steps, 2), "dtype": "float32"}
    return out


# ---------------------------------------------------------------------------
# VJP / JVP plumbing
# ---------------------------------------------------------------------------


@functools.cache
def _diff_apply(sub_k: int, dx: float) -> Any:
    """Forward pass differentiated by jax_vjp, for one (_sub_k, dx).

    Runs in float64 by default, or float32 when XLB_VJP_FP32=1 (see
    _USE_F64_DIFF). viscosity and dt arrive as traced (1,) arrays.
    """
    _diff_dtype = jnp.float64 if _USE_F64_DIFF else jnp.float32

    def fn(inputs: dict) -> dict:
        fwd_kwargs = dict(inputs)
        for k in ("v0", "inflow_profile"):
            if fwd_kwargs.get(k) is not None:
                fwd_kwargs[k] = jnp.asarray(fwd_kwargs[k], dtype=_diff_dtype)
        for k in ("viscosity", "dt"):
            fwd_kwargs[k] = jnp.asarray(fwd_kwargs[k], dtype=_diff_dtype).reshape(())
        fwd_kwargs["dx"] = dx
        return xlb_fwd(
            _use_f64=_USE_F64_DIFF,
            _sub_k=sub_k,
            **fwd_kwargs,
        )

    return fn
