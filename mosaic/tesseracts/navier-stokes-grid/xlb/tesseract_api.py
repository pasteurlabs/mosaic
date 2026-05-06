import os

# Force XLA_FLAGS to disable cublaslt and GEMM autotuning.
# --xla_gpu_autotune_level=0 prevents DEVICE_TYPE_INVALID failures on V100 (CC 7.0)
# with JAX 0.10.x when cuBLAS GEMM autotuner fails to identify the GPU device type.
# Using os.environ assignment (not setdefault) to override any Dockerfile ENV.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_autotune_level" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_autotune_level=0").strip()
del _xla_flags

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import xlb
import xlb.velocity_set

# Enable 64-bit floats in JAX.  Must be set before any JAX computation.
# This is required so that the VJP/JVP paths can run the LBM in float64 to
# avoid float32 cancellation errors that corrupt gradients near omega≈2.
jax.config.update("jax_enable_x64", True)
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable


class InputSchema(make_differentiable(
    _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
)):
    pass
class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result", "drag"])):
    pass


from xlb.compute_backend import ComputeBackend

# XLB operator imports
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stream import Stream
from xlb.precision_policy import PrecisionPolicy

# ---------------------------------------------------------------------------
# XLB one-time initialisation (needed to instantiate velocity sets)
# ---------------------------------------------------------------------------


def _make_ops(vset, pp, cb, fdtype, kind: str):  # mosaic:init
    """Build the XLB operator bundle for a given (ndim, precision, collision kind).

    `kind` selects the collision operator: "bgk" (default, fast) or "kbc"
    (entropic-stabilised; only available for D2Q9 and D3Q27).  KBC is used in
    the 2-D cylinder-wake (obstacle) path to keep the LBM dynamics stable at
    the sub-grid shear layer next to the solid mask, where plain BGK develops
    NaNs as the effective Re/omega climbs.  KBC has the same call signature as
    BGK so it is a drop-in replacement downstream.

    Re-initialise XLB global state with the correct velocity set before
    constructing operators.  XLB stores internal bookkeeping (e.g. the number
    of discrete velocities q) in a global singleton that is read by KBC during
    construction.  If xlb.init() was last called with D2Q9 (q=9) and we now
    construct a D3Q27 KBC bundle, the mismatched global q=9 vs D3Q27's q=27
    causes ``dot_general requires contracting dimensions to have the same shape,
    got (9,) and (27,)`` at jit-compile time.  Re-calling xlb.init() here is
    idempotent for BGK (which does not read the global q) and fixes the KBC
    crash for 3-D runs.
    """
    xlb.init(velocity_set=vset, default_backend=cb, default_precision_policy=pp)
    if kind == "kbc":
        collide = KBC(velocity_set=vset, precision_policy=pp, compute_backend=cb)
    else:
        collide = BGK(velocity_set=vset, precision_policy=pp, compute_backend=cb)
    return dict(
        C=jnp.array(vset.c, dtype=fdtype),
        W=jnp.array(vset.w, dtype=fdtype),
        eq=QuadraticEquilibrium(
            velocity_set=vset, precision_policy=pp, compute_backend=cb
        ),
        stream=Stream(velocity_set=vset, precision_policy=pp, compute_backend=cb),
        macro=Macroscopic(velocity_set=vset, precision_policy=pp, compute_backend=cb),
        bgk=collide,
        fdtype=fdtype,
    )


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
_OPS: dict[tuple[int, bool, str], dict] = {
    # Plain BGK bundle (used for periodic / inflow-only cases)
    (2, False, "bgk"): _make_ops(
        _vsets[(2, False)],
        PrecisionPolicy.FP32FP32,
        ComputeBackend.JAX,
        jnp.float32,
        "bgk",
    ),
    (2, True, "bgk"): _make_ops(
        _vsets[(2, True)],
        PrecisionPolicy.FP64FP64,
        ComputeBackend.JAX,
        jnp.float64,
        "bgk",
    ),
    (3, False, "bgk"): _make_ops(
        _vsets[(3, False)],
        PrecisionPolicy.FP32FP32,
        ComputeBackend.JAX,
        jnp.float32,
        "bgk",
    ),
    (3, True, "bgk"): _make_ops(
        _vsets[(3, True)],
        PrecisionPolicy.FP64FP64,
        ComputeBackend.JAX,
        jnp.float64,
        "bgk",
    ),
    # Entropic-stabilised KBC bundle (2-D and 3-D; 3-D requires D3Q27)
    (2, False, "kbc"): _make_ops(
        _vsets[(2, False)],
        PrecisionPolicy.FP32FP32,
        ComputeBackend.JAX,
        jnp.float32,
        "kbc",
    ),
    (2, True, "kbc"): _make_ops(
        _vsets[(2, True)],
        PrecisionPolicy.FP64FP64,
        ComputeBackend.JAX,
        jnp.float64,
        "kbc",
    ),
    (3, False, "kbc"): _make_ops(
        _vsets[(3, False)],
        PrecisionPolicy.FP32FP32,
        ComputeBackend.JAX,
        jnp.float32,
        "kbc",
    ),
    (3, True, "kbc"): _make_ops(
        _vsets[(3, True)],
        PrecisionPolicy.FP64FP64,
        ComputeBackend.JAX,
        jnp.float64,
        "kbc",
    ),
}

# Concrete (non-JAX) components of the D2Q9 lattice velocities extracted at
# module load time, before any JAX tracing.  Used in _compute_drag_lbm for
# Python-level branching, which must not operate on traced JAX arrays.
import numpy as _np

_D2Q9_C: list[tuple[int, int]] = [
    (int(cx), int(cy))
    for cx, cy in zip(
        _np.array(_vsets[(2, False)].c)[0],
        _np.array(_vsets[(2, False)].c)[1],
    )
]
_D2Q9_CX: list[float] = [float(c[0]) for c in _D2Q9_C]
del _np


# ---------------------------------------------------------------------------
# Helper: quadratic equilibrium (used for BCs with arbitrary slice shapes)
# This mirrors XLB QuadraticEquilibrium.jax_implementation but operates on
# arbitrary spatial slice shapes (not just full-domain arrays), so it remains
# a standalone helper for the boundary-condition code.
# ---------------------------------------------------------------------------


def _feq(  # mosaic:physics
    C: jnp.ndarray, W: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    """Quadratic equilibrium distribution (used for BCs only).

    Args:
        C:   Lattice velocities, shape (d, q).
        W:   Lattice weights, shape (q,).
        rho: Density, shape (1, *spatial).
        u:   Velocity, shape (d, *spatial).

    Returns:
        feq, shape (q, *spatial).
    """
    ndim = u.shape[0]
    w = W.reshape((-1,) + (1,) * ndim)  # (q, 1...) broadcast
    cu = jnp.einsum("dq,d...->q...", C, u)  # (q, *spatial)
    usqr = jnp.sum(u**2, axis=0, keepdims=True)  # (1, *spatial)
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)


def _make_obstacle_mask_xlb(  # mosaic:init
    obstacle: dict | None, spatial: tuple
) -> jnp.ndarray | None:
    """Rasterize geometric obstacle to a boolean JAX mask, shape (1, *spatial)."""
    if obstacle is None or not obstacle.get("shape"):
        return None
    ndim = len(spatial)
    nx, ny = spatial[0], spatial[1]
    cx = obstacle["center"][0] * nx
    cy = obstacle["center"][1] * ny
    r = obstacle["radius"] * nx  # isotropic grid
    if obstacle["shape"] in ("cylinder", "CYLINDER"):
        x = jnp.arange(nx, dtype=jnp.float32)
        y = jnp.arange(ny, dtype=jnp.float32)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        disk = (X - cx) ** 2 + (Y - cy) ** 2 < r**2  # (nx, ny)
        if ndim == 2:
            return disk[None, :, :]  # (1, nx, ny) — broadcast over q
        # 3D: infinite cylinder along z
        nz = spatial[2]
        return jnp.broadcast_to(disk[None, :, :, None], (1, nx, ny, nz))
    raise ValueError(f"XLB: unsupported obstacle shape {obstacle['shape']!r}")


def _compute_drag_lbm(  # mosaic:physics
    f: jnp.ndarray,
    C: jnp.ndarray,
    obs_mask_1: jnp.ndarray,
    dx: float,
    dt: float,
) -> jnp.ndarray:
    """Compute x-direction drag via Ladd's momentum exchange method (2-D D2Q9).

    Per-link MEM: for each fluid cell x_f and each lattice direction i such
    that (x_f + c_i) is a solid cell, the bounce-back exchanges lattice
    momentum 2 * f_i(x_f) * c_i per lattice step.  Summing over all such
    (x_f, i) pairs and extracting the x-component gives the total x-momentum
    transferred from fluid to obstacle per lattice step.

    Unit conversion (lattice → physical):
        Each link transfers lattice momentum (ρ_lb · dx^D · dx/dt) per dt
        seconds.  For 2-D per unit depth (D=2) this gives
            F_phys = Σ_links 2 f_i c_ix · ρ_phys · dx^3 / dt^2.
        XLB uses ρ_lb = 1 and we adopt the ns-grid convention of ρ_phys = 1
        (force-per-density, matching jax-cfd and phiflow surface integrals),
        so the conversion factor is simply dx^3 / dt^2.

    The previous implementation (a) only checked the cardinal-x neighbour
    for every cx≠0 direction (triple-counting cardinal surface cells and
    missing diagonal-only surface cells) and (b) applied the wrong lattice
    factor dx^2/dt^2 = 1/scale^2 instead of dx^3/dt^2, leaving the drag
    magnitude off by roughly 1/dx relative to the reference finite-volume
    drag integrals.  Both are fixed here.

    Args:
        f:          Populations after final step, shape (q, nx, ny).
        C:          Lattice velocities, shape (2, q).  Unused (kept for
                    signature compatibility); the concrete D2Q9 cx/cy are
                    taken from the module-level `_D2Q9_C` list to avoid
                    tracing issues inside jit.
        obs_mask_1: Boolean solid mask, shape (1, nx, ny).
        dx:         Physical grid spacing (domain_extent / nx).
        dt:         Physical timestep.

    Returns:
        Shape (1,) float32 — drag in physical units (force per density, 2-D).
    """
    obs = obs_mask_1[0]  # (nx, ny) bool
    fluid = ~obs

    # Per-link neighbour check: link i from fluid cell x_f bounces off solid
    # at (x_f + c_i), so we must shift the mask by -c_i to bring "solid at
    # x_f + c_i" onto the fluid cell index x_f.  Using the concrete D2Q9
    # integer offsets from `_D2Q9_C` (module-level Python tuples) keeps the
    # loop unrolled and avoids indexing the traced JAX array C.
    q = len(_D2Q9_C)

    # Match accumulator dtype to f to avoid dtype mixing when _use_f64=True.
    total = jnp.zeros((), dtype=f.dtype)
    for qi in range(q):
        cxi, cyi = _D2Q9_C[qi]
        if cxi == 0:
            continue  # only x-drag; links with cx=0 carry no x-momentum
        fi = f[qi]  # (nx, ny)
        # Solid at (x_f + c_i) → shift obs by -c_i so solid_nbr[x_f] is True
        # when (x_f + c_i) is solid.
        solid_nbr = jnp.roll(obs, shift=(-cxi, -cyi), axis=(0, 1))
        link = fluid & solid_nbr
        # Ladd's MEM: force ON the obstacle in +x from a single bounce-back
        # link is +2 f_i c_ix per lattice step.  The existing jax-cfd /
        # phiflow drag reports a NEGATIVE scalar for a cylinder in uniform
        # +x flow (force on fluid in +x = -force on obstacle in +x).  To
        # match that sign convention we accumulate with a leading minus so
        # the scalar we return has the same sign as the reference solvers'
        # surface-integral drag.
        contrib = jnp.sum(jnp.where(link, 2.0 * fi * cxi, 0.0))
        total = total - contrib

    # Lattice → physical: multiply by ρ_phys · dx^{D+1} / dt^2.  With ρ_phys
    # and ρ_lb both = 1 and D=2 (2-D per unit depth), this is dx^3 / dt^2.
    drag = (total * (dx**3) / (dt**2)).astype(jnp.float32)
    return jnp.reshape(drag, (1,))


def xlb_fwd(  # mosaic:physics
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    boundary_conditions: dict | None = None,
    obstacle: dict | None = None,
    inflow_profile: jnp.ndarray | None = None,
    _use_f64: bool = False,
    _sub_k: int | None = None,
    _collision_kind_override: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Run a 2D or 3D incompressible LBM simulation from an initial velocity field.

    Accepts physical units; internally converts to lattice units via:
        dx = domain_extent / nx
        scale = dt / dx          (u_lb = u_phys * scale)
        nu_lb = viscosity * dt / dx**2

    LBM Mach-number compressibility error is O(Ma²) where Ma = u_lb / cs,
    cs = 1/√3.  When Ma is large (e.g. dt=0.05 at N=64 gives Ma≈0.88),
    the error floor can be 10–100× above peer NS solvers.  To suppress this,
    we automatically compute the number of internal sub-steps k = ceil(Ma/0.1)
    needed to keep Ma ≤ 0.1, then run with dt_eff = dt/k and steps_eff = steps*k.
    This preserves the total physical time T = dt*steps exactly while reducing
    the lattice Ma by factor k, cutting the O(Ma²) floor by k².  The external
    interface (dt, steps, outputs) is unchanged.

    Args:
        v0:            Initial velocity in physical units, shape (nx, ny, 1, 2) or (nx, ny, nz, 3).
        viscosity:     Physical kinematic viscosity.
        dt:            Physical timestep size.
        steps:         Number of LBM timesteps.
        domain_extent: Side length of the isotropic domain.
        inflow_profile: Optional 1-D u_x(y) profile shape (ny,). Applied at x=0 each step.
        _use_f64:      If True, run the entire LBM computation in float64 for accurate
                       gradients. Output is cast back to float32 before returning.
                       This prevents float32 cancellation errors in the VJP/JVP paths,
                       especially near omega≈2 where small perturbations amplify rapidly.

    Returns:
        (result, drag): result same shape as v0; drag shape (1,) or None.
    """
    import math as _math

    ndim = v0.shape[-1]  # 2 or 3

    # Physical → lattice unit conversion (needed early for Ma-based sub-step selection)
    dx = domain_extent / v0.shape[0]
    scale = dt / dx  # u_lb = u_phys * scale  (at full dt)

    # ── Automatic sub-stepping to suppress O(Ma²) compressibility error ──────
    # LBM compressibility error is O(Ma²) where Ma = u_lb / cs = u_phys * scale / cs.
    # cs = 1/√3 for the D2Q9 / D3Q27 lattice.  We target Ma ≤ 0.1 which brings
    # the O(Ma²) error floor to ~0.01, within 5× of incompressible NS peers.
    # Sub-stepping k internal LBM steps per external dt reduces the effective
    # lattice timestep to dt_eff = dt/k, hence scale_eff = scale/k and Ma_eff =
    # Ma/k.  Total physical time T = dt * steps is preserved exactly.
    #
    # We use u_max = 1.0 as a conservative upper bound (safe for all benchmark ICs:
    # TGV max = 1.0, multimode normalised to 0.3, inflow profiles ≤ 0.5).
    # This ensures _sub_k is determined solely by the static (non-traced) inputs
    # dt and domain_extent, making it a compile-time constant for JAX JIT.
    _cs = 1.0 / _math.sqrt(3.0)
    _MA_TARGET = 0.1
    _u_max_conservative = 1.0  # upper bound on physical velocity for all benchmark ICs
    if _sub_k is None:
        # Compute _sub_k from the concrete (non-traced) scale value.  This works
        # when dt is a Python float (apply/apply_jit path) but would raise
        # ConcretizationTypeError when dt is a JAX abstract value (VJP/JVP path).
        # Callers that differentiate w.r.t. dt must pre-compute _sub_k from the
        # concrete primal dt and pass it explicitly — see _run_forward_f64.
        _Ma_full = _u_max_conservative * float(scale) / _cs
        _sub_k = max(1, _math.ceil(_Ma_full / _MA_TARGET))

    # Apply sub-stepping: use effective dt and total step count
    dt_eff = dt / _sub_k
    steps_eff = steps * _sub_k
    scale_eff = dt_eff / dx  # u_lb = u_phys * scale_eff
    nu_lb = viscosity * dt_eff / dx**2
    omega = 1.0 / (3.0 * nu_lb + 0.5)

    # Use entropic-stabilised KBC collision whenever an obstacle is present OR
    # when omega > 1.8 (low-viscosity BGK goes unstable as tau→0.5).
    # KBC has the same call signature and same equilibrium targets as BGK so
    # the rest of the loop is unchanged.  KBC is supported for D2Q9 (2-D) and
    # D3Q27 (3-D); the guard below falls back to BGK if the bundle is missing.
    #
    # When called inside jax.jit (VJP/JVP path), viscosity and dt are traced
    # arrays so omega is a traced value — `omega > 1.8` cannot be evaluated as
    # a Python bool.  The caller pre-computes _collision_kind_override from
    # concrete primal values (same pattern as _sub_k) and passes it here.
    if _collision_kind_override is not None:
        _collision_kind = _collision_kind_override
    else:
        _needs_kbc = (obstacle is not None) or (omega > 1.8)
        _collision_kind = "kbc" if _needs_kbc else "bgk"
    # Safety net: fall back to bgk if the kbc bundle is not registered.
    if _collision_kind == "kbc" and (ndim, _use_f64, "kbc") not in _OPS:
        _collision_kind = "bgk"
    ops = _OPS[(ndim, _use_f64, _collision_kind)]
    fdtype, C, W = ops["fdtype"], ops["C"], ops["W"]
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

    spatial = u0.shape[1:]
    rho0 = jnp.ones((1,) + spatial, dtype=fdtype)

    # Initialise populations from equilibrium at rho=1 using XLB operator
    f0 = xlb_eq(rho0, u0)

    # ── Standard periodic / inflow / obstacle mode ───────────────────────────
    obs_mask = _make_obstacle_mask_xlb(obstacle, u0.shape[1:])
    if obs_mask is not None and _use_f64:
        obs_mask = obs_mask.astype(jnp.bool_)

    # No-slip walls at y=0 and y=ny-1 (equilibrium BC with u=0).
    wall_y_noslip = (
        boundary_conditions is not None
        and boundary_conditions.get("y_lo", {}).get("type") == "no_slip"
        and boundary_conditions.get("y_hi", {}).get("type") == "no_slip"
    )

    if inflow_profile is not None and ndim == 2:
        nx_s, ny_s = spatial
        prof_len = inflow_profile.shape[0]
        if prof_len != ny_s:
            src_y = jnp.linspace(0, 1, prof_len)
            dst_y = jnp.linspace(0, 1, ny_s)
            ux_in_lb = (
                jnp.interp(dst_y, src_y, inflow_profile.astype(fdtype)) * scale_eff
            )
        else:
            ux_in_lb = inflow_profile.astype(fdtype) * scale_eff  # (ny,)

        # Build equilibrium inflow distribution at x=0
        # rho=1, u_x=ux_in_lb[j], u_y=0 for each j
        rho_in = jnp.ones((1, 1, ny_s), dtype=fdtype)
        ux_in_2d = ux_in_lb[None, None, :]  # (1, 1, ny) → broadcast
        uy_in_2d = jnp.zeros_like(ux_in_2d)
        u_in_lb = jnp.concatenate([ux_in_2d, uy_in_2d], axis=0)  # (2, 1, ny)
        f_inflow = xlb_eq(rho_in, u_in_lb)  # (9, 1, ny)

        def body(f, _):
            # Stream using XLB Stream operator
            f_s = xlb_stream(f)
            # Compute macroscopic quantities
            rho_s, u_s = xlb_macro(f_s)
            # BGK collision using XLB operators
            feq = xlb_eq(rho_s, u_s)
            f_next = xlb_bgk(f_s, feq, rho_s, u_s, omega)
            # Apply obstacle BC if present
            if obs_mask is not None:
                rho_wall = jnp.ones_like(rho_s)
                u_zero = jnp.zeros_like(u_s)
                feq_wall = xlb_eq(rho_wall, u_zero)
                f_next = jnp.where(obs_mask, feq_wall, f_next)
            # Apply inflow BC at x=0: set populations at x=0 slice
            f_next = f_next.at[:, 0, :].set(f_inflow[:, 0, :])
            # Apply no-slip walls at y=0 and y=ny-1 (equilibrium at u=0)
            if wall_y_noslip:
                _ny = f_next.shape[2]
                _rho_yw = jnp.ones((1, nx_s, 1), dtype=fdtype)
                _u_yw = jnp.zeros((2, nx_s, 1), dtype=fdtype)
                _f_yw = xlb_eq(_rho_yw, _u_yw)
                f_next = f_next.at[:, :, 0].set(_f_yw[:, :, 0])
                f_next = f_next.at[:, :, _ny - 1].set(_f_yw[:, :, 0])
            drag_step = (
                _compute_drag_lbm(f_next, C, obs_mask, dx, dt_eff)
                if obs_mask is not None
                else jnp.zeros((1,), dtype=fdtype)
            )
            return f_next, drag_step
    else:
        _nx_g, _ny_g = spatial[0], spatial[1]

        def body(f, _):
            # Stream using XLB Stream operator
            f_s = xlb_stream(f)
            # Compute macroscopic quantities
            rho_s, u_s = xlb_macro(f_s)
            # BGK collision using XLB operators
            feq = xlb_eq(rho_s, u_s)
            f_next = xlb_bgk(f_s, feq, rho_s, u_s, omega)
            # Apply obstacle BC if present
            if obs_mask is not None:
                rho_wall = jnp.ones_like(rho_s)
                u_zero = jnp.zeros_like(u_s)
                feq_wall = xlb_eq(rho_wall, u_zero)
                f_next = jnp.where(obs_mask, feq_wall, f_next)
            # Apply no-slip walls at y=0 and y=ny-1 (equilibrium at u=0)
            if wall_y_noslip:
                _ny = f_next.shape[2]
                _rho_yw = jnp.ones((1, _nx_g, 1), dtype=fdtype)
                _u_yw = jnp.zeros((2, _nx_g, 1), dtype=fdtype)
                _f_yw = xlb_eq(_rho_yw, _u_yw)
                f_next = f_next.at[:, :, 0].set(_f_yw[:, :, 0])
                f_next = f_next.at[:, :, _ny - 1].set(_f_yw[:, :, 0])
            drag_step = (
                _compute_drag_lbm(f_next, C, obs_mask, dx, dt_eff)
                if obs_mask is not None
                else jnp.zeros((1,), dtype=fdtype)
            )
            return f_next, drag_step

    f_final, drag_history = jax.lax.scan(body, f0, None, length=steps_eff)

    # Extract macroscopic velocity using XLB Macroscopic operator
    rho_f, u_out = xlb_macro(f_final)  # rho: (1, *spatial), u: (d, *spatial)

    # Drag computation via Ladd's momentum exchange method (2-D only).
    # We use the tail-window mean over the last half of the simulation to
    # capture the time-averaged drag (important for periodic flows like Re=100
    # cylinder wake where the instantaneous drag at the final step can be far
    # from the mean).  drag_history has shape (steps_eff, 1).
    drag = None
    if obs_mask is not None and ndim == 2:
        n_tail = max(1, steps_eff // 2)
        drag = jnp.mean(drag_history[-n_tail:], axis=0).astype(jnp.float32)

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
        if drag is not None:
            drag = drag.astype(jnp.float32)

    return result, drag


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:  # mosaic:io
    result, drag = xlb_fwd(**inputs)
    out = dict(result=result)
    out["drag"] = drag if drag is not None else jnp.zeros((1,), dtype=jnp.float32)
    return out


def _unpack_scalars(d: dict) -> dict:  # mosaic:io
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in ("viscosity", "dt"):
        if key in d:
            val = d[key]
            if not isinstance(val, (int, float)):
                d[key] = float(val[0])
    return d


def apply(inputs: InputSchema) -> OutputSchema:
    d = _unpack_scalars(inputs.model_dump())
    # Run forward pass in float64 so apply() and vjp_jit() compute the same
    # function. Without this the FD check calls apply() in float32 while
    # vjp_jit() runs in float64, and float32 quantisation noise swamps the FD
    # numerator at fine ε (omega≈2 at low viscosity). xlb_fwd casts output to
    # float32 before returning so the schema contract holds.
    d["_use_f64"] = True
    return apply_jit(d)


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt,inflow_profile:autodiff
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
    """Output shape equals input v0 shape; drag is always shape (1,)."""
    v0_info = abstract_inputs.v0
    if isinstance(v0_info, dict):
        shape = tuple(v0_info["shape"])
        dtype = v0_info.get("dtype", "float32")
    else:
        shape = v0_info.shape
        dtype = "float32"
    out = {
        "result": {"shape": shape, "dtype": dtype},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
    raw = abstract_inputs.model_dump()
    obstacle_raw = raw.get("obstacle") or {}
    _has_obstacle = bool(
        (
            obstacle_raw.get("shape")
            if isinstance(obstacle_raw, dict)
            else getattr(obstacle_raw, "shape", None)
        )
    )
    return out


# ---------------------------------------------------------------------------
# VJP / JVP plumbing
# ---------------------------------------------------------------------------
#
# xlb exposes four differentiable array-valued inputs: v0, viscosity, dt,
# inflow_profile.  The VJP must return a gradient for every
# input requested in `vjp_inputs` under its *own* path key, otherwise the
# tesseract_jax dispatcher falls back to a NaN filler (in particular the
# drag_opt harness asks for the `inflow_profile` gradient — returning only
# `{"v0": ...}` would leave the optimiser with zero/NaN gradient for the
# inflow DOF, which is exactly the symptom the suite reported).
#
# We build a single `_fwd_all` that takes the full diff-input bundle as a
# dict, promotes each array to float64, runs `xlb_fwd`, and returns the
# requested outputs.  jax.vjp then differentiates w.r.t. *every* requested
# input in one pass, and we emit the gradients under their original key.
# This mirrors the jax-cfd / phiflow pattern (filter_func + flatten_with_paths)
# but without the eqx.filter_jit boundary that previously blocked reverse
# mode here.

_DIFF_INPUT_KEYS: tuple[str, ...] = (
    "v0",
    "viscosity",
    "dt",
    "inflow_profile",
)

# Module-level cache: (v0_shape, steps, present_keys, vjp_outputs) -> jit-compiled fn.
# Avoids recompiling the XLA kernel on every HTTP request (optimizer iteration).
_vjp_compiled_cache: dict = {}


def _scalar_f64(x):  # mosaic:util
    """Coerce a scalar-like (Python float, 0-D / 1-D array) to a float64 scalar."""
    if hasattr(x, "astype"):
        return jnp.asarray(x, dtype=jnp.float64)
    return jnp.asarray(x, dtype=jnp.float64)


def _run_forward_f64(
    inputs: dict, diff_bundle: dict
) -> tuple:  # mosaic:grad:v0,viscosity,dt,inflow_profile:autodiff
    """Run xlb_fwd in float64 with diff inputs overridden from diff_bundle.

    Non-diff inputs (steps, boundary_conditions, obstacle, domain_extent) are
    read from ``inputs``.  Returns (result, drag) with float32 dtype.

    Pre-computes ``_sub_k`` from the concrete (non-traced) ``inputs["dt"]`` so
    that ``xlb_fwd`` does not need to call ``float()`` on a potentially-traced
    JAX array when dt is in the diff bundle (VJP/JVP w.r.t. dt path).
    """
    import math as _math_run

    fwd_kwargs = {}
    for k in (
        "steps",
        "domain_extent",
        "boundary_conditions",
        "obstacle",
    ):
        if k in inputs:
            fwd_kwargs[k] = inputs[k]

    # v0: always required
    v0 = diff_bundle.get("v0", inputs["v0"])
    fwd_kwargs["v0"] = jnp.asarray(v0, dtype=jnp.float64)

    # viscosity / dt: xlb_fwd consumes as Python-ish scalars via fdtype casts;
    # passing the 0-D float64 array is fine because _unpack_scalars only runs
    # at the public `apply()` boundary, not here.
    for k in ("viscosity", "dt"):
        src = diff_bundle.get(k, inputs.get(k))
        if src is None:
            continue
        fwd_kwargs[k] = _scalar_f64(src)

    # Optional fields: only pass when the caller actually supplied them
    # (either via diff_bundle or via the primal input dict).  Passing None
    # is also fine because xlb_fwd treats both as "no such BC".
    for k in ("inflow_profile",):
        if k in diff_bundle and diff_bundle[k] is not None:
            fwd_kwargs[k] = jnp.asarray(diff_bundle[k], dtype=jnp.float64)
        elif inputs.get(k) is not None:
            fwd_kwargs[k] = jnp.asarray(inputs[k], dtype=jnp.float64)

    # Pre-compute _sub_k from the concrete (primal) dt so that xlb_fwd never
    # needs to call float() on a JAX abstract tracer.  When dt is in the diff
    # bundle, it becomes a traced dual number inside jax.vjp / jax.jvp, which
    # would cause ConcretizationTypeError if xlb_fwd tried to extract a Python
    # float from it.  Using the concrete inputs["dt"] here keeps _sub_k static.
    _dt_concrete = float(inputs.get("dt", 0.05))
    _v0_shape = inputs["v0"]
    if hasattr(_v0_shape, "shape"):
        _nx = _v0_shape.shape[0]
    else:
        _nx = _v0_shape["shape"][0] if isinstance(_v0_shape, dict) else 16
    _domain_extent_concrete = float(inputs.get("domain_extent", 1.0))
    _dx_concrete = _domain_extent_concrete / _nx
    _scale_concrete = _dt_concrete / _dx_concrete
    _cs_run = 1.0 / _math_run.sqrt(3.0)
    _Ma_full_run = 1.0 * _scale_concrete / _cs_run
    _sub_k_concrete = max(1, _math_run.ceil(_Ma_full_run / 0.1))
    fwd_kwargs["_sub_k"] = _sub_k_concrete

    # Pre-compute _collision_kind from concrete primal values so xlb_fwd never
    # evaluates `omega > 1.8` on a traced JAX value inside jax.jit.
    _visc_concrete = float(inputs.get("viscosity", 0.001))
    _dt_eff_concrete = _dt_concrete / _sub_k_concrete
    _nu_lb_concrete = _visc_concrete * _dt_eff_concrete / _dx_concrete**2
    _omega_concrete = 1.0 / (3.0 * _nu_lb_concrete + 0.5)
    _obstacle_concrete = inputs.get("obstacle")
    _needs_kbc_concrete = (_obstacle_concrete is not None) or (_omega_concrete > 1.8)
    _ndim = _v0_shape.shape[-1] if hasattr(_v0_shape, "shape") else 2
    _ck = "kbc" if _needs_kbc_concrete else "bgk"
    if _ck == "kbc" and (_ndim, True, "kbc") not in _OPS:
        _ck = "bgk"
    fwd_kwargs["_collision_kind_override"] = _ck

    result, drag = xlb_fwd(_use_f64=True, **fwd_kwargs)
    return result, drag


def _build_diff_bundle(
    inputs: dict, include: tuple[str, ...]
) -> dict:  # mosaic:grad:v0,viscosity,dt,inflow_profile:autodiff
    """Build a {path: value} dict for jax.vjp / jax.jvp over `include` keys.

    Only includes paths that are actually present (non-None) in the primal
    inputs, so jax.vjp never needs to trace through a Python None.  Scalar
    inputs (viscosity, dt) are promoted from 1-element arrays or plain floats
    to 0-D float64 arrays.
    """
    bundle: dict = {}
    for k in include:
        if k not in _DIFF_INPUT_KEYS:
            continue
        v = inputs.get(k)
        if v is None:
            continue
        if k in ("viscosity", "dt"):
            bundle[k] = _scalar_f64(v)
        else:
            bundle[k] = jnp.asarray(v, dtype=jnp.float64)
    return bundle


def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    """Reverse-mode VJP over any subset of diff inputs.

    Returns a dict keyed by the requested `vjp_inputs` paths.  Scalar grads
    (viscosity, dt) are reshaped to (1,) to match the declared input schema.
    """
    # Only keep requested paths that are actually present in inputs (so a
    # caller asking for inflow_profile when none was provided gets an empty
    # bundle — jax.vjp will refuse anyway, and the harness should have
    # filtered this upstream via the `differentiable_input_paths` check).
    present = tuple(
        k for k in vjp_inputs if k in _DIFF_INPUT_KEYS and inputs.get(k) is not None
    )
    if not present:
        return {}

    diff_bundle = _build_diff_bundle(inputs, present)

    # Build a cache key from static aspects of the computation graph.
    # diff_bundle and cotangent_vector are the only traced (variable) arguments.
    v0_src = inputs.get("v0")
    v0_shape = tuple(v0_src.shape) if hasattr(v0_src, "shape") else ()
    cache_key = (
        v0_shape,
        inputs.get("steps"),
        present,
        tuple(sorted(vjp_outputs)),
    )

    if cache_key not in _vjp_compiled_cache:
        # Capture static inputs in closure once; only bundle/cotan are traced.
        _inputs_frozen = inputs
        _vjp_outputs_frozen = vjp_outputs

        def _fwd_static(bundle):
            result, drag = _run_forward_f64(_inputs_frozen, bundle)
            out = {}
            if "result" in _vjp_outputs_frozen:
                out["result"] = result
            if "drag" in _vjp_outputs_frozen:
                out["drag"] = (
                    drag if drag is not None else jnp.zeros((1,), dtype=jnp.float32)
                )
            return out

        @jax.jit
        def _vjp_compiled(bundle, cotan):
            _, vjp_func = jax.vjp(_fwd_static, bundle)
            return vjp_func(cotan)[0]

        _vjp_compiled_cache[cache_key] = _vjp_compiled

    grads = _vjp_compiled_cache[cache_key](diff_bundle, cotangent_vector)

    out: dict = {}
    for k, g in grads.items():
        g = g.astype(jnp.float32)
        if k in ("viscosity", "dt"):
            # Canonical schema has shape (1,); jax returns 0-D for promoted scalars.
            g = jnp.atleast_1d(g)
        elif g.ndim == 0:
            g = jnp.atleast_1d(g)
        out[k] = g
    return out
