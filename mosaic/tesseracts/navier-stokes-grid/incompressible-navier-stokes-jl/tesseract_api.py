from pathlib import Path
from typing import Any

import juliacall
import mosaic_shared
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable


class InputSchema(make_differentiable(
    _CanonicalInputSchema, ["v0", "viscosity", "dt"]
)):
    pass
class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    pass


# ---------------------------------------------------------------------------
# Julia module initialisation
# ---------------------------------------------------------------------------

jl = juliacall.newmodule("ins_ns")
jl.seval('using Pkg; Pkg.activate(ENV["JULIA_PROJECT"])')
jl.seval("using IncompressibleNavierStokes, Zygote")
jl.include(
    str(
        Path(mosaic_shared.__file__).parent
        / "problems"
        / "navier_stokes_grid"
        / "ns_solver.jl"
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_julia(arr: np.ndarray):  # mosaic:util
    """Convert a numpy array to a Julia array (zero-copy if contiguous)."""
    return juliacall.convert(jl.Array, np.ascontiguousarray(arr))


def _to_numpy(jl_arr) -> np.ndarray:  # mosaic:util
    """Convert a Julia array back to numpy."""
    return np.asarray(jl_arr).copy()


def _compute_drag_numpy(  # mosaic:physics
    ux: np.ndarray,
    pressure: np.ndarray,
    obstacle: dict | None,
    viscosity: float,
    domain_extent: float,
) -> np.ndarray | None:
    """Compute x-direction drag on obstacle via discrete surface integral (numpy).

    Uses the same scheme as the JAX solvers but in numpy (no gradient).

    Args:
        ux:          x-velocity (nx, ny), collocated.
        pressure:    pressure field (nx, ny).
        obstacle:    obstacle dict or None.
        viscosity:   kinematic viscosity ν.
        domain_extent: side length of domain.

    Returns:
        shape (1,) float32 or None.
    """
    if obstacle is None or not obstacle.get("shape"):
        return None
    nx, ny = ux.shape
    cx = obstacle["center"][0] * nx
    cy = obstacle["center"][1] * ny
    r = obstacle["radius"] * nx
    dx = domain_extent / nx

    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    solid_mask = (X - cx) ** 2 + (Y - cy) ** 2 < r**2
    fluid_mask = ~solid_mask

    solid_right = np.roll(solid_mask, -1, axis=0)
    solid_left = np.roll(solid_mask, 1, axis=0)
    surf_right = fluid_mask & solid_right
    surf_left = fluid_mask & solid_left

    p_drag = np.sum(
        np.where(surf_right, pressure * dx, 0.0)
        + np.where(surf_left, -pressure * dx, 0.0)
    )
    visc_drag = np.sum(
        np.where(surf_right, -viscosity * ux, 0.0)
        + np.where(surf_left, viscosity * ux, 0.0)
    )
    return np.array([p_drag + visc_drag], dtype=np.float32)


def _make_obstacle_mask(obstacle: dict, N: int) -> np.ndarray:  # mosaic:init
    """Build a binary float32 Brinkman mask of shape (N, N, 2) from obstacle dict.

    Obstacle coordinates are normalised fractions of domain_extent, so we
    multiply by N to get grid indices.  Points satisfying the circle equation
    (x-cx)^2 + (y-cy)^2 < r^2 are marked solid (mask = 1).
    """
    cx = obstacle["center"][0] * N
    cy = obstacle["center"][1] * N
    r = obstacle["radius"] * N
    x = np.arange(N, dtype=np.float32)
    y = np.arange(N, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    solid = ((X - cx) ** 2 + (Y - cy) ** 2 < r**2).astype(np.float32)
    # Broadcast scalar mask over both velocity components → (N, N, 2)
    return np.stack([solid, solid], axis=-1)


def _run(  # mosaic:physics
    v0: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    obstacle: dict | None = None,
    inflow_profile: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Forward pass.

    Returns a 2-tuple (result, drag).  drag is non-None only in channel mode
    (obstacle present).

    2-D: v0 (nx, ny, 1, 2) → result (nx, ny, 1, 2)
    3-D: v0 (nx, ny, nz, 3) → result (nx, ny, nz, 3)
    2-D channel: obstacle provided
                 → result (nx, ny, 1, 2)
                 Inflow at x=0 face: either inflow_profile (shape (N,), ux
                 only; uy=0) when given, else derived from v0[:, 0, 0].

    inflow_profile without an obstacle is not supported: plain inflow_profile
    in a periodic domain has no physical meaning in this solver (periodic
    pressure projection). Raises NotImplementedError.
    """
    # Channel mode: obstacle present. Inflow may be either an explicit
    # inflow_profile (drag_opt path) or derived from v0 (re20/re100-style).
    if obstacle is not None:
        # v0 is (nx, ny, 1, 2) — squeeze z=1 dimension
        v0_2d = np.asarray(v0, dtype=np.float32)[:, :, 0, :]  # (N, N, 2)
        N = v0_2d.shape[0]
        if inflow_profile is not None:
            # Explicit inflow_profile: 1-D u_x(y), shape (N,); u_y=0 on inlet.
            prof = np.asarray(inflow_profile, dtype=np.float32).reshape(-1)
            if prof.shape[0] != N:
                # Linear resample to the grid resolution.
                src_y = np.linspace(0.0, 1.0, prof.shape[0], dtype=np.float32)
                dst_y = np.linspace(0.0, 1.0, N, dtype=np.float32)
                prof = np.interp(dst_y, src_y, prof).astype(np.float32)
            inflow_np = np.stack([prof, np.zeros_like(prof)], axis=-1).astype(
                np.float32
            )  # (N, 2)
        else:
            # Fall-back: derive inflow column from v0 left face.
            inflow_col = v0_2d[:, 0, 0]  # (N,) — ux values at x=0 column
            inflow_np = np.stack(
                [inflow_col, np.zeros_like(inflow_col)], axis=-1
            ).astype(np.float32)  # (N, 2)
        obstacle_mask = _make_obstacle_mask(obstacle, N)  # (N, N, 2)
        jl_result = jl.ns_apply_channel_2d_drag_window(
            _to_julia(v0_2d),
            _to_julia(inflow_np),
            _to_julia(obstacle_mask),
            float(viscosity),
            float(dt),
            int(steps),
            int(N),
            float(domain_extent),
        )
        result_2d = _to_numpy(jl_result[0])  # (N, N, 2)
        mean_drag_jl = float(jl_result[1])  # scalar: tail-window mean drag
        result_4d = result_2d[:, :, np.newaxis, :]  # (N, N, 1, 2)
        drag = np.array([mean_drag_jl], dtype=np.float32)
        return result_4d, drag

    if inflow_profile is not None:
        raise NotImplementedError(
            "incompressible-navier-stokes-jl supports inflow_profile only when "
            "combined with an obstacle (channel mode)."
        )

    nx = v0.shape[0]
    ndim = v0.shape[-1]
    nz = v0.shape[2]

    if ndim == 2 and nz == 1:
        # 2-D path: squeeze the z=1 dimension before passing to Julia
        v0_in = v0[:, :, 0, :]  # (nx, ny, 2)
        result = _to_numpy(
            jl.ns_apply(
                _to_julia(v0_in),
                float(viscosity),
                float(dt),
                int(steps),
                int(nx),
                float(domain_extent),
            )
        )  # (nx, ny, 2)
        result_4d = result[:, :, np.newaxis, :]  # (nx, ny, 1, 2)
        ux = result[:, :, 0]
        drag = _compute_drag_numpy(
            ux, np.zeros_like(ux), obstacle, viscosity, domain_extent
        )
        return result_4d, drag
    else:
        # 3-D path: pass the full (nx, ny, nz, 3) array
        result = _to_numpy(
            jl.ns_apply(
                _to_julia(v0),
                float(viscosity),
                float(dt),
                int(steps),
                int(nx),
                float(domain_extent),
            )
        )  # (nx, ny, nz, 3)
        return result, None


def _vjp(  # mosaic:grad:v0,viscosity,dt:adjoint
    v0: np.ndarray,
    cotangent: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    obstacle: dict | None = None,
    inflow_profile: np.ndarray | None = None,
    cotangent_drag: np.ndarray | None = None,
) -> dict:
    """VJP: cotangent w.r.t. result -> gradients w.r.t. v0, viscosity, dt, domain_extent.

    In channel mode (obstacle present), when inflow_profile is given, the
    gradient w.r.t. inflow_profile (shape (N,)) is returned. Otherwise the
    gradient w.r.t. the inflow column derived from v0[:, 0, 0, :] is computed.
    """
    # Channel mode VJP: obstacle present. Either inflow_profile (drag_opt)
    # or v0-derived inflow (re20/re100).
    #
    # cotangent is a dict with keys "result" and/or "drag".  We pass both to
    # ns_vjp_channel_2d_drag_window so that Zygote can differentiate through
    # the tail-window mean drag AND the final velocity field in a single pullback.
    if obstacle is not None:
        v0_2d = np.asarray(v0, dtype=np.float32)[:, :, 0, :]  # (N, N, 2)
        N = v0_2d.shape[0]
        prof_len = None
        if inflow_profile is not None:
            prof_raw = np.asarray(inflow_profile, dtype=np.float32).reshape(-1)
            prof_len = prof_raw.shape[0]
            if prof_len != N:
                src_y = np.linspace(0.0, 1.0, prof_len, dtype=np.float32)
                dst_y = np.linspace(0.0, 1.0, N, dtype=np.float32)
                prof = np.interp(dst_y, src_y, prof_raw).astype(np.float32)
            else:
                prof = prof_raw
            inflow_np = np.stack([prof, np.zeros_like(prof)], axis=-1).astype(
                np.float32
            )  # (N, 2)
        else:
            inflow_col = v0_2d[:, 0, 0]  # (N,)
            inflow_np = np.stack(
                [inflow_col, np.zeros_like(inflow_col)], axis=-1
            ).astype(np.float32)  # (N, 2)
        obstacle_mask = _make_obstacle_mask(obstacle, N)

        # cotangent here is the combined cotangent on result (after the caller
        # has already merged drag_cot into it via _drag_cotangent_to_result, OR
        # it is purely a result cotangent when called from the old path).
        # In the new drag-window path, vector_jacobian_product passes us a plain
        # (N,N,2) result cotangent; the drag cotangent is passed separately.
        # To keep _vjp backward-compatible with both call sites we accept
        # cotangent as an (N,N,1,2) array (result cotangent only) and an
        # optional drag_cotangent kwarg.
        cot_2d = np.asarray(cotangent, dtype=np.float32)[:, :, 0, :]  # (N, N, 2)
        cot_drag_scalar = float(
            np.asarray(cotangent_drag).reshape(-1)[0]
            if cotangent_drag is not None
            else 0.0
        )
        result = jl.ns_vjp_channel_2d_drag_window(
            _to_julia(v0_2d),
            _to_julia(inflow_np),
            _to_julia(obstacle_mask),
            float(viscosity),
            float(dt),
            int(steps),
            int(N),
            float(domain_extent),
            _to_julia(cot_2d),
            float(cot_drag_scalar),
        )
        grad_inflow = _to_numpy(result[0]).astype(np.float32)  # (N, 2)
        out: dict[str, Any] = {
            "viscosity": np.array([float(result[1])], dtype=np.float32),
            "dt": np.array([float(result[2])], dtype=np.float32),
            "domain_extent": np.float32(float(result[3])),
        }
        if inflow_profile is not None:
            # gradient w.r.t. the 1-D ux(y) inflow_profile.
            grad_prof_full = grad_inflow[:, 0]  # (N,) on grid
            if prof_len is not None and prof_len != N:
                # Resample gradient back to the caller's profile length. Using
                # linear interpolation for shape compatibility (not adjoint-exact).
                src_y = np.linspace(0.0, 1.0, N, dtype=np.float32)
                dst_y = np.linspace(0.0, 1.0, prof_len, dtype=np.float32)
                grad_prof_full = np.interp(dst_y, src_y, grad_prof_full).astype(
                    np.float32
                )
            out["inflow_profile"] = grad_prof_full.astype(np.float32)
            # v0 is still a constant uniform flow in drag_opt; provide a
            # zero-valued gradient at the same shape for completeness.
            out["v0"] = np.zeros(
                (v0_2d.shape[0], v0_2d.shape[1], 1, v0_2d.shape[2]),
                dtype=np.float32,
            )
        else:
            # Map gradient back to v0 shape: only the x=0 column carries gradient.
            grad_v0 = np.zeros_like(v0_2d)  # (N, N, 2)
            grad_v0[:, 0, 0] = grad_inflow[:, 0]  # ux component
            grad_v0[:, 0, 1] = grad_inflow[:, 1]  # uy (≡0 but propagate)
            out["v0"] = grad_v0[:, :, np.newaxis, :].astype(np.float32)
        return out

    nx = v0.shape[0]
    ndim = v0.shape[-1]
    nz = v0.shape[2]

    if ndim == 2 and nz == 1:
        v0_in = v0[:, :, 0, :]
        cot_in = cotangent[:, :, 0, :]
        result = jl.ns_vjp(
            _to_julia(v0_in),
            _to_julia(cot_in),
            float(viscosity),
            float(dt),
            int(steps),
            int(nx),
            float(domain_extent),
        )
        grad_v0_2d = _to_numpy(result[0])
        grad_nu = float(result[1])
        grad_dt_val = float(result[2])
        grad_L = float(result[3])
        return {
            "v0": grad_v0_2d[:, :, np.newaxis, :].astype(np.float32),
            "viscosity": np.array([grad_nu], dtype=np.float32),
            "dt": np.array([grad_dt_val], dtype=np.float32),
            "domain_extent": np.float32(grad_L),
        }
    else:
        result = jl.ns_vjp(
            _to_julia(v0),
            _to_julia(cotangent),
            float(viscosity),
            float(dt),
            int(steps),
            int(nx),
            float(domain_extent),
        )
        grad_v0 = _to_numpy(result[0])
        grad_nu = float(result[1])
        grad_dt_val = float(result[2])
        grad_L = float(result[3])
        return {
            "v0": grad_v0.astype(np.float32),
            "viscosity": np.array([grad_nu], dtype=np.float32),
            "dt": np.array([grad_dt_val], dtype=np.float32),
            "domain_extent": np.float32(grad_L),
        }


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


def _obstacle_dict_ins(inputs: "InputSchema") -> dict | None:  # mosaic:io
    obs = inputs.obstacle
    if obs is None:
        return None
    return {
        "shape": obs.shape.value,
        "center": list(obs.center),
        "radius": float(obs.radius) if obs.radius is not None else 0.0,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    result, drag = _run(
        np.asarray(inputs.v0),
        float(inputs.viscosity[0]),
        float(inputs.dt[0]),
        inputs.steps,
        inputs.domain_extent,
        obstacle=_obstacle_dict_ins(inputs),
        inflow_profile=np.asarray(inputs.inflow_profile)
        if inputs.inflow_profile is not None
        else None,
    )
    out = {"result": result.astype(np.float32)}
    if drag is not None:
        out["drag"] = drag
    else:
        out["drag"] = np.zeros((1,), dtype=np.float32)
    return out


def _drag_cotangent_to_result(  # mosaic:grad:v0,viscosity,dt:adjoint
    drag_cot: np.ndarray,
    v0_shape: tuple,
    obstacle: dict,
    viscosity: float,
) -> np.ndarray:
    """Back-propagate the drag cotangent through _compute_drag_numpy into a
    cotangent on ``result`` (shape v0_shape). Since pressure is forced to zero
    in our drag computation, only the viscous term contributes and the drag is
    linear in ux; the Jacobian is a fixed mask determined by the obstacle
    surface layout.

    v0_shape: (N, N, 1, 2).
    """
    N = v0_shape[0]
    cx = obstacle["center"][0] * N
    cy = obstacle["center"][1] * N
    r = obstacle["radius"] * N
    x = np.arange(N, dtype=np.float32)
    y = np.arange(N, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    solid_mask = (X - cx) ** 2 + (Y - cy) ** 2 < r**2
    fluid_mask = ~solid_mask
    solid_right = np.roll(solid_mask, -1, axis=0)
    solid_left = np.roll(solid_mask, 1, axis=0)
    surf_right = fluid_mask & solid_right
    surf_left = fluid_mask & solid_left
    # ∂drag/∂ux[i,j] = ν * (surf_left[i,j] - surf_right[i,j])
    d_ux = viscosity * (surf_left.astype(np.float32) - surf_right.astype(np.float32))
    ct = np.zeros(v0_shape, dtype=np.float32)
    # drag_cot shape is (1,); multiply scalar.
    scale = float(np.asarray(drag_cot).reshape(-1)[0])
    ct[:, :, 0, 0] = scale * d_ux
    return ct


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt:adjoint
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    ct_result = cotangent_vector.get("result")
    ct_drag = cotangent_vector.get("drag")

    obs = _obstacle_dict_ins(inputs)
    v0_arr = np.asarray(inputs.v0)

    # Channel mode (obstacle present): drag cotangent is passed directly to
    # ns_vjp_channel_2d_drag_window inside Julia so Zygote differentiates
    # through the tail-window mean drag.  No need to manually map drag_cot to a
    # velocity cotangent via _drag_cotangent_to_result here.
    if obs is not None:
        if ct_result is None and ct_drag is None:
            return {}
        ct = (
            np.asarray(ct_result, dtype=np.float32)
            if ct_result is not None
            else np.zeros(v0_arr.shape, dtype=np.float32)
        )
        inflow_prof = (
            np.asarray(inputs.inflow_profile)
            if inputs.inflow_profile is not None
            else None
        )
        grads = _vjp(
            v0_arr,
            ct,
            float(inputs.viscosity[0]),
            float(inputs.dt[0]),
            inputs.steps,
            inputs.domain_extent,
            obstacle=obs,
            inflow_profile=inflow_prof,
            cotangent_drag=np.asarray(ct_drag) if ct_drag is not None else None,
        )
        result: dict[str, Any] = {}
        if inflow_prof is not None:
            for key in ("inflow_profile", "v0", "viscosity", "dt", "domain_extent"):
                if key in vjp_inputs and key in grads:
                    result[key] = grads[key]
        else:
            for key in ("v0", "viscosity", "dt", "domain_extent"):
                if key in vjp_inputs and key in grads:
                    result[key] = grads[key]
        return result

    # Periodic mode: drag is not used / zero.
    if ct_result is None:
        return {}
    ct = np.asarray(ct_result, dtype=np.float32)

    inflow_prof = (
        np.asarray(inputs.inflow_profile) if inputs.inflow_profile is not None else None
    )
    grads = _vjp(
        v0_arr,
        np.asarray(ct),
        float(inputs.viscosity[0]),
        float(inputs.dt[0]),
        inputs.steps,
        inputs.domain_extent,
        obstacle=obs,
        inflow_profile=inflow_prof,
    )
    out_grads: dict[str, Any] = {}
    for key in ("v0", "viscosity", "dt", "domain_extent"):
        if key in vjp_inputs and key in grads:
            out_grads[key] = grads[key]
    return out_grads


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    raw = abstract_inputs.model_dump()

    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        return {
            "result": {"shape": v0["shape"], "dtype": v0["dtype"]},
            "drag": {"shape": (1,), "dtype": "float32"},
        }
    arr = np.asarray(v0)
    return {
        "result": {"shape": list(arr.shape), "dtype": str(arr.dtype)},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
