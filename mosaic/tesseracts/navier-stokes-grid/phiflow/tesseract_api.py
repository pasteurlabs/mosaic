# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import drag_jax
from mosaic_shared.types import make_differentiable
from phi.jax.flow import (
    Box,
    CenteredGrid,
    Obstacle,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
    geom,
    math,
)
from pydantic import model_validator
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    """PhiFlow Navier-Stokes input schema with differentiable velocity and physics params."""

    @model_validator(mode="after")
    def _check_bcs(self) -> "InputSchema":
        supported = {"periodic", "no_slip", "neumann", "dirichlet"}
        for face_key, face in self.boundary_conditions.model_dump().items():
            t = face["type"]
            if t not in supported:
                raise ValueError(
                    f"phiflow supports {sorted(supported)} BCs, got '{t}' at {face_key}."
                )
        return self


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result", "drag"])):
    """PhiFlow Navier-Stokes output schema with differentiable result and drag."""


def _phiflow_extrapolation(bc_dict: dict, ndim: int):  # mosaic:io
    """Map a serialized GridBC dict to a phiflow extrapolation object."""
    if all(bc_dict[k]["type"] == "periodic" for k in bc_dict):
        return extrapolation.PERIODIC

    def _face_ext(face: dict):
        t = face["type"]
        if t == "periodic":
            return extrapolation.PERIODIC
        if t == "no_slip":
            return extrapolation.ZERO
        if t == "neumann":
            return extrapolation.ZERO_GRADIENT
        if t == "dirichlet":
            if face.get("value") is not None:
                val = math.tensor(face["value"], math.channel("vector"))
                return extrapolation.ConstantExtrapolation(val)
            return extrapolation.ZERO
        raise ValueError(f"Unknown BC type: {t!r}")

    kwargs = {
        "x": (_face_ext(bc_dict["x_lo"]), _face_ext(bc_dict["x_hi"])),
        "y": (_face_ext(bc_dict["y_lo"]), _face_ext(bc_dict["y_hi"])),
    }
    if ndim == 3:
        kwargs["z"] = (_face_ext(bc_dict["z_lo"]), _face_ext(bc_dict["z_hi"]))
    return extrapolation.combine_sides(**kwargs)


def _make_phiflow_obstacle(  # mosaic:init
    obstacle: dict | None, domain_extent: float, ndim: int
):
    """Construct a PhiFlow Obstacle from the canonical obstacle dict, or None."""
    if obstacle is None or not obstacle.get("shape"):
        return None
    L = domain_extent
    cx = obstacle["center"][0] * L
    cy = obstacle["center"][1] * L
    r = obstacle["radius"] * L
    if obstacle["shape"] in ("cylinder", "CYLINDER"):
        if ndim == 2:
            obs_geom = geom.Sphere(x=cx, y=cy, radius=r)
        else:
            cz = obstacle["center"][2] * L if len(obstacle["center"]) > 2 else L / 2.0
            obs_geom = geom.Sphere(x=cx, y=cy, z=cz, radius=r)
        return Obstacle(obs_geom)
    raise ValueError(f"PhiFlow: unsupported obstacle shape {obstacle['shape']!r}")


def _make_obstacle_mask_phiflow(  # mosaic:init
    obstacle: dict | None, nx: int, ny: int
) -> jnp.ndarray | None:
    """Rasterize obstacle to a boolean JAX mask of shape (nx, ny).

    Returns None when no obstacle is present.
    """
    if obstacle is None or not obstacle.get("shape"):
        return None
    cx = obstacle["center"][0] * nx
    cy = obstacle["center"][1] * ny
    r = obstacle["radius"] * nx  # isotropic grid
    if obstacle["shape"] in ("cylinder", "CYLINDER"):
        x = jnp.arange(nx, dtype=jnp.float32)
        y = jnp.arange(ny, dtype=jnp.float32)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        return (X - cx) ** 2 + (Y - cy) ** 2 < r**2  # (nx, ny)
    raise ValueError(f"PhiFlow: unsupported obstacle shape {obstacle['shape']!r}")


# Drag is computed via the shared canonical surface integral
# (mosaic_shared.problems.navier_stokes_grid.drag_jax).


def phiflow_fwd(  # mosaic:physics
    v0: jnp.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    boundary_conditions: dict,
    obstacle: dict | None = None,
    inflow_profile: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Run a 2D or 3D incompressible Navier-Stokes simulation using PhiFlow.

    Uses semi-Lagrangian advection, explicit diffusion, and pressure projection
    for incompressibility on a periodic domain. Dimensionality is inferred from
    v0.shape[-1] (2 → 2D, 3 → 3D).

    Args:
        v0: Initial velocity field, shape (*spatial, ndim).
        viscosity: Kinematic viscosity.
        dt: Timestep size.
        steps: Number of simulation steps.
        domain_extent: Side length of the periodic square/cubic domain (isotropic grid).
        boundary_conditions: Dict mapping face keys to BC type/value dicts.
        obstacle: Optional obstacle dict.
        inflow_profile: Optional 1-D u_x(y) profile shape (ny,). Applied at x_lo each step.

    Returns:
        (result, drag): result same shape as v0; drag shape (1,) or None.
    """
    ndim = v0.shape[-1]  # 2 for 2D, 3 for 3D
    if ndim == 2:
        # squeeze the dummy nz=1 axis: (nx, ny, 1, 2) → (nx, ny, 2)
        v0 = v0[:, :, 0, :]

    if ndim == 2:
        nx, ny = v0.shape[:2]
        spatial_str = "x,y"
        grid_kwargs = {"x": nx, "y": ny}
        bounds = Box["x,y", 0:domain_extent, 0:domain_extent]
    else:
        nx, ny, nz = v0.shape[:3]
        spatial_str = "x,y,z"
        grid_kwargs = {"x": nx, "y": ny, "z": nz}
        bounds = Box["x,y,z", 0:domain_extent, 0:domain_extent, 0:domain_extent]

    dx = domain_extent / nx
    raw_ext = _phiflow_extrapolation(boundary_conditions, ndim)
    # For 2D inflow+obstacle runs use PERIODIC so the pressure is
    # unconstrained at the domain boundary — the periodic wrap-around acts as
    # a virtual channel that allows the manual per-step inflow override to
    # drive realistic flow.  Drag uses a surface integral on the RANS mean
    # (pressure + viscous), which is correct for PERIODIC since the constant-
    # pressure offset cancels on a closed surface around the cylinder.
    # _resize_arr sees clean (nx, ny) staggered shapes, making it a no-op.
    if (
        inflow_profile is not None
        and obstacle is not None
        and obstacle.get("shape")
        and ndim == 2
    ):
        raw_ext = extrapolation.PERIODIC
    ext = raw_ext
    obstacle_obj = _make_phiflow_obstacle(obstacle, domain_extent, ndim)
    obstacles = (obstacle_obj,) if obstacle_obj is not None else ()

    # Obstacle mask for drag computation (2-D only)
    if ndim == 2 and obstacle is not None and obstacle.get("shape"):
        obs_mask = _make_obstacle_mask_phiflow(obstacle, nx, ny)
    else:
        obs_mask = None

    # Target cell shape for the lax.scan carry.  All components are trimmed/padded
    # to exactly this shape so jnp.stack works regardless of phiflow's BC-dependent
    # staggered face counts:
    #   periodic   → N faces (no change needed)
    #   neumann    → N+1 faces on the staggered axis  (trim last face)
    #   no_slip    → N-1 faces on the staggered axis  (pad one zero at far end)
    if ndim == 2:
        _cell_shape = (nx, ny)
    else:
        _cell_shape = (nx, ny, nz)

    # Pre-compute the staggered shape that phiflow expects for each velocity
    # component under the current extrapolation.  This is a Python-level shape
    # query (no JAX tracing) so the result is a plain tuple of ints.
    _probe_channel = math.stack(
        [
            math.tensor(jnp.zeros(_cell_shape), math.spatial(spatial_str))
            for _ in range(ndim)
        ],
        dim=math.channel("vector"),
    )
    _probe_cg = CenteredGrid(_probe_channel, ext, bounds=bounds, **grid_kwargs)
    _probe_sg = StaggeredGrid(_probe_cg, ext, bounds=bounds, **grid_kwargs)
    _staggered_shapes = tuple(
        _probe_sg.vector[i].values.native(spatial_str).shape for i in range(ndim)
    )

    def _resize_arr(arr: jnp.ndarray, target_shape: tuple) -> jnp.ndarray:
        """Trim or zero-pad/repeat arr along each axis to match target_shape."""
        for axis, target in enumerate(target_shape):
            current = arr.shape[axis]
            if current > target:
                # Trim: drop the extra boundary face.
                idx = [slice(None)] * ndim
                idx[axis] = slice(None, target)
                arr = arr[tuple(idx)]
            elif current < target:
                # Pad: repeat the last face value (lossless for neumann ghost face).
                # Use a concrete integer index (shape is always concrete in JAX).
                idx = [slice(None)] * ndim
                idx[axis] = current - 1
                last = arr[tuple(idx)]
                last = jnp.expand_dims(last, axis=axis)
                arr = jnp.concatenate([arr, last], axis=axis)
        return arr

    # Spatial axis name list for named-dual stacking (2D: ['x','y'], 3D: ['x','y','z'])
    _spatial_names = spatial_str.split(",")

    def faces_to_staggered(face_arr: jnp.ndarray) -> StaggeredGrid:
        """Face-centred (ndim, *spatial) carry → StaggeredGrid.

        Resizes each component from _cell_shape to the BC-dependent staggered
        shape phiflow expects, wraps in a phiml tensor, then stacks with a
        named dual 'vector' dim so StaggeredGrid accepts non-uniform shapes.
        """
        component_tensors = {
            _spatial_names[i]: math.tensor(
                _resize_arr(face_arr[i], _staggered_shapes[i]),
                math.spatial(spatial_str),
            )
            for i in range(ndim)
        }
        staggered_tensor = math.stack(component_tensors, dim=math.dual("vector"))
        return StaggeredGrid(staggered_tensor, ext, bounds=bounds, **grid_kwargs)

    def staggered_to_faces(vel: StaggeredGrid) -> jnp.ndarray:
        """StaggeredGrid → face-centred (ndim, *spatial) carry of shape _cell_shape.

        Phiflow allocates BC-dependent face counts (N+1 for neumann, N-1 for
        no_slip, N for periodic).  We resize every component to _cell_shape so
        jnp.stack produces a fixed-shape carry for lax.scan.
        """
        return jnp.stack(
            [
                _resize_arr(vel.vector[i].values.native(spatial_str), _cell_shape)
                for i in range(ndim)
            ],
            axis=0,
        )

    if inflow_profile is not None and ndim == 2:
        # Resample inflow_profile to ny if needed
        prof_len = inflow_profile.shape[0]
        if prof_len != ny:
            src_y = jnp.linspace(0, 1, prof_len)
            dst_y = jnp.linspace(0, 1, ny)
            ux_inflow = jnp.interp(dst_y, src_y, inflow_profile)
        else:
            ux_inflow = inflow_profile  # (ny,)

        def step(
            face_arr: jnp.ndarray, _: None
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            vel = faces_to_staggered(face_arr)
            vel = advect.semi_lagrangian(vel, vel, dt)
            vel = diffuse.explicit(vel, viscosity, dt)
            vel, p_field = fluid.make_incompressible(vel, obstacles)
            p_arr = p_field.values.native(spatial_str)  # (nx, ny)
            # Apply inflow: override x=0 face of u_x
            faces = staggered_to_faces(vel)  # (ndim, nx, ny)
            # faces[0] is u_x face values; set column 0 to ux_inflow
            ux_f = faces[0].at[0, :].set(ux_inflow)
            # faces[1] is u_y; set column 0 to 0
            uy_f = faces[1].at[0, :].set(0.0)
            new_faces = jnp.stack([ux_f, uy_f], axis=0)
            # Accumulate both faces and pressure for RANS drag computation
            return new_faces, (new_faces, p_arr)

    else:
        # Use explicit CG pressure solver to prevent numerical divergence in 3D
        # at step counts ≥ 20 (default solver produces NaN in 3D; 2D unaffected).
        # Tightened rel_tol/abs_tol to 1e-10 on the 3D
        # periodic path to prevent adjoint-side residual bias in fd_check.
        _cg_solve = math.Solve("CG", 1e-10, 1e-10)

        def step(face_arr: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
            vel = faces_to_staggered(face_arr)
            # Use explicit Euler differential advection to
            # fix VJP gradient magnitude bias and prevent NaN from semi-Lagrangian
            # interpolation. advect.differential(-u·∇v) is JAX-autodiffable and
            # stable for periodic forward/agreement runs (TGV, multimode, 3D).
            # Restored here: 7f9213f accidentally reverted this to semi_lagrangian.
            vel = vel + dt * advect.differential(vel, vel)
            vel = diffuse.explicit(vel, viscosity, dt)
            vel, _ = fluid.make_incompressible(vel, obstacles, solve=_cg_solve)
            return staggered_to_faces(vel), None

    # Convert IC from collocated to staggered face values once using PhiFlow's
    # native CenteredGrid→StaggeredGrid resampling.  The scan carry then stays
    # in face-value space so no smoothing accumulates across steps.
    v_arr = jnp.moveaxis(v0, -1, 0)  # (*spatial, ndim) → (ndim, *spatial)
    channel = math.stack(
        [math.tensor(v_arr[i], math.spatial(spatial_str)) for i in range(ndim)],
        dim=math.channel("vector"),
    )
    vel_init = StaggeredGrid(
        CenteredGrid(channel, ext, bounds=bounds, **grid_kwargs),
        ext,
        bounds=bounds,
        **grid_kwargs,
    )
    face_init = staggered_to_faces(vel_init)

    # fluid.make_incompressible's CG while_loop is wrapped with custom_vjp,
    # so the backward pass does an adjoint solve rather than differentiating
    # through the loop — both forward and VJP are safe inside lax.scan.
    face_final, scan_hist = jax.lax.scan(step, face_init, None, length=steps)
    # For inflow+2D branch, scan_hist is (face_hist, p_hist); unpack here.
    if inflow_profile is not None and ndim == 2:
        _face_hist, _p_hist = scan_hist
    else:
        _face_hist, _p_hist = scan_hist, None

    # Convert final face values back to collocated once.
    vel_final = faces_to_staggered(face_final)
    centered_final = CenteredGrid(vel_final, ext, bounds=bounds, **grid_kwargs)
    v_final = jnp.stack(
        [centered_final.vector[i].values.native(spatial_str) for i in range(ndim)],
        axis=0,
    )

    result = jnp.moveaxis(v_final, 0, -1)  # (ndim, *spatial) → (*spatial, ndim)
    if ndim == 2:
        # restore the dummy nz=1 axis: (nx, ny, 2) → (nx, ny, 1, 2)
        result = result[:, :, None, :]

    # --- drag from RANS-averaged pressure + velocity ----------------------
    drag = None
    if obs_mask is not None and ndim == 2 and inflow_profile is not None:
        # _face_hist shape: (steps, 2, nx, ny); _p_hist shape: (steps, nx, ny).
        # Average over the last 50% of steps to smooth Kármán shedding oscillations.
        n_tail = max(1, steps // 2)
        rans_faces = jnp.mean(_face_hist[-n_tail:], axis=0)  # (2, nx, ny)
        ux_rans = 0.5 * (rans_faces[0] + jnp.roll(rans_faces[0], 1, axis=0))
        # Surface-integral drag from RANS mean pressure + velocity.
        # Time-averaging the per-step CG pressure (collected in-loop) gives the
        # correct RANS pressure; constant offset cancels on the closed cylinder surface.
        p_rans = jnp.mean(_p_hist[-n_tail:], axis=0)  # (nx, ny)
        drag = drag_jax(ux_rans, p_rans, obs_mask, viscosity, dx)

    return result, drag


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:  # mosaic:io
    """JIT-compiled forward pass returning result and drag arrays."""
    result, drag = phiflow_fwd(**inputs)
    out = {"result": result}
    out["drag"] = drag if drag is not None else jnp.zeros((1,), dtype=jnp.float32)
    return out


_SCALAR_KEYS = ("viscosity", "dt")


def _unpack_scalars(d: dict) -> dict:  # mosaic:io
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in _SCALAR_KEYS:
        if key in d:
            d[key] = float(d[key][0])
    return d


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Run the PhiFlow Navier-Stokes forward simulation."""
    return apply_jit(_unpack_scalars(inputs.model_dump()))


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt,inflow_profile:autodiff
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute the vector-Jacobian product via JAX autodiff."""
    return vjp_jit(
        _unpack_scalars(inputs.model_dump()),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, dict[str, Any]]:
    """Calculate output shape of apply from the shape of its inputs.

    For phiflow the output ``result`` always has the same shape as ``v0``,
    and ``drag`` is always shape ``(1,)``.
    """
    raw = abstract_inputs.model_dump()
    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        v0_shape = tuple(v0["shape"])
        v0_dtype = v0["dtype"]
    else:
        import numpy as np

        arr = np.asarray(v0)
        v0_shape = arr.shape
        v0_dtype = str(arr.dtype)
    out = {
        "result": {"shape": v0_shape, "dtype": v0_dtype},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
    obstacle_raw = raw.get("obstacle") or {}
    _has_obstacle = bool(
        obstacle_raw.get("shape")
        if isinstance(obstacle_raw, dict)
        else getattr(obstacle_raw, "shape", None)
    )
    _has_inflow = raw.get("inflow_profile") is not None
    _ndim = v0_shape[-1] if v0_shape else 0
    return out


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
) -> dict[str, Any]:
    """JIT-compiled vector-Jacobian product computation."""
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    grads = vjp_func(cotangent_vector)[0]
    # Scalar physics params (viscosity, dt, etc.) are unpacked to Python floats
    # before the VJP, so their gradients come back as 0D arrays.  Tesseract
    # expects shape (1,) to match the input schema; reshape any 0D grad here.
    return jax.tree.map(jnp.atleast_1d, grads)
