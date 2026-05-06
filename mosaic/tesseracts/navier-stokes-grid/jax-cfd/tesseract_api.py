from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from pydantic import Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class InputSchema(_CanonicalInputSchema):
    density: Differentiable[Array[(1,), Float32]] = Field(
        description="Density of the fluid", default=1.0
    )
    inner_steps: int = Field(
        description="Solver sub-iterations per timestep (higher = more accurate pressure solve)",
        default=1,
    )

    @model_validator(mode="after")
    def _check_bcs(self) -> "InputSchema":
        bc = self.boundary_conditions.model_dump()
        for lo_key, hi_key in [("x_lo", "x_hi"), ("y_lo", "y_hi"), ("z_lo", "z_hi")]:
            for face_key in (lo_key, hi_key):
                face = bc[face_key]
                t = face["type"]
                if t not in ("periodic", "no_slip", "dirichlet"):
                    raise ValueError(
                        f"jax-cfd supports 'periodic' and 'no_slip' BCs only, got '{t}' "
                        f"at {face_key}. Use the phiflow solver for 'neumann' BCs."
                    )
                if t == "dirichlet" and face.get("value") is not None:
                    raise ValueError(
                        f"jax-cfd only supports zero-velocity (homogeneous) Dirichlet BCs "
                        f"at {face_key}. Set value=None or use 'no_slip'."
                    )
            if (bc[lo_key]["type"] == "periodic") != (bc[hi_key]["type"] == "periodic"):
                raise ValueError(
                    f"Periodic BC must be applied to both faces of a dimension, "
                    f"got {lo_key}='{bc[lo_key]['type']}', {hi_key}='{bc[hi_key]['type']}'."
                )
        return self


class OutputSchema(_CanonicalOutputSchema):
    pass


def _jaxcfd_bc(  # mosaic:io
    bc_dict: dict, ndim: int
) -> "cfd.boundaries.HomogeneousBoundaryConditions":
    """Map a serialized GridBC dict to jax-cfd HomogeneousBoundaryConditions."""

    def _face_bctype(face: dict) -> "cfd.boundaries.BCType":
        t = face["type"]
        if t == "periodic":
            return cfd.boundaries.BCType.PERIODIC
        # no_slip and dirichlet (value=None) both map to zero-velocity wall
        return cfd.boundaries.BCType.DIRICHLET

    pairs = [("x_lo", "x_hi"), ("y_lo", "y_hi")]
    if ndim == 3:
        pairs.append(("z_lo", "z_hi"))

    dim_bcs = []
    for lo_key, hi_key in pairs:
        lo_t = _face_bctype(bc_dict[lo_key])
        hi_t = _face_bctype(bc_dict[hi_key])
        dim_bcs.append((lo_t, hi_t))

    return cfd.boundaries.HomogeneousBoundaryConditions(tuple(dim_bcs))


def _make_obstacle_mask_jaxcfd(  # mosaic:init
    obstacle: dict | None, nx: int, ny: int
) -> jnp.ndarray | None:
    """Rasterize a 2-D obstacle to a boolean mask of shape (nx, ny).

    Coordinates are specified as fractions of the domain, so we multiply by
    the grid resolution nx (isotropic grid assumed).

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
        return (X - cx) ** 2 + (Y - cy) ** 2 < r**2  # (nx, ny) bool
    raise ValueError(f"jax-cfd: unsupported obstacle shape {obstacle['shape']!r}")


def _compute_drag(  # mosaic:physics
    ux: jnp.ndarray,
    pressure: jnp.ndarray,
    solid_mask: jnp.ndarray,
    viscosity: float,
    dx: float,
) -> jnp.ndarray:
    """Compute x-direction drag on obstacle via discrete surface integral.

    Uses fluid cells adjacent to solid cells as proxy for the obstacle surface.
    All JAX operations — fully differentiable.

    Physics:
        F_x = Σ_surface [ p * n_x - ν * (∂u_x/∂n) ] * dx

    For a surface cell adjacent to a solid in the +x direction (n_x = +1):
        pressure term:  p_cell * (+1) * dx
        viscous term:   -ν * (u_wall - u_fluid) / dx * dx
                      = -ν * (0 - u_x) = +ν * u_x  ... but we want force ON obstacle,
        Actually the sign is:
            stress on obstacle from fluid = +ν * ∂u_x/∂n_fluid
        where n_fluid points away from the obstacle (i.e. into the fluid).
        For surf_right: n_fluid = -x,  so ∂u_x/∂n_fluid ≈ (u_fluid - 0)/dx
            contribution to F_x from viscous = -ν * (u_fluid/dx) * dx * n_x_obstacle
            n_x_obstacle = +1 → viscous force = -ν * u_fluid

        For surf_left: n_fluid = +x,  so ∂u_x/∂n_fluid ≈ (u_fluid - 0)/dx
            n_x_obstacle = -1 → viscous force = +ν * u_fluid * (-(-1)) = ... hmm.

    Simplified consistent discrete formula:
        For each x-adjacent surface pair (fluid at i, solid at i±1):
            surf_right (solid at i+1, n_x = +1 toward obstacle):
                Δp   = +p[i] * dx
                Δvisc = -ν * u_x[i]  (wall is 0, gradient across dx)
            surf_left (solid at i-1, n_x = -1 toward obstacle):
                Δp   = -p[i] * dx
                Δvisc = +ν * u_x[i]

    Args:
        ux:          x-velocity, shape (nx, ny), collocated.
        pressure:    pressure field, shape (nx, ny).
        solid_mask:  boolean, True = solid cell, shape (nx, ny).
        viscosity:   kinematic viscosity ν.
        dx:          grid spacing.

    Returns:
        Scalar drag force F_x, shape (1,), float32.
    """
    fluid_mask = ~solid_mask

    # Identify surface cells: fluid adjacent to solid in x-direction
    solid_right = jnp.roll(solid_mask, -1, axis=0)  # solid at (i+1, j)
    solid_left = jnp.roll(solid_mask, 1, axis=0)  # solid at (i-1, j)
    surf_right = fluid_mask & solid_right  # normal points +x into obstacle
    surf_left = fluid_mask & solid_left  # normal points -x into obstacle

    # Pressure contribution (p * n_x * dx)
    p_drag = jnp.sum(
        jnp.where(surf_right, pressure * dx, 0.0)
        + jnp.where(surf_left, -pressure * dx, 0.0)
    )

    # Viscous contribution (-ν * du_x/dn * dx)
    # surf_right: n = +x, du/dn = (u_wall - u_fluid)/dx = -u_fluid/dx
    #   F_visc = -ν * (-u_fluid/dx) * dx = ν * u_fluid  ... wait, the sign convention:
    #   We want force on obstacle = -σ · n_fluid * dA
    #   Normal stress = ν * (du_x/dn_fluid), n_fluid = -x for surf_right
    #   du_x/dn_fluid = (u_x_fluid - 0)/dx (gradient pointing out from obstacle)
    #   F_visc_x on obstacle = -ν*(u_x/dx)*(-1) * dx = ν * u_x  ... but n_obstacle = +x
    #   force on obstacle in x = +ν * (du/dn_obstacle) * dx where n_obstacle points into fluid
    # Simpler: viscous drag opposes inflow, convention:
    #   surf_right: -ν * u_x (drag reduces with inflow velocity)
    #   surf_left:  +ν * u_x (drag increases from behind)
    visc_drag = jnp.sum(
        jnp.where(surf_right, -viscosity * ux, 0.0)
        + jnp.where(surf_left, viscosity * ux, 0.0)
    )

    drag = (p_drag + visc_drag).astype(jnp.float32)
    return jnp.reshape(drag, (1,))


def _extract_pressure_jaxcfd(v_grid, grid, bc):  # mosaic:physics
    """Extract pressure from the final velocity field via one Poisson solve.

    Uses cfd.pressure.solve_fast_diag which returns a GridArray of pressure
    corrections (proportional to physical pressure).

    Returns a (nx, ny) float32 array.
    """
    q = cfd.pressure.solve_fast_diag(v_grid)
    return q.data.astype(jnp.float32)


def cfd_fwd(  # mosaic:physics
    v0: jnp.ndarray,
    density: float,
    viscosity: float,
    dt: float,
    steps: int,
    inner_steps: int,
    domain_extent: float,
    boundary_conditions: dict,
    obstacle: dict | None = None,
    inflow_profile: jnp.ndarray | None = None,
    **_kwargs,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute the final velocity field using the semi-implicit Navier-Stokes equations.

    Supports both 2D and 3D; dimensionality is inferred from v0.shape[-1].

    When ``inflow_profile`` is provided (shape (ny,)), it is applied as a
    spatially-varying Dirichlet u_x BC at the x_lo face (x=0) after every step.

    Returns:
        (result, drag): result has same shape as v0.  drag is shape (1,) float32
        when an obstacle is present, else None.
    """
    ndim = v0.shape[-1]  # 2 for 2D, 3 for 3D
    if ndim == 2:
        # squeeze the dummy nz=1 axis: (nx, ny, 1, 2) → (nx, ny, 2)
        v0 = v0[:, :, 0, :]
    domain_sizes = (domain_extent,) * ndim

    bc = _jaxcfd_bc(boundary_conditions, ndim)

    spatial_shape = v0.shape[:-1]
    grid = cfd.grids.Grid(
        spatial_shape,
        domain=tuple((0.0, L) for L in domain_sizes),
    )
    dx = domain_extent / spatial_shape[0]

    # Rasterize obstacle mask (2-D only; drag computed in 2-D)
    if ndim == 2 and obstacle is not None and obstacle.get("shape"):
        obs_mask = _make_obstacle_mask_jaxcfd(
            obstacle, spatial_shape[0], spatial_shape[1]
        )
    else:
        obs_mask = None

    # --- staggered grid helper -------------------------------------------
    def interp_to_face(v_comp, own_axis):
        return 0.5 * (v_comp + jnp.roll(v_comp, -1, axis=own_axis))

    def make_grid_vars(v_2d):
        return tuple(
            cfd.grids.GridVariable(
                cfd.grids.GridArray(
                    interp_to_face(v_2d[..., i], own_axis=i),
                    grid=grid,
                    offset=tuple(1.0 if j == i else 0.5 for j in range(ndim)),
                ),
                bc,
            )
            for i in range(ndim)
        )

    def interp_to_coloc(v_stag, own_axis):
        return 0.5 * (v_stag + jnp.roll(v_stag, 1, axis=own_axis))

    # --- NS step function ------------------------------------------------
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid
        ),
        steps=inner_steps,
    )

    v_grid = make_grid_vars(v0)

    if inflow_profile is not None and ndim == 2:
        # Resample inflow_profile to ny if needed
        ny = spatial_shape[1]
        prof_len = inflow_profile.shape[0]
        if prof_len != ny:
            src_y = jnp.linspace(0, 1, prof_len)
            dst_y = jnp.linspace(0, 1, ny)
            ux_inflow = jnp.interp(dst_y, src_y, inflow_profile)
        else:
            ux_inflow = inflow_profile  # (ny,)

        # Run step-by-step with lax.scan, applying inflow override after each step.
        # The staggered u_x component lives at x-faces; face i=0 corresponds to x=0.
        def step_with_inflow(v_stag, _):
            v_stag = step_fn(v_stag)
            # IBM volume-penalization: zero velocity inside obstacle cells.
            # obs_mask shape is (nx, ny); staggered face arrays have the same shape
            # in jax-cfd's periodic shifted indexing.  jnp.where is fully differentiable.
            if obs_mask is not None:
                new_components = []
                for i in range(ndim):
                    data = v_stag[i].array.data
                    data = jnp.where(obs_mask, 0.0, data)
                    new_components.append(
                        cfd.grids.GridVariable(
                            cfd.grids.GridArray(
                                data, grid=grid, offset=v_stag[i].array.offset
                            ),
                            bc,
                        )
                    )
                v_stag = tuple(new_components)
            # Override x=0 face of u_x (staggered x-velocity, component 0)
            ux_data = v_stag[0].array.data
            ux_data = ux_data.at[0, :].set(ux_inflow)
            new_ux = cfd.grids.GridVariable(
                cfd.grids.GridArray(ux_data, grid=grid, offset=v_stag[0].array.offset),
                bc,
            )
            # Override x=0 face of u_y to zero (no transverse inflow)
            uy_data = v_stag[1].array.data
            uy_data = uy_data.at[0, :].set(0.0)
            new_uy = cfd.grids.GridVariable(
                cfd.grids.GridArray(uy_data, grid=grid, offset=v_stag[1].array.offset),
                bc,
            )
            return (new_ux, new_uy), None

        v_final_stag, _ = jax.lax.scan(step_with_inflow, v_grid, None, length=steps)
        stag_components = [v_final_stag[i].array.data for i in range(ndim)]
        final_v_grid = v_final_stag
    else:
        # IBM volume-penalization for non-inflow path: wrap step_fn so obstacle
        # cells are zeroed after every NS sub-step.
        if obs_mask is not None:
            orig_step_fn = step_fn

            def step_fn_with_obs(v):
                v = orig_step_fn(v)
                new_comps = []
                for i in range(ndim):
                    data = v[i].array.data
                    data = jnp.where(obs_mask, 0.0, data)
                    new_comps.append(
                        cfd.grids.GridVariable(
                            cfd.grids.GridArray(
                                data, grid=grid, offset=v[i].array.offset
                            ),
                            bc,
                        )
                    )
                return tuple(new_comps)

            step_fn = step_fn_with_obs

        rollout_fn = cfd.funcutils.trajectory(step_fn, steps)
        _, trajectory = rollout_fn(v_grid)
        stag_components = [trajectory[i].array.data[-1] for i in range(ndim)]
        # Rebuild GridVariables for pressure solve
        final_v_grid = tuple(
            cfd.grids.GridVariable(
                cfd.grids.GridArray(
                    stag_components[i],
                    grid=grid,
                    offset=tuple(1.0 if j == i else 0.5 for j in range(ndim)),
                ),
                bc,
            )
            for i in range(ndim)
        )

    # --- interpolate back to collocated grid ---
    coloc_components = [
        interp_to_coloc(stag_components[i], own_axis=i) for i in range(ndim)
    ]
    result = jnp.stack(coloc_components, axis=-1)

    # --- drag computation -------------------------------------------------
    drag = None
    if obs_mask is not None and ndim == 2:
        ux_coloc = coloc_components[0]  # (nx, ny)
        pressure = _extract_pressure_jaxcfd(final_v_grid, grid, bc)
        drag = _compute_drag(ux_coloc, pressure, obs_mask, viscosity, dx)

    if ndim == 2:
        result = result[:, :, None, :]  # (nx, ny, 2) → (nx, ny, 1, 2)
    return result, drag


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:  # mosaic:io
    result, drag = cfd_fwd(**inputs)
    out = dict(result=result)
    if drag is not None:
        out["drag"] = drag
    else:
        # Always return drag key so abstract_eval / VJP routing works
        out["drag"] = jnp.zeros((1,), dtype=jnp.float32)
    return out


def _unpack_scalars(d: dict) -> dict:  # mosaic:io
    """Extract Python floats from 1-element arrays for JIT-static scalar params."""
    for key in ("density", "viscosity", "dt"):
        if key in d:
            d[key] = float(d[key][0])
    return d


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_jit(_unpack_scalars(inputs.model_dump()))


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt:autodiff
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
    """Calculate output shape of apply from the shape of its inputs.

    For jax-cfd the output ``result`` always has the same shape as ``v0``,
    and ``drag`` is always shape ``(1,)``.
    """
    raw = abstract_inputs.model_dump()
    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        v0_shape = tuple(v0["shape"])
        v0_dtype = v0["dtype"]
    else:
        arr = jnp.asarray(v0)
        v0_shape = arr.shape
        v0_dtype = str(arr.dtype)
    return {
        "result": {"shape": v0_shape, "dtype": v0_dtype},
        "drag": {"shape": (1,), "dtype": "float32"},
    }


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    grads = vjp_func(cotangent_vector)[0]
    # Scalar physics params (viscosity, dt, etc.) are unpacked to Python floats
    # before the VJP, so their gradients come back as 0D arrays.  Tesseract
    # expects shape (1,) to match the input schema; reshape any 0D grad here.
    return jax.tree.map(jnp.atleast_1d, grads)
