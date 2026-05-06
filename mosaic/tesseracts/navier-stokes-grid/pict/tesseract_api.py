from typing import Any

import numpy as np
import PISOtorch
import PISOtorch_simulation
import torch
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import BCType


class InputSchema(_CanonicalInputSchema):
    pass


class OutputSchema(_CanonicalOutputSchema):
    pass


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

# PICT is GPU-only: the PISOtorch C++ extension requires CUDA.
# If no GPU is available the tesseract will raise at domain-creation time
# (matching PICT's own assertion).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Domain factory
# ---------------------------------------------------------------------------


def _make_vertex_grid(
    N: int, ndim: int, dtype: torch.dtype
) -> torch.Tensor:  # mosaic:init
    """Build a uniform vertex-coordinate grid with unit cell spacing.

    PICT uses unit cell spacing (dx=1), so vertex coordinates run from 0 to N
    in each spatial dimension.

    Args:
        N: Number of cells per spatial dimension.
        ndim: Spatial dimensionality (2 or 3).
        dtype: Torch dtype.

    Returns:
        Vertex coordinate tensor on CPU with shape:
          2-D: (1, 2, ny+1, nx+1)
          3-D: (1, 3, nz+1, ny+1, nx+1)
        where the channel dimension holds (x-coord, y-coord[, z-coord]).
    """
    coords_1d = torch.arange(N + 1, dtype=dtype)  # 0, 1, ..., N
    if ndim == 2:
        # meshgrid returns (ny+1, nx+1) grids when indexed='ij' with (y, x)
        gy, gx = torch.meshgrid(coords_1d, coords_1d, indexing="ij")  # (ny+1, nx+1)
        grid = torch.stack([gx, gy], dim=0)  # (2, ny+1, nx+1)
        grid = grid.unsqueeze(0)  # (1, 2, ny+1, nx+1)
    else:
        gz, gy, gx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=0)  # (3, nz+1, ny+1, nx+1)
        grid = grid.unsqueeze(0)  # (1, 3, nz+1, ny+1, nx+1)
    return grid.contiguous()


def _make_uniform_rect_grid(  # mosaic:init
    nx: int, ny: int, x0: float, y0: float, x1: float, y1: float, dtype: torch.dtype
) -> torch.Tensor:
    """Build a uniform orthogonal vertex grid for a rectangular block.

    Args:
        nx: Number of cells in x.
        ny: Number of cells in y.
        x0, y0: Lower-left corner coordinates.
        x1, y1: Upper-right corner coordinates.
        dtype: Torch dtype.

    Returns:
        Vertex coordinate tensor on CPU with shape (1, 2, ny+1, nx+1).
    """
    coords_x = torch.linspace(x0, x1, nx + 1, dtype=dtype)
    coords_y = torch.linspace(y0, y1, ny + 1, dtype=dtype)
    gy, gx = torch.meshgrid(coords_y, coords_x, indexing="ij")  # (ny+1, nx+1)
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, ny+1, nx+1)
    return grid.contiguous()


def _make_arc_block_grid(  # mosaic:init
    nx: int,
    ny: int,
    outer_rect: tuple,
    arc_cx: float,
    arc_cy: float,
    arc_r: float,
    arc_face: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a curved vertex grid for a block whose one face follows a circular arc.

    The outer boundary (the face farthest from the obstacle) is a straight
    rectangular edge.  The inner boundary (the face touching the obstacle) is a
    circular arc.  Interior vertices are obtained by linear blending (lerp)
    between the outer and inner boundaries in the normal direction.

    Args:
        nx: Number of cells in x.
        ny: Number of cells in y.
        outer_rect: (x0, y0, x1, y1) bounding box of the block in cell coords.
        arc_cx, arc_cy: Circle centre in cell coords.
        arc_r: Circle radius in cell coords.
        arc_face: Which face of the block is the arc: "-y" (bottom), "+y" (top),
            "-x" (left), "+x" (right).  The *opposite* face stays rectangular.
        dtype: Torch dtype.

    Returns:
        Vertex coordinate tensor on CPU with shape (1, 2, ny+1, nx+1).
    """
    import math

    x0, y0, x1, y1 = outer_rect

    # --- build the rectangular vertex positions (uniform) ---
    coords_x = torch.linspace(x0, x1, nx + 1, dtype=dtype)  # (nx+1,)
    coords_y = torch.linspace(y0, y1, ny + 1, dtype=dtype)  # (ny+1,)

    # --- compute the arc boundary (inner face) ---
    # The arc maps from one corner of the block to the other along the circle.
    # We parameterise it by the x- or y-coordinates on the straight far edge.
    if arc_face in ("+y", "-y"):
        # Arc face is horizontal → nx+1 arc points, one per x-column vertex.
        # The angles at the two endpoints are determined by which x-coords lie on
        # the circle at the y-level of the inner face.
        # We linearly interpolate angles between the two endpoint angles so that
        # the arc spacing matches the rectangular x-spacing.
        theta0 = math.atan2(
            y1 - arc_cy if arc_face == "+y" else y0 - arc_cy, x0 - arc_cx
        )
        theta1 = math.atan2(
            y1 - arc_cy if arc_face == "+y" else y0 - arc_cy, x1 - arc_cx
        )
        thetas = torch.linspace(theta0, theta1, nx + 1, dtype=dtype)  # (nx+1,)
        arc_x = arc_cx + arc_r * torch.cos(thetas)  # (nx+1,)
        arc_y = arc_cy + arc_r * torch.sin(thetas)  # (nx+1,)

        # Build grid: for each row j, blend between outer edge and arc.
        # arc_face="+y" → outer edge is y0 (bottom), arc is at top (j=ny).
        # arc_face="-y" → outer edge is y1 (top),    arc is at bottom (j=0).
        gx = torch.zeros(ny + 1, nx + 1, dtype=dtype)
        gy = torch.zeros(ny + 1, nx + 1, dtype=dtype)

        for j in range(ny + 1):
            if arc_face == "+y":
                # j=0 → outer (straight bottom), j=ny → arc (top)
                t = j / ny
                gx[j, :] = (1 - t) * coords_x + t * arc_x
                gy[j, :] = (1 - t) * y0 + t * arc_y
            else:  # arc_face == "-y"
                # j=0 → arc (bottom), j=ny → outer (straight top)
                t = 1.0 - j / ny
                gx[j, :] = t * arc_x + (1 - t) * coords_x
                gy[j, :] = t * arc_y + (1 - t) * y1

    else:  # arc_face in ("+x", "-x")
        # Arc face is vertical → ny+1 arc points, one per y-row vertex.
        theta0 = math.atan2(
            y0 - arc_cy, x1 - arc_cx if arc_face == "+x" else x0 - arc_cx
        )
        theta1 = math.atan2(
            y1 - arc_cy, x1 - arc_cx if arc_face == "+x" else x0 - arc_cx
        )
        thetas = torch.linspace(theta0, theta1, ny + 1, dtype=dtype)  # (ny+1,)
        arc_x = arc_cx + arc_r * torch.cos(thetas)  # (ny+1,)
        arc_y = arc_cy + arc_r * torch.sin(thetas)  # (ny+1,)

        gx = torch.zeros(ny + 1, nx + 1, dtype=dtype)
        gy = torch.zeros(ny + 1, nx + 1, dtype=dtype)

        for i in range(nx + 1):
            if arc_face == "+x":
                # i=0 → outer (straight left), i=nx → arc (right)
                t = i / nx
                gx[:, i] = (1 - t) * x0 + t * arc_x
                gy[:, i] = (1 - t) * coords_y + t * arc_y
            else:  # arc_face == "-x"
                # i=0 → arc (left), i=nx → outer (straight right)
                t = 1.0 - i / nx
                gx[:, i] = t * arc_x + (1 - t) * x1
                gy[:, i] = t * arc_y + (1 - t) * coords_y

    grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, ny+1, nx+1)
    return grid.contiguous()


def _make_corner_block_grid(  # mosaic:init
    nx: int,
    ny: int,
    outer_rect: tuple,
    arc_cx: float,
    arc_cy: float,
    arc_r: float,
    arc_corner: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a vertex grid for a corner block using bilinear (Coons) interpolation.

    The corner block has two outer straight edges (the domain boundary) and two
    inner edges that connect to the adjacent arc-facing blocks.  The inner corner
    snaps to the nearest point on the circle; the two inner edges are straight
    lines from the outer adjacent corners to that arc-endpoint.  Interior vertices
    are computed via a Coons patch (bilinear transfinite interpolation) of the
    four edges.

    Args:
        nx: Number of cells in x.
        ny: Number of cells in y.
        outer_rect: (x0, y0, x1, y1) bounding box of the block in cell coords.
        arc_cx, arc_cy: Circle centre in cell coords.
        arc_r: Circle radius in cell coords.
        arc_corner: Which corner of the block faces the obstacle:
            "bl" (bottom-left), "br" (bottom-right),
            "tl" (top-left),    "tr" (top-right).
        dtype: Torch dtype.

    Returns:
        Vertex coordinate tensor on CPU with shape (1, 2, ny+1, nx+1).
    """
    import math

    x0, y0, x1, y1 = outer_rect

    # Compute the arc endpoint — the point on the circle closest to the inner
    # corner.  This is where the two adjacent arc-block edges terminate.
    if arc_corner == "tr":
        # Inner corner is top-right (x1, y1); arc endpoint direction: (x1-cx, y1-cy)
        theta_inner = math.atan2(y1 - arc_cy, x1 - arc_cx)
    elif arc_corner == "tl":
        theta_inner = math.atan2(y1 - arc_cy, x0 - arc_cx)
    elif arc_corner == "br":
        theta_inner = math.atan2(y0 - arc_cy, x1 - arc_cx)
    else:  # "bl"
        theta_inner = math.atan2(y0 - arc_cy, x0 - arc_cx)

    arc_px = arc_cx + arc_r * math.cos(theta_inner)
    arc_py = arc_cy + arc_r * math.sin(theta_inner)

    # Define the four corners of the Coons patch:
    #   c00 = (i=0, j=0), c10 = (i=nx, j=0)
    #   c01 = (i=0, j=ny), c11 = (i=nx, j=ny)
    # The inner corner snaps to the arc point; the three outer corners are
    # the rectangle corners.
    if arc_corner == "tr":
        # inner corner at top-right → (i=nx, j=ny) = arc point
        c00 = (x0, y0)
        c10 = (x1, y0)
        c01 = (x0, y1)
        c11 = (arc_px, arc_py)
    elif arc_corner == "tl":
        # inner corner at top-left → (i=0, j=ny) = arc point
        c00 = (x0, y0)
        c10 = (x1, y0)
        c01 = (arc_px, arc_py)
        c11 = (x1, y1)
    elif arc_corner == "br":
        # inner corner at bottom-right → (i=nx, j=0) = arc point
        c00 = (x0, y0)
        c10 = (arc_px, arc_py)
        c01 = (x0, y1)
        c11 = (x1, y1)
    else:  # "bl"
        # inner corner at bottom-left → (i=0, j=0) = arc point
        c00 = (arc_px, arc_py)
        c10 = (x1, y0)
        c01 = (x0, y1)
        c11 = (x1, y1)

    # All four edges are straight (linear) between their endpoint corners.
    # The Coons patch with all-straight edges reduces to bilinear interpolation:
    #   P(s,t) = (1-t)(1-s)*c00 + (1-t)*s*c10 + t*(1-s)*c01 + t*s*c11
    s = torch.linspace(0.0, 1.0, nx + 1, dtype=dtype)  # (nx+1,)
    t = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype)  # (ny+1,)
    t2 = t.unsqueeze(1)  # (ny+1, 1)
    s2 = s.unsqueeze(0)  # (1, nx+1)

    gx = (
        (1 - t2) * (1 - s2) * c00[0]
        + (1 - t2) * s2 * c10[0]
        + t2 * (1 - s2) * c01[0]
        + t2 * s2 * c11[0]
    )
    gy = (
        (1 - t2) * (1 - s2) * c00[1]
        + (1 - t2) * s2 * c10[1]
        + t2 * (1 - s2) * c01[1]
        + t2 * s2 * c11[1]
    )

    grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, ny+1, nx+1)
    return grid.contiguous()


def _make_domain(  # mosaic:init
    viscosity_val: float,
    N: int,
    ndim: int,
    dtype: torch.dtype = torch.float32,
    inflow_profile_t: torch.Tensor | None = None,
    phys_scale: float = 1.0,
    obstacle: dict | None = None,
    in_vel: float = 0.0,
    y_walls_noslip: bool = False,
) -> tuple:
    """Create a single-block domain of size N×N (×N for 3-D).

    Three modes are supported:

    * **Periodic** (default): ``inflow_profile_t`` and ``obstacle`` are both
      ``None``.  Uses ``CreateBlockWithSize`` — all faces periodic.
    * **Inflow/channel**: ``inflow_profile_t`` is provided (shape ``(ny,)``
      torch tensor).  Walls are applied to top/bottom/right faces via
      ``CloseAllBoundaries()``, the ``"-x"`` face receives the inflow via
      ``getBoundary("-x").setVelocity(...)``, and the ``"+x"`` (outflow) face
      uses a proper advective (convective) outflow BC via
      ``makeVelocityVarying()`` + per-step
      ``PISOtorch_simulation.update_advective_boundaries()`` callback.
    * **Cylinder obstacle**: ``obstacle`` is provided as a dict with keys
      ``"shape"``, ``"center"``, and ``"radius"``.  Creates an 8-block ring
      topology where the missing center block forms the obstacle via closed
      boundaries.  An optional ``inflow_profile_t`` (torch tensor of shape
      ``(ny,)``) overrides the uniform inflow; the three left-face blocks each
      receive their matching y-slice.
    """
    # ------------------------------------------------------------------
    # Cylinder obstacle path: delegate to 8-block ring topology
    # ------------------------------------------------------------------
    if obstacle is not None:
        return _make_domain_cylinder(
            viscosity_val,
            N,
            in_vel,
            obstacle,
            dtype,
            inflow_profile_t=inflow_profile_t,
            phys_scale=phys_scale,
            y_walls_noslip=y_walls_noslip,
        )

    v = torch.tensor([viscosity_val], dtype=dtype, device=torch.device("cpu"))
    domain = PISOtorch.Domain(
        ndim,
        v,
        name="PICTDomain",
        dtype=dtype,
        device=_DEVICE,
        passiveScalarChannels=0,
    )

    # ------------------------------------------------------------------
    # Periodic path (default)
    # ------------------------------------------------------------------
    if inflow_profile_t is None:
        if ndim == 3:
            block_size = PISOtorch.Int4(x=N, y=N, z=N)
        else:
            block_size = PISOtorch.Int4(x=N, y=N)
        domain.CreateBlockWithSize(block_size, name="Block")
        return (
            domain,
            [],
            None,
            lambda v: domain.getBlock(0).setVelocity(v),
            lambda d: d.getBlock(0).velocity,
            None,  # no drag assembler for periodic domains
            None,  # no inflow_setter for periodic domains
        )

    # ------------------------------------------------------------------
    # Non-periodic paths: build an explicit vertex-coordinate grid
    # ------------------------------------------------------------------
    grid = _make_vertex_grid(N, ndim, dtype).to(_DEVICE)  # must be on device
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")

    # ------------------------------------------------------------------
    # Lid-driven cavity
    # ------------------------------------------------------------------
    if lid_velocity_t is not None:
        block.CloseAllBoundaries()

        # lid_velocity_t shape: (nx, ny, 2) — physical units, possibly autograd leaf
        # Cast to solver dtype/device and scale to PICT unit-cell coordinates;
        # these ops keep the autograd chain intact so boundary VJPs flow back
        # to the original user tensor.
        lv_t = lid_velocity_t.to(dtype=dtype, device=_DEVICE) * float(
            phys_scale
        )  # (nx, ny, 2)

        if ndim == 2:
            # 2-D lid: apply on "+y" face.  Face tensor shape: (1, 2, 1, nx).
            # The lid is a 1-cell-thick face; sample the first y-row (index 0).
            # For the canonical lid_cavity IC every row is identical, so this
            # is equivalent to picking any y slice.
            lv_face = lv_t[:, 0, :]  # (nx, 2)
            lv_face = lv_face.permute(1, 0)  # (2, nx)
            lv_face = lv_face.unsqueeze(0).unsqueeze(2)  # (1, 2, 1, nx)
        else:
            # 3-D lid: apply on "+z" face.  Face tensor shape: (1, 3, 1, ny, nx).
            # lid_velocity_t carries u_x, u_y components; u_z is set to zero.
            # Matches the phiflow 3-D cavity reference (top z-face, u_z=0).
            ux = lv_t[..., 0]  # (nx, ny)
            uy = lv_t[..., 1]  # (nx, ny)
            uz = torch.zeros_like(ux)  # (nx, ny)
            lv_stack = torch.stack([ux, uy, uz], dim=0)  # (3, nx, ny)
            lv_face = lv_stack.permute(0, 2, 1)  # (3, ny, nx)
            lv_face = lv_face.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, ny, nx)

        lv_face = lv_face.contiguous()
        lid_face_name = "+y" if ndim == 2 else "+z"
        block.CloseBoundary(lid_face_name, lv_face)
        return (
            domain,
            [],
            None,
            lambda v: domain.getBlock(0).setVelocity(v),
            lambda d: d.getBlock(0).velocity,
            None,  # no drag assembler for lid-driven cavity
            None,  # no inflow_setter for lid-driven cavity
        )

    # ------------------------------------------------------------------
    # Inflow / channel flow
    # ------------------------------------------------------------------
    if inflow_profile_t is not None:
        # Close all faces first (gives no-slip walls everywhere), then
        # override the inflow face and re-open the outflow face.
        block.CloseAllBoundaries()

        # inflow_profile_t shape: (ny,) — physical x-velocity at each y cell.
        # Keep as a live torch tensor (may be an autograd leaf) so boundary
        # VJPs flow back to the caller.  Scale to PICT coordinates via a plain
        # multiply that stays in the autograd graph.
        ip_t = inflow_profile_t.to(dtype=dtype, device=_DEVICE) * float(
            phys_scale
        )  # (ny,)

        def _build_inflow_bc(ip: torch.Tensor) -> torch.Tensor:
            """Build the inflow BC face tensor from a scaled profile (ny,)."""
            if ndim == 2:
                # Inflow tensor shape for "-x" face: (1, 2, ny, 1)
                ux = ip  # (ny,)
                uy = torch.zeros_like(ux)  # (ny,)
                bc = torch.stack([ux, uy], dim=0)  # (2, ny)
                bc = bc.unsqueeze(0).unsqueeze(-1)  # (1, 2, ny, 1)
            else:
                # 3-D: broadcast inflow_profile (ny,) across z dimension → (ny, nz)
                nz = N
                ux = ip.unsqueeze(1).expand(N, nz)  # (ny, nz)
                zero = torch.zeros_like(ux)  # (ny, nz)
                bc = torch.stack([ux, zero, zero], dim=0)  # (3, ny, nz)
                bc = bc.unsqueeze(0).unsqueeze(-1)  # (1, 3, ny, nz, 1)
            return bc.contiguous()

        # Set the inflow BC once for PrepareSolve (required so PICT knows the
        # boundary layout before the solve starts).
        inflow_bc_init = _build_inflow_bc(ip_t)
        inflow_boundary = block.getBoundary("-x")
        inflow_boundary.setVelocity(inflow_bc_init)

        # inflow_setter: re-applies the BC each PISO step inside the PRE
        # callback so that inflow_profile_t remains in PISOtorch_diff's
        # autograd graph for every step of the time loop.  Without this,
        # inflow_profile_t is set once before PrepareSolve() and is never
        # seen by the differentiable subgraph → VJP returns zero.
        def _inflow_setter() -> None:
            # Rebuild ip_t from the live inflow_profile_t each step so that
            # the multiplication (and hence the autograd edge) is re-recorded
            # by PISOtorch_diff on every step.
            ip_live = inflow_profile_t.to(dtype=dtype, device=_DEVICE) * float(
                phys_scale
            )
            inflow_boundary.setVelocity(_build_inflow_bc(ip_live))

        # Outflow face "+x": advective (convective) outflow BC.
        # Initialise with mean inflow velocity (detached — used only for init
        # and advection velocity; does not need to participate in the VJP).
        mean_ux_val = float(ip_t.detach().mean().item()) if ip_t.numel() > 0 else 0.0
        if ndim == 2:
            out_vel = torch.tensor(
                [[[[mean_ux_val]], [[0.0]]]], dtype=dtype, device=_DEVICE
            )  # (1,2,1,1)
            out_vel = out_vel.expand(1, 2, N, 1).contiguous()
        else:
            out_vel = torch.zeros(1, 3, N, N, 1, dtype=dtype, device=_DEVICE)
            out_vel[:, 0, :, :, :] = mean_ux_val
        out_bound = block.getBoundary("+x")
        out_bound.setVelocity(out_vel)
        out_bound.makeVelocityVarying()  # required by update_advective_boundaries

        # Advection velocity for update_advective_boundaries: (1, ndim) tensor
        # — normal component of the mean outflow speed; zeros for tangential.
        adv_comps = [mean_ux_val] + [0.0] * (ndim - 1)
        adv_velm = torch.tensor([adv_comps], dtype=dtype, device=_DEVICE)  # (1, ndim)

        PISOtorch_simulation.balance_boundary_fluxes(domain, [out_bound])
        return (
            domain,
            [out_bound],
            adv_velm,
            lambda v: domain.getBlock(0).setVelocity(v),
            lambda d: d.getBlock(0).velocity,
            None,  # no drag assembler for single-block channel flow
            _inflow_setter,
        )

    return (
        domain,
        [],
        None,
        lambda v: domain.getBlock(0).setVelocity(v),
        lambda d: d.getBlock(0).velocity,
        None,  # no drag assembler (fallback path)
        None,  # no inflow_setter (fallback path)
    )


def _make_domain_cylinder(  # mosaic:init
    viscosity_val: float,
    N: int,
    in_vel: float,
    obstacle: dict,
    dtype: torch.dtype,
    inflow_profile_t: torch.Tensor | None = None,
    phys_scale: float = 1.0,
    y_walls_noslip: bool = False,
) -> tuple:
    """Create an 8-block ring-topology domain with a circular obstacle.

    Args:
        viscosity_val: Kinematic viscosity ν in PICT units (already scaled).
        N: Number of cells per spatial dimension.
        in_vel: Mean x-velocity at inflow (PICT units).  Used only when
            ``inflow_profile_t`` is ``None`` (uniform inflow fallback).
        obstacle: Dict with keys ``"shape"``, ``"center"`` (list[float]),
            ``"radius"`` (float).  Coordinates are fractions of domain_extent.
        dtype: Torch dtype.
        inflow_profile_t: Optional 1-D ``(N,)`` torch tensor carrying the
            x-velocity inflow profile in physical units.  May be an autograd
            leaf; boundary VJPs flow back when ``requires_grad=True``.  The
            profile is split by y-slice across the three left-face blocks
            (``Bbl``, ``Bml``, ``Btl``).
        phys_scale: ``N/L`` scaling factor applied to ``inflow_profile_t`` to
            convert physical velocities to PICT unit-cell coordinates.

    Returns:
        ``(domain, out_bounds, adv_velm, v0_setter, assembler, drag_assembler)``
        where ``drag_assembler()`` computes the differentiable x-direction drag
        on the cylinder obstacle via a discrete pressure + viscous surface
        integral over the four obstacle-facing block boundaries.
    """
    # ------------------------------------------------------------------
    # Compute obstacle geometry in cell coordinates
    # ------------------------------------------------------------------
    cx = obstacle["center"][0] * N
    cy = obstacle["center"][1] * N
    r = obstacle["radius"] * N
    # PICT requires all spatial block dims >= 3. Enforce here so small-N debug
    # runs (e.g. N=16, radius=0.05 → 2·r≈1.6 → obs_w=2) don't trigger the C++
    # "all spatial dimensions must be at least 3" assertion on the centre blocks.
    # obs_w/obs_h are the bounding box of the circle (equal, used to partition
    # the block layout); the actual obstacle boundary follows the circular arc.
    obs_w = max(3, round(2 * r))
    obs_h = obs_w  # bounding box is square; arc blocks round out the circle

    x_pos = round(cx - obs_w / 2)
    y_pos = round(cy - obs_h / 2)
    x_pos = max(1, min(x_pos, N - obs_w - 1))
    y_pos = max(1, min(y_pos, N - obs_h - 1))

    xs = [x_pos, obs_w, N - x_pos - obs_w]  # left, mid, right column widths
    ys = [y_pos, obs_h, N - y_pos - obs_h]  # bot, mid, top row heights

    # Block corner coordinates (cell units)
    x0 = 0
    x1 = xs[0]
    x1m = xs[0]
    x2m = xs[0] + xs[1]
    x2 = xs[0] + xs[1]
    x3 = N

    y0 = 0
    y1 = ys[0]
    y1m = ys[0]
    y2m = ys[0] + ys[1]
    y2 = ys[0] + ys[1]
    y3 = N

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    v = torch.tensor([viscosity_val], dtype=dtype, device=torch.device("cpu"))
    domain = PISOtorch.Domain(
        2,
        v,
        name="CylinderDomain",
        dtype=dtype,
        device=_DEVICE,
        passiveScalarChannels=0,
    )

    # ------------------------------------------------------------------
    # 8 blocks (center block is the obstacle — not created)
    # Curved vertex grids: arc-adjacent blocks use circular arcs on their
    # obstacle-facing edge; corner blocks use Coons patch interpolation so
    # their two inner edges also conform to the circle.
    # ------------------------------------------------------------------
    Bbl = domain.CreateBlock(
        vertexCoordinates=_make_corner_block_grid(
            xs[0], ys[0], (x0, y0, x1, y1), cx, cy, r, "tr", dtype
        ).to(_DEVICE),
        name="BlockBotLeft",
    )
    Bbm = domain.CreateBlock(
        vertexCoordinates=_make_arc_block_grid(
            xs[1], ys[0], (x1m, y0, x2m, y1), cx, cy, r, "+y", dtype
        ).to(_DEVICE),
        name="BlockBotMiddle",
    )
    Bbr = domain.CreateBlock(
        vertexCoordinates=_make_corner_block_grid(
            xs[2], ys[0], (x2, y0, x3, y1), cx, cy, r, "tl", dtype
        ).to(_DEVICE),
        name="BlockBotRight",
    )
    Bml = domain.CreateBlock(
        vertexCoordinates=_make_arc_block_grid(
            xs[0], ys[1], (x0, y1m, x1, y2m), cx, cy, r, "+x", dtype
        ).to(_DEVICE),
        name="BlockMidLeft",
    )
    Bmr = domain.CreateBlock(
        vertexCoordinates=_make_arc_block_grid(
            xs[2], ys[1], (x2, y1m, x3, y2m), cx, cy, r, "-x", dtype
        ).to(_DEVICE),
        name="BlockMidRight",
    )
    Btl = domain.CreateBlock(
        vertexCoordinates=_make_corner_block_grid(
            xs[0], ys[2], (x0, y2, x1, y3), cx, cy, r, "br", dtype
        ).to(_DEVICE),
        name="BlockTopLeft",
    )
    Btm = domain.CreateBlock(
        vertexCoordinates=_make_arc_block_grid(
            xs[1], ys[2], (x1m, y2, x2m, y3), cx, cy, r, "-y", dtype
        ).to(_DEVICE),
        name="BlockTopMiddle",
    )
    Btr = domain.CreateBlock(
        vertexCoordinates=_make_corner_block_grid(
            xs[2], ys[2], (x2, y2, x3, y3), cx, cy, r, "bl", dtype
        ).to(_DEVICE),
        name="BlockTopRight",
    )

    # ------------------------------------------------------------------
    # Close obstacle faces (no-slip walls facing the obstacle cavity)
    # ------------------------------------------------------------------
    Bbm.CloseBoundary("+y")
    Bml.CloseBoundary("+x")
    Bmr.CloseBoundary("-x")
    Btm.CloseBoundary("-y")

    # ------------------------------------------------------------------
    # Ring connections
    # ------------------------------------------------------------------
    # Horizontal ring (axes=["-y"])
    Btl.ConnectBlock("+x", Btm, "-x", "-y")
    Btm.ConnectBlock("+x", Btr, "-x", "-y")
    Bbl.ConnectBlock("+x", Bbm, "-x", "-y")
    Bbm.ConnectBlock("+x", Bbr, "-x", "-y")
    # Vertical ring (axes=["-x"])
    Btr.ConnectBlock("-y", Bmr, "+y", "-x")
    Bmr.ConnectBlock("-y", Bbr, "+y", "-x")
    Btl.ConnectBlock("-y", Bml, "+y", "-x")
    Bml.ConnectBlock("-y", Bbl, "+y", "-x")

    # ------------------------------------------------------------------
    # Top/bottom BCs: periodic wrap or no-slip walls
    # ------------------------------------------------------------------
    if y_walls_noslip:
        # No-slip walls: close outer top/bottom faces (zero velocity)
        Btl.CloseBoundary("+y")
        Btm.CloseBoundary("+y")
        Btr.CloseBoundary("+y")
        Bbl.CloseBoundary("-y")
        Bbm.CloseBoundary("-y")
        Bbr.CloseBoundary("-y")
    else:
        # Periodic top-bottom (axes=["-x"])
        Btl.ConnectBlock("+y", Bbl, "-y", "-x")
        Btm.ConnectBlock("+y", Bbm, "-y", "-x")
        Btr.ConnectBlock("+y", Bbr, "-y", "-x")

    # ------------------------------------------------------------------
    # Inflow BCs (left faces, split by row height)
    # ------------------------------------------------------------------
    # When an explicit inflow profile is provided we slice it into the three
    # left-face blocks (Bbl spans y in [y0:y1], Bml spans [y1m:y2m], Btl spans
    # [y2:y3]).  The profile is kept as a live torch tensor so boundary VJPs
    # flow back to the caller's autograd leaf.  Otherwise we fall back to a
    # uniform inflow at ``in_vel``.
    cyl_inflow_setter = None  # will be set when inflow_profile_t is not None

    def _cyl_inflow_face(ip_slice: torch.Tensor) -> torch.Tensor:
        """Build a (1, 2, ny_block, 1) face tensor with u_x = ip_slice, u_y=0."""
        uy = torch.zeros_like(ip_slice)
        face = torch.stack([ip_slice, uy], dim=0)  # (2, ny_block)
        face = face.unsqueeze(0).unsqueeze(-1)  # (1, 2, ny_block, 1)
        return face.contiguous()

    if inflow_profile_t is not None:
        # PICT velocities are in physical units (m/s) — same coordinate as v0.
        # No phys_scale factor here; phys_scale only applies to time (dt) and
        # kinematic viscosity (nu), not to velocity itself.
        ip_t = inflow_profile_t.to(dtype=dtype, device=_DEVICE)  # (N,) physical units

        # Cache boundary references to avoid re-lookups in the setter closure.
        _bnd_bl = Bbl.getBoundary("-x")
        _bnd_ml = Bml.getBoundary("-x")
        _bnd_tl = Btl.getBoundary("-x")

        _bnd_bl.setVelocity(_cyl_inflow_face(ip_t[y0:y1]))
        _bnd_ml.setVelocity(_cyl_inflow_face(ip_t[y1m:y2m]))
        _bnd_tl.setVelocity(_cyl_inflow_face(ip_t[y2:y3]))

        # inflow_setter: re-applies the split inflow BCs each PISO step so that
        # inflow_profile_t stays in PISOtorch_diff's autograd graph for every
        # step of the time loop (fixing zero-VJP for drag_opt).
        def _cyl_inflow_setter() -> None:
            ip_live = inflow_profile_t.to(dtype=dtype, device=_DEVICE)
            _bnd_bl.setVelocity(_cyl_inflow_face(ip_live[y0:y1]))
            _bnd_ml.setVelocity(_cyl_inflow_face(ip_live[y1m:y2m]))
            _bnd_tl.setVelocity(_cyl_inflow_face(ip_live[y2:y3]))

        cyl_inflow_setter = _cyl_inflow_setter
    else:

        def _make_inflow_slice(ny_block: int) -> torch.Tensor:
            t = torch.zeros(1, 2, ny_block, 1, dtype=dtype, device=_DEVICE)
            t[:, 0, :, :] = in_vel  # x-component = in_vel (physical units)
            return t

        Bbl.getBoundary("-x").setVelocity(_make_inflow_slice(ys[0]))
        Bml.getBoundary("-x").setVelocity(_make_inflow_slice(ys[1]))
        Btl.getBoundary("-x").setVelocity(_make_inflow_slice(ys[2]))

    # ------------------------------------------------------------------
    # Outflow BCs (right faces, advective)
    # ------------------------------------------------------------------
    # Initialise the outflow and advection velocity from the mean inflow speed;
    # when an explicit inflow profile is present we use its mean (detached from
    # the autograd graph — these are init values, not a VJP path).
    if inflow_profile_t is not None:
        # No phys_scale: velocities are in physical units consistent with v0
        mean_in_val = float(inflow_profile_t.detach().mean().item())
    else:
        mean_in_val = float(in_vel)

    out_bound_br = Bbr.getBoundary("+x")
    out_bound_mr = Bmr.getBoundary("+x")
    out_bound_tr = Btr.getBoundary("+x")
    for ob, ny_b in [
        (out_bound_br, ys[0]),
        (out_bound_mr, ys[1]),
        (out_bound_tr, ys[2]),
    ]:
        out_vel = torch.zeros(1, 2, ny_b, 1, dtype=dtype, device=_DEVICE)
        out_vel[:, 0, :, :] = mean_in_val
        ob.setVelocity(out_vel.contiguous())
        ob.makeVelocityVarying()
    out_bounds = [out_bound_br, out_bound_mr, out_bound_tr]
    PISOtorch_simulation.balance_boundary_fluxes(domain, out_bounds)
    adv_velm = torch.tensor([[mean_in_val, 0.0]], dtype=dtype, device=_DEVICE)

    # ------------------------------------------------------------------
    # v0_setter closure: distribute (1,2,N,N) PICT tensor into 8 blocks
    # ------------------------------------------------------------------
    def _set_v0(v0_pict: torch.Tensor) -> None:
        # v0_pict shape (1, 2, N, N): dim2=y-axis, dim3=x-axis
        Bbl.setVelocity(v0_pict[:, :, y0:y1, x0:x1].contiguous())
        Bbm.setVelocity(v0_pict[:, :, y0:y1, x1m:x2m].contiguous())
        Bbr.setVelocity(v0_pict[:, :, y0:y1, x2:x3].contiguous())
        Bml.setVelocity(v0_pict[:, :, y1m:y2m, x0:x1].contiguous())
        Bmr.setVelocity(v0_pict[:, :, y1m:y2m, x2:x3].contiguous())
        Btl.setVelocity(v0_pict[:, :, y2:y3, x0:x1].contiguous())
        Btm.setVelocity(v0_pict[:, :, y2:y3, x1m:x2m].contiguous())
        Btr.setVelocity(v0_pict[:, :, y2:y3, x2:x3].contiguous())

    # ------------------------------------------------------------------
    # assembler closure: reassemble (1,2,N,N) from 8 blocks
    # ------------------------------------------------------------------
    # NOTE: PISOtorch_diff's backward calls ``block.setVelocityGrad(v_grad)``
    # which requires v_grad to be contiguous.  The backward of ``torch.cat``
    # returns stride-slices of the upstream cotangent that are not contiguous;
    # attach a ``register_hook`` to each block velocity that materialises the
    # incoming gradient before PICT consumes it.
    def _ensure_contiguous_grad(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(lambda g: g.contiguous() if g is not None else g)
        return t

    def _assemble(domain: PISOtorch.Domain) -> torch.Tensor:  # noqa: ARG001
        z_obs = torch.zeros(1, 2, ys[1], xs[1], dtype=dtype, device=_DEVICE)
        vbl = _ensure_contiguous_grad(Bbl.velocity)
        vbm = _ensure_contiguous_grad(Bbm.velocity)
        vbr = _ensure_contiguous_grad(Bbr.velocity)
        vml = _ensure_contiguous_grad(Bml.velocity)
        vmr = _ensure_contiguous_grad(Bmr.velocity)
        vtl = _ensure_contiguous_grad(Btl.velocity)
        vtm = _ensure_contiguous_grad(Btm.velocity)
        vtr = _ensure_contiguous_grad(Btr.velocity)
        row_bot = torch.cat([vbl, vbm, vbr], dim=3)
        row_mid = torch.cat([vml, z_obs, vmr], dim=3)
        row_top = torch.cat([vtl, vtm, vtr], dim=3)
        return torch.cat([row_bot, row_mid, row_top], dim=2)  # (1, 2, N, N)

    # ------------------------------------------------------------------
    # drag_assembler: differentiable x-direction drag on the cylinder
    # ------------------------------------------------------------------
    # Computes drag via a discrete surface integral over the four
    # obstacle-facing block boundaries, matching phiflow's convention
    # (physical units: pressure in Pa·m, viscous in N/m per unit depth).
    #
    # Obstacle face layout:
    #   Bml "+x" face  → fluid left of obstacle, normal +x  → +p * dx_phys
    #   Bmr "-x" face  → fluid right of obstacle, normal -x → -p * dx_phys
    #   Bbm "+y" face  → fluid below obstacle (viscous shear only)
    #   Btm "-y" face  → fluid above obstacle (viscous shear only)
    #
    # Scaling: PICT uses cell-unit coordinates (dx_pict=1).
    #   dx_phys = 1 / phys_scale  (phys_scale = N/L)
    #   p_phys  = p_pict / phys_scale^2
    #   drag_phys = sum(p_phys * dx_phys)
    #             = sum(p_pict) / phys_scale^3
    # The viscous contribution is small compared to pressure drag for the
    # Re values used in drag_opt (Re=20, 100) but is included for correctness.
    #   ux_phys = ux_pict / phys_scale
    #   nu_phys = viscosity_val / phys_scale  (viscosity_val already PICT-scaled)
    #   drag_visc = nu_phys * sum(ux_face_phys) / dx_phys  * dx_phys
    #             = (viscosity_val / phys_scale) * sum(ux_pict / phys_scale)
    #             = viscosity_val * sum(ux_pict) / phys_scale^2
    def _drag_assembler(viscosity_pict: float, ps: float) -> torch.Tensor:
        """Compute x-direction drag on the cylinder obstacle.

        Args:
            viscosity_pict: Kinematic viscosity in PICT units.
            ps: phys_scale = N/L.

        Returns:
            Shape (1,) float32 drag in physical units.
        """
        # Pressure fields (contiguous to allow backward through PISOtorch_diff)
        p_ml = Bml.pressure.contiguous()  # (1, 1, ny_mid, nx_left)
        p_mr = Bmr.pressure.contiguous()  # (1, 1, ny_mid, nx_right)

        # Unit analysis (PICT uses physical units for velocity and pressure):
        #   p_pict = p_phys  (pressure is the same in PICT and physical units)
        #   u_pict = u_phys  (velocity is the same in PICT and physical units)
        #   dy_phys = dy_pict / ps = 1 / ps  (PICT cells have unit cell spacing)
        #
        # Pressure drag (x-direction on left/right obstacle faces):
        #   F_pressure = integral(p_phys * n_x) dA = sum(p_left - p_right) * dy_phys
        #              = (p_left - p_right).sum() / ps
        p_left = p_ml[:, 0, :, -1]  # (1, ny_mid) — rightmost col of left block
        p_right = p_mr[:, 0, :, 0]  # (1, ny_mid) — leftmost col of right block
        drag_pressure = (p_left - p_right).sum() / ps

        # Viscous drag from horizontal faces (top/bottom of obstacle):
        #   tau_xy = nu_phys * du_x/dy_phys
        #          = nu_phys * (du_x_pict / dy_pict) * ps  [since dy_phys = dy_pict/ps]
        #          ≈ nu_phys * u_x_face * ps
        #
        #   F_visc = integral(tau_xy * n_y) dA = tau_xy * dx_phys = nu_phys * u_x_face * ps / ps
        #          = nu_phys * u_x_face  (per cell)
        #   Total: nu_phys * sum(u_x_face)  = (nu_pict / ps) * sum(u_x_face)
        #
        # Bottom face (obstacle below, n_y = +1): τ · n_y = +τ_xy at top row of Bbm
        # Top face (obstacle above, n_y = -1): τ · n_y = -τ_xy at bottom row of Btm
        v_bm = Bbm.velocity.contiguous()  # (1, 2, ny_bot, nx_mid)
        v_tm = Btm.velocity.contiguous()  # (1, 2, ny_top, nx_mid)
        ux_bm_top = v_bm[:, 0, -1, :]  # (1, nx_mid) — top row of Bbm
        ux_tm_bot = v_tm[:, 0, 0, :]  # (1, nx_mid) — bottom row of Btm
        nu_phys = viscosity_pict / ps
        drag_visc_bot = nu_phys * ux_bm_top.sum()
        drag_visc_top = -nu_phys * ux_tm_bot.sum()

        # Negate to match phiflow/xlb convention: drag is the force on the cylinder
        # in the upstream (-x) direction, so positive flow → negative drag value.
        drag = (
            -(drag_pressure + drag_visc_bot + drag_visc_top)
            .reshape(1)
            .to(torch.float32)
        )
        return drag

    return (
        domain,
        out_bounds,
        adv_velm,
        _set_v0,
        _assemble,
        _drag_assembler,
        cyl_inflow_setter,
    )


# ---------------------------------------------------------------------------
# Velocity conversion helpers
# ---------------------------------------------------------------------------


def _v0_to_pict(  # mosaic:io
    v0_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert canonical velocity array → PICT channels-first tensor.

    2-D: (nx, ny, 1, 2) → NCHW (1, 2, ny, nx)
    3-D: (nx, ny, nz, 3) → NCDHW (1, 3, nz, ny, nx)

    PICT stores velocity in channels-first (batch, comp, [depth,] height, width)
    with the convention H=y-axis, W=x-axis, D=z-axis.
    """
    ndim = v0_np.shape[-1]
    t = torch.tensor(v0_np, dtype=dtype, device=device)

    if ndim == 2:
        # (nx, ny, 1, 2) → (nx, ny, 2) → (2, nx, ny) → (2, ny, nx) → (1, 2, ny, nx)
        t = t.squeeze(2)  # (nx, ny, 2)
        t = t.permute(2, 0, 1)  # (2, nx, ny)
        t = t.permute(0, 2, 1)  # (2, ny, nx)  [swap x↔y for PICT H=y, W=x]
        t = t.unsqueeze(0)  # (1, 2, ny, nx)
    else:
        # (nx, ny, nz, 3) → (3, nz, ny, nx) → (1, 3, nz, ny, nx)
        # permute: (ix=0, iy=1, iz=2, comp=3) → (comp=3, iz=2, iy=1, ix=0)
        t = t.permute(3, 2, 1, 0)  # (3, nz, ny, nx)
        t = t.unsqueeze(0)  # (1, 3, nz, ny, nx)

    t = t.contiguous()
    if requires_grad:
        t.requires_grad_(True)
    return t


def _pict_to_v0(vel: torch.Tensor, N: int, ndim: int) -> torch.Tensor:  # mosaic:io
    """Convert PICT channels-first tensor → canonical velocity array.

    2-D: NCHW (1, 2, ny, nx) → (nx, ny, 1, 2)
    3-D: NCDHW (1, 3, nz, ny, nx) → (nx, ny, nz, 3)
    """
    if ndim == 2:
        t = vel.squeeze(0)  # (2, ny, nx)
        t = t.permute(0, 2, 1)  # (2, nx, ny)
        t = t.permute(1, 2, 0)  # (nx, ny, 2)
        t = t.unsqueeze(2)  # (nx, ny, 1, 2)
    else:
        t = vel.squeeze(0)  # (3, nz, ny, nx)
        t = t.permute(3, 2, 1, 0)  # (nx, ny, nz, 3)
    return t.contiguous()


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def _run_pict(  # mosaic:physics
    v0_tensor: torch.Tensor,
    viscosity_val: float,
    dt_val: float,
    steps: int,
    N: int,
    ndim: int,
    dtype: torch.dtype,
    differentiable: bool,
    lid_velocity_t: torch.Tensor | None = None,
    inflow_profile_t: torch.Tensor | None = None,
    phys_scale: float = 1.0,
    obstacle: dict | None = None,
    collect_velocity: bool = False,
    y_walls_noslip: bool = False,
) -> tuple[torch.Tensor, PISOtorch.Domain]:
    """Run PICT forward simulation.

    Args:
        v0_tensor: Velocity in PICT channels-first format, on GPU.
        viscosity_val: Kinematic viscosity ν (PICT units, already scaled).
        dt_val: Timestep size (PICT units, already scaled).
        steps: Number of timesteps.
        N: Grid resolution.
        ndim: Spatial dimensionality (2 or 3).
        dtype: Torch dtype.
        differentiable: Whether to keep the autograd graph.
        lid_velocity_t: Optional moving-lid velocity torch tensor in canonical
            layout ``(nx, ny, 2)``.  Live autograd leaf when set with
            ``requires_grad=True``; boundary VJPs flow through PISOtorch_diff.
        inflow_profile_t: Optional inflow x-velocity profile torch tensor of
            shape ``(ny,)``.  Live autograd leaf when set with
            ``requires_grad=True``; boundary VJPs flow through PISOtorch_diff.
        phys_scale: ``N/L`` scaling factor passed to ``_make_domain`` for
            converting physical boundary velocities to PICT units.
        obstacle: Optional obstacle dict (keys ``"shape"``, ``"center"``,
            ``"radius"``).  When provided, triggers 8-block ring topology.

    Returns:
        ``(result_tensor, drag_tensor, domain, velocity_mean_tensor)`` where
        result_tensor is the final velocity in PICT channels-first format,
        drag_tensor is the x-direction drag on the obstacle in physical units
        (shape (1,), or None when no obstacle is present), domain is the live
        domain, and velocity_mean_tensor is the time-averaged velocity over the
        tail window (or None when ``collect_velocity=False``).
    """
    (
        domain,
        out_bounds,
        adv_velm,
        v0_setter,
        assembler,
        drag_assembler,
        inflow_setter,
    ) = _make_domain(
        viscosity_val,
        N,
        ndim,
        dtype=dtype,
        lid_velocity_t=lid_velocity_t,
        inflow_profile_t=inflow_profile_t,
        phys_scale=phys_scale,
        obstacle=obstacle,
        in_vel=float(v0_tensor[:, 0].mean()),
        y_walls_noslip=y_walls_noslip,
    )
    domain.PrepareSolve()  # allocate internal structures first
    v0_setter(v0_tensor)
    domain.UpdateDomainData()

    # Wire PRE callback for inflow/channel mode.
    #
    # Two things happen before each PISO substep:
    #
    # 1. **Advective outflow update** (``out_bounds``): extrapolates the outflow
    #    BC from the current interior velocity using the 1st-order advective
    #    scheme.  NOTE: PISOtorch_simulation passes ``time_step`` as a CPU tensor;
    #    the advective boundary kernel operates on CUDA, so we move it here to
    #    avoid a device-mismatch error (see vortex_street_sample.py in PICT).
    #
    # 2. **Inflow BC re-application** (``inflow_setter``): re-calls
    #    ``getBoundary("-x").setVelocity(inflow_bc)`` with the live
    #    ``inflow_profile_t`` tensor each step.  PISOtorch_diff builds its
    #    autograd graph during ``sim.run()``; if the inflow BC is set only once
    #    before ``PrepareSolve()`` (outside the loop), ``inflow_profile_t`` is
    #    never seen by the differentiable subgraph and
    #    ``torch.autograd.grad(..., inflow_profile_t)`` returns None (→ zero).
    #    Re-applying it each step keeps ``inflow_profile_t`` in the graph and
    #    makes the VJP non-zero.
    #
    # 3. **Tail-window drag accumulation** (``drag_assembler``): a POST
    #    callback collects drag after every completed PISO step.  The mean over
    #    the last ``steps // 2`` steps is returned as the drag output, matching
    #    the semantics of xlb and phiflow (which use jax.lax.scan +
    #    jnp.mean(drag_history[-n_tail:])).  Approach: POST callback chosen over
    #    a step loop because it preserves the single sim.run(steps) call and
    #    keeps the full PISOtorch_diff autograd graph intact.  The mean is taken
    #    outside sim.run after collecting all per-step tensors; PyTorch autodiff
    #    through torch.stack + torch.mean is well-defined and chains correctly
    #    with the PISOtorch_diff backward pass.
    drag_history: list[torch.Tensor] = []
    velocity_history: list[torch.Tensor] = []

    prep_fn = None
    if (
        out_bounds
        or inflow_setter is not None
        or drag_assembler is not None
        or collect_velocity
    ):

        def _pre_step(domain, time_step, **_kw):
            # Re-apply inflow BC first (keep inflow_profile_t in autograd graph).
            if inflow_setter is not None:
                inflow_setter()
            # Then update advective outflow boundaries.
            if out_bounds:
                ts = time_step
                if isinstance(ts, torch.Tensor) and ts.device != _DEVICE:
                    ts = ts.to(_DEVICE)
                PISOtorch_simulation.update_advective_boundaries(
                    domain, out_bounds, adv_velm, ts
                )

        def _post_step(domain, time_step, **_kw):  # noqa: ARG001
            # Collect per-step drag after the PISO corrector has converged.
            # drag_assembler reads pressure/velocity from block boundaries which
            # are already updated at this point.
            if drag_assembler is not None:
                drag_history.append(drag_assembler(viscosity_val, phys_scale))
            if collect_velocity:
                velocity_history.append(assembler(domain).detach().clone())

        prep_fn = {"PRE": _pre_step, "POST": _post_step}

    sim = PISOtorch_simulation.Simulation(
        domain,
        substeps=1,
        time_step=float(dt_val),
        corrector_steps=4,
        non_orthogonal=False,
        differentiable=differentiable,
        log_dir=None,
        log_interval=0,
        log_images=False,
        prep_fn=prep_fn,
    )
    sim.run(steps, log_domain=False)

    result = assembler(domain)
    if drag_assembler is not None:
        if drag_history:
            # Tail-window mean: average drag over the last steps // 2 steps,
            # matching xlb/phiflow semantics (phase-independent for periodic
            # vortex-shedding flows).
            n_tail = max(1, steps // 2)
            drag = torch.stack(drag_history[-n_tail:]).mean(dim=0).to(torch.float32)
        else:
            # POST callback not fired (e.g. steps=0); fall back to final-step drag.
            drag = drag_assembler(viscosity_val, phys_scale)
    else:
        drag = None
    velocity_mean_t = None
    if collect_velocity and velocity_history:
        n_tail = max(1, steps // 2)
        velocity_mean_t = (
            torch.stack(velocity_history[-n_tail:]).mean(dim=0).to(torch.float32)
        )
    return result, drag, domain, velocity_mean_t


# ---------------------------------------------------------------------------
# Tesseract API
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: run PICT simulation and return final velocity field."""
    v0_np = np.asarray(inputs.v0)
    nu_np = np.asarray(inputs.viscosity)
    dt_np = np.asarray(inputs.dt)
    N = v0_np.shape[0]
    ndim = v0_np.shape[-1]

    # PICT uses unit cell spacing (dx=1), but the physical domain has dx = L/N
    # where L = domain_extent.  To reproduce the canonical NS equation in physical
    # coordinates, both dt and ν must be scaled by N/L.
    #
    # Derivation: let x_pict = x_phys * N/L (cell-index coordinates).
    # Then grad_phys = grad_pict * N/L and laplacian_phys = laplacian_pict * (N/L)².
    # Substituting into physical NS and dividing by N/L yields:
    #   du/d(t_phys · N/L) = -(u·∇_pict)u − ∇_pict p + (ν · N/L)·∇²_pict u
    # So the PICT-internal time step is dt_pict = dt_phys · N/L and the
    # PICT-internal viscosity is nu_pict = nu_phys · N/L.
    L = float(inputs.domain_extent)
    phys_scale = N / L  # = N / (2π) ≈ N / 6.28
    dt_pict = float(dt_np[0]) * phys_scale
    nu_pict = float(nu_np[0]) * phys_scale

    # Boundary conditions (optional).  Build live torch tensors (no grad in
    # apply) so the same helper functions work in both apply and VJP paths.
    dtype = torch.float32
    lid_velocity_t = (
        torch.tensor(
            np.ascontiguousarray(inputs.lid_velocity, dtype=np.float32),
            dtype=dtype,
            device=_DEVICE,
        )
        if inputs.lid_velocity is not None
        else None
    )
    inflow_profile_t = (
        torch.tensor(
            np.ascontiguousarray(inputs.inflow_profile, dtype=np.float32),
            dtype=dtype,
            device=_DEVICE,
        )
        if inputs.inflow_profile is not None
        else None
    )
    obstacle = inputs.obstacle.model_dump() if inputs.obstacle is not None else None
    bc = inputs.boundary_conditions
    y_walls_noslip = bc.y_lo.type == BCType.NO_SLIP and bc.y_hi.type == BCType.NO_SLIP

    v0_t = _v0_to_pict(v0_np, _DEVICE, dtype, requires_grad=False)

    result_t, drag_t, domain, velocity_mean_t = _run_pict(
        v0_tensor=v0_t,
        viscosity_val=nu_pict,
        dt_val=dt_pict,
        steps=inputs.steps,
        N=N,
        ndim=ndim,
        dtype=dtype,
        differentiable=False,
        lid_velocity_t=lid_velocity_t,
        inflow_profile_t=inflow_profile_t,
        phys_scale=phys_scale,
        obstacle=obstacle,
        collect_velocity=True,
        y_walls_noslip=y_walls_noslip,
    )

    out_np = _pict_to_v0(result_t, N, ndim).detach().cpu().numpy()
    drag_np = (
        drag_t.detach().cpu().numpy()
        if drag_t is not None
        else np.zeros((1,), dtype=np.float32)
    )
    _velocity_mean_np = (
        _pict_to_v0(velocity_mean_t, N, ndim).detach().cpu().numpy()
        if velocity_mean_t is not None
        else None
    )
    return {"result": out_np, "drag": drag_np}


def vector_jacobian_product(  # mosaic:grad:v0,viscosity,dt,lid_velocity,inflow_profile
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via torch.autograd through the differentiable PICT PISO time loop.

    Supports gradients w.r.t. ``v0``, ``lid_velocity``, and ``inflow_profile``
    through the PISO time loop via PISOtorch_diff autograd.  Gradients w.r.t.
    ``viscosity`` and ``dt`` are returned as zeros (scalar params).

    ``drag`` output is differentiable when an obstacle is present: drag is
    computed from the pressure/velocity surface integral on the obstacle faces
    and the VJP flows back through the same autograd graph.
    """
    import sys as _sys

    print(
        f"[pict VJP] vjp_inputs={sorted(vjp_inputs)} vjp_outputs={sorted(vjp_outputs)} "
        f"ct_keys={sorted(cotangent_vector.keys())}",
        file=_sys.stderr,
        flush=True,
    )

    want_result = "result" in vjp_outputs and "result" in cotangent_vector
    want_drag = "drag" in vjp_outputs and "drag" in cotangent_vector

    # If neither result nor drag is requested, there is nothing to differentiate.
    if not want_result and not want_drag:
        return {
            k: np.zeros_like(np.asarray(getattr(inputs, k)))
            for k in vjp_inputs
            if getattr(inputs, k, None) is not None
        }

    v0_np = np.asarray(inputs.v0)
    nu_np = np.asarray(inputs.viscosity)
    dt_np = np.asarray(inputs.dt)
    N = v0_np.shape[0]
    ndim = v0_np.shape[-1]
    dtype = torch.float32

    # Apply the same dt and nu scaling as in apply():
    #   dt_pict = dt_physical * N/L,  nu_pict = nu_physical * N/L
    L = float(inputs.domain_extent)
    phys_scale = N / L
    dt_pict_val = float(dt_np[0]) * phys_scale
    nu_pict_val = float(nu_np[0]) * phys_scale

    # Boundary conditions (optional).  Build live torch tensors; mark as
    # autograd leaves (requires_grad=True) for any input we need a VJP for so
    # PISOtorch_diff captures the BOUNDARY_VELOCITY gradient chain.
    lid_velocity_np = (
        np.asarray(inputs.lid_velocity) if inputs.lid_velocity is not None else None
    )
    inflow_profile_np = (
        np.asarray(inputs.inflow_profile) if inputs.inflow_profile is not None else None
    )
    obstacle = inputs.obstacle.model_dump() if inputs.obstacle is not None else None
    bc = inputs.boundary_conditions
    y_walls_noslip = bc.y_lo.type == BCType.NO_SLIP and bc.y_hi.type == BCType.NO_SLIP

    want_v0 = "v0" in vjp_inputs
    want_lid = "lid_velocity" in vjp_inputs and lid_velocity_np is not None
    want_inflow = "inflow_profile" in vjp_inputs and inflow_profile_np is not None

    # PICT's PISOtorch_diff backend tracks VELOCITY, BOUNDARY_VELOCITY, and
    # VISCOSITY gradients through the PISO time loop.  viscosity and dt are
    # still passed as Python floats so their grads remain zero, but v0,
    # lid_velocity, and inflow_profile all flow through torch.autograd when
    # passed as live tensors with requires_grad=True.
    v0_t = _v0_to_pict(v0_np, _DEVICE, dtype, requires_grad=want_v0)

    def _to_leaf(arr: np.ndarray, grad: bool) -> torch.Tensor:
        # Build a contiguous leaf tensor on device.  torch.tensor copies the
        # numpy buffer and returns a fresh leaf; numpy arrays are C-contiguous
        # by default, so the result is already contiguous.
        t = torch.tensor(
            np.ascontiguousarray(arr, dtype=np.float32), dtype=dtype, device=_DEVICE
        )
        if grad:
            t.requires_grad_(True)
        return t

    lid_velocity_t = (
        _to_leaf(lid_velocity_np, want_lid) if lid_velocity_np is not None else None
    )
    inflow_profile_t = (
        _to_leaf(inflow_profile_np, want_inflow)
        if inflow_profile_np is not None
        else None
    )

    result_t, drag_t, opt_domain, _ = _run_pict(
        v0_tensor=v0_t,
        viscosity_val=nu_pict_val,
        dt_val=dt_pict_val,
        steps=inputs.steps,
        N=N,
        ndim=ndim,
        dtype=dtype,
        differentiable=True,
        lid_velocity_t=lid_velocity_t,
        inflow_profile_t=inflow_profile_t,
        phys_scale=phys_scale,
        obstacle=obstacle,
        y_walls_noslip=y_walls_noslip,
    )

    out = {}

    # Build the list of scalar outputs to differentiate through.
    # When drag is requested, include it in the outputs list so the single
    # backward pass covers both result and drag VJPs simultaneously.
    outputs_for_grad: list[torch.Tensor] = []
    grad_outputs: list[torch.Tensor] = []

    if want_result:
        ct_np = np.asarray(cotangent_vector["result"])
        # Convert cotangent to PICT layout to avoid non-contiguous tensor in backward.
        ct_t_pict = _v0_to_pict(ct_np, _DEVICE, dtype, requires_grad=False)
        outputs_for_grad.append(result_t)
        grad_outputs.append(ct_t_pict)

    if want_drag and drag_t is not None:
        ct_drag_np = np.asarray(cotangent_vector["drag"])
        ct_drag_t = torch.tensor(
            np.ascontiguousarray(ct_drag_np, dtype=np.float32),
            dtype=dtype,
            device=_DEVICE,
        )
        outputs_for_grad.append(drag_t)
        grad_outputs.append(ct_drag_t)

    # Collect all leaf tensors we want gradients for in one backward pass so
    # the autograd graph is only traversed once.
    # mosaic:grad:v0,lid_velocity,inflow_profile:autodiff
    grad_targets: list[torch.Tensor] = []
    grad_keys: list[str] = []
    if want_v0:
        grad_targets.append(v0_t)
        grad_keys.append("v0")
    if want_lid:
        grad_targets.append(lid_velocity_t)
        grad_keys.append("lid_velocity")
    if want_inflow:
        grad_targets.append(inflow_profile_t)
        grad_keys.append("inflow_profile")

    if grad_targets and outputs_for_grad:
        grads = torch.autograd.grad(
            outputs=outputs_for_grad,
            inputs=grad_targets,
            grad_outputs=grad_outputs,
            allow_unused=True,
        )
        for key, g in zip(grad_keys, grads):
            if key == "v0":
                out[key] = (
                    _pict_to_v0(g, N, ndim).cpu().numpy()
                    if g is not None
                    else np.zeros_like(v0_np)
                )
            elif key == "lid_velocity":
                out[key] = (
                    g.detach().cpu().numpy()
                    if g is not None
                    else np.zeros_like(lid_velocity_np)
                )
            elif key == "inflow_profile":
                out[key] = (
                    g.detach().cpu().numpy()
                    if g is not None
                    else np.zeros_like(inflow_profile_np)
                )

    # mosaic:grad:viscosity:zero
    if "viscosity" in vjp_inputs:
        out["viscosity"] = np.zeros_like(nu_np)
    # mosaic:grad:dt:zero
    if "dt" in vjp_inputs:
        out["dt"] = np.zeros_like(dt_np)
    # Requested gradients for BC inputs that weren't differentiated (e.g. no
    # autograd leaf available) come back as zeros to keep the VJP shape valid.
    # mosaic:grad:lid_velocity,inflow_profile:autodiff
    if (
        "lid_velocity" in vjp_inputs
        and "lid_velocity" not in out
        and lid_velocity_np is not None
    ):
        out["lid_velocity"] = np.zeros_like(lid_velocity_np)
    if (
        "inflow_profile" in vjp_inputs
        and "inflow_profile" not in out
        and inflow_profile_np is not None
    ):
        out["inflow_profile"] = np.zeros_like(inflow_profile_np)

    opt_domain.Detach()
    return out


def abstract_eval(abstract_inputs: InputSchema):
    v0_info = abstract_inputs.v0
    if isinstance(v0_info, dict):
        shape = v0_info["shape"]
    else:
        shape = v0_info.shape
    return {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
