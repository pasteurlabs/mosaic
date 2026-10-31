# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import math
from typing import Any

import numpy as np
import PISOtorch
import PISOtorch_diff
import PISOtorch_simulation
import torch
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import (
    check_periodic_or_channel,
    solid_cells,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator
from scipy.spatial import Delaunay


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["v0", "viscosity", "inflow_profile"])
):
    """Differentiable input schema for the PICT Navier-Stokes solver."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        shape = tuple(self.v0.shape)
        if self.boundary_conditions.is_fully_periodic and self.obstacle is None:
            if len(set(shape[: shape[-1]])) != 1:
                raise ValueError(
                    "pict supports periodic domains with the same number of "
                    f"cells along every axis; got v0 of shape {shape}"
                )
            return self
        check_periodic_or_channel(self, "pict")
        return self


class OutputSchema(
    make_differentiable(_CanonicalOutputSchema, ["result", "body_force"])
):
    """Differentiable output schema for the PICT Navier-Stokes solver."""


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

# PICT is GPU-only: the PISOtorch C++ extension requires CUDA.
# If no GPU is available the tesseract will raise at domain-creation time
# (matching PICT's own assertion).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Periodic domain
# ---------------------------------------------------------------------------


def _make_periodic_domain(
    viscosity_val: torch.Tensor, N: int, ndim: int, dtype: torch.dtype
) -> PISOtorch.Domain:
    """Create a single all-periodic block of N×N (×N in 3-D) unit cells."""
    domain = PISOtorch.Domain(
        ndim,
        viscosity_val,
        name="PICTDomain",
        dtype=dtype,
        device=_DEVICE,
        passiveScalarChannels=0,
    )
    if ndim == 3:
        block_size = PISOtorch.Int4(x=N, y=N, z=N)
    else:
        block_size = PISOtorch.Int4(x=N, y=N)
    domain.CreateBlockWithSize(block_size, name="Block")
    return domain


# ---------------------------------------------------------------------------
# Velocity conversion helpers
# ---------------------------------------------------------------------------


def _v0_to_pict(
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


def _pict_to_v0(vel: torch.Tensor, N: int, ndim: int) -> torch.Tensor:
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
# Channel around a cylinder: body-fitted multi-block mesh
# ---------------------------------------------------------------------------
#
# The channel is split at the canonical grid lines i0, i1, j0, j1 that bound a
# square box of side 2D around the cylinder:
#
#     TL | TM | TR       The eight blocks outside the box are Cartesian and
#     ML | O  | MR       coincide with the canonical cells. Inside the box,
#     BL | BM | BR       four O-grid blocks (bottom, right, top, left) run from
#                        the box sides to the circle along straight spokes
#                        through the box corners.
#
# Every block is right-handed. PICT's linear solvers stop at an absolute
# residual, so the blocks are built in units of the canonical cell size dx,
# like the periodic box. dt and ν are divided by dx, and velocity and
# pressure are unchanged. Everything outside the blocks stays physical.


def _o_block_vertices(
    side: str,
    box: tuple[float, float, float, float],
    center: tuple[float, float],
    radius: float,
    n_side: int,
    n_radial: int,
) -> np.ndarray:
    """Vertex coordinates (2, nj+1, ni+1) of the O-grid block on one box side.

    The tangential axis runs along +x (bottom, top) or +y (left, right). The
    radial axis runs from the box side towards the circle for the bottom and
    left blocks and away from it for the top and right ones, which keeps every
    block right-handed.
    """
    x0, x1, y0, y1 = box
    cx, cy = center
    (ax, ay), (bx, by) = {
        "bottom": ((x0, y0), (x1, y0)),
        "top": ((x0, y1), (x1, y1)),
        "left": ((x0, y0), (x0, y1)),
        "right": ((x1, y0), (x1, y1)),
    }[side]
    t = np.linspace(0.0, 1.0, n_side + 1)
    on_side = np.stack([ax + t * (bx - ax), ay + t * (by - ay)])
    th_a = math.atan2(ay - cy, ax - cx)
    th_b = math.atan2(by - cy, bx - cx)
    th = th_a + t * ((th_b - th_a + math.pi) % (2 * math.pi) - math.pi)
    on_circle = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)])
    s = np.linspace(0.0, 1.0, n_radial + 1)
    # (2, tangential, radial), radial index 0 on the circle.
    pts = on_circle[:, :, None] + s * (on_side - on_circle)[:, :, None]
    if side == "bottom":
        return np.transpose(pts[:, :, ::-1], (0, 2, 1))
    if side == "top":
        return np.transpose(pts, (0, 2, 1))
    if side == "left":
        return pts[:, :, ::-1]
    return pts


def _cell_centers(vertices: np.ndarray) -> np.ndarray:
    """Cell centers (2, nj, ni) as the mean of each cell's four vertices."""
    v = vertices
    return 0.25 * (v[:, :-1, :-1] + v[:, 1:, :-1] + v[:, :-1, 1:] + v[:, 1:, 1:])


def _linear_weights(
    points: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and weights (m, 3) of linear interpolation on a Delaunay triangulation."""
    tri = Delaunay(points)
    simplex = tri.find_simplex(targets)
    if np.any(simplex < 0):
        raise RuntimeError("canonical cell center outside the PICT mesh")
    trans = tri.transform[simplex]
    bary = np.einsum("mij,mj->mi", trans[:, :2], targets - trans[:, 2])
    weights = np.concatenate([bary, 1.0 - bary.sum(axis=1, keepdims=True)], axis=1)
    return tri.simplices[simplex], weights


# O-grid block -> (axis of the wall-normal index, wall-adjacent cell index, next
# cell index). Axis 0 indexes rows (y), 1 columns (x). The wall is the block's
# +y, -y, +x and -x face, respectively.
_WALL_CELLS = {
    "bottom": (0, -1, -2),
    "top": (0, 0, 1),
    "left": (1, -1, -2),
    "right": (1, 0, 1),
}


class _Channel:
    """The DFG channel on PICT blocks, with transfers to and from the canonical grid.

    Block cell values are gathered into one flat vector per field (block by
    block, each row-major as PICT stores it, followed by one point per
    cylinder wall face). Every canonical cell center is a fixed linear
    combination of at most three entries of that vector: its own cell in the
    Cartesian blocks, a piecewise-linear interpolant in the box around the
    cylinder, and nothing inside the cylinder.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        center: tuple[float, float],
        radius: float,
        viscosity_val: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        self.nx, self.ny, self.dx = nx, ny, dx
        self.center, self.radius = center, radius
        self.viscosity = viscosity_val
        self.dtype = dtype
        i0 = round((center[0] - 2 * radius) / dx)
        i1 = round((center[0] + 2 * radius) / dx)
        j0 = round((center[1] - 2 * radius) / dx)
        j1 = round((center[1] + 2 * radius) / dx)
        if not (3 <= i0 and i1 <= nx - 3 and 3 <= j0 and j1 <= ny - 3):
            raise NotImplementedError(
                "pict needs at least 3 cells between the box of side 2D around "
                "the cylinder and the channel boundary"
            )
        if not (
            i0 * dx < center[0] - radius < center[0] + radius < i1 * dx
            and j0 * dx < center[1] - radius < center[1] + radius < j1 * dx
        ):
            raise NotImplementedError("the cylinder must lie inside its O-grid box")
        self.xs, self.ys = (0, i0, i1, nx), (0, j0, j1, ny)
        box = (i0 * dx, i1 * dx, j0 * dx, j1 * dx)
        # Two radial cells per dx between the circle and the box side.
        n_radial = max(3, round(2 * radius / dx))

        # name -> (vertices, canonical (i, j) slices, or None for O-grid blocks)
        layout: dict[str, tuple[np.ndarray, tuple[slice, slice] | None]] = {}
        for a, xname in enumerate("LMR"):
            for b, yname in enumerate("BMT"):
                if (a, b) == (1, 1):
                    continue
                xe = np.arange(self.xs[a], self.xs[a + 1] + 1) * dx
                ye = np.arange(self.ys[b], self.ys[b + 1] + 1) * dx
                gx, gy = np.meshgrid(xe, ye, indexing="xy")
                cells = (
                    slice(self.xs[a], self.xs[a + 1]),
                    slice(self.ys[b], self.ys[b + 1]),
                )
                layout[yname + xname] = (np.stack([gx, gy]), cells)
        for side, n_side in (
            ("bottom", i1 - i0),
            ("right", j1 - j0),
            ("top", i1 - i0),
            ("left", j1 - j0),
        ):
            verts = _o_block_vertices(side, box, center, radius, n_side, n_radial)
            layout[side] = (np.ascontiguousarray(verts), None)

        self.domain = PISOtorch.Domain(
            2,
            viscosity_val / dx,
            name="ChannelDomain",
            dtype=dtype,
            device=_DEVICE,
            passiveScalarChannels=0,
        )
        self.blocks: dict[str, Any] = {}
        self.vertices: dict[str, np.ndarray] = {}
        self.cells: dict[str, tuple[slice, slice] | None] = {}
        for name, (verts, cells) in layout.items():
            coords = torch.tensor(verts[None] / dx, dtype=dtype, device=_DEVICE)
            block = self.domain.CreateBlock(
                vertexCoordinates=coords.contiguous(), name=name
            )
            # Close every face first. The connections below replace the
            # interior ones, and the rest stay no-slip walls unless set up as
            # inlet or outlet.
            block.CloseAllBoundaries()
            self.blocks[name] = block
            self.vertices[name] = verts
            self.cells[name] = cells
        B = self.blocks
        for lo, hi in (("BL", "BM"), ("BM", "BR"), ("TL", "TM"), ("TM", "TR")):
            B[lo].ConnectBlock("+x", B[hi], "-x", "-y")
        for lo, hi in (("BL", "ML"), ("ML", "TL"), ("BR", "MR"), ("MR", "TR")):
            B[lo].ConnectBlock("+y", B[hi], "-y", "-x")
        B["ML"].ConnectBlock("+x", B["left"], "-x", "-y")
        B["right"].ConnectBlock("+x", B["MR"], "-x", "-y")
        B["BM"].ConnectBlock("+y", B["bottom"], "-y", "-x")
        B["top"].ConnectBlock("+y", B["TM"], "-y", "-x")
        # Spokes through the box corners. Their "+" axes run in opposite directions.
        B["left"].ConnectBlock("-y", B["bottom"], "-x", "-y")
        B["left"].ConnectBlock("+y", B["top"], "-x", "+y")
        B["right"].ConnectBlock("-y", B["bottom"], "+x", "+y")
        B["right"].ConnectBlock("+y", B["top"], "+x", "-y")

        self.inlets = [
            (B[n].getBoundary("-x"), self.cells[n][1]) for n in ("BL", "ML", "TL")
        ]
        self.outlets = [B[n].getBoundary("+x") for n in ("BR", "MR", "TR")]
        self._build_wall()
        self._build_transfers()

    # mosaic-intervention: body-force-integration
    # -- cylinder wall ------------------------------------------------------

    def _build_wall(self) -> None:
        """Length, fluid-side normal and first two cells' wall distances of each arc face."""
        cx, cy = self.center
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("len", "n", "d1", "d2")}
        for name, (axis, first, second) in _WALL_CELLS.items():
            verts = self.vertices[name]
            centers = _cell_centers(verts)
            if axis == 0:
                wall = verts[:, 0 if first == 0 else -1, :]
                c1, c2 = centers[:, first, :], centers[:, second, :]
            else:
                wall = verts[:, :, 0 if first == 0 else -1]
                c1, c2 = centers[:, :, first], centers[:, :, second]
            th = np.arctan2(wall[1] - cy, wall[0] - cx)
            dth = np.diff(np.unwrap(th))
            mid = th[:-1] + 0.5 * dth
            n = np.stack([np.cos(mid), np.sin(mid)])
            parts["len"].append(self.radius * np.abs(dth))
            parts["n"].append(n)
            for key, c in (("d1", c1), ("d2", c2)):
                parts[key].append((c[0] - cx) * n[0] + (c[1] - cy) * n[1] - self.radius)
        cat = {k: np.concatenate(v, axis=-1) for k, v in parts.items()}
        self.wall_xy = np.array(self.center)[:, None] + self.radius * cat["n"]
        self.wall_len, self.wall_n, self.wall_d1, self.wall_d2 = (
            torch.tensor(cat[k], dtype=self.dtype, device=_DEVICE)
            for k in ("len", "n", "d1", "d2")
        )

    def _wall_cells(self, field: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Values (C, faces) of ``field`` in the first and second cell off the wall."""
        first, second = [], []
        for name, (axis, i_first, i_second) in _WALL_CELLS.items():
            f = getattr(self.blocks[name], field)[0]
            first.append(f[:, i_first, :] if axis == 0 else f[:, :, i_first])
            second.append(f[:, i_second, :] if axis == 0 else f[:, :, i_second])
        return torch.cat(first, dim=-1), torch.cat(second, dim=-1)

    def wall_pressure(self) -> torch.Tensor:
        """Pressure on each wall face, extrapolated linearly along the wall normal."""
        p1, p2 = self._wall_cells("pressure")
        d1, d2 = self.wall_d1, self.wall_d2
        return (p1 - d1 * (p2 - p1) / (d2 - d1))[0]

    def body_force(self) -> torch.Tensor:
        """Force (F_x, F_y) of the fluid on the cylinder, pressure plus viscous.

        Integrates the traction −p n + ν (∂u_t/∂n) t over the arc faces, with
        n the wall normal into the fluid and t the tangent. The wall pressure
        and the shear rate are second-order one-sided fits to the first two
        cells off the wall.
        """
        u1, u2 = self._wall_cells("velocity")
        n = self.wall_n
        tangent = torch.stack([-n[1], n[0]])
        ut1, ut2 = (u1 * tangent).sum(0), (u2 * tangent).sum(0)
        d1, d2 = self.wall_d1, self.wall_d2
        shear = (ut1 * d2**2 - ut2 * d1**2) / (d1 * d2 * (d2 - d1))
        nu = self.viscosity.to(device=_DEVICE, dtype=self.dtype)
        traction = -self.wall_pressure() * n + nu * shear * tangent
        return (traction * self.wall_len).sum(dim=1)

    # mosaic-intervention-end: body-force-integration
    # -- transfers between the blocks and the canonical grid ----------------

    def _build_transfers(self) -> None:
        dx, nx, ny = self.dx, self.nx, self.ny
        i0, i1, j0, j1 = self.xs[1], self.xs[2], self.ys[1], self.ys[2]
        offsets: dict[str, int] = {}
        centers = []
        offset = 0
        for name, verts in self.vertices.items():
            c = _cell_centers(verts).reshape(2, -1)
            offsets[name] = offset
            offset += c.shape[1]
            centers.append(c)
        points = np.concatenate([*centers, self.wall_xy], axis=1).T

        # Canonical cell (i, j), flattened as i * ny + j, from three sources.
        idx = np.zeros((nx * ny, 3), dtype=np.int64)
        w = np.zeros((nx * ny, 3))
        for name, cells in self.cells.items():
            if cells is None:
                continue
            ii, jj = np.meshgrid(
                np.arange(cells[0].start, cells[0].stop),
                np.arange(cells[1].start, cells[1].stop),
                indexing="xy",
            )  # (nj, ni), PICT's row-major order
            dst = (ii * ny + jj).ravel()
            idx[dst, 0] = offsets[name] + np.arange(ii.size)
            w[dst, 0] = 1.0
        X, Y = np.meshgrid(
            (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx, indexing="ij"
        )
        solid = solid_cells(nx, ny, dx, self.center, self.radius)
        in_box = np.zeros((nx, ny), dtype=bool)
        in_box[i0:i1, j0:j1] = True
        targets = in_box & ~solid
        # Interpolate from the cells within two cell sizes of the box (the
        # O-grid and a ring of Cartesian cells) and the wall points.
        sources = np.flatnonzero(
            (points[:, 0] > (i0 - 2) * dx)
            & (points[:, 0] < (i1 + 2) * dx)
            & (points[:, 1] > (j0 - 2) * dx)
            & (points[:, 1] < (j1 + 2) * dx)
        )
        tri_idx, tri_w = _linear_weights(
            points[sources], np.stack([X[targets], Y[targets]], axis=1)
        )
        dst = np.flatnonzero(targets.ravel())
        idx[dst] = sources[tri_idx]
        w[dst] = tri_w
        idx[solid.ravel()] = 0
        w[solid.ravel()] = 0.0
        self.to_canonical = (
            torch.tensor(idx, device=_DEVICE),
            torch.tensor(w, dtype=self.dtype, device=_DEVICE),
        )
        self.solid = torch.tensor(solid, device=_DEVICE)

        # O-grid cells take the canonical field bilinearly interpolated at
        # their centers.
        self.from_canonical = {}
        for name, cells in self.cells.items():
            if cells is not None:
                continue
            c = _cell_centers(self.vertices[name]).reshape(2, -1)
            fx, fy = c[0] / dx - 0.5, c[1] / dx - 0.5
            ia, ja = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
            ax, ay = fx - ia, fy - ja
            corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
            ids = np.stack([(ia + di) * ny + (ja + dj) for di, dj in corners], 1)
            ws = np.stack(
                [(ax if di else 1 - ax) * (ay if dj else 1 - ay) for di, dj in corners],
                1,
            )
            self.from_canonical[name] = (
                torch.tensor(ids, device=_DEVICE),
                torch.tensor(ws, dtype=self.dtype, device=_DEVICE),
            )

    def set_velocity(self, v0: torch.Tensor) -> None:
        """Set the block velocities from a canonical (nx, ny, 1, 2) field."""
        flat = v0[:, :, 0, :].reshape(-1, 2)
        for name, block in self.blocks.items():
            cells = self.cells[name]
            if cells is not None:
                vel = v0[cells[0], cells[1], 0, :].permute(2, 1, 0)
            else:
                ids, ws = self.from_canonical[name]
                nj, ni = (s - 1 for s in self.vertices[name].shape[1:])
                vel = (flat[ids] * ws[..., None]).sum(1).T.reshape(2, nj, ni)
            block.setVelocity(vel[None].contiguous())

    def set_inflow(self, profile: torch.Tensor) -> None:
        """Impose u_x = profile(y), u_y = 0 on the inlet faces."""
        for bound, rows in self.inlets:
            ux = profile[rows]
            face = torch.stack([ux, torch.zeros_like(ux)])[None, :, :, None]
            bound.setVelocity(face.contiguous())

    def init_outflow(self, profile: torch.Tensor) -> None:
        """Start the outlet faces from the inlet profile, as advected outflow."""
        for bound, (_, rows) in zip(self.outlets, self.inlets, strict=True):
            ux = profile[rows].detach()
            face = torch.stack([ux, torch.zeros_like(ux)])[None, :, :, None]
            bound.setVelocity(face.contiguous())
            bound.makeVelocityVarying()
        PISOtorch_simulation.balance_boundary_fluxes(self.domain, self.outlets)

    def _to_canonical(self, field: str, wall: torch.Tensor) -> torch.Tensor:
        """Canonical (nx, ny, C) from a block field and its wall values (C, faces)."""
        flat = torch.cat(
            [getattr(b, field)[0].flatten(1) for b in self.blocks.values()] + [wall],
            dim=1,
        )
        ids, ws = self.to_canonical
        return (flat[:, ids] * ws).sum(-1).T.reshape(self.nx, self.ny, -1)

    def velocity(self) -> torch.Tensor:
        """Velocity (nx, ny, 1, 2) at the canonical cell centers, zero in the cylinder."""
        wall = torch.zeros(2, self.wall_len.numel(), dtype=self.dtype, device=_DEVICE)
        return self._to_canonical("velocity", wall)[:, :, None, :]

    def pressure(self) -> torch.Tensor:
        """Kinematic pressure (nx, ny, 1), zero on the outlet face and in the cylinder.

        The outlet value is the mean over the face of the last two cell
        columns extrapolated linearly onto it.
        """
        p_out = torch.cat(
            [
                1.5 * self.blocks[n].pressure[0, 0, :, -1]
                - 0.5 * self.blocks[n].pressure[0, 0, :, -2]
                for n in ("BR", "MR", "TR")
            ]
        ).mean()
        p = self._to_canonical("pressure", self.wall_pressure()[None]) - p_out
        return torch.where(self.solid[:, :, None], torch.zeros_like(p), p)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class _LogRecordCollector(logging.Handler):
    """Logging handler that keeps the exceptions attached to log records."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.exceptions: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and record.exc_info[1] is not None:
            self.exceptions.append(record.exc_info[1])


# mosaic-intervention: warm-start-pressure
def _warm_cg(
    matrix: PISOtorch.CSRmatrix,
    rhs: torch.Tensor,
    guess: torch.Tensor,
    tol: float,
    max_iter: int,
    adjoint_guess: list[torch.Tensor | None],
) -> torch.Tensor:
    """Differentiable CG solve of ``matrix @ x = rhs`` that starts from ``guess``.

    Mirrors the CG branch of PISOtorch_diff.linear_solve_GPU, which always
    starts from zero. The gradient is the same, since the solution does not
    depend on the starting point. The adjoint solve starts from the mean-free
    part of ``adjoint_guess[0]`` and leaves its result there for the next one.
    """
    A = matrix.clone()
    A.detach()
    conv = PISOtorch.ConvergenceCriterion.NORM2_NORMALIZED
    tol_t = PISOtorch_diff._get_solver_tolerance_torch(tol, rhs.dtype)
    max_iter_t = torch.IntTensor([max_iter])

    def solve(b: torch.Tensor, x: torch.Tensor, transpose: bool) -> None:
        PISOtorch_diff._linear_solve_wrapper(
            A,
            b,
            x,
            max_iter_t,
            tol_t,
            conv,
            False,  # use_BiCG
            False,  # matrix_rank_deficient
            0,  # residual_reset_step
            transpose,
            False,  # print_residual
            False,  # return_best_result
            is_FWD=not transpose,
        )

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, A_val: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x = guess.clone()
            solve(b, x, transpose=False)
            ctx.save_for_backward(A_val, x)
            return x

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_x: torch.Tensor) -> tuple:
            _, x = ctx.saved_tensors
            last = adjoint_guess[0]
            grad_b = torch.zeros_like(grad_x) if last is None else last - last.mean()
            solve(grad_x, grad_b, transpose=True)
            adjoint_guess[0] = grad_b.detach()
            grad_A_val = None
            if ctx.needs_input_grad[0]:
                grad_A = A.WithZeroValue()
                PISOtorch.SparseOuterProduct(grad_b, x, grad_A)
                grad_A_val = -grad_A.value
            return grad_A_val, grad_b

    return _Solve.apply(matrix.value, rhs)


class _WarmStartSimulation(PISOtorch_simulation.Simulation):
    """PICT's PISO loop with each pressure solve warm-started.

    PICT starts every pressure CG solve from zero. Here the k-th corrector of
    a step starts from the k-th corrector's pressure one step earlier. The
    correctors of one step solve for different pressures, but each changes
    little from step to step. ``solve_tol`` replaces the pressure tolerance in
    the solves, and PICT's own tolerance only gates the start of the run.
    """

    def __init__(self, *args: Any, solve_tol: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._solve_tol = solve_tol
        self._n_solves = 0
        self._last = [None] * self.corrector_steps
        self._last_adjoint = [[None] for _ in range(self.corrector_steps)]

    def linear_solve(
        self,
        csrMat: PISOtorch.CSRmatrix,
        rhs: torch.Tensor,
        x: torch.Tensor | None = None,
        use_BiCG: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, bool]:
        # The pressure solves are the only CG solves without a starting point.
        if use_BiCG or x is not None:
            return super().linear_solve(csrMat, rhs, x, use_BiCG, *args, **kwargs)
        k = self._n_solves % self.corrector_steps
        self._n_solves += 1
        last = self._last[k]
        guess = torch.zeros_like(rhs) if last is None else last - last.mean()
        if self.differentiable:
            result = _warm_cg(
                csrMat,
                rhs,
                guess,
                self._solve_tol,
                self.linear_solve_max_iterations,
                self._last_adjoint[k],
            )
            ok = True
        else:
            kwargs["tol"] = self._solve_tol
            result, ok = super().linear_solve(
                csrMat, rhs, guess, use_BiCG, *args, **kwargs
            )
        self._last[k] = result.detach()
        return result, ok


# mosaic-intervention-end: warm-start-pressure
def _simulate(
    domain: PISOtorch.Domain,
    dt_val: float,
    steps: int,
    differentiable: bool,
    prep_fn: dict | None = None,
    warm_start_tol: float | None = None,
    **sim_kwargs: Any,
) -> None:
    """Run ``steps`` PISO steps on a prepared domain and raise if PICT stops early.

    With ``warm_start_tol``, pressure solves start from the previous step's
    pressure and stop at that tolerance.
    """
    # mosaic-intervention: warm-start-pressure
    if warm_start_tol is None:
        sim_cls = PISOtorch_simulation.Simulation
    else:
        sim_cls = functools.partial(_WarmStartSimulation, solve_tol=warm_start_tol)
    # mosaic-intervention-end: warm-start-pressure
    sim = sim_cls(
        domain,
        substeps=1,
        time_step=float(dt_val),
        # PISO is conventionally run with 2 pressure correctors (Issa 1986);
        # 4 was a conservative default and roughly doubles per-step wall.
        # Re=20 cylinder & decaying-TGV regression checks remained within
        # the existing status_checks bounds, so we run with 2 by default.
        corrector_steps=2,
        differentiable=differentiable,
        log_dir=None,
        log_interval=0,
        log_images=False,
        prep_fn=prep_fn,
        **sim_kwargs,
    )
    # Simulation.run() catches a failed linear solve, logs it and stops early
    # while still returning True, which would hand back a partially integrated
    # field. Collect the logged error so it can be raised here instead.
    solve_errors = _LogRecordCollector()
    pict_log = logging.getLogger("PISOsim")
    pict_log.addHandler(solve_errors)
    try:
        sim.run(steps, log_domain=False)
    finally:
        pict_log.removeHandler(solve_errors)

    if solve_errors.exceptions:
        raise RuntimeError(
            f"PICT linear solve failed in step {sim.total_step + 1} of {steps}: "
            f"{solve_errors.exceptions[-1]}"
        )
    # A NaN blow-up also stops the run early; the non-finite result reports it.
    finite = all(
        bool(torch.isfinite(domain.getBlock(i).velocity).all())
        for i in range(domain.getNumBlocks())
    )
    if sim.total_step != steps and finite:
        raise RuntimeError(
            f"PICT simulation stopped after {sim.total_step} of {steps} steps"
        )


_PERIODIC_SOLVE_TOL = 1e-6


def _run_periodic(
    v0_tensor: torch.Tensor,
    viscosity_val: torch.Tensor,
    dt_val: float,
    steps: int,
    N: int,
    ndim: int,
    dtype: torch.dtype,
    differentiable: bool,
) -> tuple[torch.Tensor, PISOtorch.Domain]:
    """Run PICT on the periodic box and return the final velocity in PICT layout."""
    domain = _make_periodic_domain(viscosity_val, N, ndim, dtype)
    domain.PrepareSolve()  # allocate internal structures first
    domain.getBlock(0).setVelocity(v0_tensor)
    domain.UpdateDomainData()
    # PICT's float32 default tolerance for the advection and pressure solves is
    # 1e-5. Its residual error in the forward and adjoint solves puts a floor of
    # about 1e-3 under the VJP's agreement with finite differences on the
    # periodic box; 1e-6 lowers it to about 7e-5.
    _simulate(
        domain,
        dt_val,
        steps,
        differentiable,
        non_orthogonal=False,
        advection_tol=_PERIODIC_SOLVE_TOL,
        pressure_tol=_PERIODIC_SOLVE_TOL,
    )
    return domain.getBlock(0).velocity, domain


def _run_channel(
    inputs: InputSchema,
    v0: torch.Tensor,
    profile: torch.Tensor,
    viscosity_val: torch.Tensor,
    differentiable: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PISOtorch.Domain]:
    """Run PICT on the channel around the cylinder.

    Returns the canonical velocity (nx, ny, 1, 2), pressure (nx, ny, 1), the
    body force history (steps, 2) and the domain.
    """
    nx, ny = inputs.v0.shape[:2]
    obstacle = inputs.obstacle
    channel = _Channel(
        nx,
        ny,
        inputs.dx,
        (float(obstacle.center[0]), float(obstacle.center[1])),
        float(obstacle.radius),
        viscosity_val,
        torch.float32,
    )
    channel.set_inflow(profile)
    channel.init_outflow(profile)
    domain = channel.domain
    # PICT's pressure residual is absolute. The pressure solves start from the
    # previous step's pressure and stop at 1e-8, where Cd matches a solve from
    # zero at 1e-8 to 2e-5 at 10 and 20 cells per diameter. At 1e-6 the
    # warm-started pressure lags behind the flow and Cd at 20 still drifts by 1e-3
    # over the final 40% of the run. PICT refuses to start when the boundary
    # flux imbalance exceeds its own pressure tolerance, and in float32 the
    # balanced inlet and outlet still differ by ~4e-8 of the flux (1.4e-6 at
    # 40 cells per diameter), so that one stays above twice the imbalance.
    imbalance = abs(float(domain.GetBoundaryFluxBalance()))
    pressure_tol = max(1e-6, 2 * imbalance)
    domain.PrepareSolve()
    channel.set_velocity(v0)
    domain.UpdateDomainData()

    # PICT has no pressure outlet. The outlet velocity is advected out with
    # the mean inflow speed (a convective outflow, zero normal gradient at
    # steady state), pressure is Neumann there, and the returned pressure is
    # shifted to zero on the outlet face.
    adv_vel = torch.tensor(
        [[float(profile.detach().mean()), 0.0]], dtype=torch.float32, device=_DEVICE
    )
    forces: list[torch.Tensor] = []

    def _pre_step(domain: PISOtorch.Domain, time_step: Any, **_kw: Any) -> None:
        # Re-applied every step so that the inflow profile stays in the
        # autograd graph that PISOtorch_diff records during the run.
        channel.set_inflow(profile)
        PISOtorch_simulation.update_advective_boundaries(
            domain, channel.outlets, adv_vel, time_step.to(_DEVICE)
        )

    # mosaic-intervention: body-force-integration
    def _post_step(domain: PISOtorch.Domain, **_kw: Any) -> None:
        forces.append(channel.body_force())

    # mosaic-intervention-end: body-force-integration

    _simulate(
        domain,
        float(np.asarray(inputs.dt)[0]) / inputs.dx,
        inputs.steps,
        differentiable,
        prep_fn={"PRE": _pre_step, "POST": _post_step},
        # PICT's skew correction makes the pressure matrix non-symmetric while
        # it stays singular (pressure is Neumann on every boundary), and on this
        # mesh CG and BiCGStab both break down in the first step. The O-grid
        # is close to orthogonal at the cylinder, and the skewed cells sit at
        # the box corners, a diameter away.
        non_orthogonal=False,
        pressure_tol=pressure_tol,
        # mosaic-intervention: warm-start-pressure
        warm_start_tol=1e-8,
        # mosaic-intervention-end: warm-start-pressure
    )
    return channel.velocity(), channel.pressure(), torch.stack(forces), domain


# ---------------------------------------------------------------------------
# Tesseract API
# ---------------------------------------------------------------------------


def _periodic_scaling(inputs: InputSchema) -> tuple[float, float, float]:
    """``(phys_scale, dt_pict, nu_pict)`` for PICT's unit-cell periodic box.

    With x_pict = x_phys · N/L and velocity unchanged, the physical NS equation
    divided by N/L keeps its form with dt_pict = dt_phys · N/L and
    nu_pict = nu_phys · N/L.
    """
    phys_scale = np.asarray(inputs.v0).shape[0] / inputs.extent[0]
    return (
        phys_scale,
        float(np.asarray(inputs.dt)[0]) * phys_scale,
        float(np.asarray(inputs.viscosity)[0]) * phys_scale,
    )


def _profile_tensor(inputs: InputSchema, requires_grad: bool) -> torch.Tensor:
    profile = np.ascontiguousarray(inputs.inflow_profile, dtype=np.float32)
    return torch.tensor(profile, device=_DEVICE).requires_grad_(requires_grad)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run PICT and return the final velocity, plus pressure and body force on the channel."""
    dtype = torch.float32
    if inputs.obstacle is not None:
        v0 = torch.tensor(np.asarray(inputs.v0), dtype=dtype, device=_DEVICE)
        nu = torch.tensor(np.asarray(inputs.viscosity), dtype=dtype)
        with torch.no_grad():
            velocity, pressure, force, _ = _run_channel(
                inputs, v0, _profile_tensor(inputs, False), nu, differentiable=False
            )
        return {
            "result": velocity.cpu().numpy(),
            "pressure": pressure.cpu().numpy(),
            "body_force": force.cpu().numpy(),
        }

    v0_np = np.asarray(inputs.v0)
    N = v0_np.shape[0]
    ndim = v0_np.shape[-1]
    _, dt_pict, nu_pict = _periodic_scaling(inputs)
    # Viscosity is passed as a CPU torch tensor (PISOtorch requirement).
    nu_pict_t = torch.tensor([nu_pict], dtype=dtype, device=torch.device("cpu"))
    v0_t = _v0_to_pict(v0_np, _DEVICE, dtype, requires_grad=False)
    result_t, _ = _run_periodic(
        v0_t, nu_pict_t, dt_pict, inputs.steps, N, ndim, dtype, differentiable=False
    )
    return {"result": _pict_to_v0(result_t, N, ndim).detach().cpu().numpy()}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via torch.autograd through the differentiable PICT PISO time loop.

    Differentiates ``result`` and, on the channel, ``body_force`` w.r.t. ``v0``,
    ``inflow_profile`` and ``viscosity``. ``dt`` enters as a Python scalar, so
    its gradient is returned as zeros.
    """
    dtype = torch.float32
    v0_np = np.asarray(inputs.v0)
    N = v0_np.shape[0]
    ndim = v0_np.shape[-1]
    want = {
        "v0": "v0" in vjp_inputs,
        "viscosity": "viscosity" in vjp_inputs,
        "inflow_profile": "inflow_profile" in vjp_inputs
        and inputs.inflow_profile is not None,
    }

    outputs: list[torch.Tensor] = []
    cotangents: list[torch.Tensor] = []
    if inputs.obstacle is not None:
        v0_t = torch.tensor(v0_np, dtype=dtype, device=_DEVICE)
        # Viscosity must be a CPU tensor (PISOtorch requirement).
        nu_t = torch.tensor(np.asarray(inputs.viscosity), dtype=dtype)
        profile_t = _profile_tensor(inputs, want["inflow_profile"])
        v0_t.requires_grad_(want["v0"])
        nu_t.requires_grad_(want["viscosity"])
        velocity, _, force, domain = _run_channel(
            inputs, v0_t, profile_t, nu_t, differentiable=True
        )
        for key, tensor in (("result", velocity), ("body_force", force)):
            if key in vjp_outputs and key in cotangent_vector:
                outputs.append(tensor)
                cotangents.append(
                    torch.tensor(
                        np.asarray(cotangent_vector[key]), dtype=dtype, device=_DEVICE
                    )
                )
        nu_scale = 1.0
    else:
        phys_scale, dt_pict, nu_pict = _periodic_scaling(inputs)
        v0_t = _v0_to_pict(v0_np, _DEVICE, dtype, requires_grad=want["v0"])
        # Viscosity must be a CPU tensor (PISOtorch requirement).
        nu_t = torch.tensor([nu_pict], dtype=dtype, requires_grad=want["viscosity"])
        profile_t = None
        result_t, domain = _run_periodic(
            v0_t, nu_t, dt_pict, inputs.steps, N, ndim, dtype, differentiable=True
        )
        if "result" in vjp_outputs and "result" in cotangent_vector:
            outputs.append(result_t)
            # In PICT layout, so the backward pass gets a contiguous tensor.
            ct = np.asarray(cotangent_vector["result"])
            cotangents.append(_v0_to_pict(ct, _DEVICE, dtype))
        # nu_pict = nu_phys * phys_scale ⇒ dL/dnu_phys = dL/dnu_pict * phys_scale.
        nu_scale = phys_scale

    leaves = {"v0": v0_t, "viscosity": nu_t, "inflow_profile": profile_t}
    keys = [k for k, wanted in want.items() if wanted]
    grads: list[torch.Tensor | None] = [None] * len(keys)
    # mosaic-intervention: cotangent-scaling
    # DEVIATION FROM PICT. PICT's linear solves stop once the RMS of their
    # residual falls below an absolute tolerance (by default 1e-5 in float32),
    # and the adjoint solves of its backward pass use the same one. For small
    # cotangents they stop after few or no iterations, so PICT's VJP is not
    # linear in the cotangent: with cotangents of about 1e-4 per entry, the
    # gradient through a chain of calls is wrong by tens of percent. Here the
    # cotangents are scaled to a largest entry of 1 and the gradients scaled
    # back. That is exact for a linear VJP and makes the adjoint tolerance
    # relative to the cotangent.
    scale = max((float(c.abs().max()) for c in cotangents), default=0.0)
    if keys and outputs and scale > 0:
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=[leaves[k] for k in keys],
            grad_outputs=[c / scale for c in cotangents],
            allow_unused=True,
        )
        grads = [None if g is None else g * scale for g in grads]
    # mosaic-intervention-end: cotangent-scaling
    out: dict[str, Any] = {}
    for key, g in zip(keys, grads, strict=True):
        if g is None:
            out[key] = np.zeros_like(np.asarray(getattr(inputs, key)))
        elif key == "v0" and inputs.obstacle is None:
            out[key] = _pict_to_v0(g, N, ndim).cpu().numpy()
        elif key == "viscosity":
            out[key] = (g.detach().cpu().numpy() * nu_scale).astype(np.float32)
        else:
            out[key] = g.detach().cpu().numpy()
    if "dt" in vjp_inputs:
        out["dt"] = np.zeros_like(np.asarray(inputs.dt))
    if "inflow_profile" in vjp_inputs and "inflow_profile" not in out:
        out["inflow_profile"] = np.zeros_like(np.asarray(inputs.inflow_profile))
    domain.Detach()
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, dict[str, Any]]:
    """Return output shapes and dtypes without running the simulation."""
    v0_info = abstract_inputs.v0
    shape = v0_info["shape"] if isinstance(v0_info, dict) else v0_info.shape
    out = {"result": {"shape": shape, "dtype": "float32"}}
    if abstract_inputs.obstacle is not None:
        out["pressure"] = {"shape": tuple(shape[:3]), "dtype": "float32"}
        out["body_force"] = {"shape": (abstract_inputs.steps, 2), "dtype": "float32"}
    return out
