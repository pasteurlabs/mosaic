# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM tesseract for incompressible Navier-Stokes on Cartesian grids.

Periodic mode (fully periodic boundary conditions), solved with icoFoam (PISO), or
with ``foamRun -solver incompressibleFluid`` when ``forcing`` or ``linear_drag`` is set:
  - 2-D: N×N Cartesian grid, cyclic x/y, empty z patches  (v0 shape (N,N,1,2))
  - 3-D: N×N×N Cartesian grid, cyclic x/y/z patches       (v0 shape (N,N,N,3))
  - IC written from v0 as nonuniform OpenFOAM volVectorField
  - Final velocity field parsed back to same shape as v0

Channel mode (a cylinder obstacle in the DFG 2D-1 channel, 2-D only), solved
with ``foamRun -solver incompressibleFluid`` (PIMPLE):
  - Body-fitted mesh: an O-grid around the cylinder inside a Cartesian channel
  - Inlet profile from ``inflow_profile``, zero-gradient outlet with p = 0,
    no-slip walls and cylinder
  - Force on the cylinder from the ``forces`` function object at every step
  - Velocity and pressure sampled at the canonical cell centers

Forward only: no VJP/JVP.
"""

import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
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
from pydantic import model_validator

# OpenFOAM Foundation v12 environment (installed from openfoam.org)
_OF_BASHRC = "/opt/openfoam12/etc/bashrc"

# OpenFOAM case files, with @@NAME@@ placeholders filled in by _render.
_TEMPLATES = Path(__file__).parent / "case_templates"


def _render(template: str, dest: Path, **values: object) -> None:
    """Write case_templates/``template`` to ``dest`` with its placeholders filled."""
    text = (_TEMPLATES / template).read_text()
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", str(value))
    unfilled = re.findall(r"@@\w+@@", text)
    if unfilled:
        raise RuntimeError(f"{template}: placeholders not filled: {unfilled}")
    dest.write_text(text)


class InputSchema(_CanonicalInputSchema):
    """Input schema for the OpenFOAM Navier-Stokes tesseract."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "openfoam")
        return self


class OutputSchema(_CanonicalOutputSchema):
    """Output schema for the OpenFOAM Navier-Stokes tesseract."""


# ---------------------------------------------------------------------------
# OpenFOAM subprocess helper
# ---------------------------------------------------------------------------


def _run_of(cmd: str, cwd: Path) -> None:
    """Run an OpenFOAM command by sourcing the OF environment first."""
    result = subprocess.run(
        ["bash", "-c", f". {_OF_BASHRC} && {cmd}"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'{cmd}' failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Case file writers
# ---------------------------------------------------------------------------


def _write_block_mesh_dict(system_dir: Path, N: int, L: float, ndim: int) -> None:
    """blockMeshDict: N×N (2-D, one cell thick in z) or N×N×N (3-D) periodic grid."""
    _render(
        f"periodic/system/blockMeshDict.{ndim}d",
        system_dir / "blockMeshDict",
        N=N,
        L=L,
        DZ=L / N,
    )


def _write_control_dict(
    system_dir: Path,
    dt: float,
    end_time: float,
    *,
    foam_run: bool = False,
    write_every: int | None = None,
) -> None:
    """Write the periodic-mode controlDict.

    By default only the final time is written. With ``write_every``, a time
    directory is kept every ``write_every`` steps, for the snapshots.
    """
    n_steps = max(1, round(end_time / dt))
    _render(
        "periodic/system/controlDict",
        system_dir / "controlDict",
        APPLICATION=(
            "foamRun;\nsolver          incompressibleFluid;" if foam_run else "icoFoam;"
        ),
        END_TIME=f"{end_time:.10g}",
        DELTA_T=f"{dt:.10g}",
        WRITE_INTERVAL=n_steps if write_every is None else write_every,
        PURGE_WRITE=1 if write_every is None else 0,
    )


def _write_fv_schemes(system_dir: Path) -> None:
    _render("periodic/system/fvSchemes", system_dir / "fvSchemes")


def _write_fv_solution(system_dir: Path) -> None:
    # GAMG instead of PCG/DIC for the pressure, as in the channel case: the
    # forward checks are unchanged and the 2-D solve is up to 2.3x faster at
    # N = 256; 3-D moves by less than 8% either way.
    # The momentum predictor starts from the previous step's U, and a loose
    # tolerance leaves part of each step's update unsolved. On fine grids,
    # where steps are small and many, that becomes an error floor.
    _render("periodic/system/fvSolution", system_dir / "fvSolution")


def _write_transport_properties(constant_dir: Path, nu: float) -> None:
    _render(
        "common/constant/physicalProperties",
        constant_dir / "physicalProperties",
        NU=f"{nu:.10g}",
    )


def _write_turbulence_properties(constant_dir: Path) -> None:
    _render(
        "periodic/constant/turbulenceProperties", constant_dir / "turbulenceProperties"
    )


def _write_vector_field(
    case_dir: Path, name: str, dimensions: str, field: np.ndarray
) -> None:
    """Write 0/``name`` from a v0-shaped field as a nonuniform internalField.

    2-D: shape (N, N, 1, 2) — cell ordering ix + iy*N (x fastest).
    3-D: shape (N, N, N, 3) — cell ordering ix + iy*N + iz*N*N (x fastest).
    """
    ndim = field.shape[-1]
    N = field.shape[0]
    if ndim == 2:
        n_cells = N * N
        vel = field[:, :, 0, :]  # (N, N, 2)
        # Transpose so iy is outer (C-order ravel → ix + iy*N)
        vx = vel[:, :, 0].T.ravel()
        vy = vel[:, :, 1].T.ravel()
        lines = "\n".join(f"({vx[k]:.8g} {vy[k]:.8g} 0)" for k in range(n_cells))
    else:
        n_cells = N * N * N
        # Cell ordering: ix + iy*N + iz*N²  (x fastest → transpose to (nz,ny,nx) for C-ravel)
        vx = field[:, :, :, 0].transpose(2, 1, 0).ravel()
        vy = field[:, :, :, 1].transpose(2, 1, 0).ravel()
        vz = field[:, :, :, 2].transpose(2, 1, 0).ravel()
        lines = "\n".join(
            f"({vx[k]:.8g} {vy[k]:.8g} {vz[k]:.8g})" for k in range(n_cells)
        )
    _render(
        f"periodic/0/vector.{ndim}d",
        case_dir / "0" / name,
        OBJECT=name,
        DIMENSIONS=dimensions,
        N_CELLS=n_cells,
        VALUES=lines,
    )


def _write_initial_u(case_dir: Path, v0: np.ndarray) -> None:
    """Write 0/U from v0."""
    _write_vector_field(case_dir, "U", "[0 1 -1 0 0 0 0]", v0)


def _write_initial_p(case_dir: Path, ndim: int) -> None:
    _render(f"periodic/0/p.{ndim}d", case_dir / "0" / "p")


# ---------------------------------------------------------------------------
# Periodic mode with forcing or linear drag: foamRun incompressibleFluid
# ---------------------------------------------------------------------------
# icoFoam has no momentum source, so a run with ``forcing`` or ``linear_drag``
# uses foamRun's incompressibleFluid solver with fvModels instead. PIMPLE
# with one outer corrector and two pressure correctors is the PISO scheme icoFoam
# runs. Runs without a source keep icoFoam, so their results do not change.


def _write_source_numerics(system_dir: Path, constant_dir: Path) -> None:
    """fvSchemes, fvSolution and momentumTransport for the source-term case."""
    _render("common/system/fvSchemes.incompressibleFluid", system_dir / "fvSchemes")
    _render("source/system/fvSolution", system_dir / "fvSolution")
    _render("common/constant/momentumTransport", constant_dir / "momentumTransport")


def _write_forcing(case_dir: Path, forcing: np.ndarray) -> None:
    """Write 0/F, the body force at the cell centers, in 0/U's layout.

    It sits in the start time directory, not constant/, so decomposePar splits it
    over the processors along with U and p.
    """
    _write_vector_field(case_dir, "F", "[0 1 -2 0 0 0 0]", forcing)


def _write_fv_models(constant_dir: Path, has_forcing: bool, linear_drag: float) -> None:
    """FvModels adding ``F - linear_drag * U`` to the momentum equation.

    The drag is a ``semiImplicitSource`` with an implicit coefficient, written
    as the incompressibleFluid/planarPoiseuille tutorial writes its momentum
    source. The force varies in space, which ``semiImplicitSource`` cannot
    express, so a coded fvModel adds it from 0/F.
    """
    models = []
    if linear_drag != 0.0:
        drag = (_TEMPLATES / "source/fvModels/linearDrag").read_text()
        models.append(drag.replace("@@IMPLICIT@@", f"{-linear_drag:.17g}"))
    if has_forcing:
        # mosaic-intervention: coded-body-force
        models.append((_TEMPLATES / "source/fvModels/bodyForce").read_text())
        # mosaic-intervention-end: coded-body-force
    # The template ends in a newline after @@MODELS@@, and so does each model.
    _render(
        "source/constant/fvModels",
        constant_dir / "fvModels",
        MODELS="\n".join(models).removesuffix("\n"),
    )


# ---------------------------------------------------------------------------
# Channel mode: body-fitted DFG 2D-1 case
# ---------------------------------------------------------------------------

# Radial cell growth in the O-grid, cylinder cell size over square cell size.
_RADIAL_GRADING = 0.5
# Cell growth along the wake block, outlet cell size over the cell size next
# to the cylinder.
_WAKE_GRADING = 10.0


def _write_channel_block_mesh_dict(
    system_dir: Path,
    extent: tuple[float, float],
    dx: float,
    center: tuple[float, float],
    radius: float,
) -> None:
    """Write the blockMeshDict of a 2-D channel with an O-grid around the cylinder.

    The lines x = cx ± 2r and y = cy ± 2r cut the channel into 3 × 3 blocks.
    The outer blocks are Cartesian with spacing ``dx``, except that the cells
    downstream of the cylinder grow towards the outlet. The center square
    holds four O-grid blocks between its sides and the cylinder, with
    4 × 4r/dx cells around the cylinder and 2r/dx cells radially, shrinking
    towards the wall. One cell thick in z.
    """
    lx, ly = extent
    cx, cy = center
    half_side = 2 * radius
    if not (
        cx - half_side > 0
        and cx + half_side < lx
        and cy - half_side > 0
        and cy + half_side < ly
    ):
        raise NotImplementedError(
            "openfoam needs the square of side 2D around the cylinder to lie "
            "inside the channel"
        )
    xs = (0.0, cx - half_side, cx + half_side, lx)
    ys = (0.0, cy - half_side, cy + half_side, ly)
    nx = [max(1, round((xs[i + 1] - xs[i]) / dx)) for i in range(3)]
    ny = [max(1, round((ys[j + 1] - ys[j]) / dx)) for j in range(3)]
    # Geometric growth from dx to _WAKE_GRADING dx over the wake block.
    wake = xs[3] - xs[2]
    q = (wake - dx) / (wake - _WAKE_GRADING * dx)
    nx[2] = max(1, round(1 + math.log(_WAKE_GRADING) / math.log(q)))
    n_side, n_rad = nx[1], max(2, nx[1] // 2)

    # 2-D vertices: the 4 × 4 block corners (index 4 j + i), then the points
    # on the cylinder at 225°, 315°, 45° and 135°. The z = dz copies follow.
    c = radius / math.sqrt(2)
    pts = [(x, y) for y in ys for x in xs]
    pts += [(cx - c, cy - c), (cx + c, cy - c), (cx + c, cy + c), (cx - c, cy + c)]
    n = len(pts)
    dz = dx
    vertices = [f"({x:.10g} {y:.10g} 0)" for x, y in pts]
    vertices += [f"({x:.10g} {y:.10g} {dz:.10g})" for x, y in pts]

    def corner(i: int, j: int) -> int:
        return 4 * j + i

    def hex_block(v: list[int], cells: tuple[int, int], grading: str) -> str:
        ids = " ".join(str(k) for k in [*v, *(k + n for k in v)])
        return f"hex ({ids}) ({cells[0]} {cells[1]} 1) simpleGrading ({grading} 1)"

    blocks = [
        hex_block(
            [corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1)],
            (nx[i], ny[j]),
            f"{_WAKE_GRADING if i == 2 else 1} 1",
        )
        for j in range(3)
        for i in range(3)
        if (i, j) != (1, 1)
    ]
    # O-grid blocks run counter-clockwise, with local x along the square side
    # and local y inwards from the square to the cylinder.
    square = [corner(1, 1), corner(2, 1), corner(2, 2), corner(1, 2)]
    circle = [16, 17, 18, 19]
    for k in range(4):
        v = [square[k], square[(k + 1) % 4], circle[(k + 1) % 4], circle[k]]
        blocks.append(hex_block(v, (n_side, n_rad), f"1 {_RADIAL_GRADING}"))

    edges = []
    for k in range(4):
        angle = math.radians(270 + 90 * k)
        mid = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        for z, off in ((0.0, 0), (dz, n)):
            edges.append(
                f"arc {circle[k] + off} {circle[(k + 1) % 4] + off} "
                f"({mid[0]:.10g} {mid[1]:.10g} {z:.10g})"
            )

    def face(a: int, b: int) -> str:
        # a → b runs along the boundary with the fluid on its left, so the
        # face normal points out of the domain.
        return f"({a} {b} {b + n} {a + n})"

    patches = {
        "inlet": ("patch", [face(corner(0, j + 1), corner(0, j)) for j in range(3)]),
        "outlet": ("patch", [face(corner(3, j), corner(3, j + 1)) for j in range(3)]),
        "walls": (
            "wall",
            [face(corner(i, 0), corner(i + 1, 0)) for i in range(3)]
            + [face(corner(i + 1, 3), corner(i, 3)) for i in range(3)],
        ),
        "cylinder": (
            "wall",
            [face(circle[(k + 1) % 4], circle[k]) for k in range(4)],
        ),
    }
    boundary = "\n".join(
        f"    {name}\n    {{\n        type {kind};\n"
        f"        faces ({' '.join(faces)});\n    }}"
        for name, (kind, faces) in patches.items()
    )
    nl = "\n    "
    _render(
        "channel/system/blockMeshDict",
        system_dir / "blockMeshDict",
        VERTICES=nl.join(vertices),
        BLOCKS=nl.join(blocks),
        EDGES=nl.join(edges),
        BOUNDARY=boundary,
    )


def _write_channel_control_dict(
    system_dir: Path, dt: float, n_steps: int, center: tuple[float, float]
) -> None:
    """Write the foamRun controlDict, recording the force on the cylinder at every step."""
    _render(
        "channel/system/controlDict",
        system_dir / "controlDict",
        END_TIME=f"{n_steps * dt:.15g}",
        DELTA_T=f"{dt:.15g}",
        WRITE_INTERVAL=n_steps,
        CX=f"{center[0]:.10g}",
        CY=f"{center[1]:.10g}",
    )


def _write_channel_numerics(system_dir: Path, constant_dir: Path, nu: float) -> None:
    """fvSchemes, fvSolution and the laminar transport model for channel mode.

    Second-order in space and implicit Euler in time, which damps the start-up
    transient on the way to the steady state.
    """
    _render("common/system/fvSchemes.incompressibleFluid", system_dir / "fvSchemes")
    _render("channel/system/fvSolution", system_dir / "fvSolution")
    _write_transport_properties(constant_dir, nu)
    _render("common/constant/momentumTransport", constant_dir / "momentumTransport")


def _parse_vectors(text: str) -> np.ndarray:
    """``(n, 3)`` array from the body of an OpenFOAM ``List<vector>``."""
    return np.array(
        [r.split() for r in re.findall(r"\(([^()]+)\)", text)], dtype=np.float64
    )


def _read_cell_centers(workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mesh cell centers ``(n_cells, 3)`` and inlet face centers ``(n_faces, 3)``."""
    _run_of("foamPostProcess -func writeCellCentres -time 0", workdir)
    text = (workdir / "0" / "C").read_text()
    cells = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    inlet = re.search(
        r"inlet\s*\{[^{}]*?value\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    for name in ("C", "Cx", "Cy", "Cz"):
        (workdir / "0" / name).unlink(missing_ok=True)
    return _parse_vectors(cells.group(1)), _parse_vectors(inlet.group(1))


def _bilinear(field: np.ndarray, dx: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a cell-centered ``(nx, ny, c)`` field at (x, y)."""
    nx, ny = field.shape[:2]
    fx = np.clip(x / dx - 0.5, 0, nx - 1)
    fy = np.clip(y / dx - 0.5, 0, ny - 1)
    i = np.minimum(fx.astype(int), nx - 2)
    j = np.minimum(fy.astype(int), ny - 2)
    wx, wy = (fx - i)[:, None], (fy - j)[:, None]
    return (
        (1 - wx) * (1 - wy) * field[i, j]
        + wx * (1 - wy) * field[i + 1, j]
        + (1 - wx) * wy * field[i, j + 1]
        + wx * wy * field[i + 1, j + 1]
    )


def _vector_list(v: np.ndarray) -> str:
    rows = "\n".join(f"({a:.10g} {b:.10g} 0)" for a, b in v)
    return f"nonuniform List<vector>\n{len(v)}\n(\n{rows}\n)"


def _write_channel_fields(
    case_dir: Path, u_cells: np.ndarray, u_inlet: np.ndarray
) -> None:
    """Write 0/U and 0/p with the inlet profile, zero-gradient outlet at p = 0 and no-slip walls."""
    _render(
        "channel/0/U",
        case_dir / "0" / "U",
        INTERNAL_FIELD=_vector_list(u_cells),
        INLET_VALUE=_vector_list(u_inlet),
    )
    _render("channel/0/p", case_dir / "0" / "p")


def _sample(workdir: Path, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """U ``(n, 2)`` and p ``(n,)`` at ``points`` at the final time.

    Uses the ``probes`` function object with cell-point interpolation, which
    is exact at cell centers and honors the boundary values on the walls.
    """
    locations = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in points)
    _render("channel/system/sample", workdir / "system" / "sample", LOCATIONS=locations)
    _run_of("foamPostProcess -dict system/sample -latestTime", workdir)
    (out_dir,) = (workdir / "postProcessing" / "probes").iterdir()

    def last_row(name: str) -> str:
        lines = (out_dir / name).read_text().splitlines()
        return next(ln for ln in reversed(lines) if ln and not ln.startswith("#"))

    u = _parse_vectors(last_row("U"))
    p = np.array(last_row("p").split()[1:], dtype=np.float64)
    if len(u) != len(points) or len(p) != len(points):
        raise RuntimeError(
            f"probes returned {len(u)} velocities and {len(p)} pressures for "
            f"{len(points)} points"
        )
    if np.abs(u).max() > 1e100 or np.abs(p).max() > 1e100:
        raise RuntimeError("some sample points lie outside the mesh")
    return u[:, :2], p


def _read_forces(workdir: Path, depth: float) -> np.ndarray:
    """Total force on the cylinder per unit depth at every internal step, ``(n, 2)``."""
    (path,) = (workdir / "postProcessing" / "forces").glob("*/forces.dat")
    rows = [
        ln.replace("(", " ").replace(")", " ").split()
        for ln in path.read_text().splitlines()
        if ln and not ln.startswith("#")
    ]
    data = np.array(rows, dtype=np.float64)
    # Columns: time, pressure force (x y z), viscous force (x y z), then the
    # moments.
    return (data[:, 1:3] + data[:, 4:6]) / depth


# Fewest cells per MPI process. Smaller meshes run serially.
_CELLS_PER_PROCESS = 10_000


def _physical_cores() -> int:
    """Physical cores among the CPUs this process may run on.

    One rank per physical core: OpenFOAM's ranks busy-wait in MPI, and two
    of them on hardware threads of the same core slow each other down.
    """
    cpus = os.sched_getaffinity(0)
    siblings = set()
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings.add(path.read_text().strip())
        except OSError:
            return len(cpus)
    return len(siblings)


def _decompose(workdir: Path, n_cells: int) -> int:
    """Split the case over the available cores; returns the number of processes."""
    n_procs = min(_physical_cores(), n_cells // _CELLS_PER_PROCESS)
    if n_procs < 2:
        return 1
    _render(
        "common/system/decomposeParDict",
        workdir / "system" / "decomposeParDict",
        N_PROCS=n_procs,
    )
    _run_of("decomposePar", workdir)
    return n_procs


def _parallel(cmd: str, n_procs: int) -> str:
    """``cmd`` run on ``n_procs`` MPI processes of a decomposed case."""
    return f"mpirun -np {n_procs} {cmd} -parallel" if n_procs > 1 else cmd


def _run_channel(inputs: InputSchema) -> dict:
    """Solve the DFG channel on a body-fitted mesh and return canonical-grid outputs."""
    v0 = np.asarray(inputs.v0, dtype=np.float64)[:, :, 0, :]
    nx, ny = v0.shape[:2]
    dx = inputs.dx
    extent = inputs.extent
    center = tuple(float(c) for c in inputs.obstacle.center)
    radius = float(inputs.obstacle.radius)
    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    profile = np.asarray(inputs.inflow_profile, dtype=np.float64)
    y_canonical = (np.arange(ny) + 0.5) * dx

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        (workdir / "0").mkdir(parents=True)
        (workdir / "constant").mkdir()
        (workdir / "system").mkdir()

        _write_channel_block_mesh_dict(workdir / "system", extent, dx, center, radius)
        n_steps = inputs.steps
        _write_channel_control_dict(workdir / "system", dt, n_steps, center)
        _write_channel_numerics(workdir / "system", workdir / "constant", nu)
        _run_of("blockMesh", workdir)

        cells, inlet = _read_cell_centers(workdir)
        u_cells = _bilinear(v0, dx, cells[:, 0], cells[:, 1])
        u_inlet = np.zeros((len(inlet), 2))
        u_inlet[:, 0] = np.interp(inlet[:, 1], y_canonical, profile)
        _write_channel_fields(workdir, u_cells, u_inlet)

        _run_of("foamRun", workdir)

        forces = _read_forces(workdir, depth=dx)
        if len(forces) != n_steps:
            raise RuntimeError(f"expected {n_steps} force samples, got {len(forces)}")

        xc = (np.arange(nx) + 0.5) * dx
        X, Y = np.meshgrid(xc, y_canonical, indexing="ij")
        fluid = ~solid_cells(nx, ny, dx, center, radius)
        points = np.stack([X[fluid], Y[fluid], np.full(fluid.sum(), dx / 2)], axis=1)
        u, p = _sample(workdir, points)

    velocity = np.zeros((nx, ny, 1, 2), dtype=np.float32)
    velocity[fluid, 0] = u
    pressure = np.zeros((nx, ny, 1), dtype=np.float32)
    pressure[fluid, 0] = p
    return {
        "result": velocity,
        "pressure": pressure,
        "body_force": forces.astype(np.float32),
    }


def _output_time_dirs(workdir: Path) -> list[Path]:
    """Time directories after the start, in time order."""
    time_dirs = []
    for d in workdir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d))
            except ValueError:
                pass
    if not time_dirs:
        raise RuntimeError(f"No output time directories found in {workdir}")
    return [d for _, d in sorted(time_dirs, key=lambda x: x[0])]


def _read_final_velocity(workdir: Path, N: int, ndim: int) -> np.ndarray:
    """Read the latest time directory and return velocity as v0-shaped float32."""
    return _read_velocity(_output_time_dirs(workdir)[-1], N, ndim)


def _read_velocity(latest_dir: Path, N: int, ndim: int) -> np.ndarray:
    """Read U from one time directory and return it as v0-shaped float32.

    2-D cell ordering: cell = ix + iy*N  → flat[k]: ix = k%N, iy = k//N
    3-D cell ordering: cell = ix + iy*N + iz*N² → flat[k]: ix=k%N, iy=(k//N)%N, iz=k//(N²)
    """
    text = (latest_dir / "U").read_text()

    m = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\(\s*(.*?)\s*\)\s*;",
        text,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"Could not parse internalField from {latest_dir / 'U'}")

    rows = re.findall(r"\(([^)]+)\)", m.group(1))
    n_cells = N**ndim
    if len(rows) != n_cells:
        raise RuntimeError(f"Expected {n_cells} velocity rows, got {len(rows)}")

    flat = np.array(
        [[float(x) for x in r.split()] for r in rows], dtype=np.float32
    )  # (n_cells, 3)

    if ndim == 2:
        # C-order reshape gives [iy, ix, 3]; transpose to [ix, iy, 3]
        vel = flat.reshape(N, N, 3).transpose(1, 0, 2)  # (N, N, 3)
        return vel[:, :, None, :2].copy()  # (N, N, 1, 2)
    else:
        # C-order reshape gives [iz, iy, ix, 3]; transpose to [ix, iy, iz, 3]
        vel = flat.reshape(N, N, N, 3).transpose(2, 1, 0, 3)  # (N, N, N, 3)
        return vel.copy()  # (N, N, N, 3)


# ---------------------------------------------------------------------------
# Tesseract interface
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Run OpenFOAM and return the final velocity.

    With an obstacle, also return the final pressure and the force history.
    """
    # A zero force and a zero drag are no source: they keep icoFoam, as without them.
    forcing = None if inputs.forcing is None else np.asarray(inputs.forcing, np.float32)
    if forcing is not None and not forcing.any():
        forcing = None
    drag = 0.0 if inputs.linear_drag is None else float(np.ravel(inputs.linear_drag)[0])
    has_source = forcing is not None or drag != 0.0
    if inputs.obstacle is not None:
        # The schema allows forcing and linear_drag only without an obstacle.
        return _run_channel(inputs)

    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    N = inputs.v0.shape[0]
    ndim = inputs.v0.shape[-1]
    L = float(inputs.domain_extent)
    end_time = inputs.steps * dt
    v0 = np.asarray(inputs.v0, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        (workdir / "0").mkdir(parents=True)
        (workdir / "constant").mkdir()
        (workdir / "system").mkdir()

        _write_block_mesh_dict(workdir / "system", N, L, ndim)
        _write_control_dict(
            workdir / "system",
            dt,
            end_time,
            foam_run=has_source,
            write_every=inputs.snapshot_every,
        )
        _write_transport_properties(workdir / "constant", nu)
        if has_source:
            _write_source_numerics(workdir / "system", workdir / "constant")
            _write_fv_models(workdir / "constant", forcing is not None, drag)
        else:
            _write_fv_schemes(workdir / "system")
            _write_fv_solution(workdir / "system")
            _write_turbulence_properties(workdir / "constant")
        if forcing is not None:
            # before 0/U, which _write_forcing borrows as a template
            _write_forcing(workdir, forcing)
        _write_initial_u(workdir, v0)
        _write_initial_p(workdir, ndim)

        _run_of("blockMesh", workdir)
        n_procs = _decompose(workdir, N**ndim)
        solver = "foamRun" if has_source else "icoFoam"
        _run_of(_parallel(solver, n_procs), workdir)
        if inputs.snapshot_every is None:
            if n_procs > 1:
                _run_of("reconstructPar -latestTime", workdir)
            return {"result": _read_final_velocity(workdir, N, ndim)}

        if n_procs > 1:
            _run_of("reconstructPar", workdir)
        frames = np.stack(
            [_read_velocity(d, N, ndim) for d in _output_time_dirs(workdir)]
        )
    return {"result": frames[-1], "snapshots": frames}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Return output shapes and dtypes without running the solver."""
    v0 = abstract_inputs.v0
    shape = v0["shape"] if isinstance(v0, dict) else tuple(v0.shape)
    out = {"result": {"shape": shape, "dtype": "float32"}}
    if abstract_inputs.obstacle is not None:
        out["pressure"] = {"shape": shape[:3], "dtype": "float32"}
        out["body_force"] = {"shape": (abstract_inputs.steps, 2), "dtype": "float32"}
    if abstract_inputs.snapshot_every is not None:
        frames = abstract_inputs.steps // abstract_inputs.snapshot_every
        out["snapshots"] = {"shape": (frames, *shape), "dtype": "float32"}
    return out
