# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAFoam tesseract for the DFG 2D-1 cylinder channel.

DAPimpleFoam solves the channel on the OpenFOAM tesseract's body-fitted O-grid:
  - Inlet profile from ``inflow_profile``, zero-gradient outlet with p = 0,
    no-slip walls and cylinder
  - Force on the cylinder per unit depth at every step
  - Velocity and pressure sampled at the canonical cell centers

DAFoam's time-accurate discrete adjoint gives the VJP of ``body_force`` with
respect to ``inflow_profile`` for time-averaged drags (:func:`_trailing_window`).
DAFoam runs as a subprocess under its own Python (dafoam_driver.py).
"""

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

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
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator

_DAFOAM_ENV = "/home/dafoamuser/dafoam/loadDAFoam.sh"
_DRIVER = Path(__file__).with_name("dafoam_driver.py")


class InputSchema(make_differentiable(_CanonicalInputSchema, ["inflow_profile"])):
    """Input schema for the DAFoam Navier-Stokes tesseract."""

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> "InputSchema":
        check_periodic_or_channel(self, "dafoam")
        if self.obstacle is None:
            raise ValueError(
                "dafoam supports only the 2-D channel with an inflow_profile and "
                "a cylinder"
            )
        return self


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["body_force"])):
    """Output schema for the DAFoam Navier-Stokes tesseract."""


def _run(cmd: str, cwd: Path) -> None:
    """Run a command with the DAFoam (OpenFOAM v2506) environment loaded."""
    # The Tesseract runtime exports its sys.path as PYTHONPATH
    # (pasteurlabs/tesseract-core#883), which would make DAFoam's Python import
    # the runtime's packages.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        ["bash", "-c", f". {_DAFOAM_ENV} && {cmd}"],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'{cmd}' failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-5000:]}\nSTDERR:\n{result.stderr[-5000:]}"
        )


# ---------------------------------------------------------------------------
# Case files
# ---------------------------------------------------------------------------

# Radial cell growth in the O-grid, cylinder cell size over square cell size.
_RADIAL_GRADING = 0.5
# Cell growth along the wake block, outlet cell size over the cell size next
# to the cylinder.
_WAKE_GRADING = 10.0

_FOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


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
    towards the wall. One cell thick in z, with symmetry front and back planes.
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
            "dafoam needs the square of side 2D around the cylinder to lie "
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
    content = (
        _FOAM_HEADER.format(cls="dictionary", obj="blockMeshDict")
        + f"""\
scale   1;

vertices
(
    {nl.join(vertices)}
);

blocks
(
    {nl.join(blocks)}
);

edges
(
    {nl.join(edges)}
);

boundary
(
{boundary}
);

defaultPatch
{{
    name    frontAndBack;
    type    symmetry;
}}
"""
    )
    (system_dir / "blockMeshDict").write_text(content)


def _write_numerics(case_dir: Path, dt: float, steps: int, nu: float) -> None:
    """Write controlDict, fvSchemes, fvSolution and the laminar transport model.

    The schemes match the OpenFOAM tesseract's channel case.
    """
    system, constant = case_dir / "system", case_dir / "constant"
    # The adjoint reads the state of every step back, so every step is written.
    (system / "controlDict").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="controlDict")
        + f"""\
application     DAPimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {steps * dt:.15g};
deltaT          {dt:.15g};
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  16;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
"""
    )
    # div(pc) is the first-order convection scheme of the adjoint's
    # preconditioner matrix.
    (system / "fvSchemes").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="fvSchemes")
        + """\
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(pc)         bounded Gauss upwind;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWaveFrozen; }
"""
    )
    # The discrete adjoint linearizes the converged residual of each step, so
    # PIMPLE iterates every step to convergence. DAFoam's residual assumes an
    # unrelaxed final outer iteration (UEqn.relax(1.0)), hence UFinal and
    # pFinal 1. Convergence is checked on p only, because the symmetry planes
    # make Uz a solved component that stays near zero, and its normalized
    # residual never reaches a tolerance.
    (system / "fvSolution").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="fvSolution")
        + """\
solvers
{
    "p.*"   { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0.001; maxIter 200; minIter 1; }
    pFinal  { $p; relTol 0; }
    "U.*"   { solver smoothSolver; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 200; minIter 1; }
}
relaxationFactors
{
    fields    { p 0.3; pFinal 1; }
    equations { U 0.7; UFinal 1; }
}
PIMPLE
{
    nNonOrthogonalCorrectors 1;
    nCorrectors              2;
    nOuterCorrectors         50;
    residualControl { p { tolerance 1e-9; relTol 0; } }
}
"""
    )
    # DAFoam has no laminar model. Its "dummy" RAS model with nut = 0 is laminar
    # flow, and it reads Pr and Prt.
    (constant / "transportProperties").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="transportProperties")
        + f"transportModel Newtonian;\nnu {nu:.15g};\nPr 1.0;\nPrt 1.0;\n"
    )
    (constant / "turbulenceProperties").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="turbulenceProperties")
        + "simulationType RAS;\nRAS { RASModel dummy; turbulence on; printCoeffs off; }\n"
    )


def _parse_vectors(text: str) -> np.ndarray:
    """``(n, 3)`` array from the body of an OpenFOAM ``List<vector>``."""
    return np.array(
        [r.split() for r in re.findall(r"\(([^()]+)\)", text)], dtype=np.float64
    )


def _read_cell_centers(workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mesh cell centers ``(n_cells, 3)`` and inlet face centers ``(n_faces, 3)``."""
    _run("postProcess -func writeCellCentres -time 0", workdir)
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
    rows = "\n".join(f"({a:.15g} {b:.15g} 0)" for a, b in v)
    return f"nonuniform List<vector>\n{len(v)}\n(\n{rows}\n)"


def _write_fields(case_dir: Path, u_cells: np.ndarray, u_inlet: np.ndarray) -> None:
    """Write 0/U, 0/p and 0/nut for the inlet profile, outlet and no-slip walls."""
    zero = case_dir / "0"
    (zero / "U").write_text(
        _FOAM_HEADER.format(cls="volVectorField", obj="U")
        + f"""\
dimensions      [0 1 -1 0 0 0 0];
internalField   {_vector_list(u_cells)};
boundaryField
{{
    inlet        {{ type fixedValue; value {_vector_list(u_inlet)}; }}
    outlet       {{ type zeroGradient; }}
    walls        {{ type fixedValue; value uniform (0 0 0); }}
    cylinder     {{ type fixedValue; value uniform (0 0 0); }}
    frontAndBack {{ type symmetry; }}
}}
"""
    )
    (zero / "p").write_text(
        _FOAM_HEADER.format(cls="volScalarField", obj="p")
        + """\
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    walls        { type zeroGradient; }
    cylinder     { type zeroGradient; }
    frontAndBack { type symmetry; }
}
"""
    )
    (zero / "nut").write_text(
        _FOAM_HEADER.format(cls="volScalarField", obj="nut")
        + """\
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet        { type calculated; value uniform 0; }
    outlet       { type calculated; value uniform 0; }
    walls        { type calculated; value uniform 0; }
    cylinder     { type calculated; value uniform 0; }
    frontAndBack { type symmetry; }
}
"""
    )


def _sample(workdir: Path, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """U ``(n, 2)`` and p ``(n,)`` at ``points`` at the final time.

    Uses the ``probes`` function object with cell-point interpolation, which
    is exact at cell centers and honors the boundary values on the walls.
    """
    locations = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in points)
    (workdir / "system" / "sample").write_text(
        _FOAM_HEADER.format(cls="dictionary", obj="sample")
        + f"""\
functions
{{
    probes
    {{
        type                probes;
        libs                (sampling);
        fields              (U p);
        interpolationScheme cellPoint;
        probeLocations
        (
{locations}
        );
    }}
}}
"""
    )
    # OpenFOAM v2506's -dict stands in for the controlDict, so the probes sit
    # under functions, and the fields to load are named with -fields.
    _run("postProcess -dict system/sample -fields '(U p)' -latestTime", workdir)
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


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def _solve(
    inputs: InputSchema, workdir: Path, window: int | None = None
) -> tuple[dict, np.ndarray]:
    """Build and solve the case in ``workdir``.

    Returns the driver's results (see dafoam_driver.py) and the linear map
    from ``inflow_profile`` to the inlet face velocities, ``(n_faces, ny)``.
    """
    v0 = np.asarray(inputs.v0, dtype=np.float64)[:, :, 0, :]
    ny = v0.shape[1]
    dx = inputs.dx
    profile = np.asarray(inputs.inflow_profile, dtype=np.float64)

    (workdir / "0").mkdir(parents=True)
    (workdir / "constant").mkdir()
    (workdir / "system").mkdir()
    _write_channel_block_mesh_dict(
        workdir / "system",
        inputs.extent,
        dx,
        tuple(float(c) for c in inputs.obstacle.center),
        float(inputs.obstacle.radius),
    )
    # dt arrives in float32. Its shortest decimal form (0.01, not
    # 0.0099999998) keeps the time directory names OpenFOAM writes equal to
    # the ones DAFoam reads back.
    dt = float(str(np.float32(inputs.dt[0])))
    _write_numerics(workdir, dt, inputs.steps, float(inputs.viscosity[0]))
    _run("blockMesh", workdir)

    cells, inlet = _read_cell_centers(workdir)
    y_canonical = (np.arange(ny) + 0.5) * dx
    inlet_map = np.stack(
        [np.interp(inlet[:, 1], y_canonical, e) for e in np.eye(ny)], axis=1
    )
    u_inlet = np.zeros((len(inlet), 2))
    u_inlet[:, 0] = inlet_map @ profile
    _write_fields(workdir, _bilinear(v0, dx, cells[:, 0], cells[:, 1]), u_inlet)

    request = {
        "inlet_u": u_inlet[:, 0].tolist(),
        # One cell thick in z, as deep as a cell is wide.
        "depth": dx,
        "steps": inputs.steps,
        "u_ref": float(profile.mean()),
        "window": window,
    }
    (workdir / "request.json").write_text(json.dumps(request))
    _run(f"python {_DRIVER} . request.json result.npz", workdir)
    with np.load(workdir / "result.npz") as result:
        return dict(result), inlet_map


def _trailing_window(cotangent: np.ndarray) -> tuple[int, float]:
    """``(window, weight)`` of a cotangent that weights F_x over the last steps.

    DAFoam differentiates time averages of a force, so the VJP supports
    cotangents with one value ``weight`` on the x force of the last ``window``
    steps and zeros elsewhere, as for a drag coefficient averaged over a
    trailing window. ``window`` is 0 for an all-zero cotangent.
    """
    fx, fy = cotangent[:, 0], cotangent[:, 1]
    if not fx.any() and not fy.any():
        return 0, 0.0
    window = int(len(fx) - np.flatnonzero(fx)[0]) if fx.any() else 0
    if (
        fy.any()
        or window == 0
        or not np.allclose(fx[-window:], fx[-1], rtol=1e-6, atol=0)
    ):
        raise NotImplementedError(
            "dafoam's VJP supports only body_force cotangents that weight the "
            "x force uniformly over a trailing window of steps, as for a "
            "time-averaged drag"
        )
    return window, float(fx[-1])


# ---------------------------------------------------------------------------
# Tesseract interface
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Run DAFoam and return the final velocity and pressure and the force history."""
    v0 = np.asarray(inputs.v0)
    nx, ny = v0.shape[:2]
    dx = inputs.dx
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        result, _ = _solve(inputs, workdir)

        xc = (np.arange(nx) + 0.5) * dx
        yc = (np.arange(ny) + 0.5) * dx
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        fluid = ~solid_cells(nx, ny, dx, inputs.obstacle.center, inputs.obstacle.radius)
        points = np.stack([X[fluid], Y[fluid], np.full(fluid.sum(), dx / 2)], axis=1)
        u, p = _sample(workdir, points)

    velocity = np.zeros((nx, ny, 1, 2), dtype=np.float32)
    velocity[fluid, 0] = u
    pressure = np.zeros((nx, ny, 1), dtype=np.float32)
    pressure[fluid, 0] = p
    return {
        "result": velocity,
        "pressure": pressure,
        "body_force": result["body_force"].astype(np.float32),
    }


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Return output shapes and dtypes without running the solver."""
    v0 = abstract_inputs.v0
    shape = v0["shape"] if isinstance(v0, dict) else tuple(v0.shape)
    return {
        "result": {"shape": shape, "dtype": "float32"},
        "pressure": {"shape": shape[:3], "dtype": "float32"},
        "body_force": {"shape": (abstract_inputs.steps, 2), "dtype": "float32"},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """VJP of ``body_force`` with respect to ``inflow_profile`` via DAFoam's adjoint."""
    profile = np.asarray(inputs.inflow_profile)
    window, weight = _trailing_window(np.asarray(cotangent_vector["body_force"]))
    if window == 0:
        return {"inflow_profile": np.zeros_like(profile, dtype=np.float32)}
    with tempfile.TemporaryDirectory() as tmpdir:
        result, inlet_map = _solve(inputs, Path(tmpdir) / "case", window)
    # The adjoint differentiates the window mean; the cotangent weights the sum.
    grad = weight * window * (inlet_map.T @ result["d_fx_d_inlet_u"])
    return {"inflow_profile": grad.astype(np.float32)}
