# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM icoFoam tesseract for incompressible NS on periodic Cartesian grids.

Wraps OpenFOAM icoFoam (transient, laminar, incompressible) with:
  - 2-D: N×N Cartesian grid, cyclic x/y, empty z patches  (v0 shape (N,N,1,2))
  - 3-D: N×N×N Cartesian grid, cyclic x/y/z patches       (v0 shape (N,N,N,3))
  - IC written from v0 as nonuniform OpenFOAM volVectorField
  - Final velocity field parsed back to same shape as v0
  - No VJP/JVP (forward-only baseline)

Channel mode (activated when ``inputs.obstacle`` is not None):
  - INLET/OUTLET BCs in x, cyclic BCs in y, empty z patches (2-D only)
  - No-slip obstacle enforcement via fvConstraints fixedValue + topoSet cylinderToCell (OF12)
  - Uses foamRun -solver incompressibleFluid (NOT legacy icoFoam) so fvConstraints are applied
"""

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

# OpenFOAM Foundation v12 environment (installed from openfoam.org)
_OF_BASHRC = "/opt/openfoam12/etc/bashrc"


class InputSchema(_CanonicalInputSchema):
    """Input schema for the OpenFOAM icoFoam Navier-Stokes tesseract."""

    pass


class OutputSchema(_CanonicalOutputSchema):
    """Output schema for the OpenFOAM icoFoam Navier-Stokes tesseract."""

    pass


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
    """blockMeshDict: N×N (2-D) or N×N×N (3-D) Cartesian periodic grid."""
    if ndim == 2:
        # Quasi-2-D: one cell thick in z, empty patches front/back.
        dz = L / N
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
scale   1;

vertices
(
    (0 0 0)      // 0
    ({L} 0 0)    // 1
    ({L} {L} 0)  // 2
    (0 {L} 0)    // 3
    (0 0 {dz})   // 4
    ({L} 0 {dz}) // 5
    ({L} {L} {dz}) // 6
    (0 {L} {dz}) // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N} {N} 1) simpleGrading (1 1 1)
);

boundary
(
    x_lo
    {{
        type cyclic;
        neighbourPatch x_hi;
        faces ((0 4 7 3));
    }}
    x_hi
    {{
        type cyclic;
        neighbourPatch x_lo;
        faces ((1 2 6 5));
    }}
    y_lo
    {{
        type cyclic;
        neighbourPatch y_hi;
        faces ((0 1 5 4));
    }}
    y_hi
    {{
        type cyclic;
        neighbourPatch y_lo;
        faces ((3 7 6 2));
    }}
    front
    {{
        type empty;
        faces ((0 3 2 1));
    }}
    back
    {{
        type empty;
        faces ((4 5 6 7));
    }}
);
"""
    else:
        # Full 3-D cube: N×N×N cells, all six faces cyclic.
        content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
scale   1;

vertices
(
    (0 0 0)      // 0
    ({L} 0 0)    // 1
    ({L} {L} 0)  // 2
    (0 {L} 0)    // 3
    (0 0 {L})    // 4
    ({L} 0 {L})  // 5
    ({L} {L} {L}) // 6
    (0 {L} {L})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N} {N} {N}) simpleGrading (1 1 1)
);

boundary
(
    x_lo
    {{
        type cyclic;
        neighbourPatch x_hi;
        faces ((0 4 7 3));
    }}
    x_hi
    {{
        type cyclic;
        neighbourPatch x_lo;
        faces ((1 2 6 5));
    }}
    y_lo
    {{
        type cyclic;
        neighbourPatch y_hi;
        faces ((0 1 5 4));
    }}
    y_hi
    {{
        type cyclic;
        neighbourPatch y_lo;
        faces ((3 7 6 2));
    }}
    z_lo
    {{
        type cyclic;
        neighbourPatch z_hi;
        faces ((0 3 2 1));
    }}
    z_hi
    {{
        type cyclic;
        neighbourPatch z_lo;
        faces ((4 5 6 7));
    }}
);
"""
    (system_dir / "blockMeshDict").write_text(content)


def _write_control_dict(
    system_dir: Path, dt: float, end_time: float, *, use_foam_run: bool = False
) -> None:
    n_steps = max(1, round(end_time / dt))
    if use_foam_run:
        # foamRun -solver incompressibleFluid supports fvModels/fvConstraints (OF12).
        # Legacy icoFoam (applications/legacy) does NOT load fvModels and therefore
        # silently ignores the fixedValueConstraint — the obstacle is never enforced.
        app_line = "application     foamRun;\nsolver          incompressibleFluid;"
    else:
        app_line = "application     icoFoam;"
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
{app_line}
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.10g};
deltaT          {dt:.10g};
writeControl    timeStep;
writeInterval   {n_steps};
purgeWrite      1;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable false;
"""
    (system_dir / "controlDict").write_text(content)


def _write_fv_schemes(system_dir: Path) -> None:
    content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""
    (system_dir / "fvSchemes").write_text(content)


def _write_fv_solution(system_dir: Path, *, use_pimple: bool = False) -> None:
    # foamRun -solver incompressibleFluid uses PIMPLE; legacy icoFoam uses PISO.
    # PIMPLE (foamRun incompressibleFluid) also requires a UFinal solver entry
    # because the PIMPLE outer-corrector loop solves for UFinal at the last corrector.
    loop_dict = "PIMPLE" if use_pimple else "PISO"
    if use_pimple:
        u_solver = """\
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
    UFinal { $U; }"""
    else:
        u_solver = """\
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }"""
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }}
    pFinal {{ $p; relTol 0; }}
{u_solver}
}}
{loop_dict}
{{
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;
    pRefCell                 0;
    pRefValue                0;
}}
"""
    (system_dir / "fvSolution").write_text(content)


def _write_transport_properties(constant_dir: Path, nu: float) -> None:
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      physicalProperties;
}}
viscosityModel  constant;
nu              {nu:.10g};
"""
    (constant_dir / "physicalProperties").write_text(content)


def _write_turbulence_properties(constant_dir: Path) -> None:
    content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType  laminar;
"""
    (constant_dir / "turbulenceProperties").write_text(content)


def _write_initial_u(case_dir: Path, v0: np.ndarray) -> None:
    """Write 0/U from v0 as a nonuniform OpenFOAM internalField.

    2-D: v0 shape (N, N, 1, 2) — cell ordering ix + iy*N (x fastest).
    3-D: v0 shape (N, N, N, 3) — cell ordering ix + iy*N + iz*N*N (x fastest).
    """
    ndim = v0.shape[-1]
    N = v0.shape[0]

    if ndim == 2:
        n_cells = N * N
        vel = v0[:, :, 0, :]  # (N, N, 2)
        # Transpose so iy is outer (C-order ravel → ix + iy*N)
        vx = vel[:, :, 0].T.ravel()
        vy = vel[:, :, 1].T.ravel()
        lines = "\n".join(f"({vx[k]:.8g} {vy[k]:.8g} 0)" for k in range(n_cells))
        boundary = """\
    x_lo  { type cyclic; value $internalField; }
    x_hi  { type cyclic; value $internalField; }
    y_lo  { type cyclic; value $internalField; }
    y_hi  { type cyclic; value $internalField; }
    front { type empty; }
    back  { type empty; }"""
    else:
        n_cells = N * N * N
        vel = v0  # (N, N, N, 3)
        # Cell ordering: ix + iy*N + iz*N²  (x fastest → transpose to (nz,ny,nx) for C-ravel)
        vx = vel[:, :, :, 0].transpose(2, 1, 0).ravel()
        vy = vel[:, :, :, 1].transpose(2, 1, 0).ravel()
        vz = vel[:, :, :, 2].transpose(2, 1, 0).ravel()
        lines = "\n".join(
            f"({vx[k]:.8g} {vy[k]:.8g} {vz[k]:.8g})" for k in range(n_cells)
        )
        boundary = """\
    x_lo  { type cyclic; value $internalField; }
    x_hi  { type cyclic; value $internalField; }
    y_lo  { type cyclic; value $internalField; }
    y_hi  { type cyclic; value $internalField; }
    z_lo  { type cyclic; value $internalField; }
    z_hi  { type cyclic; value $internalField; }"""

    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
{n_cells}
(
{lines}
)
;

boundaryField
{{
{boundary}
}}
"""
    (case_dir / "0" / "U").write_text(content)


def _write_initial_p(case_dir: Path, ndim: int) -> None:
    if ndim == 2:
        boundary = """\
    x_lo  { type cyclic; }
    x_hi  { type cyclic; }
    y_lo  { type cyclic; }
    y_hi  { type cyclic; }
    front { type empty; }
    back  { type empty; }"""
    else:
        boundary = """\
    x_lo  { type cyclic; }
    x_hi  { type cyclic; }
    y_lo  { type cyclic; }
    y_hi  { type cyclic; }
    z_lo  { type cyclic; }
    z_hi  { type cyclic; }"""

    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{boundary}
}}
"""
    (case_dir / "0" / "p").write_text(content)


# ---------------------------------------------------------------------------
# Channel-mode case writers
# ---------------------------------------------------------------------------


def _write_block_mesh_dict_channel(system_dir: Path, N: int, L: float) -> None:
    """BlockMeshDict for a 2-D channel: INLET/OUTLET in x, cyclic y, empty z.

    Vertex numbering matches the periodic case (same hex block):
      0=(0,0,0)  1=(L,0,0)  2=(L,L,0)  3=(0,L,0)
      4=(0,0,dz) 5=(L,0,dz) 6=(L,L,dz) 7=(0,L,dz)

    Patch face winding (right-hand normal pointing outward):
      INLET   (x=0) : (0 3 7 4)  → normal in -x direction; reversed so OF sees +outward
      OUTLET  (x=L) : (1 5 6 2)  → normal in +x direction
      WALL_BOT(y=0) : (0 4 5 1)  → normal in -y direction
      WALL_TOP(y=L) : (3 2 6 7)  → normal in +y direction
      FRONT   (z=0) : (0 1 2 3)  → normal in -z direction
      BACK    (z=dz): (4 7 6 5)  → normal in +z direction

    WALL_BOT / WALL_TOP are declared as cyclic neighbours so the y-direction
    is periodic, consistent with other solvers.
    """
    dz = L / N
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
scale   1;

vertices
(
    (0 0 0)        // 0
    ({L} 0 0)      // 1
    ({L} {L} 0)    // 2
    (0 {L} 0)      // 3
    (0 0 {dz})     // 4
    ({L} 0 {dz})   // 5
    ({L} {L} {dz}) // 6
    (0 {L} {dz})   // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N} {N} 1) simpleGrading (1 1 1)
);

boundary
(
    INLET
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    OUTLET
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    WALL_BOT
    {{
        type cyclic;
        neighbourPatch WALL_TOP;
        faces ((0 1 5 4));
    }}
    WALL_TOP
    {{
        type cyclic;
        neighbourPatch WALL_BOT;
        faces ((3 7 6 2));
    }}
    FRONT
    {{
        type empty;
        faces ((0 3 2 1));
    }}
    BACK
    {{
        type empty;
        faces ((4 5 6 7));
    }}
);
"""
    (system_dir / "blockMeshDict").write_text(content)


def _write_initial_u_channel(case_dir: Path, v0: np.ndarray, inflow_vel: float) -> None:
    """Write 0/U for channel mode (2-D only).

    Internal field is taken from v0 (same x-fastest ordering as periodic case).
    Boundary conditions:
      INLET    — fixedValue at inflow_vel
      OUTLET   — inletOutlet (prevents back-flow)
      WALL_BOT — cyclic
      WALL_TOP — cyclic
      FRONT    — empty
      BACK     — empty
    """
    N = v0.shape[0]
    n_cells = N * N
    vel = v0[:, :, 0, :]  # (N, N, 2)
    vx = vel[:, :, 0].T.ravel()
    vy = vel[:, :, 1].T.ravel()
    lines = "\n".join(f"({vx[k]:.8g} {vy[k]:.8g} 0)" for k in range(n_cells))

    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
{n_cells}
(
{lines}
)
;

boundaryField
{{
    INLET    {{ type fixedValue; value uniform ({inflow_vel:.8g} 0 0); }}
    OUTLET   {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
    WALL_BOT {{ type cyclic; }}
    WALL_TOP {{ type cyclic; }}
    FRONT    {{ type empty; }}
    BACK     {{ type empty; }}
}}
"""
    (case_dir / "0" / "U").write_text(content)


def _write_initial_p_channel(case_dir: Path) -> None:
    """Write 0/p for channel mode.

    INLET  — zeroGradient (pressure free at inlet)
    OUTLET — fixedValue 0  (reference pressure)
    WALL_BOT/TOP — cyclic
    FRONT/BACK — empty
    """
    content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    INLET    { type zeroGradient; }
    OUTLET   { type fixedValue; value uniform 0; }
    WALL_BOT { type cyclic; }
    WALL_TOP { type cyclic; }
    FRONT    { type empty; }
    BACK     { type empty; }
}
"""
    (case_dir / "0" / "p").write_text(content)


def _write_topo_set_dict(system_dir: Path, cx: float, cy: float, r: float) -> None:
    """Write system/topoSetDict to select cylinder cells as 'obstacleZone'.

    ``cx``, ``cy``, ``r`` must be in physical (mesh) coordinates, not normalized.
    The cylinder axis runs in the z-direction through (cx, cy).
    We use z-extents beyond the mesh so all cells on the axis are captured.
    """
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
actions
(
    {{
        name    obstacleZone;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        point1  ({cx:.8g} {cy:.8g} -1.0);
        point2  ({cx:.8g} {cy:.8g}  2.0);
        radius  {r:.8g};
    }}
);
"""
    (system_dir / "topoSetDict").write_text(content)


def _write_fv_constraints(system_dir: Path) -> None:
    """Write system/fvConstraints with fixedValue constraint on 'obstacleZone'.

    OpenFOAM 12 (openfoam.org) separates fvModels (source terms) from
    fvConstraints (hard constraints).  The correct file is ``system/fvConstraints``
    (NOT ``constant/fvModels``; also NOT ``constant/fvConstraints`` — OF12 reads
    fvConstraints from ``system/``, matching tutorials like channel395 and ductSecondaryFlow).

    The OF12 ``fixedValue`` fvConstraint (TypeName "fixedValue", not the longer
    "fixedValueConstraint") enforces U = (0 0 0) on all cells in the obstacleZone
    cellSet at every time step by calling ``fvConstraints().constrain(UEqn)`` inside
    the PIMPLE loop.

    Previous attempt wrote ``constant/fvModels`` with ``type fixedValueConstraint``
    and flat ``field``/``value`` keys.  Both were wrong:
      1. ``fixedValueConstraint`` is not a registered fvModel type (it is an fvConstraint);
         OF12 throws "Unknown fvModel fixedValueConstraint" and aborts.
      2. Even if the type name were correct, ``fixedValueConstraint`` reads its field
         values from a ``fieldValues {}`` subdictionary, not flat keys — the flat-key
         format would have triggered ``FatalIOError`` on ``subDict("fieldValues")``.
      3. Legacy ``icoFoam`` (applications/legacy) does NOT call ``fvConstraints`` at all
         and ignores the file entirely regardless of its content — the channel solver
         must use ``foamRun -solver incompressibleFluid`` which does call
         ``fvConstraints().constrain()`` in the PIMPLE loop.
    """
    content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvConstraints;
}
brinkman
{
    type            fixedValue;
    active          yes;
    select          cellSet;
    cellSet         obstacleZone;
    fieldValues
    {
        U           uniform (0 0 0);
    }
}
"""
    (system_dir / "fvConstraints").write_text(content)


def _run_openfoam_channel(
    v0: np.ndarray,
    inflow_vel: float,
    nu: float,
    dt: float,
    end_time: float,
    N: int,
    L: float,
    obstacle_dict: dict | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run foamRun (incompressibleFluid) in channel mode (2-D only) and return (velocity, pressure).

    Steps:
      1. Write blockMeshDict (INLET/OUTLET/cyclic-y/empty-z) + run blockMesh
      2. If obstacle present: write topoSetDict + run topoSet to create cellSet,
         then write system/fvConstraints with fixedValue constraint (OF12 no-slip enforcement)
      3. Write controlDict / fvSchemes / fvSolution / physicalProperties / turbulenceProperties
      4. Write initial U (channel BCs) and p (channel BCs)
      5. Run foamRun -solver incompressibleFluid
      6. Read and return velocity and pressure fields

    NOTE: channel mode uses ``foamRun -solver incompressibleFluid`` (OF12 modern path),
    NOT legacy ``icoFoam``.  The modern solver calls ``fvConstraints().constrain()`` at
    every time step, which is required for ``fixedValue`` (fvConstraints) to
    enforce no-slip on the obstacle cellSet.  Legacy ``icoFoam`` does not load fvConstraints
    at all and silently ignores those files, producing unconstrained flow.
    The modern solver requires PIMPLE (not PISO) in fvSolution, and a UFinal solver entry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        (workdir / "0").mkdir(parents=True)
        (workdir / "constant").mkdir()
        (workdir / "system").mkdir()

        # Solver config (must be written before blockMesh, which requires controlDict)
        # Channel mode: use foamRun + incompressibleFluid (supports fvModels/fvConstraints).
        _write_control_dict(workdir / "system", dt, end_time, use_foam_run=True)

        # Mesh
        _write_block_mesh_dict_channel(workdir / "system", N, L)
        _run_of("blockMesh", workdir)

        # Obstacle cellSet + no-slip constraint
        if obstacle_dict is not None:
            cx = obstacle_dict["center"][0] * L
            cy = obstacle_dict["center"][1] * L
            r = obstacle_dict["radius"] * L
            _write_topo_set_dict(workdir / "system", cx, cy, r)
            _run_of("topoSet", workdir)
            _write_fv_constraints(workdir / "system")
        _write_fv_schemes(workdir / "system")
        # Channel mode: use PIMPLE (required by foamRun incompressibleFluid).
        _write_fv_solution(workdir / "system", use_pimple=True)
        _write_transport_properties(workdir / "constant", nu)
        _write_turbulence_properties(workdir / "constant")

        # Initial conditions
        _write_initial_u_channel(workdir, v0, inflow_vel)
        _write_initial_p_channel(workdir)

        # Solve — foamRun picks up the solver from controlDict.
        _run_of("foamRun", workdir)

        result = _read_final_velocity(workdir, N, ndim=2)
        pressure = _read_final_pressure(workdir, N, ndim=2)

    return result, pressure


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------


def _read_final_pressure(workdir: Path, N: int, ndim: int) -> np.ndarray | None:
    """Read the latest time directory and return pressure as float32 array.

    2-D: shape (N, N)   (same x-fastest ordering as _read_final_velocity)
    3-D: shape (N, N, N)
    Returns None if the pressure field cannot be parsed.
    """
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
        return None
    latest_dir = max(time_dirs, key=lambda x: x[0])[1]
    p_file = latest_dir / "p"
    if not p_file.exists():
        return None
    text = p_file.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\(\s*(.*?)\s*\)\s*;",
        text,
        re.DOTALL,
    )
    if not m:
        # Try uniform
        mu = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", text)
        if mu:
            val = float(mu.group(1))
            if ndim == 2:
                return np.full((N, N), val, dtype=np.float32)
            return np.full((N, N, N), val, dtype=np.float32)
        return None
    rows = m.group(1).split()
    n_cells = N**ndim
    if len(rows) != n_cells:
        return None
    flat = np.array([float(r) for r in rows], dtype=np.float32)
    if ndim == 2:
        # Same C-order reshape as velocity: [iy, ix] → [ix, iy]
        return flat.reshape(N, N).T.copy()
    else:
        return flat.reshape(N, N, N).transpose(2, 1, 0).copy()


def _compute_drag_numpy_of(
    ux: np.ndarray,
    pressure: np.ndarray,
    obstacle: dict | None,
    viscosity: float,
    domain_extent: float,
) -> np.ndarray | None:
    """Compute x-direction drag on obstacle via discrete surface integral (OpenFOAM).

    Args:
        ux:          x-velocity (N, N), collocated.
        pressure:    pressure field (N, N).
        obstacle:    obstacle dict or None.
        viscosity:   kinematic viscosity ν.
        domain_extent: side length of domain.

    Returns:
        shape (1,) float32 or None.
    """
    if obstacle is None or not obstacle.get("shape"):
        return None
    N = ux.shape[0]
    cx = obstacle["center"][0] * N
    cy = obstacle["center"][1] * N
    r = obstacle["radius"] * N
    dx = domain_extent / N

    x = np.arange(N, dtype=np.float32)
    y = np.arange(N, dtype=np.float32)
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


def _read_final_velocity(workdir: Path, N: int, ndim: int) -> np.ndarray:
    """Read the latest time directory and return velocity as v0-shaped float32.

    2-D cell ordering: cell = ix + iy*N  → flat[k]: ix = k%N, iy = k//N
    3-D cell ordering: cell = ix + iy*N + iz*N² → flat[k]: ix=k%N, iy=(k//N)%N, iz=k//(N²)
    """
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

    latest_dir = max(time_dirs, key=lambda x: x[0])[1]
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
    """Run OpenFOAM icoFoam and return the final velocity and drag."""
    if inputs.inflow_profile is not None:
        raise NotImplementedError(
            "openfoam does not support inflow_profile. "
            "Use jax-cfd, phiflow, xlb, or lettuce for inflow-profile optimisation."
        )

    nu = float(inputs.viscosity[0])
    dt = float(inputs.dt[0])
    N = inputs.v0.shape[0]
    ndim = inputs.v0.shape[-1]
    L = float(inputs.domain_extent)
    end_time = inputs.steps * dt
    v0 = np.asarray(inputs.v0, dtype=np.float32)

    # Obstacle dict for drag computation and channel-mode dispatch
    obs = inputs.obstacle
    obstacle_dict = None
    if obs is not None:
        obstacle_dict = {
            "shape": obs.shape.value,
            "center": list(obs.center),
            "radius": float(obs.radius) if obs.radius is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # Channel mode: activated when an obstacle is specified (2-D only).
    # Uses INLET/OUTLET BCs in x, cyclic y, Brinkman penalization.
    # ------------------------------------------------------------------
    if obstacle_dict is not None and ndim == 2:
        # Derive inflow velocity as the mean x-velocity at the left column of v0.
        inflow_vel = float(v0[:, 0, 0, 0].mean())

        result, pressure = _run_openfoam_channel(
            v0=v0,
            inflow_vel=inflow_vel,
            nu=nu,
            dt=dt,
            end_time=end_time,
            N=N,
            L=L,
            obstacle_dict=obstacle_dict,
        )

        ux = result[:, :, 0, 0]  # (N, N)
        p = pressure if pressure is not None else np.zeros((N, N), dtype=np.float32)
        drag = _compute_drag_numpy_of(ux, p, obstacle_dict, nu, L)

        out = {"result": result}
        out["drag"] = drag if drag is not None else np.zeros((1,), dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # Periodic mode: fully periodic Cartesian domain (original behaviour).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        (workdir / "0").mkdir(parents=True)
        (workdir / "constant").mkdir()
        (workdir / "system").mkdir()

        _write_block_mesh_dict(workdir / "system", N, L, ndim)
        _write_control_dict(workdir / "system", dt, end_time)
        _write_fv_schemes(workdir / "system")
        _write_fv_solution(workdir / "system")
        _write_transport_properties(workdir / "constant", nu)
        _write_turbulence_properties(workdir / "constant")
        _write_initial_u(workdir, v0)
        _write_initial_p(workdir, ndim)

        _run_of("blockMesh", workdir)
        _run_of("icoFoam", workdir)

        result = _read_final_velocity(workdir, N, ndim)

        # Compute drag if there is an obstacle (2-D only).
        # In periodic mode this is a post-hoc surface integral on the periodic field.
        drag = None
        if obstacle_dict is not None and ndim == 2:
            pressure = _read_final_pressure(workdir, N, ndim)
            ux = result[:, :, 0, 0]  # (N, N) from (N, N, 1, 2)
            p = pressure if pressure is not None else np.zeros((N, N), dtype=np.float32)
            drag = _compute_drag_numpy_of(ux, p, obstacle_dict, nu, L)

    out = {"result": result}
    out["drag"] = drag if drag is not None else np.zeros((1,), dtype=np.float32)
    return out


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Return output shapes and dtypes without running the solver."""
    v0 = abstract_inputs.v0
    shape = v0["shape"] if isinstance(v0, dict) else tuple(v0.shape)
    return {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
