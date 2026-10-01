# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFOAM-AD tesseract: icoFoam with a reverse-mode discrete adjoint.

Runs icoFoamADR, a PISO transient laminar solver built against DAFoam's
OpenFOAM-AD (OpenFOAM v2506). The whole time loop is taped, so the VJP
is the exact discrete adjoint of the final velocity field with respect
to the initial condition.

Periodic mode only: N*N (2-D) or N*N*N (3-D) Cartesian grid with cyclic
patches. Obstacle and inflow-profile cases are not supported.

Two deviations from stock icoFoam, both applied to the forward path as well
so that apply and the VJP describe the same discrete map:
  - GAMG for pressure. PCG diverges in the AD build.
  - The Rhie-Chow time correction uses a constant coefficient in place of
    fvcDdtPhiCoeff's limiter, which is non-smooth at the scale of a
    finite-difference step and so puts AD and FD 0.6 to 14 percent apart along
    random directions. See src/icoFoamADR/icoFoamADR.C.
"""

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
from mosaic_shared.schema_types import make_differentiable

_OF_BASHRC = "/home/dafoamuser/dafoam/OpenFOAM/OpenFOAM-AD/etc/bashrc"


class InputSchema(make_differentiable(_CanonicalInputSchema, ["v0"])):
    """OpenFOAM-AD Navier-Stokes inputs with a differentiable initial condition."""


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    """OpenFOAM-AD Navier-Stokes outputs with a differentiable velocity field."""


def _run_of(cmd: str, cwd: Path) -> None:
    """Run an OpenFOAM command with the OpenFOAM-AD environment sourced."""
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


def _write_block_mesh_dict(system_dir: Path, N: int, L: float, ndim: int) -> None:
    """Write blockMeshDict for an N**ndim Cartesian periodic grid."""
    if ndim == 2:
        dz = L / N
        vertices = (
            f"    (0 0 0)\n    ({L} 0 0)\n    ({L} {L} 0)\n    (0 {L} 0)\n"
            f"    (0 0 {dz})\n    ({L} 0 {dz})\n    ({L} {L} {dz})\n    (0 {L} {dz})"
        )
        blocks = f"hex (0 1 2 3 4 5 6 7) ({N} {N} 1) simpleGrading (1 1 1)"
        boundary = """\
    x_lo { type cyclic; neighbourPatch x_hi; faces ((0 4 7 3)); }
    x_hi { type cyclic; neighbourPatch x_lo; faces ((1 2 6 5)); }
    y_lo { type cyclic; neighbourPatch y_hi; faces ((0 1 5 4)); }
    y_hi { type cyclic; neighbourPatch y_lo; faces ((3 7 6 2)); }
    front { type empty; faces ((0 3 2 1)); }
    back  { type empty; faces ((4 5 6 7)); }"""
    else:
        vertices = (
            f"    (0 0 0)\n    ({L} 0 0)\n    ({L} {L} 0)\n    (0 {L} 0)\n"
            f"    (0 0 {L})\n    ({L} 0 {L})\n    ({L} {L} {L})\n    (0 {L} {L})"
        )
        blocks = f"hex (0 1 2 3 4 5 6 7) ({N} {N} {N}) simpleGrading (1 1 1)"
        boundary = """\
    x_lo { type cyclic; neighbourPatch x_hi; faces ((0 4 7 3)); }
    x_hi { type cyclic; neighbourPatch x_lo; faces ((1 2 6 5)); }
    y_lo { type cyclic; neighbourPatch y_hi; faces ((0 1 5 4)); }
    y_hi { type cyclic; neighbourPatch y_lo; faces ((3 7 6 2)); }
    z_lo { type cyclic; neighbourPatch z_hi; faces ((0 3 2 1)); }
    z_hi { type cyclic; neighbourPatch z_lo; faces ((4 5 6 7)); }"""

    (system_dir / "blockMeshDict").write_text(
        f"""\
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
{vertices}
);

blocks
(
    {blocks}
);

edges ();

boundary
(
{boundary}
);

mergePatchPairs ();
"""
    )


def _write_control_dict(system_dir: Path, dt: float, steps: int) -> None:
    (system_dir / "controlDict").write_text(
        f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     icoFoamADR;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {steps * dt:.10g};
deltaT          {dt:.10g};
writeControl    timeStep;
writeInterval   {steps};
purgeWrite      1;
writeFormat     ascii;
writePrecision  16;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable false;
"""
    )


def _write_fv_schemes(system_dir: Path) -> None:
    (system_dir / "fvSchemes").write_text(
        """\
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
    )


def _write_fv_solution(system_dir: Path, p_tol: float, u_tol: float) -> None:
    """Write fvSolution with GAMG for pressure; PCG diverges in the AD build."""
    (system_dir / "fvSolution").write_text(
        f"""\
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
        solver          GAMG;
        smoother        DIC;
        tolerance       {p_tol:.10g};
        relTol          0;
        nCellsInCoarsestLevel 1;
    }}
    pFinal {{ $p; }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       {u_tol:.10g};
        relTol          0;
    }}
}}
PISO
{{
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;
    pRefCell                 0;
    pRefValue                0;
}}
"""
    )


def _write_transport_properties(constant_dir: Path, nu: float) -> None:
    (constant_dir / "transportProperties").write_text(
        f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
nu              {nu:.10g};
"""
    )


def _cell_order(field: np.ndarray) -> np.ndarray:
    """Flatten a v0-shaped array to OpenFOAM cell order (x fastest), (n_cells, 3)."""
    ndim = field.shape[-1]
    N = field.shape[0]
    out = np.zeros((N**ndim, 3), dtype=np.float64)
    if ndim == 2:
        vel = field[:, :, 0, :]
        out[:, 0] = vel[:, :, 0].T.ravel()
        out[:, 1] = vel[:, :, 1].T.ravel()
    else:
        for c in range(3):
            out[:, c] = field[:, :, :, c].transpose(2, 1, 0).ravel()
    return out


def _from_cell_order(flat: np.ndarray, N: int, ndim: int) -> np.ndarray:
    """Inverse of _cell_order: (n_cells, 3) back to v0 shape, float32."""
    flat = np.asarray(flat, dtype=np.float32)
    if ndim == 2:
        vel = flat.reshape(N, N, 3).transpose(1, 0, 2)
        return vel[:, :, None, :2].copy()
    vel = flat.reshape(N, N, N, 3).transpose(2, 1, 0, 3)
    return vel.copy()


def _write_initial_u(case_dir: Path, v0: np.ndarray) -> None:
    ndim = v0.shape[-1]
    rows = _cell_order(v0)
    lines = "\n".join(f"({r[0]:.17g} {r[1]:.17g} {r[2]:.17g})" for r in rows)
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

    (case_dir / "0" / "U").write_text(
        f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
{len(rows)}
(
{lines}
)
;

boundaryField
{{
{boundary}
}}
"""
    )


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

    (case_dir / "0" / "p").write_text(
        f"""\
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
    )


def _read_final_velocity(workdir: Path, N: int, ndim: int) -> np.ndarray:
    """Parse the latest written time directory into a v0-shaped float32 array."""
    times = []
    for d in workdir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            if t > 0:
                times.append((t, d))
    if not times:
        raise RuntimeError(f"no output time directories in {workdir}")

    text = (max(times, key=lambda x: x[0])[1] / "U").read_text()
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\(\s*(.*?)\s*\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not parse internalField in {workdir}")

    rows = re.findall(r"\(([^)]+)\)", match.group(1))
    if len(rows) != N**ndim:
        raise RuntimeError(f"expected {N**ndim} rows, got {len(rows)}")

    flat = np.array([[float(x) for x in r.split()] for r in rows])
    return _from_cell_order(flat, N, ndim)


def _reject_unsupported(inputs: InputSchema) -> None:
    if inputs.inflow_profile is not None:
        raise NotImplementedError(
            "openfoam-ad is periodic-only and does not support inflow_profile"
        )
    if inputs.obstacle is not None:
        raise NotImplementedError(
            "openfoam-ad is periodic-only and does not support obstacles"
        )


def _build_case(workdir: Path, inputs: InputSchema, v0: np.ndarray) -> None:
    (workdir / "0").mkdir(parents=True)
    (workdir / "constant").mkdir()
    (workdir / "system").mkdir()

    N = v0.shape[0]
    ndim = v0.shape[-1]
    _write_block_mesh_dict(workdir / "system", N, float(inputs.domain_extent), ndim)
    _write_control_dict(workdir / "system", float(inputs.dt[0]), int(inputs.steps))
    _write_fv_schemes(workdir / "system")
    # The finite-difference reference in gradient/fd_check is only as accurate
    # as the forward map, so the linear solves are tightened
    _write_fv_solution(workdir / "system", p_tol=1e-12, u_tol=1e-12)
    _write_transport_properties(workdir / "constant", float(inputs.viscosity[0]))
    _write_initial_u(workdir, v0)
    _write_initial_p(workdir, ndim)

    _run_of("blockMesh", workdir)


def apply(inputs: InputSchema) -> OutputSchema:
    """Run icoFoamADR and return the final velocity field."""
    _reject_unsupported(inputs)
    v0 = np.asarray(inputs.v0, dtype=np.float64)
    N, ndim = v0.shape[0], v0.shape[-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        _build_case(workdir, inputs, v0)
        _run_of("icoFoamADR", workdir)
        result = _read_final_velocity(workdir, N, ndim)

    return {"result": result, "drag": np.zeros((1,), dtype=np.float32)}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode discrete adjoint of the final velocity w.r.t. v0."""
    _reject_unsupported(inputs)
    if vjp_inputs != {"v0"}:
        raise NotImplementedError(
            f"openfoam-ad differentiates only v0, got {sorted(vjp_inputs)}"
        )
    if "result" not in vjp_outputs:
        raise NotImplementedError(
            f"openfoam-ad differentiates only result, got {sorted(vjp_outputs)}"
        )

    v0 = np.asarray(inputs.v0, dtype=np.float64)
    N, ndim = v0.shape[0], v0.shape[-1]
    cotangent = np.asarray(cotangent_vector["result"], dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "case"
        _build_case(workdir, inputs, v0)

        rows = _cell_order(cotangent)
        (workdir / "cotangent.dat").write_text(
            "\n".join(f"{r[0]:.17g} {r[1]:.17g} {r[2]:.17g}" for r in rows) + "\n"
        )

        _run_of("icoFoamADR -vjp", workdir)

        values = np.fromstring(
            (workdir / "gradient.dat").read_text(), sep="\n", dtype=np.float64
        )
        if values.size != 3 * N**ndim:
            raise RuntimeError(
                f"expected {3 * N**ndim} gradient entries, got {values.size}"
            )

    return {"v0": _from_cell_order(values.reshape(-1, 3), N, ndim)}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Return output shapes and dtypes without running the solver."""
    v0 = abstract_inputs.v0
    shape = v0["shape"] if isinstance(v0, dict) else tuple(v0.shape)
    return {
        "result": {"shape": shape, "dtype": "float32"},
        "drag": {"shape": (1,), "dtype": "float32"},
    }
