# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native OpenCFD buoyantBoussinesqPimpleFoam, forward-only comparison solver.

Laminar equations: div(U)=0; DU/Dt=-grad(p)+nu*laplace(U)+b*theta*e_y;
Dtheta/Dt=kappa*laplace(theta). OpenFOAM transports T=Tref+Tscale*theta,
where Tscale=1 kelvin, beta=1e-3/K, and g=(0,-b/(beta*Tscale),0).
The dimensionless product beta*Tscale fixes the density scale; constant gravity is
absorbed in pressure, leaving exactly +b*theta vertically. Pr=nu/kappa.
No radiation or turbulent diffusivity is enabled. Positive nu/kappa required.

The canonical output uses the native pressure-corrected face flux phi/area,
not interpolated cell velocity, and cell-centered T-Tref. Initial cell velocity
is reconstructed from canonical faces. OpenFOAM retains collocated momentum
and pressure internally, so separate calls are NOT an exact recurrent restart.
This adapter is not differentiable and is not used for corrector training.
"""

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mosaic_shared.problems.buoyancy_grid import InputSchema as BaseInput
from mosaic_shared.problems.buoyancy_grid import OutputSchema as BaseOutput
from pydantic import Field, model_validator
from tesseract_core.runtime import Array, Float32, ShapeDType

_BETA = 1e-3
_TREF = 300.0
_TEMPERATURE_SCALE = 1.0
_HEADER = "FoamFile\n{{ version 2.0; format ascii; class {cls}; object {name}; }}\n"


class InputSchema(BaseInput):
    """Constant-property, laminar Boussinesq inputs without derivative support."""

    diffusivity: Array[(1,), Float32] = Field(
        default_factory=lambda: np.array([0.001], np.float32),
        description="Positive scalar diffusivity; the native Prandtl number is nu/kappa.",
    )

    @model_validator(mode="after")
    def _check_transport(self) -> "InputSchema":
        if isinstance(self.state0, (dict, ShapeDType)):
            return self
        if float(self.viscosity[0]) <= 0 or float(self.diffusivity[0]) <= 0:
            raise ValueError(
                "OpenFOAM requires positive viscosity and diffusivity (Pr=nu/kappa)."
            )
        if (
            _BETA * _TEMPERATURE_SCALE * np.max(np.abs(np.asarray(self.state0)[..., 2]))
            >= 0.1
        ):
            raise ValueError(
                "Scalar exceeds the small density-variation range of this Boussinesq mapping."
            )
        return self


class OutputSchema(BaseOutput):
    """Pressure-corrected face velocities and the transported scalar."""


def _run(command: list[str], directory: Path) -> None:
    paths = [
        Path(
            os.environ.get(
                "OPENFOAM_BASHRC", "/usr/lib/openfoam/openfoam2412/etc/bashrc"
            )
        )
    ]
    paths.extend(Path("/opt").glob("openfoam*/etc/bashrc"))
    bashrc = next((path for path in paths if path.is_file()), None)
    if bashrc is None:
        raise RuntimeError("OpenFOAM environment not found; set OPENFOAM_BASHRC.")
    process = subprocess.run(
        [
            "bash",
            "-c",
            f". {shlex.quote(str(bashrc))} >/dev/null && exec {shlex.join(command)}",
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    if process.returncode:
        raise RuntimeError(
            f"{command[0]} failed ({process.returncode}):\n{process.stdout[-8000:]}\n{process.stderr[-2000:]}"
        )


def _dictionary(path: Path, body: str) -> None:
    path.write_text(_HEADER.format(cls="dictionary", name=path.name) + body)


def _write_case(
    directory: Path,
    n: int,
    extent: float,
    nu: float,
    kappa: float,
    buoyancy: float,
    dt: float,
    steps: int,
) -> None:
    system, constant = directory / "system", directory / "constant"
    for part in (system, constant, directory / "0"):
        part.mkdir(parents=True, exist_ok=True)
    h = extent / n
    _dictionary(
        system / "blockMeshDict",
        f"""
convertToMeters 1;
vertices ((0 0 0) ({extent} 0 0) ({extent} {extent} 0) (0 {extent} 0)
          (0 0 {h}) ({extent} 0 {h}) ({extent} {extent} {h}) (0 {extent} {h}));
blocks (hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1 1 1));
edges ();
boundary (
 walls {{ type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2)); }}
 frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""",
    )
    _dictionary(
        system / "controlDict",
        f"""
application buoyantBoussinesqPimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {dt * steps:.17g};
deltaT {dt:.17g}; adjustTimeStep no;
writeControl timeStep; writeInterval {steps}; purgeWrite 1;
writeFormat ascii; writePrecision 17; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
""",
    )
    _dictionary(
        system / "fvSchemes",
        """
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes {
 default none;
 div(phi,U) Gauss linear;
 div(phi,T) Gauss limitedLinear 1;
 div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
fluxRequired { default no; p_rgh; }
""",
    )
    _dictionary(
        system / "fvSolution",
        """
solvers {
 p_rgh { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; }
 p_rghFinal { $p_rgh; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-11; relTol 0; }
 UFinal { $U; }
 T { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-11; relTol 0; }
 TFinal { $T; }
}
PIMPLE { momentumPredictor yes; nOuterCorrectors 2; nCorrectors 2;
 nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
relaxationFactors { equations { U 1; T 1; } }
""",
    )
    _dictionary(
        constant / "transportProperties",
        f"""
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.17g};
beta [0 0 0 -1 0 0 0] {_BETA};
TRef [0 0 0 1 0 0 0] {_TREF};
Pr {nu / kappa:.17g}; Prt 1;
rhoRef 1; CpRef 1;
""",
    )
    _dictionary(constant / "turbulenceProperties", "simulationType laminar;\n")
    _dictionary(constant / "radiationProperties", "radiationModel none;\n")
    (constant / "g").write_text(
        _HEADER.format(cls="uniformDimensionedVectorField", name="g")
        + f"dimensions [0 1 -2 0 0 0 0];\nvalue (0 {-buoyancy / (_BETA * _TEMPERATURE_SCALE):.17g} 0);\n"
    )


def _labels(path: Path) -> np.ndarray:
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(), flags=re.DOTALL)
    match = re.search(r"\n\s*(\d+)\s*\(\s*([^()]*)\)", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Cannot parse ASCII label list {path}.")
    values = np.fromstring(match[2], sep=" ", dtype=np.int64)
    if values.size != int(match[1]):
        raise ValueError(f"Wrong label count in {path}.")
    return values


def _face_map(
    directory: Path, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbor = _labels(directory / "constant/polyMesh/neighbour")
    owner = _labels(directory / "constant/polyMesh/owner")[: len(neighbor)]
    ox, oy = owner % n, owner // n
    nx, ny = neighbor % n, neighbor // n
    x_face = (np.abs(nx - ox) == 1) & (oy == ny)
    y_face = (np.abs(ny - oy) == 1) & (ox == nx)
    if len(neighbor) != 2 * n * (n - 1) or not np.all(x_face | y_face):
        raise ValueError("Unexpected blockMesh cell/face topology.")
    axis = np.where(x_face, 0, 1)
    i, j = np.minimum(ox, nx), np.minimum(oy, ny)
    sign = np.where(x_face, np.sign(nx - ox), np.sign(ny - oy))
    return i, j, axis, sign


def _field(
    path: Path, values: np.ndarray, dimensions: str, wall: str, surface: bool = False
) -> None:
    vector = values.ndim == 2
    kind = "vector" if vector else "scalar"
    cls = ("surface" if surface else "vol") + (
        "VectorField" if vector else "ScalarField"
    )
    rows = (
        "\n".join("(" + " ".join(f"{v:.17g}" for v in row) + ")" for row in values)
        if vector
        else "\n".join(f"{v:.17g}" for v in values)
    )
    path.write_text(
        _HEADER.format(cls=cls, name=path.name)
        + f"dimensions {dimensions};\ninternalField nonuniform List<{kind}>\n"
        + f"{len(values)}\n(\n{rows}\n);\n"
        + f"boundaryField {{ walls {{ {wall} }} frontAndBack {{ type empty; }} }}\n"
    )


def _write_fields(
    directory: Path, state: np.ndarray, extent: float, faces: tuple
) -> None:
    n = state.shape[0]
    ux = np.pad(state[:-1, :, 0], ((1, 1), (0, 0)))
    uy = np.pad(state[:, :-1, 1], ((0, 0), (1, 1)))
    velocity = np.stack(
        [(ux[1:] + ux[:-1]) / 2, (uy[:, 1:] + uy[:, :-1]) / 2, np.zeros((n, n))],
        axis=-1,
    )
    _field(
        directory / "0/U",
        velocity.transpose(1, 0, 2).reshape(-1, 3),
        "[0 1 -1 0 0 0 0]",
        "type fixedValue; value uniform (0 0 0);",
    )
    _field(
        directory / "0/T",
        (state[..., 2].T * _TEMPERATURE_SCALE + _TREF).ravel(),
        "[0 0 0 1 0 0 0]",
        "type zeroGradient;",
    )
    _field(
        directory / "0/p_rgh",
        np.zeros(n * n),
        "[0 2 -2 0 0 0 0]",
        "type fixedFluxPressure; value uniform 0;",
    )
    _field(
        directory / "0/alphat",
        np.zeros(n * n),
        "[0 2 -1 0 0 0 0]",
        "type fixedValue; value uniform 0;",
    )
    i, j, axis, sign = faces
    flux = state[i, j, axis] * sign * (extent / n) ** 2
    _field(
        directory / "0/phi",
        flux,
        "[0 3 -1 0 0 0 0]",
        "type calculated; value uniform 0;",
        surface=True,
    )


def _read_scalar(path: Path, count: int) -> np.ndarray:
    text = path.read_text()
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        return np.full(count, float(uniform[1]))
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(\s*([^()]*)\)\s*;",
        text,
        re.DOTALL,
    )
    if match is None or int(match[1]) != count:
        raise ValueError(f"Invalid scalar field {path}.")
    values = np.fromstring(match[2], sep=" ")
    if values.size != count or not np.isfinite(values).all():
        raise ValueError(f"Invalid scalar values in {path}.")
    return values


def apply(inputs: InputSchema) -> dict[str, Any]:
    """Run native laminar Boussinesq dynamics and return canonical face fluxes."""
    state = np.asarray(inputs.state0, dtype=np.float64)
    n, extent = state.shape[0], inputs.domain_extent
    nu, kappa, buoyancy, dt = (
        float(getattr(inputs, key)[0])
        for key in ("viscosity", "diffusivity", "buoyancy", "dt")
    )
    with tempfile.TemporaryDirectory(prefix="mosaic-buoyancy-openfoam-") as tmp:
        directory = Path(tmp)
        _write_case(directory, n, extent, nu, kappa, buoyancy, dt, inputs.steps)
        _run(["blockMesh"], directory)
        faces = _face_map(directory, n)
        _write_fields(directory, state, extent, faces)
        _run(["buoyantBoussinesqPimpleFoam"], directory)
        times = [
            (float(path.name), path)
            for path in directory.iterdir()
            if path.is_dir() and re.fullmatch(r"[\d.eE+-]+", path.name)
        ]
        time, final = max(times)
        if not np.isclose(time, dt * inputs.steps, rtol=1e-8, atol=1e-12):
            raise RuntimeError(
                f"OpenFOAM stopped at {time}, expected {dt * inputs.steps}."
            )
        scalar = (
            _read_scalar(final / "T", n * n).reshape(n, n).T - _TREF
        ) / _TEMPERATURE_SCALE
        i, j, axis, sign = faces
        flux = _read_scalar(final / "phi", len(i))
        result = np.zeros_like(state)
        result[..., 2] = scalar
        result[i, j, axis] = flux * sign / (extent / n) ** 2
    if not np.isfinite(result).all():
        raise RuntimeError("OpenFOAM returned a nonfinite state.")
    return {"result": result.astype(np.float32)}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Preserve the canonical state shape without starting OpenFOAM."""
    state = abstract_inputs.state0
    shape = tuple(state["shape"]) if isinstance(state, dict) else tuple(state.shape)
    return {"result": {"shape": shape, "dtype": "float32"}}
