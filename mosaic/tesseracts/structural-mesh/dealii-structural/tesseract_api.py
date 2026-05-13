"""Linear elasticity topology optimisation on a structured hexahedral mesh.

Uses deal.II Q1 finite elements (Step-8 pattern) as a C++ subprocess.
Python writes JSON + rho.npy to a tempdir, runs the compiled struct_solver
binary, and reads back compliance.txt.  Forward-only.

SIMP stiffness:
    E(ρ) = xmin·E_max + (1−xmin)·E_max·ρ^penal

Objective:
    C = F^T U  (structural compliance)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pydantic import Field
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)

# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

_DEALII_SOLVER = os.environ.get(
    "DEALII_STRUCT_SOLVER", "/opt/dealii_struct/build/struct_solver"
)


# ---------------------------------------------------------------------------
# Schema (subclass with material parameters; forward-only — no Differentiable)
# ---------------------------------------------------------------------------


class InputSchema(_CanonicalInputSchema):
    """Inputs for deal.II structural solver, extended with SIMP material parameters."""

    E_max: float = Field(
        default=70000.0,
        description="Young's modulus of the fully solid material [MPa].",
    )
    nu: float = Field(
        default=0.3,
        description="Poisson's ratio.",
    )
    xmin: float = Field(
        default=1e-3,
        description="Void stiffness ratio (E_min = xmin * E_max).",
    )
    penal: float = Field(
        default=3.0,
        description="SIMP penalisation exponent (E(ρ) = E_min + (E_max−E_min)·ρ^penal).",
    )


class OutputSchema(_CanonicalOutputSchema):
    pass


# ---------------------------------------------------------------------------
# Mesh helper: infer nx, ny, nz from HexMesh
# ---------------------------------------------------------------------------


def _infer_grid_dims(
    inputs: InputSchema,
) -> tuple[int, int, int, float, float, float]:  # mosaic:io
    """Infer structured grid dimensions from the HexMesh point array.

    The benchmark always builds a structured hex mesh from
    ``np.linspace(0, L, n+1)`` in each direction.  Given the unique
    coordinate counts we can recover nx, ny, nz exactly.

    Returns:
        (nx, ny, nz, Lx, Ly, Lz)
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float32)

    xs = np.unique(np.round(pts[:, 0], 6))
    ys = np.unique(np.round(pts[:, 1], 6))
    zs = np.unique(np.round(pts[:, 2], 6))

    nx = len(xs) - 1
    ny = len(ys) - 1
    nz = len(zs) - 1

    Lx = float(xs[-1] - xs[0])
    Ly = float(ys[-1] - ys[0])
    Lz = float(zs[-1] - zs[0])

    return nx, ny, nz, Lx, Ly, Lz


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_inputs(inputs: InputSchema, wd: Path) -> None:  # mosaic:io
    """Serialise inputs to ``input.json`` and ``rho.npy`` in *wd*."""
    nx, ny, nz, Lx, Ly, Lz = _infer_grid_dims(inputs)

    hm = inputs.hex_mesh
    bc = inputs.boundary_conditions

    # Active density slice
    rho_active = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float32)
    np.save(str(wd / "rho.npy"), rho_active)

    # Dirichlet BC
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int32)
    dv = (
        np.asarray(bc.dirichlet.values, dtype=np.float32)
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 3), dtype=np.float32)
    )

    # Neumann BC
    nm = np.asarray(bc.neumann.mask if bc.neumann else [], dtype=np.int32)
    nv = (
        np.asarray(bc.neumann.values, dtype=np.float32)
        if bc.neumann
        else np.zeros((0, 3), dtype=np.float32)
    )

    payload = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "E_max": float(inputs.E_max),
        "nu": float(inputs.nu),
        "xmin": float(inputs.xmin),
        "penal": float(inputs.penal),
        "rho_file": "rho.npy",
        "dirichlet_mask": dm.tolist(),
        "dirichlet_values": dv.tolist(),
        "neumann_mask": nm.tolist(),
        "neumann_values": nv.tolist(),
    }

    with open(wd / "input.json", "w") as f:
        json.dump(payload, f)


def _run_solver(wd: Path) -> None:  # mosaic:physics
    """Invoke the deal.II struct_solver binary."""
    cmd = [_DEALII_SOLVER, str(wd / "input.json")]
    result = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II struct_solver failed:\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )


def _parse_outputs(wd: Path) -> OutputSchema:  # mosaic:io
    """Read compliance.txt written by the C++ solver."""
    with open(wd / "compliance.txt") as f:
        compliance = float(f.read().strip())
    return OutputSchema(compliance=np.float32(compliance))


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve linear elasticity and return compliance.

    Args:
        inputs: Validated InputSchema with density field, mesh, BCs, material params.

    Returns:
        OutputSchema with compliance (scalar).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        _write_inputs(inputs, wd)
        _run_solver(wd)
        return _parse_outputs(wd)


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
