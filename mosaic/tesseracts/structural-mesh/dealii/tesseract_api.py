"""Linear elasticity topology optimisation on a structured hexahedral mesh.

Uses deal.II Q1 finite elements (Step-8 pattern) as a C++ subprocess.
Python writes JSON + rho.npy to a tempdir, runs the compiled struct_solver
binary, and reads back displacement.npy, von_mises.npy, compliance.txt
(plus gradient.npy for the VJP).

SIMP stiffness:
    E(ρ) = xmin·E_max + (1−xmin)·E_max·ρ^penal

Objective:
    C = F^T U  (structural compliance)

Gradient (analytic, self-adjoint):
    dC/dρ_e = −(1−xmin)·E_max·penal·ρ_e^(penal−1) / E_e · local_compliance_e
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mosaic_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from pydantic import Field
from tesseract_core.runtime import ShapeDType

# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

_DEALII_SOLVER = os.environ.get(
    "DEALII_STRUCT_SOLVER", "/opt/dealii_struct/build/struct_solver"
)


# ---------------------------------------------------------------------------
# Schema (subclass with material parameters)
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
    """Outputs for deal.II structural solver (canonical interface)."""


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


def _run_solver(  # mosaic:physics
    wd: Path,
    compute_gradient: bool = False,
    compute_disp_gradient: bool = False,
) -> None:
    """Invoke the deal.II struct_solver binary."""
    cmd = [_DEALII_SOLVER, str(wd / "input.json")]
    if compute_gradient:
        cmd.append("--gradient")
    if compute_disp_gradient:
        cmd.append("--disp-gradient")
    result = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II struct_solver failed:\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )


def _parse_outputs(inputs: InputSchema, wd: Path) -> OutputSchema:  # mosaic:io
    """Read displacement.npy, von_mises.npy and compliance.txt written by the C++ solver."""
    displacement = np.load(str(wd / "displacement.npy")).astype(np.float32)
    von_mises = np.load(str(wd / "von_mises.npy")).astype(np.float32)
    with open(wd / "compliance.txt") as f:
        compliance = float(f.read().strip())

    # Pad von_mises to the full rho capacity if needed
    n_rho = len(np.asarray(inputs.rho))
    if len(von_mises) < n_rho:
        vm_full = np.zeros(n_rho, dtype=np.float32)
        vm_full[: len(von_mises)] = von_mises
        von_mises = vm_full

    return OutputSchema(
        compliance=np.float32(compliance),
        von_mises_stress=von_mises,
        displacement=displacement,
    )


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve linear elasticity and return compliance, von Mises, displacement.

    Args:
        inputs: Validated InputSchema with density field, mesh, BCs, material params.

    Returns:
        OutputSchema with compliance (scalar), von_mises_stress (n_cells,),
        and displacement (n_nodes, 3).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        _write_inputs(inputs, wd)
        _run_solver(wd)
        return _parse_outputs(inputs, wd)


def vector_jacobian_product(  # mosaic:grad:rho:analytic
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via analytic SIMP sensitivity: dC/drho and/or d(cotan^T u)/drho.

    Runs the forward solve and, depending on active cotangents:
      - compliance: uses ``--gradient`` flag; C++ writes ``gradient.npy``
        (n_active_cells, analytic dC/drho per cell).
      - displacement: writes ``cotan_disp.npy`` and uses ``--disp-gradient``
        flag; C++ solves K lambda = cotan_disp and writes ``disp_gradient.npy``
        (n_active_cells, analytic d(cotan^T u)/drho per cell).

    Both modes can be active simultaneously; a single forward + factored
    system is used by the C++ binary.

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name -> cotangent scalar/array.

    Returns:
        Dict mapping "rho" -> gradient array matching inputs.rho shape.
    """
    if "rho" not in vjp_inputs:
        return {}

    cot_c = float(cotangent_vector.get("compliance", 0.0))
    cot_disp_raw = cotangent_vector.get("displacement", None)
    cot_disp = (
        np.asarray(cot_disp_raw, dtype=np.float32) if cot_disp_raw is not None else None
    )

    has_compliance = abs(cot_c) > 0.0
    has_displacement = cot_disp is not None and np.any(cot_disp != 0.0)

    hm = inputs.hex_mesh
    n_active = hm.n_faces
    grad_full = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        _write_inputs(inputs, wd)

        if has_displacement:
            # Write cotangent displacement in input node ordering as (n_nodes*3,) flat array.
            # The C++ binary reads cotan_disp.npy as (n_nodes, 3) C-order float32.
            n_nodes = hm.n_points
            cot_flat = cot_disp.reshape(-1)[: n_nodes * 3].astype(np.float32)
            # Pad to full n_nodes*3 if needed
            if len(cot_flat) < n_nodes * 3:
                cot_full = np.zeros(n_nodes * 3, dtype=np.float32)
                cot_full[: len(cot_flat)] = cot_flat
                cot_flat = cot_full
            cot_arr = cot_flat.reshape(n_nodes, 3)
            np.save(str(wd / "cotan_disp.npy"), cot_arr)

        _run_solver(
            wd,
            compute_gradient=has_compliance,
            compute_disp_gradient=has_displacement,
        )

        if has_compliance:
            gradient = np.load(str(wd / "gradient.npy")).astype(np.float32)
            grad_full[:n_active] += (gradient[:n_active] * cot_c).astype(np.float32)

        if has_displacement:
            disp_grad = np.load(str(wd / "disp_gradient.npy")).astype(np.float32)
            grad_full[:n_active] += disp_grad[:n_active].astype(np.float32)

    return {"rho": grad_full}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver.

    Args:
        abstract_inputs: InputSchema with shape/dtype metadata (no values).

    Returns:
        Dict mapping output names to ShapeDType descriptors.
    """
    d = abstract_inputs.model_dump()
    points = d["hex_mesh"]["points"]
    n_nodes = points["shape"][0] if isinstance(points, dict) else len(points)
    faces = d["hex_mesh"]["faces"]
    n_cells = faces["shape"][0] if isinstance(faces, dict) else len(faces)
    return {
        "compliance": ShapeDType(shape=(), dtype="float32"),
        "von_mises_stress": ShapeDType(shape=(n_cells,), dtype="float32"),
        "displacement": ShapeDType(shape=(n_nodes, 3), dtype="float32"),
    }
