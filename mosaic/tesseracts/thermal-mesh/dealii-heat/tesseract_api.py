# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thermal topology optimisation on a structured hexahedral mesh.

Uses deal.II Q1 finite elements as a C++ subprocess.  Python writes JSON +
rho.npy + source.npy to a tempdir, runs the compiled heat_solver binary, and
reads back compliance.txt.  This solver is forward-only — it serves as the
reference solver for ground-truth temperature targets in inverse experiments.

SIMP conductivity:
    k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

Objective:
    C = ∮_Γ_N q_n · T dΓ
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pydantic import Field
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)

# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

_DEALII_SOLVER = os.environ.get(
    "DEALII_HEAT_SOLVER", "/opt/dealii_heat/build/heat_solver"
)


# ---------------------------------------------------------------------------
# Schema (subclass with material parameters; forward-only — no Differentiable)
# ---------------------------------------------------------------------------


class InputSchema(_CanonicalInputSchema):
    """Inputs for deal.II heat solver, extended with SIMP material parameters."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(_CanonicalOutputSchema):
    """Outputs for deal.II heat solver (forward-only; canonical interface)."""


# ---------------------------------------------------------------------------
# Mesh helper: infer nx, ny, nz from HexMesh
# ---------------------------------------------------------------------------


def _infer_grid_dims(
    inputs: InputSchema,
) -> tuple[int, int, int, float, float, float]:  # mosaic:io
    """Infer structured grid dimensions from the HexMesh point array."""
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
    """Serialise inputs to ``input.json``, ``rho.npy``, and ``source.npy`` in *wd*."""
    nx, ny, nz, Lx, Ly, Lz = _infer_grid_dims(inputs)

    hm = inputs.hex_mesh
    bc = inputs.boundary_conditions

    # Active density slice
    rho_active = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float32)
    np.save(str(wd / "rho.npy"), rho_active)

    # Active source slice
    source_active = np.asarray(inputs.source[: hm.n_faces], dtype=np.float32)
    np.save(str(wd / "source.npy"), source_active)

    # Dirichlet BC
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int32)
    dv = (
        np.asarray(bc.dirichlet.values, dtype=np.float32)
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1), dtype=np.float32)
    )

    # Neumann BC
    nm = np.asarray(bc.neumann.mask if bc.neumann else [], dtype=np.int32)
    nv = (
        np.asarray(bc.neumann.values, dtype=np.float32)
        if bc.neumann
        else np.zeros((0, 1), dtype=np.float32)
    )

    payload = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "k_max": float(inputs.k_max),
        "p_exp": float(inputs.p_exp),
        "rho_file": "rho.npy",
        "source_file": "source.npy",
        "dirichlet_mask": dm.tolist(),
        "dirichlet_values": dv.tolist(),
        "neumann_mask": nm.tolist(),
        "neumann_values": nv.tolist(),
    }

    with open(wd / "input.json", "w") as f:
        json.dump(payload, f)


def _run_solver(wd: Path) -> None:  # mosaic:physics
    """Invoke the deal.II heat_solver binary."""
    cmd = [_DEALII_SOLVER, str(wd / "input.json")]
    result = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II solver failed:\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )


def _parse_outputs(inputs: InputSchema, wd: Path) -> OutputSchema:  # mosaic:io
    """Read compliance.txt and (if present) temperature.npy to compute id-error."""
    with open(wd / "compliance.txt") as f:
        compliance = float(f.read().strip())

    target_temp = np.asarray(inputs.target_temperature, dtype=np.float32)
    temp_path = wd / "temperature.npy"
    if temp_path.exists() and target_temp.size > 1:
        temperature = np.load(str(temp_path)).astype(np.float32)
        n = min(len(temperature), len(target_temp))
        id_error = np.float32(np.sum((temperature[:n] - target_temp[:n]) ** 2))
    else:
        id_error = np.float32(0.0)

    return OutputSchema(
        thermal_compliance=np.float32(compliance),
        identification_error=id_error,
    )


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return thermal compliance.

    Args:
        inputs: Validated InputSchema with density field, mesh, BCs, material params.

    Returns:
        OutputSchema with thermal_compliance (scalar) and identification_error (scalar).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        _write_inputs(inputs, wd)
        _run_solver(wd)
        return _parse_outputs(inputs, wd)


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
