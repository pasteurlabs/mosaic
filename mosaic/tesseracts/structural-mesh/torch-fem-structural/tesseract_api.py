# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SIMP topology optimisation of a 3D cantilever via torch-fem.

Uses meyer-nils/torch-fem `Solid` (HEX8) with `IsotropicElasticity3D` and
PyTorch autograd for compliance sensitivities.

SIMP stiffness, applied by scaling the elastic tensor (exact, since `C` is
linear in `E` at fixed `ν`):

    E(ρ) = E_min + (E_max − E_min) · ρ^p,    E_min = xmin · E_max
    C(ρ) = [xmin + (1 − xmin) · ρ^p] · C(E_max, ν)

Objective is the full external work `C = Fᵀu`, with no factor of ½ — matching
the canonical OutputSchema and the other structural-mesh solvers.

Gradients w.r.t. ρ flow through `model.solve(differentiable_parameters=...)`,
which implements the implicit-function-theorem adjoint for the linear system.
"""

from typing import Any

import numpy as np
import torch
from mosaic_shared.problems.structural_mesh import InputSchema as _CanonicalInputSchema
from mosaic_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from tesseract_core.runtime import ShapeDType
from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    """Inputs for the torch-fem structural solver (canonical + SIMP material params)."""

    E_max: float = Field(
        default=70_000.0,
        description="Young's modulus of fully solid material (ρ = 1).",
    )
    nu: float = Field(
        default=0.3,
        description="Poisson's ratio (density-independent).",
    )
    xmin: float = Field(
        default=1e-3,
        description="Void stiffness ratio E_min / E_max.",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p in E(ρ) = E_min + (E_max−E_min)·ρ^p.",
    )


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    """torch-fem structural solver output schema."""


# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------

# torch-fem allocates auxiliary tensors without an explicit ``device=``, so the
# default device must be set globally or they land on CPU and collide with our
# CUDA tensors.
#
# Sparse linear solve: on GPU we use Jacobi-preconditioned CG in torch, so the
# forward and adjoint solves stay on the device. On CPU-only hosts torch-fem
# picks the solver automatically (SciPy).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SOLVE_KWARGS: dict[str, Any] = (
    {"method": "cg", "preconditioner": "jacobi"} if _DEVICE.type == "cuda" else {}
)
_DTYPE = torch.float64

torch.set_default_dtype(_DTYPE)
torch.set_default_device(_DEVICE)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

# Hex faces as node-index quads, for the HEX8 connectivity emitted by
# ``mosaic.benchmarks.problems.shared.mesh.hex_mesh_arrays``:
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
#   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
_HEX_FACES = np.array(
    [
        [0, 1, 2, 3],  # z−
        [4, 5, 6, 7],  # z+
        [0, 1, 5, 4],  # y−
        [1, 2, 6, 5],  # x+
        [2, 3, 7, 6],  # y+
        [0, 3, 7, 4],  # x−
    ],
    dtype=np.int64,
)


def _quad_areas(quad_pts: np.ndarray) -> np.ndarray:
    """Area of each planar quad, split into two triangles. ``quad_pts``: (n, 4, 3)."""
    p0, p1, p2, p3 = quad_pts[:, 0], quad_pts[:, 1], quad_pts[:, 2], quad_pts[:, 3]
    a1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)
    a2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=-1)
    return a1 + a2


def _lump_traction(
    points: np.ndarray,
    cells: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Consistent nodal load vector for a uniform surface traction.

    Every hex face whose four nodes carry the same nonzero Neumann group tag
    contributes ``traction · A_face / 4`` to each of them. Accumulating over
    faces yields the exact consistent load vector for HEX8, with the
    trapezoidal edge and corner weights falling out of the sum.

    Keying on faces rather than a coordinate plane lets the full-face load and
    the single-element corner patch share one path; the resultant is
    ``F_total`` in both cases.

    Expects a traction [force/area], as supplied to the surface-integrating
    solvers listed in ``_traction_solvers`` in the problem's ``physics.py``.
    """
    forces = np.zeros((n_nodes, 3), dtype=np.float64)
    if values.size == 0 or not np.any(mask > 0):
        return forces

    face_nodes = cells[:, _HEX_FACES]  # (n_cells, 6, 4)
    tags = mask[face_nodes]  # (n_cells, 6, 4)

    first = tags[..., :1]
    loaded = ((first > 0) & np.all(tags == first, axis=-1, keepdims=True))[..., 0]
    if not np.any(loaded):
        return forces

    sel_nodes = face_nodes[loaded]  # (n_loaded, 4)
    sel_group = tags[loaded][:, 0] - 1  # 0-based group index
    areas = _quad_areas(points[sel_nodes])  # (n_loaded,)

    contrib = values[sel_group] * (areas / 4.0)[:, None]  # (n_loaded, 3)
    np.add.at(forces, sel_nodes.ravel(), np.repeat(contrib, 4, axis=0))
    return forces


def _build_bcs(
    points: np.ndarray,
    cells: np.ndarray,
    bc: dict[str, Any],
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (constraints, displacements, forces), each shape (n_nodes, 3).

    ``mask`` holds 1-based group ids; ``values`` is (n_groups, 3), or None for
    zero displacement. All three DOFs of a marked node are constrained.
    """
    constraints = torch.zeros(n_nodes, 3, dtype=torch.bool, device=_DEVICE)
    displacements = torch.zeros(n_nodes, 3, dtype=_DTYPE, device=_DEVICE)

    d_bc = bc.get("dirichlet") or {}
    d_mask = np.zeros(n_nodes, dtype=np.int32)
    raw = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_mask[: min(n_nodes, raw.size)] = raw[:n_nodes]

    constrained = np.nonzero(d_mask > 0)[0]
    if constrained.size:
        idx = torch.as_tensor(constrained, dtype=torch.long, device=_DEVICE)
        constraints[idx, :] = True
        d_vals_raw = d_bc.get("values")
        if d_vals_raw is not None:
            d_vals = np.atleast_2d(np.asarray(d_vals_raw, dtype=np.float64))
            prescribed = d_vals[d_mask[constrained] - 1]  # (n_constrained, 3)
            displacements[idx, :] = torch.as_tensor(
                prescribed, dtype=_DTYPE, device=_DEVICE
            )

    n_bc = bc.get("neumann") or {}
    n_mask = np.zeros(n_nodes, dtype=np.int32)
    raw = np.asarray(n_bc.get("mask", []), dtype=np.int32)
    n_mask[: min(n_nodes, raw.size)] = raw[:n_nodes]

    n_vals_raw = n_bc.get("values")
    n_vals = (
        np.atleast_2d(np.asarray(n_vals_raw, dtype=np.float64))
        if n_vals_raw is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    forces_np = _lump_traction(points, cells, n_mask, n_vals, n_nodes)
    forces = torch.as_tensor(forces_np, dtype=_DTYPE, device=_DEVICE)

    return constraints, displacements, forces


# ---------------------------------------------------------------------------
# Forward solve
# ---------------------------------------------------------------------------


def _compliance(
    inputs: dict[str, Any], want_grad: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the SIMP cantilever and return ``(compliance, rho_tensor)``.

    ``want_grad`` makes ``rho`` a live autograd leaf so the caller can
    differentiate the returned compliance through the FE solve.
    """
    hm = inputs["hex_mesh"]
    n_nodes = int(hm["n_points"])
    n_cells = int(hm["n_faces"])

    # Inputs are padded to a max size; the active slice is the leading
    # n_points / n_faces entries.
    points = np.asarray(hm["points"], dtype=np.float64)[:n_nodes]
    cells = np.asarray(hm["faces"], dtype=np.int64)[:n_cells]
    rho_np = np.asarray(inputs["rho"], dtype=np.float64)[:n_cells]

    E_max = float(inputs.get("E_max", 70_000.0))
    nu = float(inputs.get("nu", 0.3))
    xmin = float(inputs.get("xmin", 1e-3))
    p_exp = float(inputs.get("p_exp", 3.0))

    rho = torch.as_tensor(rho_np, dtype=_DTYPE, device=_DEVICE)
    if want_grad:
        rho.requires_grad_(True)

    nodes = torch.as_tensor(points, dtype=_DTYPE, device=_DEVICE)
    elements = torch.as_tensor(cells, dtype=torch.long, device=_DEVICE)

    material = IsotropicElasticity3D(E=E_max, nu=nu)
    model = Solid(nodes, elements, material)
    model.material = material.vectorize(model.n_elem)

    # Clamp is gradient-transparent inside [0, 1], where the harness keeps rho.
    simp = xmin + (1.0 - xmin) * torch.clamp(rho, 0.0, 1.0) ** p_exp
    model.material.C = simp[:, None, None, None, None] * model.material.C

    constraints, displacements, forces = _build_bcs(
        points, cells, inputs["boundary_conditions"], n_nodes
    )
    model.constraints = constraints
    model.displacements = displacements
    model.forces = forces

    u, *_ = model.solve(
        differentiable_parameters=rho if want_grad else None,
        **_SOLVE_KWARGS,
    )

    compliance = torch.inner(forces.ravel(), u.ravel())
    return compliance, rho


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve linear elasticity and return the SIMP compliance."""
    compliance, _ = _compliance(inputs.model_dump(), want_grad=False)
    return {"compliance": np.float32(compliance.detach().cpu().numpy())}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via ``torch.autograd.grad`` through the torch-fem solve.

    The cotangent is applied as a scale on the scalar objective; it is not
    assumed to be unit.
    """
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance"}

    inputs_dict = inputs.model_dump()
    n_rho = int(np.asarray(inputs_dict["rho"]).shape[0])
    n_cells = int(inputs_dict["hex_mesh"]["n_faces"])

    if not vjp_inputs:
        return {}

    ct = cotangent_vector.get("compliance") if "compliance" in vjp_outputs else None
    if ct is None:
        # No cotangent path to any requested output — gradient is identically zero.
        return {"rho": np.zeros(n_rho, dtype=np.float32)}

    compliance, rho = _compliance(inputs_dict, want_grad=True)
    ct_t = torch.as_tensor(
        np.asarray(ct, dtype=np.float64), dtype=_DTYPE, device=_DEVICE
    )
    (grad,) = torch.autograd.grad(compliance * ct_t.reshape(()), rho)

    # Re-pad to the declared input width; only the active cells carry gradient.
    out = np.zeros(n_rho, dtype=np.float32)
    out[:n_cells] = grad.detach().cpu().numpy().astype(np.float32)[:n_cells]
    return {"rho": out}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, ShapeDType]:
    """Shape inference without running the solver."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
