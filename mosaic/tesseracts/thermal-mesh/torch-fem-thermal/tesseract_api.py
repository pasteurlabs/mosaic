# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thermal topology optimisation on a structured hexahedral mesh via torch-fem.

Uses Meyer-Nils/torch-fem `SolidHeat` (HEX8 Galerkin) with PyTorch autograd
for SIMP topology-optimisation sensitivities.  This is the first PyTorch
backend on thermal-mesh and the first GPU-native thermal VJP path in Mosaic.

SIMP conductivity:
    k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

Objective:
    C = ∮_Γ_N q_n · T dΓ    (implemented as Σ f_ext · T_nodal)

Source-identification objective:
    E = Σ_i (T_i − T_target_i)^2

Gradients (rho and source) flow through `torch.autograd` over the entire
FE solve thanks to `model.solve(differentiable_parameters=...)` in torch-fem.
"""

from typing import Any

import numpy as np
import torch
from pydantic import Field
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from tesseract_shared.types import make_differentiable
from torchfem import SolidHeat
from torchfem.materials import IsotropicConductivity3D


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho", "source"])):
    """Inputs for torch-fem thermal solver (canonical + SIMP material params)."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    """Torch-FEM thermal solver output schema."""


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

# Auto-detect CUDA.  torch-fem internally allocates many auxiliary index /
# buffer tensors via ``torch.arange(...)`` / ``torch.zeros(...)`` without an
# explicit ``device=`` kwarg, so we set ``torch.set_default_device(_DEVICE)``
# below to ensure those allocations land on the chosen device instead of
# silently reverting to CPU (which would trigger device mismatches once our
# mesh / material / force tensors are on CUDA).
#
# Sparse linear solve fallback: PyTorch's native CUDA sparse solver support
# is limited; torch-fem's GPU ``_solve_gpu`` path requires CuPy, which we do
# not ship in this tesseract to keep the image small.  Instead we pass
# ``device="cpu"`` to ``model.solve(...)`` — torch-fem then moves just the
# (A, b) sparse system to CPU for a SciPy ``spsolve``, and the resulting ``x``
# is returned on the original (``b.device``) device so the surrounding autograd
# graph (assembly, boundary conditions, compliance integral, VJP) remains on
# GPU.  For the grid sizes exercised in thermal-mesh (n_dofs <= ~17k at N=128,
# with ny=N/2, nz=1), the per-iteration GPU↔CPU copy of the COO is negligible
# relative to the assembly cost we gain back by running on GPU.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SPARSE_SOLVE_DEVICE = "cpu"  # SciPy-backed fallback; see comment above.
_DTYPE = torch.float64  # torch-fem defaults to float64 for solver stability

# torch-fem internals assume torch.get_default_dtype() matches the tensors we
# pass in (see sparse.py:eval_residual where the NR solver allocates a DU in
# default dtype and assigns into it from our float64 BC tensors).  Align the
# global default so the Newton iterate, BC buffer, and residual are all
# float64 end-to-end.  Similarly, align the default device so torch-fem's
# internal ``torch.zeros/arange(...)`` calls land on the same device as the
# tensors we construct explicitly with ``device=_DEVICE``.
torch.set_default_dtype(_DTYPE)
torch.set_default_device(_DEVICE)


# ---------------------------------------------------------------------------
# torch-fem sparse_solve device patch
# ---------------------------------------------------------------------------
# torch-fem's ``sparse.sparse_solve`` moves only (A, b) when the caller
# passes ``device="cpu"``, but ``compute_B()`` (null-space rigid-body modes)
# allocates on the default device (CUDA) and ``_solve_cpu`` then calls
# ``B.data.numpy()`` without a ``.cpu()`` — raising
# ``TypeError: can't convert cuda:0 device type tensor to numpy``.
#
# We wrap the function to also migrate ``B``, ``M`` (preconditioner), and
# ``x0`` (warm-start) to the target device before the underlying solve runs.
# This is the smallest possible change that lets us keep the outer autograd
# graph on GPU while routing the numerical spsolve through SciPy on CPU.
import torchfem.sparse as _tfem_sparse  # noqa: E402

_ORIG_SPARSE_SOLVE = _tfem_sparse.sparse_solve


def _sparse_solve_device_safe(  # mosaic:util
    A: Any,
    b: Any,
    B: Any = None,
    stol: float = 1e-10,
    device: Any = None,
    method: Any = None,
    M: Any = None,
    x0: Any = None,
) -> Any:
    if device is not None:
        if B is not None and hasattr(B, "to"):
            B = B.to(device)
        if x0 is not None and hasattr(x0, "to"):
            x0 = x0.to(device)
        # M is a scipy.sparse LinearOperator when non-None on the CPU path; it
        # does not need a torch .to() call.  If M is a torch Tensor (rare), we
        # migrate it too; otherwise pass through unchanged.
        if (
            M is not None
            and hasattr(M, "to")
            and not callable(getattr(M, "matvec", None))
        ):
            M = M.to(device)
    return _ORIG_SPARSE_SOLVE(A, b, B, stol, device, method, M, x0)


_tfem_sparse.sparse_solve = _sparse_solve_device_safe

# SIMP parameters (baked into the InputSchema defaults but kept here as fallbacks)
_K_MIN_RATIO = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mesh(  # mosaic:init
    points_np: np.ndarray, cells_np: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy mesh arrays to torch tensors on the chosen device."""
    nodes = torch.as_tensor(points_np, dtype=_DTYPE, device=_DEVICE)
    elements = torch.as_tensor(cells_np, dtype=torch.long, device=_DEVICE)
    return nodes, elements


def _build_bc_tensors(  # mosaic:init
    n_nodes: int,
    n_cells: int,
    points_np: np.ndarray,
    cells_np: np.ndarray,
    bc_dict: dict,
    source_np: np.ndarray,
    cell_volume: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build constraint / displacement / force tensors (all shape (n_nodes, 1)).

    - Dirichlet nodes: constraints[i]=True, displacements[i]=value.
    - Neumann face flux: lumped to nodes as q_n · face_area / 4 per right-face
      node.  Uses the canonical "right face" identification (x == x_max) that
      matches the heated-block problem; the mask itself marks which nodes the
      flux applies to.
    - Volumetric source: lumped to nodes as source_e · vol_e / 8 per cell node.

    All boundary-condition tensors are allocated on _DEVICE with the
    appropriate dtype (bool for the constraints mask, _DTYPE for the nodal
    displacement/force values) so they can be assigned directly to the
    corresponding ``model.*`` properties without a device hop.
    """
    constraints = torch.zeros(n_nodes, 1, dtype=torch.bool, device=_DEVICE)
    displacements = torch.zeros(n_nodes, 1, dtype=_DTYPE, device=_DEVICE)
    forces = torch.zeros(n_nodes, 1, dtype=_DTYPE, device=_DEVICE)

    # --- Dirichlet ---
    d_bc = bc_dict.get("dirichlet") or {}
    d_mask = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_vals_raw = d_bc.get("values")
    d_vals = (
        np.asarray(d_vals_raw, dtype=np.float64)
        if d_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )
    if d_mask.size > 0:
        mask = np.zeros(n_nodes, dtype=np.int32)
        mask[: min(n_nodes, d_mask.size)] = d_mask[:n_nodes]
        constrained = np.where(mask > 0)[0]
        if constrained.size > 0:
            groups = mask[constrained] - 1
            prescribed = (
                d_vals[groups, 0]
                if d_vals.size > 0
                else np.zeros(constrained.size, dtype=np.float64)
            )
            c_idx = torch.as_tensor(constrained, dtype=torch.long, device=_DEVICE)
            constraints[c_idx, 0] = True
            displacements[c_idx, 0] = torch.as_tensor(
                prescribed, dtype=_DTYPE, device=_DEVICE
            )

    # --- Neumann flux on the right (x=x_max) face ---
    # Consistent FEM lumping: for each face element with uniform q_n, the
    # contribution to each of its 4 face nodes is  f_i = q_n · A_face / 4.
    # Summed across every face element sharing a node, this gives the correct
    # total  Σ_i f_i = q_n · Ly · Lz  (interior face nodes pick up 4 elements,
    # edge nodes 2, corner nodes 1).
    n_bc = bc_dict.get("neumann") or {}
    n_mask = np.asarray(n_bc.get("mask", []), dtype=np.int32)
    n_vals_raw = n_bc.get("values")
    n_vals = (
        np.asarray(n_vals_raw, dtype=np.float64)
        if n_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )
    if n_mask.size > 0 and n_vals.size > 0:
        mask = np.zeros(n_nodes, dtype=np.int32)
        mask[: min(n_nodes, n_mask.size)] = n_mask[:n_nodes]

        # Infer dy, dz from mesh
        ys = np.unique(np.round(points_np[:, 1], 7))
        zs = np.unique(np.round(points_np[:, 2], 7))
        dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
        dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0

        x_max = float(points_np[:, 0].max())
        tol = 1e-8 * max(x_max, 1.0) + 1e-10

        # A cell element is a "right-face" element if it has exactly 4 nodes on
        # the plane x == x_max AND any of those 4 nodes are marked by the mask.
        # Walk all cells and contribute  q_n·A/4  per face node.
        q_n = float(n_vals[0, 0])
        A_face = dy * dz
        per_node = q_n * A_face / 4.0

        cells_active = cells_np[:n_cells]
        right_node_mask = np.abs(points_np[:, 0] - x_max) < tol

        # For each cell, find nodes that are both on the x_max plane and
        # flagged by the Neumann mask.  If exactly 4 such nodes exist the
        # cell contributes to that face.
        add_indices: list[np.ndarray] = []
        for cell_nodes in cells_active:
            is_face_and_marked = right_node_mask[cell_nodes] & (mask[cell_nodes] > 0)
            if int(is_face_and_marked.sum()) == 4:
                add_indices.append(cell_nodes[is_face_and_marked])
        if add_indices:
            face_nodes_flat = np.concatenate(add_indices)  # (n_contrib,)
            idx = torch.as_tensor(face_nodes_flat, dtype=torch.long, device=_DEVICE)
            vals = torch.full((idx.shape[0],), per_node, dtype=_DTYPE, device=_DEVICE)
            forces[:, 0] = forces[:, 0].scatter_add(0, idx, vals)

    # --- Volumetric source (lumped): f_i += source_e · vol_e / 8 per cell node
    if source_np is not None and np.any(source_np != 0.0):
        cells_t = torch.as_tensor(cells_np[:n_cells], dtype=torch.long, device=_DEVICE)
        source_t = torch.as_tensor(source_np[:n_cells], dtype=_DTYPE, device=_DEVICE)
        node_contrib = source_t * (cell_volume / 8.0)  # (n_cells,)
        # Scatter-add node_contrib to each of the 8 nodes of each cell
        flat_idx = cells_t.reshape(-1)  # (n_cells*8,)
        flat_vals = node_contrib.unsqueeze(1).expand(-1, 8).reshape(-1)
        forces[:, 0].scatter_add_(0, flat_idx, flat_vals)

    return constraints, displacements, forces


def _cell_volume_from_points(points_np: np.ndarray) -> float:  # mosaic:util
    """Infer structured-grid cell volume from unique coordinates."""
    xs = np.unique(np.round(points_np[:, 0], 7))
    ys = np.unique(np.round(points_np[:, 1], 7))
    zs = np.unique(np.round(points_np[:, 2], 7))
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    return dx * dy * dz


def _forward_torchfem(  # mosaic:physics
    points_np: np.ndarray,
    cells_np: np.ndarray,
    bc_dict: dict,
    rho_t: torch.Tensor,
    source_t: torch.Tensor,
    k_max: float,
    p_exp: float,
    differentiable_params: tuple | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a torch-fem SolidHeat forward solve and return (T_nodal, f_internal).

    Notes:
        - ``rho_t`` and ``source_t`` are (n_cells,) tensors on ``_DEVICE``.  If
          ``differentiable_params`` is not None they must be leaves with
          ``requires_grad=True`` (or share a graph with one).
        - SIMP is implemented by scaling ``material.KAPPA`` by
          ``k_min_ratio + (1 - k_min_ratio) * rho^p`` after ``.vectorize``;
          the base material is created at ``kappa = k_max`` so the final
          per-element conductivity is ``k = k_max · (k_min_ratio + (1-k_min_ratio)·ρ^p)
          = k_min + (k_max-k_min)·ρ^p``.
        - The source contributes to ``model.forces`` through ``source_t``; if
          ``source_t`` is a live autograd leaf its grad flows back too.
    """
    n_cells = cells_np.shape[0]
    n_nodes = points_np.shape[0]

    nodes, elements = _build_mesh(points_np, cells_np)

    # Base material at k_max; we scale per-element below via SIMP.
    material = IsotropicConductivity3D(kappa=float(k_max))
    model = SolidHeat(nodes, elements, material)

    # Vectorise the material across elements (creates KAPPA of shape (n_elem, 3, 3))
    model.material = material.vectorize(model.n_elem)

    # SIMP scaling: k = k_min + (k_max - k_min) * rho^p
    #             = k_max · (k_min_ratio + (1 - k_min_ratio) · rho^p)
    # Clip rho into [0, 1] for numerical safety but keep gradient.
    rho_clip = torch.clamp(rho_t, 0.0, 1.0)
    simp_scale = _K_MIN_RATIO + (1.0 - _K_MIN_RATIO) * rho_clip**p_exp  # (n_cells,)
    # model.material.KAPPA shape (n_cells, 3, 3); multiply elementwise along elem dim
    model.material.KAPPA = simp_scale[:, None, None] * model.material.KAPPA

    # Boundary conditions and forces (source baked into forces via scatter-add)
    cell_volume = _cell_volume_from_points(points_np)
    constraints, displacements, forces_base = _build_bc_tensors(
        n_nodes,
        n_cells,
        points_np,
        cells_np,
        bc_dict,
        source_np=None,  # source handled separately below so it stays in the graph
        cell_volume=cell_volume,
    )

    # Add source contribution as a differentiable scatter_add from source_t into
    # a per-node forces tensor.  We always build the source nodal contribution
    # so torch-fem's NewtonRaphsonAdjoint sees source_t as a live dependency of
    # ``model.forces`` (otherwise ``eval_residual`` has no path back to source
    # and ``torch.autograd.grad`` returns None for that parameter).
    cells_t = torch.as_tensor(cells_np[:n_cells], dtype=torch.long, device=_DEVICE)
    node_contrib = source_t * (cell_volume / 8.0)  # (n_cells,)
    flat_idx = cells_t.reshape(-1)
    flat_vals = node_contrib.unsqueeze(1).expand(-1, 8).reshape(-1)
    # ``index_add`` is functional (differentiable); use ``view_as`` to re-stack
    # into the (n_nodes, 1) layout that ``model.forces`` expects, and combine
    # with the base forces via a plain addition.
    src_nodal = torch.zeros(n_nodes, dtype=_DTYPE, device=_DEVICE).index_add(
        0, flat_idx, flat_vals
    )  # (n_nodes,)
    forces_final = forces_base + src_nodal.unsqueeze(1)  # (n_nodes, 1)

    model.constraints = constraints
    model.displacements = displacements
    model.forces = forces_final

    # Linear solve: torch-fem routes through ``device=_SPARSE_SOLVE_DEVICE``,
    # moving the assembled sparse A and rhs b to that device for the numerical
    # solve (SciPy ``spsolve`` on CPU — CuPy not installed in this tesseract),
    # then returns the solution on the original ``b.device`` (= _DEVICE).  The
    # returned tensor path goes through SolidHeat's custom autograd so
    # gradients flow back to differentiable_parameters on _DEVICE.
    u_k, f_k, *_ = model.solve(
        differentiable_parameters=differentiable_params,
        device=_SPARSE_SOLVE_DEVICE,
    )
    # Return u (nodal T), f_k (full internal force = f_ext at equilibrium),
    # and the Neumann-only sub-force needed to compute the boundary integral
    # ∮_Γ_N q_n T dΓ = f_Neumann^T · T.  Without this split, compliance and
    # source-gradient would pick up the volumetric source contribution too,
    # which disagrees with the peer solvers that integrate over Γ_N only.
    return u_k, f_k, forces_base


def _apply_core(inputs_dict: dict, want_grad: bool) -> dict:  # mosaic:physics
    """Shared forward solver used by both ``apply`` and ``vector_jacobian_product``.

    Args:
        inputs_dict: Raw dict form of the InputSchema.
        want_grad: If True, input tensors are set up as live autograd leaves.
    """
    hm = inputs_dict["hex_mesh"]
    n_cells = int(hm["n_faces"])
    n_nodes = int(hm["n_points"])
    points_np = np.asarray(hm["points"][:n_nodes], dtype=np.float64)
    cells_np = np.asarray(hm["faces"][:n_cells], dtype=np.int64)

    rho_np = np.asarray(inputs_dict["rho"][:n_cells], dtype=np.float64)
    source_np = np.asarray(inputs_dict["source"][:n_cells], dtype=np.float64)
    target_np = np.asarray(inputs_dict.get("target_temperature", []), dtype=np.float64)

    k_max = float(inputs_dict.get("k_max", 1.0))
    p_exp = float(inputs_dict.get("p_exp", 3.0))

    rho_t = torch.as_tensor(rho_np, dtype=_DTYPE, device=_DEVICE)
    source_t = torch.as_tensor(source_np, dtype=_DTYPE, device=_DEVICE)
    if want_grad:
        rho_t.requires_grad_(True)
        source_t.requires_grad_(True)
        diff_params = (rho_t, source_t)
    else:
        diff_params = None

    u_k, _f_k, f_neumann = _forward_torchfem(
        points_np,
        cells_np,
        inputs_dict["boundary_conditions"],
        rho_t,
        source_t,
        k_max,
        p_exp,
        diff_params,
    )

    # Compliance C = ∮_Γ_N q_n · T dΓ = f_Neumann^T · T (boundary work only,
    # not the total internal work).  Using f_k (full internal force) would add
    # the volumetric source contribution and break agreement with the peer
    # solvers' surface-integral compliance.
    thermal_compliance = torch.inner(f_neumann.reshape(-1), u_k.reshape(-1))

    # Identification error: Σ (T_nodal - T_target)^2
    T_nodal = u_k.reshape(-1)
    if target_np.size > 0:
        target_t = torch.as_tensor(target_np, dtype=_DTYPE, device=_DEVICE)
        n = min(T_nodal.shape[0], target_t.shape[0])
        diff = T_nodal[:n] - target_t[:n]
        identification_error = torch.sum(diff * diff)
    else:
        identification_error = torch.zeros((), dtype=_DTYPE, device=_DEVICE)

    return {
        "thermal_compliance": thermal_compliance,
        "identification_error": identification_error,
        "rho_t": rho_t,
        "source_t": source_t,
    }


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + identification error.

    Uses torch-fem SolidHeat with SIMP-scaled IsotropicConductivity3D material.
    """
    out = _apply_core(inputs.model_dump(), want_grad=False)
    tc = out["thermal_compliance"].detach().cpu().numpy().astype(np.float32)
    ide = out["identification_error"].detach().cpu().numpy().astype(np.float32)
    return {
        "thermal_compliance": np.float32(tc),
        "identification_error": np.float32(ide),
    }


def vector_jacobian_product(  # mosaic:grad:rho,source:autodiff
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via torch.autograd.grad through the torch-fem SolidHeat solve.

    Supports gradients w.r.t. ``rho`` and ``source``; both flow through
    ``differentiable_parameters=(rho, source)`` — torch-fem's custom autograd
    implements the implicit-function adjoint for the linear FE system.
    """
    assert vjp_inputs <= {"rho", "source"}
    assert vjp_outputs <= {"thermal_compliance", "identification_error"}

    if not vjp_inputs:
        return {}

    inputs_dict = inputs.model_dump()
    out = _apply_core(inputs_dict, want_grad=True)

    rho_t = out["rho_t"]
    source_t = out["source_t"]

    # Build the scalar loss by summing each output against its cotangent so a
    # single backward pass yields the full VJP over whatever outputs we're asked
    # about.  Missing / zero cotangents contribute nothing.
    loss_terms: list[torch.Tensor] = []
    for key, tensor in (
        ("thermal_compliance", out["thermal_compliance"]),
        ("identification_error", out["identification_error"]),
    ):
        if key not in vjp_outputs:
            continue
        if key not in cotangent_vector:
            continue
        ct_val = cotangent_vector[key]
        ct_t = torch.as_tensor(
            np.asarray(ct_val, dtype=np.float64), dtype=_DTYPE, device=_DEVICE
        )
        if ct_t.ndim == 0 and tensor.ndim == 0:
            loss_terms.append(tensor * ct_t)
        else:
            # Flatten both sides and align sizes
            t_flat = tensor.reshape(-1)
            c_flat = ct_t.reshape(-1)
            n = min(t_flat.shape[0], c_flat.shape[0])
            loss_terms.append(torch.sum(t_flat[:n] * c_flat[:n]))

    n_rho = int(np.asarray(inputs_dict["rho"]).shape[0])
    n_source = int(np.asarray(inputs_dict["source"]).shape[0])
    n_cells_active = int(inputs_dict["hex_mesh"]["n_faces"])

    result: dict[str, Any] = {}

    if not loss_terms:
        # No cotangent path — return zeros for every requested input.
        if "rho" in vjp_inputs:
            result["rho"] = np.zeros(n_rho, dtype=np.float32)
        if "source" in vjp_inputs:
            result["source"] = np.zeros(n_source, dtype=np.float32)
        return result

    loss = sum(loss_terms)

    targets: list[torch.Tensor] = []
    keys: list[str] = []
    if "rho" in vjp_inputs:
        targets.append(rho_t)
        keys.append("rho")
    if "source" in vjp_inputs:
        targets.append(source_t)
        keys.append("source")

    grads = torch.autograd.grad(loss, targets, allow_unused=True)

    for key, g in zip(keys, grads, strict=False):
        n_full = n_rho if key == "rho" else n_source
        arr = np.zeros(n_full, dtype=np.float32)
        if g is not None:
            g_np = g.detach().cpu().numpy().astype(np.float32)
            n_copy = min(arr.shape[0], g_np.shape[0], n_cells_active)
            arr[:n_copy] = g_np[:n_copy]
        result[key] = arr

    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
