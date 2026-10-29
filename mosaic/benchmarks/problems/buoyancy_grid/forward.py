# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coupled-flow refinement and diagnostics on the closed staggered grid."""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext, get_kernel_config, kernel
from mosaic.benchmarks.core.reference import load_reference
from mosaic.benchmarks.core.runner import get_last_apply_error, safe_apply
from mosaic.benchmarks.core.utils import active_solvers
from mosaic.benchmarks.problems.shared.forward import agreement as shared_agreement
from mosaic.mosaic_shared.problems.buoyancy_grid import restrict_state


def _state(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[-1] != 3 or result.shape[0] != result.shape[1]:
        raise ValueError("Expected a square packed buoyancy state (N,N,3).")
    if not np.isfinite(result).all():
        raise ValueError("Buoyancy state must be finite.")
    return result


def physical_diagnostics(
    state: np.ndarray, initial: np.ndarray, domain_extent: float = 1.0
) -> dict[str, float]:
    """Measure closed-wall MAC divergence and scalar/velocity quantities.

    Scalar mass is a continuum invariant, but semi-Lagrangian transport need not
    preserve it discretely, so its drift is reported rather than assumed zero.
    Buoyancy can increase kinetic energy. Padding is only a state-format check:
    the physical normal boundary faces are omitted from the public state.
    """
    state, initial = _state(state), _state(initial)
    if state.shape != initial.shape:
        raise ValueError("Initial and final states must have the same shape.")
    dx = domain_extent / state.shape[0]
    ux = np.pad(state[:-1, :, 0], ((1, 1), (0, 0)))
    uy = np.pad(state[:, :-1, 1], ((0, 0), (1, 1)))
    divergence = (np.diff(ux, axis=0) + np.diff(uy, axis=1)) / dx
    scalar, scalar0 = state[..., 2], initial[..., 2]
    initial_mass = float(np.mean(scalar0))
    return {
        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
        "scalar_mass_rel_drift": float(
            abs(np.mean(scalar) - initial_mass) / max(abs(initial_mass), 1e-12)
        ),
        "scalar_min": float(np.min(scalar)),
        "scalar_max": float(np.max(scalar)),
        "kinetic_energy": float(
            0.5
            * (np.sum(state[:-1, :, 0] ** 2) + np.sum(state[:, :-1, 1] ** 2))
            / scalar.size
        ),
        "wall_normal_padding_max": float(
            max(np.max(np.abs(state[-1, :, 0])), np.max(np.abs(state[:, -1, 1])))
        ),
    }


def _failure(stage: str) -> dict:
    return {
        "metrics": {
            "valid": False,
            "status": "failed",
            "failure_type": "error",
            "stage": stage,
            "error": get_last_apply_error() or "Solver returned no finite state.",
        }
    }


@kernel(
    sweep_mode="default",
    ic_sweep=True,
    selector_fn=active_solvers,
    snapshot_filename="fields.npz",
    snapshot_prefixes=("state",),
)
def refinement(t: Any, ctx: KernelContext) -> dict:
    """Compare the same solver at N,dt and 2N,dt/2 at equal physical time.

    This is a joint space/time refinement discrepancy, not an exact-error
    estimate or a claim of convergence order. Face and cell averaging transfers
    the refined state without treating staggered velocities as collocated data.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    coarse = safe_apply(t, inputs, ctx.output_key)
    if coarse is None:
        return _failure("coarse")
    n = np.asarray(ctx.ic).shape[0]
    dt, steps = float(np.asarray(inputs["dt"])[0]), int(inputs["steps"])
    fine_physics = {**ctx.phys, "N": 2 * n, "dt": dt / 2, "steps": 2 * steps}
    ic_name = ctx.run.get("ic", {}).get("name", next(iter(ctx.cfg.make_ic)))
    fine_ic = ctx.cfg.make_ic[ic_name](
        L=ctx.domain_extent, seed=ctx.seed, **fine_physics
    )
    fine_inputs = ctx.make_inputs(ctx.name, fine_ic, **fine_physics)
    fine = safe_apply(t, fine_inputs, ctx.output_key)
    if fine is None:
        return _failure("refined")
    coarse, fine = _state(coarse), _state(fine)
    reference = restrict_state(fine, 2)
    if coarse.shape != reference.shape:
        raise ValueError("Coarse and restricted refined state shapes differ.")
    velocity_error = float(
        np.linalg.norm(coarse[..., :2] - reference[..., :2])
        / max(np.linalg.norm(reference[..., :2]), 1e-12)
    )
    scalar_error = float(
        np.linalg.norm(coarse[..., 2] - reference[..., 2])
        / max(np.linalg.norm(reference[..., 2]), 1e-12)
    )
    error = float(np.sqrt((velocity_error**2 + scalar_error**2) / 2))
    return {
        "metrics": {
            "valid": True,
            "error": error,
            "velocity_rel_error": velocity_error,
            "scalar_rel_error": scalar_error,
            "reference_N": 2 * n,
            "reference_dt": dt / 2,
            "reference_steps": 2 * steps,
            "physical_time": dt * steps,
        },
        "snapshot": coarse,
    }


@kernel(
    sweep_mode="default",
    ic_sweep=True,
    selector_fn=active_solvers,
    snapshot_filename="fields.npz",
    snapshot_prefixes=("state",),
)
def physical_laws(t: Any, ctx: KernelContext) -> dict:
    """Report physical diagnostics, using mass drift as the numeric status error.

    Other curves are observational: a passing mass-drift check does not certify
    every physical law, and kinetic energy is not expected to decay under buoyancy.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **ctx.phys)
    result = safe_apply(t, inputs, ctx.output_key)
    if result is None:
        return _failure("forward")
    diagnostics = physical_diagnostics(result, ctx.ic, ctx.domain_extent)
    return {
        "metrics": {
            "valid": True,
            "error": diagnostics["scalar_mass_rel_drift"],
            **diagnostics,
        },
        "snapshot": np.asarray(result),
    }


_REFERENCE_DIR = Path(__file__).with_name("references")


def require_openfoam_reference(n: int, physics: dict, ic: dict) -> np.ndarray:
    """Reject absent, altered, or mismatched independent references, never use consensus."""
    manifest = json.loads((_REFERENCE_DIR / "forward_agreement.json").read_text())
    expected = manifest["benchmark"]
    from .physics import interacting_plumes

    ic_defaults = inspect.signature(interacting_plumes).parameters
    if ic.get("name") != expected["ic"] or ic.get("seed", 0) != expected["seed"]:
        raise ValueError("OpenFOAM reference initial condition does not match this run")
    for key, value in expected.items():
        if key in {"ic", "seed"}:
            continue
        default = (
            ic_defaults[key].default
            if key in {"position_jitter", "width_jitter", "amplitude_jitter"}
            else np.nan
        )
        if not np.isclose(physics.get(key, default), value, rtol=1e-7, atol=0):
            raise ValueError(f"OpenFOAM reference physics mismatch: {key}")
    archive = _REFERENCE_DIR / "forward_agreement.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["reference_sha256"]:
        raise ValueError("OpenFOAM reference archive hash mismatch")
    if n not in manifest["sweep_values"]:
        raise ValueError(f"No OpenFOAM reference for N={n}")
    index = manifest["sweep_values"].index(n)
    reference = load_reference("buoyancy-grid", "forward/agreement", index)
    if reference is None:
        raise FileNotFoundError(f"Required OpenFOAM reference is missing for N={n}")
    if reference.shape != (n, n, 3) or not np.isfinite(reference).all():
        raise ValueError("Invalid OpenFOAM reference state")
    return reference


def _openfoam_agreement_aggregate(by_solver: Any, **kwargs: Any) -> dict:
    run = kwargs["run"]
    physics = {"domain_extent": kwargs["cfg"].domain_extent, **run["physics"]}
    for n in kwargs["sweep_values"]:
        require_openfoam_reference(int(n), physics, run["ic"])
    aggregate = get_kernel_config(shared_agreement)["aggregate_fn"]
    result = aggregate(by_solver, **kwargs)
    result["extras"]["reference_label"] = "OpenFOAM numerical reference"
    return result


@kernel(
    **{
        **get_kernel_config(shared_agreement),
        "aggregate_fn": _openfoam_agreement_aggregate,
    }
)
def openfoam_agreement(t: Any, ctx: KernelContext) -> dict:
    """Compare to the shipped, independently refined OpenFOAM trajectory."""
    require_openfoam_reference(int(ctx.phys["N"]), ctx.phys, ctx.run["ic"])
    return shared_agreement(t, ctx)
