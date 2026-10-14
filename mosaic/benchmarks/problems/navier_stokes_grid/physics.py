# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input factory and diagnostics for ns-grid."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from mosaic.benchmarks.core.config import SolverSpec
from mosaic.benchmarks.core.utils import fixed_cfl_dt_steps
from mosaic.benchmarks.problems.shared.diagnostics import FLUID_DIAGNOSTICS

from .ics import _uniform_flow

_LBM_SOLVERS = {"xlb"}


def make_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    lbm_N_base: int | None = None,
    cfl_N_base: int | None = None,
    obstacle: dict | None = None,
    U_mean: float = 0.5,
    **_: Any,
) -> dict:
    """Build solver input dict, applying LBM dt-scaling when lbm_N_base is set.

    ``cfl_N_base`` rescales ``dt`` and ``steps`` for every solver so the CFL
    number and end time stay fixed across N (see :func:`fixed_cfl_dt_steps`);
    LBM scaling then applies on top.

    When ic is 1-D (shape (N,)) it is treated as an inflow profile for drag
    optimisation (v0 = uniform background at U_mean, ic → inflow_profile field).
    """
    if ic.ndim == 1:
        N = ic.shape[0]
        _dt, _steps = fixed_cfl_dt_steps(dt, steps, N, cfl_N_base=cfl_N_base)
        if spec.key in _LBM_SOLVERS and lbm_N_base is not None:
            _dt = _dt * min(1.0, lbm_N_base / N)
            _steps = max(1, round(_steps * max(1.0, N / lbm_N_base)))
        base = {
            "v0": _uniform_flow(N, U=U_mean),
            "inflow_profile": ic,
            "viscosity": jnp.array([nu], dtype=jnp.float32),
            "dt": jnp.array([_dt], dtype=jnp.float32),
            "steps": _steps,
            "domain_extent": float(domain_extent),
        }
        if obstacle is not None:
            base["obstacle"] = obstacle
            base["boundary_conditions"] = {
                "x_lo": {"type": "periodic"},
                "x_hi": {"type": "periodic"},
                "y_lo": {"type": "no_slip"},
                "y_hi": {"type": "no_slip"},
            }
        return {**base, **spec.input_overrides}

    N = ic.shape[0]
    _dt, _steps = fixed_cfl_dt_steps(dt, steps, N, cfl_N_base=cfl_N_base)
    if spec.key in _LBM_SOLVERS and lbm_N_base is not None:
        _dt = _dt * min(1.0, lbm_N_base / N)
        _steps = max(1, round(_steps * max(1.0, N / lbm_N_base)))

    base = {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }
    if obstacle is not None:
        base["obstacle"] = obstacle
        base["boundary_conditions"] = {
            "x_lo": {"type": "neumann"},
            "x_hi": {"type": "neumann"},
            "y_lo": {"type": "no_slip"},
            "y_hi": {"type": "no_slip"},
        }
    return {**base, **spec.input_overrides}


DIAGNOSTICS = FLUID_DIAGNOSTICS
