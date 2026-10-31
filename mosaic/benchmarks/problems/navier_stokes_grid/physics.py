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

from . import cylinder


def make_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    resolution: int | None = None,
    t_end: float = cylinder.T_END,
    **physics: Any,
) -> dict:
    """Build the solver input dict.

    Physics with a ``resolution`` (cells per cylinder diameter) describe the
    DFG 2D-1 channel, where ``ic`` is the inlet profile and everything but the
    simulated time ``t_end`` follows from the benchmark (see
    :func:`.cylinder.channel_inputs`). All other physics describe the
    periodic box.
    """
    if resolution is not None:
        base = cylinder.channel_inputs(ic, resolution, t_end)
    else:
        base = _periodic_inputs(ic, **physics)
    return {**base, **spec.input_overrides}


def _periodic_inputs(
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    cfl_N_base: int | None = None,
    **_: Any,
) -> dict:
    """Periodic-box inputs.

    ``cfl_N_base`` applies :func:`fixed_cfl_dt_steps` to every solver.
    """
    N = ic.shape[0]
    _dt, _steps = fixed_cfl_dt_steps(dt, steps, N, cfl_N_base=cfl_N_base)
    return {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }


DIAGNOSTICS = FLUID_DIAGNOSTICS
