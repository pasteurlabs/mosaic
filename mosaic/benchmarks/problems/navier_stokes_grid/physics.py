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
from .kolmogorov import forcing_field

_LBM_SOLVERS = {"xlb"}


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
        base = _periodic_inputs(spec, ic, **physics)
    return {**base, **spec.input_overrides}


def _periodic_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    lbm_N_base: int | None = None,
    cfl_N_base: int | None = None,
    forcing_wavenumber: int | None = None,
    forcing_amplitude: float = 1.0,
    linear_drag: float | None = None,
    **_: Any,
) -> dict:
    """Periodic-box inputs, applying LBM dt-scaling when ``lbm_N_base`` is set.

    ``cfl_N_base`` applies :func:`fixed_cfl_dt_steps` to every solver, before
    any LBM scaling. ``forcing_wavenumber`` adds the Kolmogorov force of that
    wavenumber and ``forcing_amplitude``, and ``linear_drag`` a drag -αu.
    """
    N = ic.shape[0]
    _dt, _steps = fixed_cfl_dt_steps(dt, steps, N, cfl_N_base=cfl_N_base)
    if spec.key in _LBM_SOLVERS and lbm_N_base is not None:
        _dt = _dt * min(1.0, lbm_N_base / N)
        _steps = max(1, round(_steps * max(1.0, N / lbm_N_base)))
    inputs = {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }
    if forcing_wavenumber is not None:
        inputs["forcing"] = jnp.asarray(
            forcing_field(N, domain_extent, forcing_wavenumber, forcing_amplitude)
        )
    if linear_drag is not None:
        inputs["linear_drag"] = jnp.array([linear_drag], dtype=jnp.float32)
    return inputs


DIAGNOSTICS = FLUID_DIAGNOSTICS
