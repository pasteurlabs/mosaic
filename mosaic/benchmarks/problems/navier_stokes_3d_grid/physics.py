# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input factory and diagnostics for ns-3d-grid."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from mosaic.benchmarks.core.config import SolverSpec
from mosaic.benchmarks.core.utils import fixed_cfl_dt_steps
from mosaic.benchmarks.problems.shared.diagnostics import FLUID_DIAGNOSTICS


def make_inputs(
    spec: SolverSpec,
    ic: jax.Array,
    *,
    nu: float,
    dt: float,
    steps: int,
    domain_extent: float = 2 * jnp.pi,
    cfl_N_base: int | None = None,
    **_: Any,
) -> dict:
    """Build solver input dict.

    ``cfl_N_base`` applies :func:`fixed_cfl_dt_steps` to every solver.

    For standard 3-D periodic runs ic has shape (N, N, N, 3) and is passed
    directly as v0.
    """
    N = ic.shape[0]
    _dt, _steps = fixed_cfl_dt_steps(dt, steps, N, cfl_N_base=cfl_N_base)

    base = {
        "v0": ic,
        "viscosity": jnp.array([nu], dtype=jnp.float32),
        "dt": jnp.array([_dt], dtype=jnp.float32),
        "steps": _steps,
        "domain_extent": float(domain_extent),
    }
    return {**base, **spec.input_overrides}


DIAGNOSTICS = FLUID_DIAGNOSTICS
