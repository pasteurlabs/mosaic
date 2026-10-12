# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check coupled-flow gradients away from interpolation knots at rest."""

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.problems.shared.gradient import fd_check as shared_fd_check


@kernel()
def fd_check(t: Any, ctx: KernelContext) -> dict:
    """Develop the initially resting plume before centered finite differences.

    Transport schemes can have derivative kinks at zero velocity: backtraced
    samples lie on interpolation knots, or an upwind flux changes direction.
    Centered differences can then disagree with a one-sided AD derivative.
    The warmup is fixed and is not part of the differentiated transition.
    """
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **{**ctx.phys, "steps": 4})
    state = jnp.asarray(apply_tesseract(t, inputs)[ctx.output_key])
    return shared_fd_check(t, replace(ctx, ic=state))
