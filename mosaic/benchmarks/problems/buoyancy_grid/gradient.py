# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check coupled-flow gradients away from interpolation knots at rest."""

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.problems.shared.gradient import fd_check as shared_fd_check
from mosaic.benchmarks.problems.shared.gradient import param_sweep as shared_param_sweep


def _warm_context(t: Any, ctx: KernelContext) -> KernelContext:
    inputs = ctx.make_inputs(ctx.name, ctx.ic, **{**ctx.phys, "steps": 4})
    state = jnp.asarray(apply_tesseract(t, inputs)[ctx.output_key])
    return replace(ctx, ic=state)


@kernel(sweep_mode="none", catch_label="VJP failed")
def fd_check(t: Any, ctx: KernelContext) -> dict:
    """Develop the initially resting plume before centered finite differences.

    Transport schemes can have derivative kinks at zero velocity: backtraced
    samples lie on interpolation knots, or an upwind flux changes direction.
    Centered differences can then disagree with a one-sided AD derivative.
    The warmup is fixed and is not part of the differentiated transition.
    """
    return shared_fd_check(t, _warm_context(t, ctx))


@kernel(sweep_mode="default", horizons_shared=True, catch_label="VJP failed")
def param_sweep(t: Any, ctx: KernelContext) -> dict:
    """Sweep transitions from a fixed four-step developed state.

    Preparation uses the current sweep physics but always four native steps.
    The shared kernel differentiates only the requested transition, preserving
    its epsilon sweep and horizon aggregation metadata.
    """
    return shared_param_sweep(t, _warm_context(t, ctx))
