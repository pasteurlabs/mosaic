# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference check of a loss on a chain of solver calls.

Training a model through a solver calls the solver once per interval and
differentiates through the whole chain. Its loss compares every interval with
a target, normalised by the target's norm, so the cotangent a call receives
is of the order of the misfit divided by that norm: far smaller than those of
the single-call checks, whose loss is the output's squared norm. A VJP that
is only accurate for cotangents of order one, for example because its adjoint
linear solves stop at an absolute tolerance, passes those checks and fails
this one.

The chain runs the Kolmogorov flow from a turbulent state of the ground truth
on the coarse configuration of ``forward/kolmogorov``. Its targets are the
solver's own chain from that state, and the chain that is differentiated
starts from the state plus a small divergence-free offset, so the misfit is
the solver's own sensitivity to that offset.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.problems.shared.diagnostics import project_divergence_free

from .kolmogorov import (
    DOMAIN_EXTENT,
    GROUND_TRUTH_SEEDS,
    initial_state,
    load_ground_truth,
)

# Seed of the offset and the finite-difference directions.
_SEED = 0


def chain(step: Callable[[Any], Any], v0: Any, intervals: int) -> list[Any]:
    """The states after each of ``intervals`` calls of ``step`` from ``v0``."""
    states, state = [], v0
    for _ in range(intervals):
        state = step(state)
        states.append(state)
    return states


def chain_loss(states: list[Any], targets: list[Any], xp: Any = jnp) -> Any:
    """Mean over the intervals of ‖s − t‖² / ‖t‖²."""
    return xp.mean(
        xp.stack(
            [
                xp.sum((s - t) ** 2) / xp.sum(t**2)
                for s, t in zip(states, targets, strict=True)
            ]
        )
    )


def divergence_free_direction(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """A unit-norm random divergence-free field of ``shape`` (N, N, 1, 2).

    The Nyquist modes are dropped first: their wavenumber has no sign, so the
    projection cannot make them divergence-free in a real field.
    """
    n = shape[0]
    hat = np.fft.fft2(rng.standard_normal(shape), axes=(0, 1))
    hat[n // 2] = 0
    hat[:, n // 2] = 0
    v = np.fft.ifft2(hat, axes=(0, 1)).real
    v = project_divergence_free(v, DOMAIN_EXTENT)
    return (v / np.linalg.norm(v)).astype(np.float32)


@kernel(sweep_mode="none", catch_label="VJP failed")
def chain_fd_check(t: Any, ctx: KernelContext) -> dict:
    """Gradient of :func:`chain_loss` with respect to the start, against central differences.

    Records the same per-direction ``rel_error`` and ``cosine`` per ε as
    ``fd_check``. Directions are divergence-free, like the solver's states,
    and ε is relative to the start's RMS. The losses of the finite
    differences are summed in float64.
    """
    phys = dict(ctx.phys)
    n = int(phys.pop("N"))
    intervals = int(phys.pop("intervals"))
    offset = float(phys.pop("offset"))
    fd_cfg = ctx.run.get("fd", {})
    eps_values = fd_cfg["eps_values"]
    n_dirs = fd_cfg["n_dirs"]

    def step(v: Any) -> Any:
        inputs = ctx.make_inputs(ctx.name, v, **phys)
        return apply_tesseract(t, inputs)[ctx.output_key]

    def host_loss(v0: np.ndarray, targets: list[np.ndarray]) -> float:
        states = [np.asarray(s, dtype=np.float64) for s in chain(step, v0, intervals)]
        return float(chain_loss(states, targets, np))

    rng = np.random.default_rng(_SEED)
    seed = GROUND_TRUTH_SEEDS["test"][0]
    reference = initial_state(load_ground_truth(ctx.cfg.name), seed, n)
    targets = [np.asarray(s) for s in chain(step, jnp.asarray(reference), intervals)]
    if not all(np.isfinite(s).all() for s in targets):
        raise FloatingPointError("the chain from the ground-truth state diverged")
    start = reference + offset * np.linalg.norm(reference) * divergence_free_direction(
        reference.shape, rng
    )
    start = start.astype(np.float32)
    ic_scale = float(np.sqrt(np.mean(start**2)))

    g = jax.grad(lambda v: chain_loss(chain(step, v, intervals), targets))(
        jnp.asarray(start)
    )
    g = np.asarray(g)
    if not np.isfinite(g).all():
        raise ValueError("VJP returned a non-finite gradient")
    dirs = [divergence_free_direction(start.shape, rng) for _ in range(n_dirs)]
    vjp_arr = np.array([float(np.vdot(g, d)) for d in dirs])
    targets64 = [np.asarray(s, dtype=np.float64) for s in targets]

    eps_sweep: dict = {}
    for eps in eps_values:
        h = eps * ic_scale
        fd_arr = np.array(
            [
                (
                    host_loss(start + h * d, targets64)
                    - host_loss(start - h * d, targets64)
                )
                / (2 * h)
                for d in dirs
            ]
        )
        denom = np.maximum(np.maximum(np.abs(fd_arr), np.abs(vjp_arr)), 1e-30)
        eps_sweep[eps] = {
            "rel_error": (np.abs(fd_arr - vjp_arr) / denom).tolist(),
            "cosine": float(
                np.dot(fd_arr, vjp_arr)
                / (np.linalg.norm(fd_arr) * np.linalg.norm(vjp_arr) + 1e-30)
            ),
            "fd_arr": fd_arr.tolist(),
        }
    return {
        "metrics": {
            "ic_scale": ic_scale,
            "loss": host_loss(start, targets64),
            "vjp_arr": vjp_arr.tolist(),
            "eps_sweep": eps_sweep,
        },
        "snapshot": g,
        "shared": {"ic": start},
    }
