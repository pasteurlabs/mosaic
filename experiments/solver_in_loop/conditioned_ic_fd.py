"""Versioned IC-only finite differences; does not modify admission or model FD.

Center the field before scalar reduction and specify componentwise RMS
perturbation amplitude. Return the entire fixed sweep without selecting an
acceptable epsilon. Callers must preserve the earlier diagnostic separately.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

PROTOCOL = "ic_fd_centered_rms_v1"
DEFAULT_EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def conditioned_ic_fd(
    forward: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    *,
    direction_seed: int = 117,
    cotangent_seed: int = 116,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
) -> dict:
    """Measure fixed centered-linear directional FD at every prescribed step.

    This diagnostic has no admission decision. In particular it neither picks
    the best epsilon nor replaces the frozen model-parameter gradient check.
    """
    steps = tuple(float(eps) for eps in epsilons)
    if not steps or any(not np.isfinite(eps) or eps <= 0 for eps in steps):
        raise ValueError("finite positive RMS perturbations required")
    initial = jnp.asarray(initial)
    if not jnp.issubdtype(initial.dtype, jnp.inexact):
        raise ValueError("floating-point initial field required")
    # Generate identical random arrays for float32/float64 paired audits.
    direction = jax.random.normal(
        jax.random.PRNGKey(direction_seed), initial.shape, dtype=jnp.float32
    ).astype(initial.dtype)
    direction = direction / jnp.sqrt(jnp.mean(direction**2))
    baseline = jax.lax.stop_gradient(forward(initial))
    cotangent = jax.random.normal(
        jax.random.PRNGKey(cotangent_seed), baseline.shape, dtype=jnp.float32
    ).astype(baseline.dtype)
    cotangent = cotangent / jnp.linalg.norm(cotangent)

    def objective(value: jax.Array) -> jax.Array:
        return jnp.vdot(forward(value) - baseline, cotangent).real

    gradient = jax.grad(objective)(initial)
    ad = float(jnp.vdot(gradient, direction).real)
    sweep = []
    for epsilon in steps:
        plus = initial + epsilon * direction
        minus = initial - epsilon * direction
        positive = float(objective(plus))
        negative = float(objective(minus))
        fd = (positive - negative) / (2 * epsilon)
        sweep.append(
            {
                "epsilon_rms": epsilon,
                "realized_positive_rms": float(
                    jnp.sqrt(jnp.mean((plus - initial) ** 2))
                ),
                "realized_negative_rms": float(
                    jnp.sqrt(jnp.mean((initial - minus) ** 2))
                ),
                "positive_centered_objective": positive,
                "negative_centered_objective": negative,
                "finite_difference": fd,
                "autodiff": ad,
                "relative_error": abs(fd - ad) / (abs(fd) + abs(ad) + 1e-12),
            }
        )
    return {
        "protocol": PROTOCOL,
        "scope": "IC field only; no admission override; model-parameter FD unchanged",
        "dtype": str(initial.dtype),
        "input_shape": list(initial.shape),
        "direction_seed": direction_seed,
        "cotangent_seed": cotangent_seed,
        "direction_rms": float(jnp.sqrt(jnp.mean(direction**2))),
        "direction_l2": float(jnp.linalg.norm(direction)),
        "finite_gradient": bool(jnp.isfinite(gradient).all()),
        "finite_baseline": bool(jnp.isfinite(baseline).all()),
        "sweep": sweep,
    }
