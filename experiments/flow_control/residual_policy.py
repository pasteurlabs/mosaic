"""Small shared policy that starts from the observed-field linear controller."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .baselines import linear_action_baseline
from .control import ControlConfig, Task, actuator_basis

MODES = tuple(
    (kx, ky)
    for kx in range(5)
    for ky in range(-4, 5)
    if 0 < kx * kx + ky * ky <= 16 and (kx > 0 or ky > 0)
)


class ResidualPolicy(eqx.Module):
    """64 physical features →32 hidden units →64 control-latent corrections."""

    hidden: eqx.nn.Linear
    output: eqx.nn.Linear
    slots: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, config: ControlConfig) -> None:
        hidden_key, output_key = jax.random.split(key)
        self.hidden = eqx.nn.Linear(2 * len(MODES) + 16, 32, key=hidden_key)
        output = eqx.nn.Linear(32, config.control_slots * 8, key=output_key)
        self.output = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            output,
            (jnp.zeros_like(output.weight), jnp.zeros_like(output.bias)),
        )
        self.slots = config.control_slots

    def __call__(self, features: jax.Array, baseline_latents: jax.Array) -> jax.Array:
        """Return baseline latents plus a learned correction; initialization is zero."""
        correction = self.output(jax.nn.gelu(self.hidden(features)))
        return baseline_latents + correction.reshape(self.slots, 8)


def init_residual_policy(seed: int, config: ControlConfig) -> ResidualPolicy:
    """Use paired initialization across every training method."""
    return ResidualPolicy(jax.random.PRNGKey(seed), config)


def attach_features(
    task: Task, zero_terminal: np.ndarray, config: ControlConfig
) -> Task:
    """Cache label-free physical observations and the linear baseline for one task.

    Fourier features are initial vorticity divided by wavenumber and velocity
    scale, retaining the complete real low-frequency disk |k|≤4. They describe
    the initial modes that can interact with forced modes. Additional features
    are actuator projections of goal-minus-unforced flow and clipped controls.
    No generating controls, fine targets or optimized action labels are read.
    """
    initial, goal, zero = map(np.asarray, (task.initial, task.goal, zero_terminal))
    shape = (config.coarse_n, config.coarse_n, 1, 2)
    if any(
        field.shape != shape or not np.isfinite(field).all()
        for field in (initial, goal, zero)
    ):
        raise ValueError("residual features require finite canonical coarse fields")
    spectrum = np.fft.fft2(initial[:, :, 0], axes=(0, 1), norm="forward")
    modes = []
    for kx, ky in MODES:
        ux, uy = spectrum[kx, ky]
        normalized_curl = (
            1j * (kx * uy - ky * ux) / (np.hypot(kx, ky) * config.velocity_scale)
        )
        modes.extend((normalized_curl.real, normalized_curl.imag))
    basis = np.asarray(actuator_basis(config.coarse_n, config.domain_extent))
    duration = config.control_slots * config.steps_per_slot * config.dt
    projection = np.einsum("mxyzc,xyzc->m", basis, goal - zero) / (
        config.coarse_n**2 * duration
    )
    controls = linear_action_baseline(
        goal,
        zero,
        basis,
        control_slots=config.control_slots,
        control_bound=config.control_bound,
        duration=duration,
    )
    latents = np.arctanh(
        np.clip(controls / config.control_bound, -1 + 1e-6, 1 - 1e-6)
    ).astype(np.float32)
    features = np.concatenate(
        (modes, projection / config.control_bound, controls[0] / config.control_bound)
    ).astype(np.float32)
    return replace(task, policy_features=features, baseline_latents=latents)
