"""Amortized periodic flow control with the same coarse and fine solver.

Numerical entry points must run in cluster allocations. Targets and actuators
are sampled at the fine grid first; task labels are retained for fair imitation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.problems.navier_stokes_grid.corrector import spectral_restrict
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _multimode
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _unforced_solver_advance,
)

_WAVES = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class ControlConfig:
    """Physical task and gate thresholds, frozen before inspecting outcomes."""

    coarse_n: int = 64
    reference_n: int = 192
    dt: float = 0.02
    steps_per_slot: int = 4
    control_slots: int = 8
    nu: float = 0.001
    domain_extent: float = 2 * np.pi
    control_bound: float = 0.25
    velocity_scale: float = 0.5
    effort_weight: float = 0.01
    reference_temporal_factor: int = 3
    audit_temporal_factor: int = 6
    reference_tolerance: float = 0.005
    gradient_tolerance: float = 0.05
    gradient_epsilons: tuple[float, ...] = (0.001, 0.01, 0.0001)
    shooting_updates: int = 25
    shooting_lr: float = 0.05

    def __post_init__(self) -> None:
        if self.coarse_n != 64 or self.reference_n != 192:
            raise ValueError("this protocol fixes coarse64/reference192")
        for value in (
            self.dt,
            self.domain_extent,
            self.control_bound,
            self.velocity_scale,
            self.reference_tolerance,
            self.gradient_tolerance,
            self.shooting_lr,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "control scales and tolerances must be finite/positive"
                )
        if not np.isfinite(self.nu) or self.nu < 0:
            raise ValueError("viscosity must be finite and nonnegative")
        if not np.isfinite(self.effort_weight) or self.effort_weight < 0:
            raise ValueError("effort_weight must be finite and nonnegative")
        for value in (
            self.steps_per_slot,
            self.control_slots,
            self.reference_temporal_factor,
            self.audit_temporal_factor,
            self.shooting_updates,
        ):
            if int(value) != value or value < 1:
                raise ValueError("step counts must be positive integers")
        if self.audit_temporal_factor <= self.reference_temporal_factor:
            raise ValueError("the audit must use smaller timesteps than the reference")
        if not self.gradient_epsilons or any(
            not np.isfinite(value) or value <= 0 for value in self.gradient_epsilons
        ):
            raise ValueError("gradient steps must be finite and positive")


@dataclass(frozen=True)
class Task:
    """A goal pair, including its feasible generating control demonstration."""

    seed: int
    initial: np.ndarray
    goal: np.ndarray
    fine_initial: np.ndarray
    fine_goal: np.ndarray
    generating_controls: np.ndarray
    initial_seed: int | None = None
    goal_seed: int | None = None
    fine_goal_rollout: np.ndarray | None = None
    policy_features: np.ndarray | None = None
    baseline_latents: np.ndarray | None = None


@lru_cache(maxsize=16)
def _actuator_constants(n: int, domain_extent: float) -> np.ndarray:
    """Construct static constants accurately; never cache a traced JAX value."""
    if n < 4 or not np.isfinite(domain_extent) or domain_extent <= 0:
        raise ValueError("actuators require a resolved periodic grid")
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    modes = []
    for kx, ky in _WAVES:
        # Integer torus phases give identical samples for equivalent phases.
        # Physical coordinates x=L*i/n cancel L in the phase 2*pi*k*x/L.
        phase = 2 * np.pi * ((kx * x + ky * y) % n) / n
        normal = np.asarray((ky, -kx), dtype=np.float64)
        normal *= np.sqrt(2.0 / (kx * kx + ky * ky))
        for trigonometric in (np.cos, np.sin):
            modes.append(trigonometric(phase)[..., None, None] * normal)
    constants = np.stack(modes).astype(np.float32)
    constants.setflags(write=False)
    return constants


def actuator_basis(n: int, domain_extent: float = 2 * np.pi) -> jax.Array:
    """Eight vector-RMS-orthonormal divergence-free physical force modes.

    Canonical Fourier nodes share the same physical modes at both resolutions.
    All selected wavevectors have equal nonzero component sizes, so centered
    finite differences also see zero divergence, up to float32 storage error.
    Constants are computed in float64 and cast once; only controls require AD.
    """
    return jnp.asarray(_actuator_constants(n, float(domain_extent)))


def controls_from_latent(latents: jax.Array, config: ControlConfig) -> jax.Array:
    """Use the same smooth bounded action map for every learned method."""
    return config.control_bound * jnp.tanh(latents)


def control_effort(controls: jax.Array, config: ControlConfig) -> jax.Array:
    """Normalized integrated force energy for equal-duration orthonormal modes."""
    return jnp.mean((controls / config.control_bound) ** 2)


def terminal_loss(
    final: jax.Array,
    goal: jax.Array,
    controls: jax.Array,
    config: ControlConfig,
) -> jax.Array:
    """Terminal velocity mismatch plus normalized time-integrated control energy."""
    return jnp.mean(
        (final - goal) ** 2
    ) / config.velocity_scale**2 + config.effort_weight * control_effort(
        controls, config
    )


def rollout(
    t: Any,
    ctx: Any,
    initial: jax.Array,
    controls: jax.Array,
    config: ControlConfig,
    *,
    temporal_factor: int = 1,
    return_states: bool = False,
) -> jax.Array:
    """Advance bounded controls with native state retained across every step.

    Each native step receives half a force kick before and after its unforced
    drift. Canonical/native reconciliation retains the final kick on the next
    call. Controls stay JAX arrays throughout, including backward evaluation.
    """
    if not np.isclose(float(ctx.phys["nu"]), config.nu, rtol=0, atol=1e-12):
        raise ValueError("solver context viscosity differs from frozen control config")
    if not np.isclose(
        float(ctx.domain_extent), config.domain_extent, rtol=0, atol=1e-12
    ):
        raise ValueError("solver context domain differs from frozen control config")
    if temporal_factor < 1 or int(temporal_factor) != temporal_factor:
        raise ValueError("temporal_factor must be a positive integer")
    if controls.shape != (config.control_slots, 8):
        raise ValueError("controls must have shape (control_slots,8)")
    if initial.ndim != 4 or initial.shape[-2:] != (1, 2):
        raise ValueError("initial velocity must use canonical(N,N,1,2) layout")
    if initial.shape[0] != initial.shape[1]:
        raise ValueError("this control protocol requires square grids")
    state = jnp.asarray(initial)
    basis = actuator_basis(state.shape[0], config.domain_extent)
    native = None
    states = [state] if return_states else None
    dt = config.dt / temporal_factor
    for slot in range(config.control_slots):
        force = jnp.einsum("m,mxyzc->xyzc", controls[slot], basis)
        for _ in range(config.steps_per_slot * temporal_factor):
            state, native = _unforced_solver_advance(
                t,
                ctx,
                state + 0.5 * dt * force,
                dt=dt,
                steps=1,
                native_state=native,
            )
            state = state + 0.5 * dt * force
        if return_states:
            states.append(state)
    return jnp.stack(states) if return_states else state


def objective(
    t: Any,
    ctx: Any,
    initial: jax.Array,
    goal: jax.Array,
    controls: jax.Array,
    config: ControlConfig,
    *,
    temporal_factor: int = 1,
) -> jax.Array:
    """Evaluate the shared physical objective without requiring action labels."""
    final = rollout(t, ctx, initial, controls, config, temporal_factor=temporal_factor)
    return terminal_loss(final, goal, controls, config)


def generate_tasks(
    t: Any, ctx: Any, config: ControlConfig, seeds: list[int] | tuple[int, ...]
) -> list[Task]:
    """Generate reachable fine-grid goals; preserve every generating control."""
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("task seeds must be nonempty, unique and nonnegative")
    tasks = []
    for seed in seeds:
        initial = _multimode(
            config.reference_n,
            L=config.domain_extent,
            seed=int(seed),
            k0=4,
            sigma_k=1,
            amplitude=config.velocity_scale,
        )
        goal_seed = 10_000_000 + int(seed)
        rng = np.random.default_rng(goal_seed)
        raw = rng.standard_normal((config.control_slots, 8))
        smooth = (2 * raw + np.roll(raw, 1, axis=0) + np.roll(raw, -1, axis=0)) / 4
        controls = (0.6 * config.control_bound * np.tanh(smooth)).astype(np.float32)
        goal_rollout = rollout(
            t,
            ctx,
            initial,
            jnp.asarray(controls),
            config,
            temporal_factor=config.reference_temporal_factor,
            return_states=True,
        )
        goal = goal_rollout[-1]
        task = Task(
            seed=int(seed),
            initial=np.asarray(spectral_restrict(initial, config.coarse_n)),
            goal=np.asarray(spectral_restrict(goal, config.coarse_n)),
            fine_initial=np.asarray(initial),
            fine_goal=np.asarray(goal),
            generating_controls=controls,
            initial_seed=int(seed),
            goal_seed=goal_seed,
            fine_goal_rollout=np.asarray(goal_rollout),
        )
        if not all(
            np.isfinite(a).all() for a in (task.initial, task.goal, task.fine_goal)
        ):
            raise RuntimeError(f"nonfinite generated task {seed}")
        tasks.append(task)
        print(f"flow-control task seed={seed} generated", flush=True)
    return tasks


def gradient_check(
    t: Any, ctx: Any, task: Task, config: ControlConfig
) -> dict[str, Any]:
    """Probe one fixed coefficient point in three directions at all step sizes."""
    point = jnp.zeros_like(jnp.asarray(task.generating_controls))

    def loss(controls: jax.Array) -> jax.Array:
        return objective(t, ctx, task.initial, task.goal, controls, config)

    gradient = jax.grad(loss)(point)
    norm = float(jnp.linalg.norm(gradient))
    rng = np.random.default_rng(np.random.SeedSequence([task.seed, 117]))
    directions = [gradient / max(norm, 1e-20)]
    for _ in range(2):
        direction = jnp.asarray(rng.standard_normal(point.shape), dtype=point.dtype)
        directions.append(direction / jnp.linalg.norm(direction))
    checks = []
    for index, direction in enumerate(directions):
        ad = float(jnp.sum(gradient * direction))
        for epsilon in config.gradient_epsilons:
            finite_difference = float(
                (loss(point + epsilon * direction) - loss(point - epsilon * direction))
                / (2 * epsilon)
            )
            checks.append(
                {
                    "direction": index,
                    "kind": "gradient_aligned" if index == 0 else "random",
                    "epsilon": epsilon,
                    "autodiff": ad,
                    "finite_difference": finite_difference,
                    "absolute_error": abs(ad - finite_difference),
                    "relative_error": abs(ad - finite_difference)
                    / max(abs(ad), abs(finite_difference), 1e-12),
                    "directional_signal_fraction": abs(ad) / max(norm, 1e-20),
                }
            )
    primary = [row for row in checks if row["epsilon"] == config.gradient_epsilons[0]]
    return {
        "gradient_norm": norm,
        "primary_epsilon": config.gradient_epsilons[0],
        "checks": checks,
        "passed": bool(
            np.isfinite(norm)
            and norm > 0
            and all(
                np.isfinite(row["relative_error"])
                and row["relative_error"] < config.gradient_tolerance
                for row in primary
            )
        ),
    }


def direct_shooting(
    t: Any,
    ctx: Any,
    task: Task,
    config: ControlConfig,
    *,
    initial_controls: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optimize bounded controls, with an explicitly recorded optional warm start."""
    started = time.perf_counter()
    requested = np.zeros((config.control_slots, 8), dtype=np.float32)
    initialization = "zero" if initial_controls is None else "provided_controls"
    if initial_controls is not None:
        requested = np.asarray(initial_controls, dtype=np.float32)
        if requested.shape != (config.control_slots, 8):
            raise ValueError("initial controls must have shape (control_slots, 8)")
        if (
            not np.isfinite(requested).all()
            or np.max(np.abs(requested)) > config.control_bound
        ):
            raise ValueError(
                "initial controls must be finite and within the shared bound"
            )
    # tanh cannot represent the exact closed boundary at finite latents.
    # Preserve and report the small explicit inward transformation.
    normalized = np.clip(requested / config.control_bound, -1 + 1e-6, 1 - 1e-6)
    latents = jnp.asarray(np.arctanh(normalized), dtype=jnp.float32)
    realized = np.asarray(controls_from_latent(latents, config))
    optimizer = optax.chain(
        optax.clip_by_global_norm(5), optax.adam(config.shooting_lr)
    )
    state = optimizer.init(latents)

    def loss(value: jax.Array) -> jax.Array:
        return objective(
            t, ctx, task.initial, task.goal, controls_from_latent(value, config), config
        )

    trace = []
    best, best_loss = latents, np.inf
    completed, failure = True, None
    rollout_count = gradient_count = failed_count = 0
    for _update in range(config.shooting_updates):
        rollout_count += 1
        gradient_count += 1
        try:
            value, gradient = jax.value_and_grad(loss)(latents)
            number = float(value)
            if not np.isfinite(number) or not bool(jnp.all(jnp.isfinite(gradient))):
                raise FloatingPointError("nonfinite shooting objective or gradient")
        except Exception as error:
            completed, failure = False, f"{type(error).__name__}: {error}"
            failed_count += 1
            break
        if number < best_loss:
            best, best_loss = latents, number
        trace.append(number)
        updates, state = optimizer.update(gradient, state, latents)
        latents = optax.apply_updates(latents, updates)
    last_loss = None
    if completed:
        rollout_count += 1
        try:
            last_loss = float(loss(latents))
            if not np.isfinite(last_loss):
                raise FloatingPointError("nonfinite final shooting objective")
            if last_loss < best_loss:
                best, best_loss = latents, last_loss
        except Exception as error:
            completed, failure = False, f"{type(error).__name__}: {error}"
            failed_count += 1
    controls = np.asarray(controls_from_latent(best, config))
    return {
        "controls": controls,
        "latents": np.asarray(best),
        "initialization": initialization,
        "initial_controls_sha256": hashlib.sha256(requested.tobytes()).hexdigest(),
        "initial_controls_max_transform_error": float(
            np.max(np.abs(realized - requested))
        ),
        "initial_controls": realized,
        "trace": trace,
        "initial_loss": trace[0] if trace else None,
        "final_loss": float(best_loss),
        "last_iterate_loss": last_loss,
        "completed": completed,
        "failure": failure,
        "wall_time_s": time.perf_counter() - started,
        "rollout_count": rollout_count,
        "gradient_rollout_count": gradient_count,
        "failed_rollout_count": failed_count,
        "updates": len(trace),
    }


class ControlPolicy(eqx.Module):
    """Small shared field-to-control MLP; returns unbounded action latents."""

    layers: tuple[eqx.nn.Linear, ...]
    slots: int = eqx.field(static=True)
    velocity_scale: float = eqx.field(static=True)

    def __init__(self, key: jax.Array, config: ControlConfig) -> None:
        keys = jax.random.split(key, 3)
        self.layers = (
            eqx.nn.Linear(16 * 16 * 4, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, config.control_slots * 8, key=keys[2]),
        )
        self.slots = config.control_slots
        self.velocity_scale = config.velocity_scale

    def __call__(self, initial: jax.Array, goal: jax.Array) -> jax.Array:
        """Encode both observed fields without inspecting generating controls."""
        field = jnp.concatenate([jnp.asarray(initial), jnp.asarray(goal)], axis=-1)[
            :, :, 0
        ]
        if field.shape[:2] != (64, 64):
            raise ValueError("policy observes the common64² fields")
        pooled = field.reshape(16, 4, 16, 4, 4).mean(axis=(1, 3))
        hidden = pooled.reshape(-1) / self.velocity_scale
        for layer in self.layers[:-1]:
            hidden = jax.nn.gelu(layer(hidden))
        return self.layers[-1](hidden).reshape(self.slots, 8)


def init_policy(
    seed: int, config: ControlConfig, *, architecture: str = "field_mlp"
) -> Any:
    """Initialize all policy-training methods from the same model seed."""
    if architecture == "linear_residual":
        from .residual_policy import init_residual_policy

        return init_residual_policy(seed, config)
    if architecture != "field_mlp":
        raise ValueError(f"Unknown policy architecture: {architecture}")
    return ControlPolicy(jax.random.PRNGKey(seed), config)


def policy_latents(model: ControlPolicy, task: Task) -> jax.Array:
    """Compute action latents before applying the common bound."""
    from .residual_policy import ResidualPolicy

    if isinstance(model, ResidualPolicy):
        if task.policy_features is None or task.baseline_latents is None:
            raise ValueError("residual policies require cached task features")
        return model(
            jnp.asarray(task.policy_features), jnp.asarray(task.baseline_latents)
        )
    return model(jnp.asarray(task.initial), jnp.asarray(task.goal))


def policy_controls(
    model: ControlPolicy, task: Task, config: ControlConfig
) -> jax.Array:
    """Compute bounded coefficients from the shared field observations."""
    return controls_from_latent(policy_latents(model, task), config)


def train_policy(
    t: Any,
    ctx: Any,
    tasks: list[Task],
    config: ControlConfig,
    *,
    seed: int = 0,
    updates: int = 500,
    lr: float = 1e-3,
    wall_time_budget_s: float | None = None,
    model: Any | None = None,
    checkpoint_updates: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Train directly on terminal objectives; no expert action labels are used."""
    if not tasks or updates < 1 or lr <= 0:
        raise ValueError("training needs tasks and positive updates/learning rate")
    if wall_time_budget_s is not None and wall_time_budget_s <= 0:
        raise ValueError("wall_time_budget_s must be positive")
    if any(
        int(value) != value or not 1 <= value <= updates for value in checkpoint_updates
    ):
        raise ValueError(
            "checkpoint updates must be integers within the training budget"
        )
    checkpoint_models = {}
    started = time.perf_counter()
    model = init_policy(seed, config) if model is None else model
    optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adam(lr))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.default_rng(seed)
    trace = []
    completed, failure = True, None
    attempted = failed = 0
    for update in range(updates):
        if (
            wall_time_budget_s is not None
            and time.perf_counter() - started >= wall_time_budget_s
        ):
            break
        task = tasks[int(rng.integers(len(tasks)))]

        def loss(candidate: ControlPolicy, task: Task = task) -> jax.Array:
            return objective(
                t,
                ctx,
                task.initial,
                task.goal,
                policy_controls(candidate, task, config),
                config,
            )

        attempted += 1
        try:
            value, gradient = eqx.filter_value_and_grad(loss)(model)
            number, norm = float(value), float(optax.tree.norm(gradient))
            if not np.isfinite(number) or not np.isfinite(norm):
                raise FloatingPointError("nonfinite policy objective or gradient")
        except Exception as error:
            completed, failure = False, f"{type(error).__name__}: {error}"
            failed += 1
            break
        change, state = optimizer.update(
            gradient, state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, change)
        jax.block_until_ready((model, state))
        if update + 1 in checkpoint_updates:
            checkpoint_models[update + 1] = model
        trace.append(
            {
                "update": update + 1,
                "task_seed": task.seed,
                "loss": number,
                "gradient_norm": norm,
                "wall_time_s": time.perf_counter() - started,
            }
        )
        if (update + 1) % 25 == 0:
            print(
                f"control policy seed={seed} update={update + 1} loss={number:g}",
                flush=True,
            )
    return {
        "model": model,
        "checkpoint_models": checkpoint_models
        if checkpoint_updates
        else {len(trace): model},
        "trace": trace,
        "completed": completed,
        "failure": failure,
        "updates": len(trace),
        "wall_time_s": time.perf_counter() - started,
        "rollout_count": attempted,
        "gradient_rollout_count": attempted,
        "failed_rollout_count": failed,
    }


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def run_gate(
    t: Any, ctx: Any, config: ControlConfig, seeds: list[int] | tuple[int, ...]
) -> dict[str, Any]:
    """Admit task generation, coefficient gradients and direct inverse control."""
    started = time.perf_counter()
    from experiments.flow_control.baselines import linear_action_baseline

    tasks = generate_tasks(t, ctx, config, seeds)
    rows, coarse_rollouts, fine_rollouts, zero_rollouts = [], [], [], []
    shooting_controls, linear_controls, linear_rollouts = [], [], []
    for task in tasks:
        gradient = gradient_check(t, ctx, task, config)
        shooting = direct_shooting(t, ctx, task, config)
        controls = jnp.asarray(shooting["controls"])
        coarse = np.asarray(
            rollout(t, ctx, task.initial, controls, config, return_states=True)
        )
        fine = np.asarray(
            rollout(
                t,
                ctx,
                task.fine_initial,
                controls,
                config,
                temporal_factor=config.reference_temporal_factor,
                return_states=True,
            )
        )
        zero = np.asarray(
            rollout(
                t,
                ctx,
                task.fine_initial,
                jnp.zeros_like(controls),
                config,
                temporal_factor=config.reference_temporal_factor,
                return_states=True,
            )
        )
        zero_coarse = rollout(t, ctx, task.initial, jnp.zeros_like(controls), config)
        linear = linear_action_baseline(
            task.goal,
            np.asarray(zero_coarse),
            np.asarray(actuator_basis(config.coarse_n, config.domain_extent)),
            control_slots=config.control_slots,
            control_bound=config.control_bound,
            duration=config.control_slots * config.steps_per_slot * config.dt,
        )
        linear_fine = np.asarray(
            rollout(
                t,
                ctx,
                task.fine_initial,
                jnp.asarray(linear),
                config,
                temporal_factor=config.reference_temporal_factor,
                return_states=True,
            )
        )
        audit = np.asarray(
            rollout(
                t,
                ctx,
                task.fine_initial,
                controls,
                config,
                temporal_factor=config.audit_temporal_factor,
            )
        )
        goal_audit = np.asarray(
            rollout(
                t,
                ctx,
                task.fine_initial,
                jnp.asarray(task.generating_controls),
                config,
                temporal_factor=config.audit_temporal_factor,
            )
        )
        fine_loss = float(
            terminal_loss(
                jnp.asarray(fine[-1]), jnp.asarray(task.fine_goal), controls, config
            )
        )
        zero_loss = float(
            terminal_loss(
                jnp.asarray(zero[-1]),
                jnp.asarray(task.fine_goal),
                jnp.zeros_like(controls),
                config,
            )
        )
        finite = all(
            np.isfinite(a).all()
            for a in (coarse, fine, zero, audit, goal_audit, linear_fine)
        )
        row = {
            "seed": task.seed,
            "initial_seed": task.initial_seed,
            "goal_seed": task.goal_seed,
            "gradient_check": gradient,
            "shooting": {
                key: value
                for key, value in shooting.items()
                if key not in {"controls", "latents", "initial_controls"}
            },
            "goal_temporal_error": _relative(goal_audit, task.fine_goal),
            "controlled_temporal_error": _relative(audit, fine[-1]),
            "fine_objective": fine_loss,
            "fine_terminal_mse_normalized": float(
                np.mean((fine[-1] - task.fine_goal) ** 2) / config.velocity_scale**2
            ),
            "weighted_control_effort": float(
                config.effort_weight * control_effort(controls, config)
            ),
            "linear_fine_objective": float(
                terminal_loss(
                    jnp.asarray(linear_fine[-1]),
                    jnp.asarray(task.fine_goal),
                    jnp.asarray(linear),
                    config,
                )
            ),
            "linear_fine_terminal_error": _relative(linear_fine[-1], task.fine_goal),
            "linear_control_effort": float(control_effort(jnp.asarray(linear), config)),
            "fine_zero_objective": zero_loss,
            "fine_terminal_error": _relative(fine[-1], task.fine_goal),
            "zero_terminal_error": _relative(zero[-1], task.fine_goal),
            "control_effort": float(control_effort(controls, config)),
            "finite": bool(finite),
        }
        row["passed"] = bool(
            finite
            and gradient["passed"]
            and shooting["completed"]
            and shooting["initial_loss"] is not None
            and shooting["final_loss"] < shooting["initial_loss"]
            and fine_loss < zero_loss
            and row["goal_temporal_error"] < config.reference_tolerance
            and row["controlled_temporal_error"] < config.reference_tolerance
        )
        rows.append(row)
        coarse_rollouts.append(coarse)
        fine_rollouts.append(fine)
        zero_rollouts.append(zero)
        shooting_controls.append(np.asarray(controls))
        linear_controls.append(linear)
        linear_rollouts.append(linear_fine)
        print(
            f"flow-control gate seed={task.seed} passed={row['passed']} "
            f"initial={shooting['initial_loss']} final={shooting['final_loss']}",
            flush=True,
        )
    return {
        "metrics": {
            "passed": all(row["passed"] for row in rows),
            "tasks": rows,
            "wall_time_s": time.perf_counter() - started,
            "task_seeds": [task.seed for task in tasks],
            "reference_scope": "same_solver192_time_refinement",
            "neural_training_started": False,
        },
        "arrays": {
            "initial": np.stack([task.initial for task in tasks]),
            "goal": np.stack([task.goal for task in tasks]),
            "fine_initial": np.stack([task.fine_initial for task in tasks]),
            "fine_goal": np.stack([task.fine_goal for task in tasks]),
            "initial_seeds": np.asarray([task.initial_seed for task in tasks]),
            "goal_seeds": np.asarray([task.goal_seed for task in tasks]),
            "fine_goal_rollouts": np.stack([task.fine_goal_rollout for task in tasks]),
            "shooting_controls": np.stack(shooting_controls),
            "linear_controls": np.stack(linear_controls),
            "fine_linear_rollouts": np.stack(linear_rollouts),
            "generating_controls": np.stack(
                [task.generating_controls for task in tasks]
            ),
            "coarse_shooting_rollouts": np.stack(coarse_rollouts),
            "fine_shooting_rollouts": np.stack(fine_rollouts),
            "fine_zero_rollouts": np.stack(zero_rollouts),
            "times": np.arange(config.control_slots + 1)
            * config.steps_per_slot
            * config.dt,
        },
    }
