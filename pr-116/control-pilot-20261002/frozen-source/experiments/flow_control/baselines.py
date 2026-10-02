"""Action-space SPSA and fixed-label imitation with explicit oracle accounting.

SPSA perturbs the H×M action latents, never the neural network parameters. Both
sides use the same bounded tanh controls and physical objective as direct AD.
Imitation labels are fixed before fitting. Extra expert optimization is charged
explicitly. Known generating controls need no extra label computation: common
goal-generation cost belongs in shared dataset metadata for every method.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def linear_action_baseline(
    goal: np.ndarray,
    zero_control_terminal: np.ndarray,
    basis: np.ndarray,
    *,
    control_slots: int,
    control_bound: float,
    duration: float,
) -> np.ndarray:
    """Project the terminal discrepancy onto constant bounded actuator forcing.

    This diagnostic ignores nonlinear response to forcing. It uses one
    zero-control terminal field, never generating controls or optimized labels.
    The shared Fourier basis is orthogonal, so clipping the unconstrained
    least-squares coefficients also solves its box-constrained linear problem.
    """
    if control_slots < 1 or control_bound <= 0 or not np.isfinite(control_bound):
        raise ValueError("positive control slots and finite positive bound required")
    if duration <= 0 or not np.isfinite(duration):
        raise ValueError("duration must be finite and positive")
    goal, terminal, basis = map(np.asarray, (goal, zero_control_terminal, basis))
    if (
        goal.shape != terminal.shape
        or basis.ndim != goal.ndim + 1
        or basis.shape[1:] != goal.shape
    ):
        raise ValueError("actuator basis and terminal field shapes must agree")
    if not all(np.all(np.isfinite(value)) for value in (goal, terminal, basis)):
        raise ValueError("linear baseline requires finite fields and actuators")
    matrix = np.asarray(basis.reshape(basis.shape[0], -1).T, dtype=np.float64)
    gram = matrix.T @ matrix
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0) or not np.allclose(
        gram, np.diag(diagonal), rtol=1e-6, atol=1e-6 * float(np.max(diagonal))
    ):
        raise ValueError(
            "clipped least squares requires the shared orthogonal actuator basis"
        )
    discrepancy = np.asarray((goal - terminal).reshape(-1), dtype=np.float64) / duration
    coefficients = (matrix.T @ discrepancy) / diagonal
    bounded = np.clip(coefficients, -control_bound, control_bound).astype(np.float32)
    return np.broadcast_to(bounded, (control_slots, basis.shape[0])).copy()


@dataclass
class Work:
    """Wall time includes oracle time; these two fields must not be added."""

    wall_time_s: float = 0.0
    oracle_wall_time_s: float = 0.0
    rollout_count: int = 0
    gradient_rollout_count: int = 0
    failed_rollout_count: int = 0
    optimizer_updates: int = 0
    label_examples: int = 0


@dataclass
class CostLedger:
    """Separate offline label generation from policy fitting and evaluation."""

    phases: dict[str, Work] = field(default_factory=dict)

    def phase(self, name: str) -> Work:
        """Get one phase's mutable counters."""
        return self.phases.setdefault(name, Work())

    def total(self) -> Work:
        """Sum disjoint phases without double-counting nested oracle time."""
        return Work(
            **{
                key: sum(getattr(work, key) for work in self.phases.values())
                for key in asdict(Work())
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable counters, including failures and offline costs."""
        return {
            "phases": {key: asdict(value) for key, value in self.phases.items()},
            "total": asdict(self.total()),
            "accounting": (
                "wall_time_s includes oracle_wall_time_s; each scalar objective "
                "query counts one requested full rollout, including failed queries. "
                "This is not a count of backend-internal replay solves."
            ),
        }


class CountedObjective:
    """Charge every action query and retain the best actually evaluated action."""

    def __init__(
        self,
        objective: Callable[[jax.Array], jax.Array],
        ledger: CostLedger,
        *,
        phase: str,
    ) -> None:
        self.objective = objective
        self.work = ledger.phase(phase)
        self.best_loss = float("inf")
        self.best_controls: np.ndarray | None = None

    def __call__(self, controls: jax.Array) -> float:
        """Synchronize each query so timing includes completed solver work."""
        self.work.rollout_count += 1
        started = time.perf_counter()
        try:
            value = float(jax.block_until_ready(self.objective(controls)))
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite action objective")
            if value < self.best_loss:
                self.best_loss = value
                self.best_controls = np.asarray(controls).copy()
            return value
        except Exception:
            self.work.failed_rollout_count += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.work.oracle_wall_time_s += elapsed
            self.work.wall_time_s += elapsed


def antithetic_action_gradient(
    latents: jax.Array,
    objective: Callable[[jax.Array], float],
    *,
    control_bound: float,
    perturbation: float,
    key: jax.Array,
    directions: int = 1,
) -> tuple[jax.Array, dict[str, Any]]:
    """Estimate the latent-action gradient with 2×directions physical rollouts.

    The perturbation is in latent coefficient units. The tanh map guarantees
    feasible physical controls without asymmetric clipping at action bounds.
    The reported pair mean is a smoothing diagnostic, not an unperturbed loss.
    """
    if control_bound <= 0 or not np.isfinite(control_bound):
        raise ValueError("control_bound must be finite and positive")
    if perturbation <= 0 or not np.isfinite(perturbation) or directions < 1:
        raise ValueError("positive perturbation and at least one direction required")
    estimate = jnp.zeros_like(latents)
    pairs = []
    for direction_key in jax.random.split(key, directions):
        direction = jax.random.rademacher(
            direction_key, latents.shape, dtype=latents.dtype
        )
        plus = control_bound * jnp.tanh(latents + perturbation * direction)
        minus = control_bound * jnp.tanh(latents - perturbation * direction)
        plus_loss, minus_loss = objective(plus), objective(minus)
        if not np.isfinite(plus_loss) or not np.isfinite(minus_loss):
            raise FloatingPointError("nonfinite antithetic objective")
        estimate = (
            estimate + ((plus_loss - minus_loss) / (2 * perturbation)) * direction
        )
        pairs.append([float(plus_loss), float(minus_loss)])
    return estimate / directions, {
        "paired_loss_mean": float(np.mean(pairs)),
        "query_losses": pairs,
        "rollout_count": 2 * directions,
        "perturbation_latent_units": perturbation,
    }


def optimise_spsa_actions(
    objective: Callable[[jax.Array], jax.Array],
    initial_latents: jax.Array,
    *,
    control_bound: float,
    seed: int,
    updates: int,
    lr: float,
    perturbation: float,
    directions: int = 1,
    ledger: CostLedger | None = None,
    phase: str = "label_generation",
) -> dict[str, Any]:
    """Optional black-box action teacher; return its best queried feasible label.

    This costs 2+2×directions×updates rollouts: initial, antithetic pairs, final.
    Selecting the best queried action requires no uncharged evaluation.
    """
    if updates < 0 or lr <= 0 or not np.isfinite(lr):
        raise ValueError("nonnegative updates and finite positive lr required")
    ledger = CostLedger() if ledger is None else ledger
    work = ledger.phase(phase)
    counted = CountedObjective(objective, ledger, phase=phase)
    accounted_before = work.wall_time_s
    started = time.perf_counter()
    try:
        latents = jnp.asarray(initial_latents)
        optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adam(lr))
        state = optimizer.init(latents)
        counted(control_bound * jnp.tanh(latents))
        trace = []
        for update in range(updates):
            gradient, diagnostic = antithetic_action_gradient(
                latents,
                counted,
                control_bound=control_bound,
                perturbation=perturbation,
                key=jax.random.fold_in(jax.random.PRNGKey(seed), update),
                directions=directions,
            )
            delta, state = optimizer.update(gradient, state, latents)
            latents = optax.apply_updates(latents, delta)
            work.optimizer_updates += 1
            trace.append(
                {
                    "update": update + 1,
                    **diagnostic,
                    "best_queried_loss": counted.best_loss,
                }
            )
        counted(control_bound * jnp.tanh(latents))
        work.label_examples += 1
    finally:
        elapsed = time.perf_counter() - started
        already_charged = work.wall_time_s - accounted_before
        work.wall_time_s += max(0.0, elapsed - already_charged)
    return {
        "controls": counted.best_controls,
        "loss": counted.best_loss,
        "trace": trace,
        "costs": ledger.to_dict(),
    }


def _training_inputs(
    tasks: Sequence[Any], updates: int, lr: float, budget: float | None
) -> None:
    if not tasks or updates < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("training requires tasks, positive updates and positive lr")
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError("wall_time_budget_s must be finite and positive")


def train_spsa_policy(
    t: Any,
    ctx: Any,
    tasks: Sequence[Any],
    config: Any,
    *,
    seed: int = 0,
    updates: int = 100,
    lr: float = 1e-3,
    perturbation: float = 0.05,
    directions: int = 1,
    wall_time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Train the shared policy using estimated action gradients and exact policy VJPs."""
    from .control import init_policy, objective, policy_latents

    _training_inputs(tasks, updates, lr, wall_time_budget_s)
    ledger = CostLedger()
    work = ledger.phase("policy_training")
    accounted_before = work.wall_time_s
    started = time.perf_counter()
    trace = []
    completed, failure = True, None
    try:
        model = init_policy(seed, config)
        optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adam(lr))
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        rng = np.random.default_rng(seed)
        for update in range(updates):
            if (
                wall_time_budget_s is not None
                and time.perf_counter() - started >= wall_time_budget_s
            ):
                break
            task_index = int(rng.integers(len(tasks)))
            task = tasks[task_index]
            latents, pullback = eqx.filter_vjp(
                lambda current, selected=task: policy_latents(current, selected), model
            )
            counted = CountedObjective(
                lambda controls, selected=task: objective(
                    t, ctx, selected.initial, selected.goal, controls, config
                ),
                ledger,
                phase="policy_training",
            )
            try:
                estimate, diagnostic = antithetic_action_gradient(
                    latents,
                    counted,
                    control_bound=config.control_bound,
                    perturbation=perturbation,
                    key=jax.random.fold_in(jax.random.PRNGKey(seed), update),
                    directions=directions,
                )
            except FloatingPointError as error:
                completed, failure = False, str(error)
                break
            gradients = pullback(jax.lax.stop_gradient(estimate))[0]
            gradient_norm = float(optax.tree.norm(gradients))
            if not np.isfinite(gradient_norm):
                completed, failure = False, "nonfinite SPSA policy gradient"
                break
            delta, state = optimizer.update(
                gradients, state, eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, delta)
            jax.block_until_ready((model, state))
            work.optimizer_updates += 1
            trace.append(
                {
                    "update": update + 1,
                    "task_index": task_index,
                    "initial_seed": getattr(task, "initial_seed", None),
                    "goal_seed": getattr(task, "goal_seed", None),
                    "gradient_norm": gradient_norm,
                    **diagnostic,
                }
            )
    finally:
        elapsed = time.perf_counter() - started
        already_charged = work.wall_time_s - accounted_before
        work.wall_time_s += max(0.0, elapsed - already_charged)
    return {
        "model": model,
        "trace": trace,
        "completed": completed,
        "failure": failure,
        "updates": len(trace),
        "wall_time_s": work.wall_time_s,
        "rollout_count": work.rollout_count,
        "gradient_rollout_count": 0,
        "costs": ledger.to_dict(),
    }


def train_imitation_policy(
    tasks: Sequence[Any],
    controls: Sequence[np.ndarray],
    config: Any,
    *,
    label_costs: CostLedger,
    seed: int = 0,
    updates: int = 100,
    lr: float = 1e-3,
    wall_time_budget_s: float | None = None,
    checkpoint_updates: Sequence[int] = (),
) -> dict[str, Any]:
    """Fit fixed training labels, charging their generation against the total budget.

    Labels may be known generating controls or separately optimized actions.
    The caller must provide the extra label-generation ledger explicitly. For
    known generating controls this is an empty ledger: common goal generation
    must be reported separately for every method, not charged only to imitation.
    For optimized/reused labels each comparison reports their full extra cost. Tasks
    must be the training split; held-out task controls must never enter fitting.
    Requested checkpoints retain immutable model snapshots without resetting
    Adam. Evaluate them afterward so validation work is outside training costs.
    """
    from .control import init_policy, policy_controls

    _training_inputs(tasks, updates, lr, wall_time_budget_s)
    if any(
        int(value) != value or not 1 <= value <= updates for value in checkpoint_updates
    ):
        raise ValueError(
            "checkpoint updates must be integers within the training budget"
        )
    requested_checkpoints = set(checkpoint_updates)
    checkpoint_models = {}
    if len(tasks) != len(controls):
        raise ValueError("one fixed action label is required per training task")
    expected_shape = (config.control_slots, 8)
    labels = []
    for value in controls:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"labels must be finite arrays with shape {expected_shape}"
            )
        if np.any(np.abs(value) > config.control_bound + 1e-7):
            raise ValueError("imitation labels exceed the shared physical action bound")
        labels.append(jax.lax.stop_gradient(jnp.asarray(value)))
    ledger = copy.deepcopy(label_costs)
    offline_wall = ledger.total().wall_time_s
    work = ledger.phase("policy_training")
    accounted_before = work.wall_time_s
    started = time.perf_counter()
    trace = []
    completed, failure = True, None
    try:
        model = init_policy(seed, config)
        optimizer = optax.chain(optax.clip_by_global_norm(5), optax.adam(lr))
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        rng = np.random.default_rng(seed)
        for update in range(updates):
            if (
                wall_time_budget_s is not None
                and offline_wall + time.perf_counter() - started >= wall_time_budget_s
            ):
                break
            task_index = int(rng.integers(len(tasks)))
            task, target = tasks[task_index], labels[task_index]

            def loss(
                current: Any, selected: Any = task, label: jax.Array = target
            ) -> jax.Array:
                prediction = policy_controls(current, selected, config)
                return jnp.mean(((prediction - label) / config.control_bound) ** 2)

            value, gradients = eqx.filter_value_and_grad(loss)(model)
            gradient_norm = float(optax.tree.norm(gradients))
            if not np.isfinite(float(value)) or not np.isfinite(gradient_norm):
                completed, failure = False, "nonfinite imitation loss or gradient"
                break
            delta, state = optimizer.update(
                gradients, state, eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, delta)
            jax.block_until_ready((value, model, state))
            work.optimizer_updates += 1
            if update + 1 in requested_checkpoints:
                checkpoint_models[update + 1] = model
            trace.append(
                {
                    "update": update + 1,
                    "task_index": task_index,
                    "action_mse": float(value),
                }
            )
    finally:
        elapsed = time.perf_counter() - started
        already_charged = work.wall_time_s - accounted_before
        work.wall_time_s += max(0.0, elapsed - already_charged)
    total = ledger.total()
    return {
        "model": model,
        "checkpoint_models": checkpoint_models,
        "trace": trace,
        "completed": completed,
        "failure": failure,
        "updates": len(trace),
        "wall_time_s": total.wall_time_s,
        "rollout_count": total.rollout_count,
        "gradient_rollout_count": total.gradient_rollout_count,
        "label_generation_wall_time_s": offline_wall,
        "costs": ledger.to_dict(),
    }
