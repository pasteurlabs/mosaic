# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a buoyancy corrector through recurrent solver steps.

Every learning-rate candidate sees the same initialization, update count and
training windows. Only validation chooses a candidate. Correctness gates check
references and gradients; held-out rollouts compare against the uncorrected solver.
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.utils import active_differentiable_solvers

from .corrector import (
    apply_corrector,
    init_corrector,
    normalized_loss,
    restrict_state,
)


def trajectory(
    step: Any, initial: Any, frames: int, params: Any = None, scales: Any = None
) -> np.ndarray:
    """Capture all states, synchronizing each remote solver result."""
    states = [np.asarray(initial)]
    state = jnp.asarray(initial)
    for _ in range(frames):
        state = step(state)
        if params is not None:
            state = apply_corrector(params, state, scales)
        states.append(np.asarray(state))
    result = np.stack(states)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite rollout")
    return result


def window_loss(
    params: Any,
    targets: jax.Array,
    step: Any,
    scales: jax.Array,
) -> jax.Array:
    """Average intermediate-state errors on a recurrent training window."""
    state = targets[0]
    loss = jnp.asarray(0.0)
    for target in targets[1:]:
        advanced = step(state)
        state = apply_corrector(params, advanced, scales)
        loss = loss + normalized_loss(state, target, scales)
    return loss / (len(targets) - 1)


def directional_fd(loss: Any, params: Any, epsilon: float = 0.003) -> dict:
    """Check a full recurrent training gradient along its normalized direction."""
    gradient = jax.grad(loss)(params)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(gradient)))
    if not np.isfinite(float(norm)) or float(norm) < 1e-9:
        raise ValueError("training gradient is nonfinite or too small to validate")
    direction = jax.tree.map(lambda g: g / norm, gradient)
    plus = jax.tree.map(lambda p, d: p + epsilon * d, params, direction)
    minus = jax.tree.map(lambda p, d: p - epsilon * d, params, direction)
    fd = float((loss(plus) - loss(minus)) / (2 * epsilon))
    ad = float(norm)
    if not np.isfinite(fd):
        raise FloatingPointError("nonfinite finite-difference probe")
    return {
        "autodiff": ad,
        "finite_difference": fd,
        "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1e-8),
        "epsilon": epsilon,
    }


def shuffled_windows(updates: int, count: int, seed: int) -> list[int]:
    """Visit every valid window once per seeded, independently shuffled epoch."""
    if updates < 1 or count < 1:
        raise ValueError("updates and window count must be positive")
    rng = np.random.default_rng(seed)
    starts = []
    while len(starts) < updates:
        starts.extend(rng.permutation(count).tolist())
    return starts[:updates]


def checkpoint_updates(updates: int, interval: int) -> list[int]:
    """Always validate the identity and final model, including partial intervals."""
    if updates < 1 or interval < 1:
        raise ValueError("updates and validation interval must be positive")
    return sorted({0, updates, *range(interval, updates + 1, interval)})


def select_candidate(candidates: list[dict]) -> int:
    """Choose by validation loss, then earliest update, then learning-rate order."""
    return min(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["validation_loss"],
            candidates[i]["selected_update"],
            i,
        ),
    )


def fit(
    params: Any,
    loss: Any,
    validate: Any,
    updates: int,
    lr: float,
    validation_interval: int,
) -> tuple:
    """Complete the whole budget while retaining the best validation checkpoint."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(loss)
    history, checkpoints = [], []
    training_seconds, validation_seconds = 0.0, 0.0
    selected_params, best_loss, selected_update = params, float("inf"), 0
    scheduled = set(checkpoint_updates(updates, validation_interval))

    def evaluate(current: Any, update: int) -> None:
        nonlocal selected_params, best_loss, selected_update, validation_seconds
        started = time.perf_counter()
        score = float(validate(current))
        validation_seconds += time.perf_counter() - started
        if not np.isfinite(score) or score < 0:
            raise FloatingPointError(f"invalid validation loss at update {update}")
        checkpoints.append({"update": update, "loss": score})
        # Strict comparison preserves the identity/earliest checkpoint on ties.
        if score < best_loss:
            selected_params, best_loss, selected_update = current, score, update

    evaluate(params, 0)
    for iteration in range(updates):
        started = time.perf_counter()
        value, gradient = value_and_grad(params, iteration)
        if not np.isfinite(float(value)) or not all(
            np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient)
        ):
            raise FloatingPointError(f"nonfinite training at update {iteration}")
        delta, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, delta)
        if not all(np.isfinite(np.asarray(p)).all() for p in jax.tree.leaves(params)):
            raise FloatingPointError(
                f"nonfinite model parameters at update {iteration + 1}"
            )
        history.append(float(value))
        training_seconds += time.perf_counter() - started
        if iteration + 1 in scheduled:
            evaluate(params, iteration + 1)
    return selected_params, {
        "learning_rate": lr,
        "training_loss": history,
        "completed_updates": len(history),
        "validation_loss": best_loss,
        "selected_update": selected_update,
        "validation_checkpoints": checkpoints,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
    }


def rollout_errors(states: np.ndarray, targets: np.ndarray, scales: Any) -> dict:
    """Per-field normalized errors, retaining frame zero for aligned plots."""
    error = ((states - targets) / np.asarray(scales)) ** 2
    velocity = error[..., :2].mean(axis=(1, 2, 3))
    scalar = error[..., 2].mean(axis=(1, 2))
    if not np.isfinite(velocity).all() or not np.isfinite(scalar).all():
        raise FloatingPointError("nonfinite normalized rollout error")
    return {
        "loss": ((velocity + scalar) / 2).tolist(),
        "velocity_mse": velocity.tolist(),
        "scalar_mse": scalar.tolist(),
    }


def divergence_rms(states: np.ndarray, domain_extent: float) -> list[float]:
    """Cell divergence from packed face velocities and implicit zero walls."""
    ux = np.pad(states[..., 0], ((0, 0), (1, 0), (0, 0)))
    uy = np.pad(states[..., 1], ((0, 0), (0, 0), (1, 0)))
    divergence = (
        (np.diff(ux, axis=1) + np.diff(uy, axis=2)) * states.shape[1] / domain_extent
    )
    return np.sqrt(np.mean(divergence**2, axis=(1, 2))).tolist()


@kernel(
    sweep_mode="none",
    selector_fn=active_differentiable_solvers,
    snapshot_filename="corrector_fields.npz",
    snapshot_prefixes=("reference", "uncorrected", "solver_in_loop"),
)
def solver_in_loop(t: Any, ctx: KernelContext) -> dict:
    """Train/evaluate one solver with no cross-solver accuracy ranking."""
    cfg = ctx.run.get("training", {})
    phys = dict(ctx.phys)
    n = int(phys.pop("N", 24))
    steps = int(phys.pop("steps", 2))
    dt = float(phys.pop("dt", 0.04))
    phys.pop("domain_extent", None)
    updates = int(cfg.get("updates", 16))
    horizon = int(cfg.get("unroll", 4))
    frames = int(cfg.get("eval_frames", 12))
    train_frames = int(cfg.get("train_frames", max(frames, horizon)))
    factor = int(cfg.get("reference_factor", 2))
    validation_interval = int(cfg.get("validation_interval", 64))
    window_seed = int(cfg.get("window_seed", 0))
    lrs = [float(lr) for lr in cfg.get("learning_rates", [0.003])]
    seeds = [
        int(cfg.get(key, default))
        for key, default in (
            ("train_seed", 0),
            ("validation_seed", 1),
            ("test_seed", 100),
        )
    ]
    if (
        min(n, steps, updates, horizon, frames, validation_interval) < 1
        or train_frames < horizon
        or factor < 2
        or len(set(seeds)) != 3
        or not lrs
        or not all(np.isfinite(lr) and lr > 0 for lr in lrs)
    ):
        raise ValueError(
            "invalid training budget, reference factor, learning rates or seeds"
        )

    def make_step(subdivision: int = 1) -> Any:
        def step(state: jax.Array) -> jax.Array:
            inputs = ctx.make_inputs(
                ctx.name,
                state,
                domain_extent=ctx.domain_extent,
                dt=dt / subdivision,
                steps=steps * subdivision,
                **phys,
            )
            return apply_tesseract(t, inputs)["result"]

        return step

    step = make_step()
    fine_step = make_step(factor)
    ic_name = ctx.run.get("ic", {}).get("name", "plume")
    make_initial = ctx.cfg.make_ic[ic_name]
    references = []
    fine_initials = []
    for seed, length in zip(seeds, (train_frames, frames, frames), strict=True):
        initial = make_initial(
            N=n * factor,
            L=ctx.domain_extent,
            seed=seed,
        )
        fine_initials.append(initial)
        fine = trajectory(fine_step, initial, length)
        references.append(np.stack([restrict_state(s, factor) for s in fine]))
    train, validation, test = references
    velocity_scale = max(float(np.sqrt(np.mean(train[..., :2] ** 2))), 1e-3)
    scalar_scale = max(float(np.sqrt(np.mean(train[..., 2] ** 2))), 1e-3)
    scales = jnp.asarray(
        [velocity_scale, velocity_scale, scalar_scale], dtype=jnp.float32
    )

    # Admit every target split independently; test data never select a model.
    audits = {}
    baselines = {}
    for split, initial, reference in zip(
        ("train", "validation", "test"), fine_initials, references, strict=True
    ):
        length = len(reference) - 1
        audit = trajectory(make_step(2 * factor), initial, length)
        audit = np.stack([restrict_state(state, factor) for state in audit])
        refinement_error = float(
            np.sqrt(np.mean(rollout_errors(audit, reference, scales)["loss"][1:]))
        )
        baseline = trajectory(step, reference[0], length)
        baselines[split] = baseline
        coarse_gap = float(
            np.sqrt(np.mean(rollout_errors(baseline, reference, scales)["loss"][1:]))
        )
        audit_gap_ratio = refinement_error / max(coarse_gap, 1e-12)
        audits[split] = {
            "time_refinement_error": refinement_error,
            "coarse_gap": coarse_gap,
            "audit_gap_ratio": audit_gap_ratio,
        }
        if refinement_error > float(cfg.get("reference_tolerance", 0.05)):
            raise ValueError(
                f"{split} reference time refinement failed: {refinement_error:.6g}"
            )
        if not np.isfinite(audit_gap_ratio) or audit_gap_ratio > float(
            cfg.get("reference_gap_tolerance", 0.25)
        ):
            raise ValueError(
                f"{split} reference uncertainty too large relative to coarse gap: {audit_gap_ratio:.6g}"
            )

    params0 = init_corrector(
        int(cfg.get("model_seed", 0)),
        int(cfg.get("hidden_channels", 8)),
    )
    train_jax = jnp.asarray(train)
    check_loss = lambda p: window_loss(p, train_jax[: horizon + 1], step, scales)
    fd = directional_fd(check_loss, params0, float(cfg.get("fd_epsilon", 0.003)))
    if fd["relative_error"] > float(cfg.get("fd_tolerance", 0.05)):
        raise ValueError(f"recurrent gradient finite-difference check failed: {fd}")
    # Also test hidden-layer and temporal derivatives away from zero readout.
    probe = jax.tree.map(lambda p: p, params0)
    probe[-1]["w"] = 0.01 * jax.random.normal(
        jax.random.PRNGKey(17), probe[-1]["w"].shape, dtype=jnp.float32
    )
    fd_nonzero = directional_fd(check_loss, probe, float(cfg.get("fd_epsilon", 0.003)))
    if fd_nonzero["relative_error"] > float(cfg.get("fd_tolerance", 0.05)):
        raise ValueError(f"nonzero-model recurrent gradient check failed: {fd_nonzero}")

    joined_inputs = ctx.make_inputs(
        ctx.name,
        jnp.asarray(train[0]),
        domain_extent=ctx.domain_extent,
        dt=dt,
        steps=steps * 2,
        **phys,
    )
    joined = np.asarray(apply_tesseract(t, joined_inputs)["result"])
    split = np.asarray(step(step(jnp.asarray(train[0]))))
    recurrence_error = float(np.max(np.abs((joined - split) / np.asarray(scales))))
    if not np.isfinite(recurrence_error) or recurrence_error > float(
        cfg.get("recurrence_tolerance", 1e-4)
    ):
        raise ValueError(f"solver restart equivalence failed: {recurrence_error}")

    count = len(train) - horizon
    starts = shuffled_windows(updates, count, window_seed)

    def recurrent_loss(params: Any, iteration: int) -> jax.Array:
        start = starts[iteration]
        return window_loss(params, train_jax[start : start + horizon + 1], step, scales)

    baseline = baselines["test"]
    snapshots = {"reference": test, "uncorrected": baseline}
    metrics = {
        "status": "ok",
        "reference": {
            "spatial_factor": factor,
            "time_refinement_error": max(
                a["time_refinement_error"] for a in audits.values()
            ),
            "audit_gap_ratio": max(a["audit_gap_ratio"] for a in audits.values()),
            "splits": audits,
        },
        "gradient_check": fd,
        "gradient_check_nonzero": fd_nonzero,
        "recurrence_error": recurrence_error,
        "scales": np.asarray(scales).tolist(),
        "seeds": dict(zip(("train", "validation", "test"), seeds, strict=True)),
        "updates": updates,
        "unroll": horizon,
        "eval_frames": frames,
        "learning_rates": lrs,
        "train_frames": train_frames,
        "window_seed": window_seed,
        "window_starts": starts,
        "validation_interval": validation_interval,
        "uncorrected": rollout_errors(baseline, test, scales),
    }

    def validate(params: Any) -> float:
        rollout = trajectory(step, validation[0], frames, params, scales)
        return float(np.mean(rollout_errors(rollout, validation, scales)["loss"][1:]))

    candidates = []
    models = []
    for lr in lrs:
        params, candidate = fit(
            params0, recurrent_loss, validate, updates, lr, validation_interval
        )
        candidates.append(candidate)
        models.append(params)
    selected = select_candidate(candidates)
    rollout = trajectory(step, test[0], frames, models[selected], scales)
    snapshots["solver_in_loop"] = rollout
    metrics["solver_in_loop"] = {
        **rollout_errors(rollout, test, scales),
        "selected_candidate": selected,
        "selected_update": candidates[selected]["selected_update"],
        "candidates": candidates,
        "total_training_seconds": sum(c["training_seconds"] for c in candidates),
        "total_validation_seconds": sum(c["validation_seconds"] for c in candidates),
        "total_seconds": sum(
            c["training_seconds"] + c["validation_seconds"] for c in candidates
        ),
    }
    # Generic optimization reporting otherwise picks the first nested loss,
    # which is the uncorrected baseline rather than the trained rollout.
    metrics["final_loss"] = metrics["solver_in_loop"]["loss"][-1]
    metrics["divergence_rms"] = {
        arm: divergence_rms(rollout, ctx.domain_extent)
        for arm, rollout in snapshots.items()
    }
    return {"metrics": metrics, "snapshots": snapshots}
