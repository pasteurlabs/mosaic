# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-solver buoyancy corrector experiment with matched training arms.

This is an execution/gradient benchmark, not a CI assertion that SOL beats
supervision. Every learning-rate candidate sees the same initialization,
update count and training trajectory. Only validation chooses a candidate.
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

ARMS = ("supervised", "solver_in_loop", "stop_gradient")


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
    *,
    stop_solver_gradient: bool = False,
) -> jax.Array:
    """Average intermediate-state errors on a recurrent training window."""
    state = targets[0]
    loss = jnp.asarray(0.0)
    for target in targets[1:]:
        advanced = step(state)
        if stop_solver_gradient:
            advanced = jax.lax.stop_gradient(advanced)
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


def fit(params: Any, loss: Any, updates: int, lr: float) -> tuple:
    """Run a fixed update budget and reject incomplete/nonfinite training."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params)
    # The solver is a remote Tesseract callback. Keep the outer loop in Python
    # so each failure names the iteration, without compiling a giant RPC graph.
    value_and_grad = jax.value_and_grad(loss)
    history = []
    started = time.perf_counter()
    for iteration in range(updates):
        value, gradient = value_and_grad(params, iteration)
        if not np.isfinite(float(value)) or not all(
            np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient)
        ):
            raise FloatingPointError(f"nonfinite training at update {iteration}")
        delta, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, delta)
        history.append(float(value))
    if not all(np.isfinite(np.asarray(p)).all() for p in jax.tree.leaves(params)):
        raise FloatingPointError("nonfinite final model parameters")
    return params, history, time.perf_counter() - started


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
    snapshot_prefixes=("reference", "uncorrected", *ARMS),
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
        min(n, steps, updates, horizon, frames) < 1
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
    references = []
    fine_initials = []
    for seed, length in zip(seeds, (train_frames, frames, frames), strict=True):
        initial = ctx.cfg.make_ic["plume"](
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
    scales = jnp.asarray([velocity_scale, velocity_scale, scalar_scale])

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
        jax.random.PRNGKey(17), probe[-1]["w"].shape
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

    # Fixed one-step pairs are constructed once from restricted reference states.
    preprocessing_started = time.perf_counter()
    fixed = jnp.asarray(np.stack([np.asarray(step(s)) for s in train_jax[:-1]]))
    fixed.block_until_ready()
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    count = len(train) - horizon

    def supervised_loss(params: Any, iteration: int) -> jax.Array:
        start = iteration % count
        return (
            sum(
                normalized_loss(
                    apply_corrector(params, fixed[i], scales), train_jax[i + 1], scales
                )
                for i in range(start, start + horizon)
            )
            / horizon
        )

    def recurrent_loss(params: Any, iteration: int, stop: bool = False) -> jax.Array:
        start = iteration % count
        return window_loss(
            params,
            train_jax[start : start + horizon + 1],
            step,
            scales,
            stop_solver_gradient=stop,
        )

    losses = {
        "supervised": supervised_loss,
        "solver_in_loop": recurrent_loss,
        "stop_gradient": lambda p, i: recurrent_loss(p, i, True),
    }
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
        "uncorrected": rollout_errors(baseline, test, scales),
    }
    for arm in ARMS:
        candidates = []
        models = []
        for lr in lrs:
            params, history, elapsed = fit(params0, losses[arm], updates, lr)
            rollout = trajectory(step, validation[0], frames, params, scales)
            score = float(
                np.mean(rollout_errors(rollout, validation, scales)["loss"][1:])
            )
            candidates.append(
                {
                    "learning_rate": lr,
                    "validation_loss": score,
                    "training_loss": history,
                    "training_seconds": elapsed,
                    "completed_updates": len(history),
                }
            )
            models.append(params)
        selected = int(np.argmin([c["validation_loss"] for c in candidates]))
        rollout = trajectory(step, test[0], frames, models[selected], scales)
        snapshots[arm] = rollout
        metrics[arm] = {
            **rollout_errors(rollout, test, scales),
            "selected_candidate": selected,
            "candidates": candidates,
            "preprocessing_seconds": preprocessing_seconds
            if arm == "supervised"
            else 0.0,
            "total_training_seconds": sum(c["training_seconds"] for c in candidates)
            + (preprocessing_seconds if arm == "supervised" else 0.0),
        }
    metrics["divergence_rms"] = {
        arm: divergence_rms(rollout, ctx.domain_extent)
        for arm, rollout in snapshots.items()
    }
    return {"metrics": metrics, "snapshots": snapshots}
