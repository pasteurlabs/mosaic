# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver-in-the-loop training benchmark for periodic 2D Navier--Stokes.

Each candidate Tesseract supplies the recurrent coarse-grid transition and
its VJP.  The data, neural corrector, optimiser, rollout loss, and evaluation
are benchmark-owned so differences in training outcomes can be attributed to
the solver interface under test.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.utils import active_differentiable_solvers

from .corrector import (
    corrected_velocity,
    divergence_rms,
    init_corrector,
    reference_trajectory,
    relative_l2,
    spectral_restrict,
)
from .ics import _multimode

_DATASET_LOCK = threading.Lock()


def _dataset_digest(config: dict[str, Any]) -> str:
    """Stable short identifier for generated reference data."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@lru_cache(maxsize=8)
def _cached_reference_dataset(
    *,
    n: int,
    reference_factor: int,
    viscosity: float,
    dt: float,
    frame_steps: int,
    n_frames: int,
    reference_substeps: int,
    domain_extent: float,
    seeds: tuple[int, ...],
    k0: float,
    sigma_k: float,
    amplitude: float,
) -> np.ndarray:
    """Generate and cache coarse views of deterministic fine-grid trajectories."""
    trajectories: list[np.ndarray] = []
    reference_n = n * reference_factor
    for seed in seeds:
        fine_initial = _multimode(
            reference_n,
            L=domain_extent,
            seed=seed,
            k0=k0,
            sigma_k=sigma_k,
            amplitude=amplitude,
        )
        fine = reference_trajectory(
            fine_initial,
            viscosity=viscosity,
            dt=dt,
            frame_steps=frame_steps,
            n_frames=n_frames,
            substeps=reference_substeps,
            domain_extent=domain_extent,
        )
        trajectories.append(np.stack([spectral_restrict(frame, n) for frame in fine]))
    return np.stack(trajectories).astype(np.float32)


def make_reference_dataset(
    *,
    physics: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    training: dict[str, Any],
    domain_extent: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build the train/test trajectory tensors used by every solver."""
    n = int(physics["N"])
    train_seeds = tuple(int(v) for v in dataset.get("train_seeds", [0, 1, 2, 3]))
    test_seeds = tuple(int(v) for v in dataset.get("test_seeds", [100, 101]))
    train_frames = int(dataset.get("train_frames", 16))
    eval_frames = int(evaluation.get("rollout_frames", 32))
    unroll = int(training.get("unroll", 4))
    n_frames = max(train_frames, eval_frames, unroll)
    config = {
        "n": n,
        "reference_factor": int(dataset.get("reference_factor", 2)),
        "viscosity": float(physics["nu"]),
        "dt": float(physics["dt"]),
        "frame_steps": int(physics["steps"]),
        "n_frames": n_frames,
        "reference_substeps": int(dataset.get("reference_substeps", 2)),
        "domain_extent": float(domain_extent),
        "seeds": train_seeds + test_seeds,
        "k0": float(dataset.get("k0", 6.0)),
        "sigma_k": float(dataset.get("sigma_k", 1.0)),
        "amplitude": float(dataset.get("amplitude", 0.3)),
    }
    # functools.lru_cache is thread-safe for its dictionary, but concurrent
    # cache misses may still execute the wrapped function more than once.
    # Reference generation is substantial enough to serialize that first miss.
    with _DATASET_LOCK:
        all_trajectories = _cached_reference_dataset(**config)
    n_train = len(train_seeds)
    train = all_trajectories[:n_train, : train_frames + 1]
    test = all_trajectories[n_train:, : eval_frames + 1]
    return train, test, _dataset_digest(config)


def _solver_advance(
    t: Any,
    ctx: KernelContext,
    velocity: jax.Array,
    *,
    frame_steps: int,
) -> jax.Array:
    """Advance one canonical correction interval through a Tesseract."""
    physics = {**ctx.phys, "steps": frame_steps}
    inputs = ctx.make_inputs(ctx.name, velocity, **physics)
    outputs = apply_tesseract(t, inputs)
    if ctx.output_key not in outputs:
        raise RuntimeError(
            f"Solver '{ctx.name}' did not return output {ctx.output_key!r}"
        )
    return outputs[ctx.output_key]


def _window_loss(
    params: Any,
    targets: jax.Array,
    *,
    t: Any,
    ctx: KernelContext,
    frame_steps: int,
    velocity_scale: float,
) -> jax.Array:
    """Mean normalized state error over one recurrent training window."""
    state = targets[0]
    losses: list[jax.Array] = []
    for target in targets[1:]:
        provisional = _solver_advance(t, ctx, state, frame_steps=frame_steps)
        state = corrected_velocity(
            params,
            provisional,
            velocity_scale=velocity_scale,
            domain_extent=ctx.domain_extent,
        )
        losses.append(jnp.sum((state - target) ** 2) / (jnp.sum(target**2) + 1e-12))
    return jnp.mean(jnp.stack(losses))


def _directional_fd(
    loss_fn: Any,
    params: Any,
    grads: Any,
    key: jax.Array,
    *,
    epsilon: float,
) -> float:
    """Relative error of one end-to-end directional finite difference."""
    direction = jax.tree_util.tree_map(
        lambda p, k: jax.random.normal(k, p.shape, dtype=p.dtype),
        params,
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(params),
            jax.random.split(key, len(jax.tree_util.tree_leaves(params))),
        ),
    )
    direction_norm = optax.tree.norm(direction)
    direction = jax.tree_util.tree_map(
        lambda value: value / (direction_norm + 1e-12), direction
    )
    plus = jax.tree_util.tree_map(lambda p, d: p + epsilon * d, params, direction)
    minus = jax.tree_util.tree_map(lambda p, d: p - epsilon * d, params, direction)
    fd = (loss_fn(plus) - loss_fn(minus)) / (2.0 * epsilon)
    ad = sum(
        jnp.vdot(g, d)
        for g, d in zip(
            jax.tree_util.tree_leaves(grads),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )
    return float(jnp.abs(fd - ad) / (jnp.abs(fd) + jnp.abs(ad) + 1e-12))


def _train_corrector(
    t: Any,
    ctx: KernelContext,
    train: np.ndarray,
    *,
    frame_steps: int,
    training: dict[str, Any],
    velocity_scale: float,
) -> tuple[Any, list[float], list[float], float | None, bool]:
    """Train one solver-specific corrector with a fixed stochastic schedule."""
    max_updates = int(training.get("max_updates", 100))
    unroll = int(training.get("unroll", 4))
    lr = float(training.get("lr", 1e-4))
    clip_norm = float(training.get("clip_norm", 1.0))
    seed = int(training.get("seed", 2026))
    model_seed = int(training.get("model_seed", 0))
    hidden_channels = int(training.get("hidden_channels", 32))
    kernel_size = int(training.get("kernel_size", 5))
    fd_epsilon = float(training.get("fd_epsilon", 1e-2))

    params = init_corrector(
        jax.random.PRNGKey(model_seed),
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr),
    )
    opt_state = optimiser.init(params)
    rng = np.random.RandomState(seed)
    losses: list[float] = []
    grad_norms: list[float] = []
    fd_error: float | None = None
    completed = True

    for update in range(max_updates):
        trajectory_idx = int(rng.randint(train.shape[0]))
        max_start = train.shape[1] - unroll - 1
        start = int(rng.randint(max_start + 1)) if max_start > 0 else 0
        targets = jnp.asarray(train[trajectory_idx, start : start + unroll + 1])

        loss_fn = partial(
            _window_loss,
            targets=targets,
            t=t,
            ctx=ctx,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
        )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        loss_value = float(loss)
        grad_norm = float(optax.tree.norm(grads))
        if not np.isfinite(loss_value) or not np.isfinite(grad_norm):
            completed = False
            break
        if update == 0 and bool(training.get("check_grad", True)):
            fd_error = _directional_fd(
                loss_fn,
                params,
                grads,
                jax.random.PRNGKey(model_seed + 1),
                epsilon=fd_epsilon,
            )
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(loss_value)
        grad_norms.append(grad_norm)
    return params, losses, grad_norms, fd_error, completed


def _evaluate_rollout(
    t: Any,
    ctx: KernelContext,
    params: Any,
    reference: np.ndarray,
    *,
    frame_steps: int,
    velocity_scale: float,
    corrected: bool,
) -> tuple[np.ndarray, list[float]]:
    """Roll out one held-out sequence with or without neural corrections."""
    state = jnp.asarray(reference[0])
    states = [np.asarray(state)]
    errors = [0.0]
    for target in reference[1:]:
        state = _solver_advance(t, ctx, state, frame_steps=frame_steps)
        if corrected:
            state = corrected_velocity(
                params,
                state,
                velocity_scale=velocity_scale,
                domain_extent=ctx.domain_extent,
            )
        state_np = np.asarray(state)
        states.append(state_np)
        errors.append(relative_l2(state_np, target))
    return np.stack(states), errors


def _first_unstable(errors: list[float], threshold: float) -> int:
    """Return first bad frame, or the full completed horizon."""
    for idx, error in enumerate(errors):
        if not np.isfinite(error) or error > threshold:
            return idx
    return len(errors) - 1


@kernel(
    sweep_mode="none",
    selector_fn=active_differentiable_solvers,
    snapshot_filename="corrector_fields.npz",
    snapshot_prefixes=(
        "loss",
        "grad_norm",
        "error_corrected",
        "error_uncorrected",
        "rollout_corrected",
        "rollout_uncorrected",
    ),
)
def solver_in_loop(t: Any, ctx: KernelContext) -> dict:
    """Train and evaluate one neural corrector through one candidate solver."""
    training = ctx.run.get("training", {})
    dataset_cfg = ctx.run.get("dataset", {})
    evaluation = ctx.run.get("evaluation", {})
    frame_steps = int(ctx.phys["steps"])

    train, test, dataset_hash = make_reference_dataset(
        physics=ctx.phys,
        dataset=dataset_cfg,
        evaluation=evaluation,
        training=training,
        domain_extent=ctx.domain_extent,
    )
    velocity_scale = float(np.sqrt(np.mean(train**2)) + 1e-8)

    started = time.perf_counter()
    params, losses, grad_norms, fd_error, completed = _train_corrector(
        t,
        ctx,
        train,
        frame_steps=frame_steps,
        training=training,
        velocity_scale=velocity_scale,
    )
    training_wall = time.perf_counter() - started

    corrected_errors: list[list[float]] = []
    uncorrected_errors: list[list[float]] = []
    first_corrected: np.ndarray | None = None
    first_uncorrected: np.ndarray | None = None
    for reference in test:
        corrected_rollout, corr_err = _evaluate_rollout(
            t,
            ctx,
            params,
            reference,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=True,
        )
        uncorrected_rollout, raw_err = _evaluate_rollout(
            t,
            ctx,
            params,
            reference,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=False,
        )
        corrected_errors.append(corr_err)
        uncorrected_errors.append(raw_err)
        if first_corrected is None:
            first_corrected = corrected_rollout
            first_uncorrected = uncorrected_rollout

    corrected_error = np.mean(np.asarray(corrected_errors), axis=0)
    uncorrected_error = np.mean(np.asarray(uncorrected_errors), axis=0)
    final_corrected = float(corrected_error[-1])
    final_uncorrected = float(uncorrected_error[-1])
    mean_corrected = float(np.mean(corrected_error[1:]))
    mean_uncorrected = float(np.mean(uncorrected_error[1:]))
    threshold = float(evaluation.get("stable_error_threshold", 1.0))
    interval_time = float(ctx.phys["dt"]) * frame_steps

    metrics = {
        "final_rollout_error": final_corrected,
        "uncorrected_rollout_error": final_uncorrected,
        "mean_rollout_error": mean_corrected,
        "uncorrected_mean_rollout_error": mean_uncorrected,
        "improvement_pct": 100.0
        * (final_uncorrected - final_corrected)
        / (final_uncorrected + 1e-12),
        "stable_horizon": _first_unstable(corrected_error.tolist(), threshold)
        * interval_time,
        "uncorrected_stable_horizon": _first_unstable(
            uncorrected_error.tolist(), threshold
        )
        * interval_time,
        "final_train_loss": losses[-1] if losses else None,
        "best_train_loss": min(losses) if losses else None,
        "n_updates": len(losses),
        "training_wall_time_s": training_wall,
        "seconds_per_update": training_wall / max(len(losses), 1),
        "final_grad_norm": grad_norms[-1] if grad_norms else None,
        "end_to_end_fd_rel_error": fd_error,
        "final_divergence_rms": divergence_rms(first_corrected[-1], ctx.domain_extent)
        if first_corrected is not None
        else None,
        "completed": completed,
        "dataset_hash": dataset_hash,
        "state_restart": True,
    }
    snapshots = {
        "loss": np.asarray(losses, dtype=np.float32),
        "grad_norm": np.asarray(grad_norms, dtype=np.float32),
        "error_corrected": np.asarray(corrected_error, dtype=np.float32),
        "error_uncorrected": np.asarray(uncorrected_error, dtype=np.float32),
    }
    if first_corrected is not None and first_uncorrected is not None:
        snapshots["rollout_corrected"] = first_corrected
        snapshots["rollout_uncorrected"] = first_uncorrected
    return {
        "metrics": metrics,
        "snapshots": snapshots,
        "shared": {
            "reference_rollout": test[0],
            "evaluation_times": np.arange(test.shape[1], dtype=np.float32)
            * interval_time,
        },
    }


__all__ = ["make_reference_dataset", "solver_in_loop"]
