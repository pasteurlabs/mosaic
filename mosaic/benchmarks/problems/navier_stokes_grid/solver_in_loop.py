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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_jax import apply_tesseract

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.utils import active_differentiable_solvers

from .corrector import (
    centered_divergence_rms,
    corrected_velocity,
    divergence_rms,
    enstrophy,
    init_corrector,
    kinetic_energy,
    reference_trajectory,
    relative_l2,
    spectral_restrict,
)
from .ics import _multimode, _tgv

_DATASET_LOCK = threading.Lock()


def _dataset_digest(config: dict[str, Any]) -> str:
    """Stable short identifier for generated reference data."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@lru_cache(maxsize=8)
def _cached_reference_dataset(
    *,
    reference_kind: str,
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
        if reference_kind == "pseudo_spectral_multimode":
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
        elif reference_kind == "analytic_tgv":
            fine_initial = np.asarray(
                _tgv(reference_n, L=domain_extent),
                dtype=np.float32,
            )
            # Integer translations preserve the exact periodic TGV solution
            # while giving train/test trajectories distinct phases.
            shift = (
                int(seed) % reference_n,
                (7 * int(seed) + 3) % reference_n,
            )
            fine_initial = np.roll(fine_initial, shift=shift, axis=(0, 1))
            frame_dt = dt * frame_steps
            decay_rate = 2.0 * viscosity * (2.0 * np.pi / domain_extent) ** 2
            fine = np.stack(
                [
                    fine_initial * np.exp(-decay_rate * frame_dt * frame)
                    for frame in range(n_frames + 1)
                ]
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown reference_kind: {reference_kind!r}")
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
        "reference_kind": str(
            dataset.get("reference_kind", "pseudo_spectral_multimode")
        ),
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
    model: Any,
    targets: jax.Array,
    *,
    t: Any,
    ctx: KernelContext,
    frame_steps: int,
    velocity_scale: float,
    differentiate_solver: bool,
) -> jax.Array:
    """Mean normalized state error over one recurrent training window."""
    state = targets[0]
    losses: list[jax.Array] = []
    for target in targets[1:]:
        provisional = _solver_advance(t, ctx, state, frame_steps=frame_steps)
        if not differentiate_solver:
            # Keep the identical recurrent forward trajectory but cut the
            # backward graph at every solver transition. This counterfactual
            # measures what the same corrector can learn without the solver VJP.
            provisional = jax.lax.stop_gradient(provisional)
        state = corrected_velocity(
            model,
            provisional,
            velocity_scale=velocity_scale,
            domain_extent=ctx.domain_extent,
        )
        losses.append(jnp.sum((state - target) ** 2) / (jnp.sum(target**2) + 1e-12))
    return jnp.mean(jnp.stack(losses))


def _directional_fd(
    loss_fn: Any,
    model: Any,
    grads: Any,
    key: jax.Array,
    *,
    epsilon: float,
) -> float:
    """Relative error of one end-to-end directional finite difference."""
    dynamic, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, structure = jax.tree_util.tree_flatten(dynamic)
    direction = jax.tree_util.tree_unflatten(
        structure,
        [
            jax.random.normal(part_key, leaf.shape, dtype=leaf.dtype)
            for leaf, part_key in zip(
                leaves,
                jax.random.split(key, len(leaves)),
                strict=True,
            )
        ],
    )
    direction_norm = optax.tree.norm(direction)
    direction = jax.tree_util.tree_map(
        lambda value: value / (direction_norm + 1e-12), direction
    )
    plus = eqx.combine(
        jax.tree_util.tree_map(
            lambda parameter, value: parameter + epsilon * value,
            dynamic,
            direction,
        ),
        static,
    )
    minus = eqx.combine(
        jax.tree_util.tree_map(
            lambda parameter, value: parameter - epsilon * value,
            dynamic,
            direction,
        ),
        static,
    )
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
    differentiate_solver: bool,
    model_seed: int,
) -> tuple[Any, list[float], list[float], list[float], float | None, bool]:
    """Train one solver-specific corrector with a fixed stochastic schedule."""
    max_updates = int(training.get("max_updates", 100))
    unroll = int(training.get("unroll", 4))
    lr = float(training.get("lr", 1e-4))
    clip_norm = float(training.get("clip_norm", 1.0))
    seed = int(training.get("seed", 2026))
    hidden_channels = int(training.get("hidden_channels", 32))
    kernel_size = int(training.get("kernel_size", 5))
    architecture = str(training.get("architecture", "periodic_residual_cnn"))
    fd_epsilon = float(training.get("fd_epsilon", 1e-2))

    model = init_corrector(
        jax.random.PRNGKey(model_seed),
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        architecture=architecture,
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr),
    )
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.RandomState(seed)
    losses: list[float] = []
    grad_norms: list[float] = []
    update_times: list[float] = []
    fd_error: float | None = None
    completed = True

    for update in range(max_updates):
        update_started = time.perf_counter()
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
            differentiate_solver=differentiate_solver,
        )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        loss_value = float(loss)
        grad_norm = float(optax.tree.norm(grads))
        if not np.isfinite(loss_value) or not np.isfinite(grad_norm):
            completed = False
            break
        if (
            update == 0
            and differentiate_solver
            and bool(training.get("check_grad", True))
        ):
            fd_error = _directional_fd(
                loss_fn,
                model,
                grads,
                jax.random.PRNGKey(model_seed + 1),
                epsilon=fd_epsilon,
            )
        updates, opt_state = optimiser.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        losses.append(loss_value)
        grad_norms.append(grad_norm)
        update_times.append(time.perf_counter() - update_started)
    return model, losses, grad_norms, update_times, fd_error, completed


def _evaluate_rollout(
    t: Any,
    ctx: KernelContext,
    model: Any,
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
                model,
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


def _rollout_log_gain(
    baseline_errors: np.ndarray | list[float],
    corrected_errors: np.ndarray | list[float],
) -> float:
    """Return rollout-wide geometric error reduction on non-initial frames."""
    baseline = np.asarray(baseline_errors, dtype=np.float64)[1:]
    corrected = np.asarray(corrected_errors, dtype=np.float64)[1:]
    if baseline.shape != corrected.shape or baseline.size == 0:
        raise ValueError("rollout errors must have matching non-empty shapes")
    return float(np.mean(np.log((baseline + 1e-12) / (corrected + 1e-12))))


def _median_steady_update_time(update_times: list[float]) -> float | None:
    """Median update time after the compile/FD-heavy first optimizer update."""
    steady = update_times[1:] if len(update_times) > 1 else update_times
    return float(np.median(steady)) if steady else None


def _mean_curve(curves: list[list[float]]) -> np.ndarray:
    """Average completed prefixes of repeated optimization or rollout curves."""
    if not curves:
        return np.asarray([], dtype=np.float32)
    common_length = min(len(curve) for curve in curves)
    if common_length == 0:
        return np.asarray([], dtype=np.float32)
    return np.mean(
        np.stack([np.asarray(curve[:common_length]) for curve in curves]),
        axis=0,
    ).astype(np.float32)


def _mean_optional(values: list[float | None]) -> float | None:
    """Average present scalar measurements from repeated model seeds."""
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _std(values: list[float]) -> float:
    """Population standard deviation, including the well-defined singleton case."""
    return float(np.std(np.asarray(values, dtype=np.float64)))


@kernel(
    sweep_mode="none",
    selector_fn=active_differentiable_solvers,
    snapshot_filename="corrector_fields.npz",
    snapshot_prefixes=(
        "loss",
        "loss_seed_std",
        "loss_stop_gradient",
        "loss_stop_gradient_seed_std",
        "grad_norm",
        "grad_norm_stop_gradient",
        "update_time",
        "update_time_stop_gradient",
        "error_corrected",
        "error_corrected_seed_std",
        "error_stop_gradient",
        "error_stop_gradient_seed_std",
        "error_uncorrected",
        "rollout_corrected",
        "rollout_stop_gradient",
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

    configured_seeds = training.get("model_seeds")
    if configured_seeds is None:
        model_seeds = (int(training.get("model_seed", 0)),)
    else:
        model_seeds = tuple(int(seed) for seed in configured_seeds)
    if not model_seeds:
        raise ValueError("training.model_seeds must contain at least one seed")

    # Evaluate the solver itself once: these trajectories do not depend on a
    # neural-model seed. A long native call and a sequence of short canonical
    # restarts expose coupling penalties separately from integration accuracy.
    uncorrected_errors: list[list[float]] = []
    native_final_errors: list[float] = []
    first_uncorrected: np.ndarray | None = None
    for reference in test:
        uncorrected_rollout, raw_err = _evaluate_rollout(
            t,
            ctx,
            None,
            reference,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=False,
        )
        native_final = _solver_advance(
            t,
            ctx,
            jnp.asarray(reference[0]),
            frame_steps=frame_steps * (reference.shape[0] - 1),
        )
        uncorrected_errors.append(raw_err)
        native_final_errors.append(relative_l2(np.asarray(native_final), reference[-1]))
        if first_uncorrected is None:
            first_uncorrected = uncorrected_rollout

    uncorrected_error = np.mean(np.asarray(uncorrected_errors), axis=0)

    models: list[Any] = []
    losses_by_seed: list[list[float]] = []
    grad_norms_by_seed: list[list[float]] = []
    update_times_by_seed: list[list[float]] = []
    fd_errors: list[float | None] = []
    completed_by_seed: list[bool] = []
    training_walls: list[float] = []
    stop_gradient_losses_by_seed: list[list[float]] = []
    stop_gradient_grad_norms_by_seed: list[list[float]] = []
    stop_gradient_update_times_by_seed: list[list[float]] = []
    stop_gradient_completed_by_seed: list[bool] = []
    stop_gradient_training_walls: list[float] = []
    corrected_errors_by_seed: list[np.ndarray] = []
    stop_gradient_errors_by_seed: list[np.ndarray] = []
    first_corrected: np.ndarray | None = None
    first_stop_gradient: np.ndarray | None = None

    for seed_idx, model_seed in enumerate(model_seeds):
        seed_training = {
            **training,
            # One end-to-end FD check is enough for a shared architecture and
            # solver VJP; repeating it for initialisation replicates adds cost
            # without strengthening the paired training comparison.
            "check_grad": bool(training.get("check_grad", True)) and seed_idx == 0,
        }
        started = time.perf_counter()
        (
            model,
            losses,
            grad_norms,
            update_times,
            fd_error,
            completed,
        ) = _train_corrector(
            t,
            ctx,
            train,
            frame_steps=frame_steps,
            training=seed_training,
            velocity_scale=velocity_scale,
            differentiate_solver=True,
            model_seed=model_seed,
        )
        training_walls.append(time.perf_counter() - started)

        stop_gradient_started = time.perf_counter()
        (
            stop_gradient_model,
            stop_gradient_losses,
            stop_gradient_grad_norms,
            stop_gradient_update_times,
            _stop_gradient_fd_error,
            stop_gradient_completed,
        ) = _train_corrector(
            t,
            ctx,
            train,
            frame_steps=frame_steps,
            training=seed_training,
            velocity_scale=velocity_scale,
            differentiate_solver=False,
            model_seed=model_seed,
        )
        stop_gradient_training_walls.append(time.perf_counter() - stop_gradient_started)

        seed_corrected_errors: list[list[float]] = []
        seed_stop_gradient_errors: list[list[float]] = []
        for reference_idx, reference in enumerate(test):
            corrected_rollout, corr_err = _evaluate_rollout(
                t,
                ctx,
                model,
                reference,
                frame_steps=frame_steps,
                velocity_scale=velocity_scale,
                corrected=True,
            )
            stop_gradient_rollout, stop_gradient_err = _evaluate_rollout(
                t,
                ctx,
                stop_gradient_model,
                reference,
                frame_steps=frame_steps,
                velocity_scale=velocity_scale,
                corrected=True,
            )
            seed_corrected_errors.append(corr_err)
            seed_stop_gradient_errors.append(stop_gradient_err)
            if seed_idx == 0 and reference_idx == 0:
                first_corrected = corrected_rollout
                first_stop_gradient = stop_gradient_rollout

        models.append(model)
        losses_by_seed.append(losses)
        grad_norms_by_seed.append(grad_norms)
        update_times_by_seed.append(update_times)
        fd_errors.append(fd_error)
        completed_by_seed.append(completed)
        stop_gradient_losses_by_seed.append(stop_gradient_losses)
        stop_gradient_grad_norms_by_seed.append(stop_gradient_grad_norms)
        stop_gradient_update_times_by_seed.append(stop_gradient_update_times)
        stop_gradient_completed_by_seed.append(stop_gradient_completed)
        corrected_errors_by_seed.append(
            np.mean(np.asarray(seed_corrected_errors), axis=0)
        )
        stop_gradient_errors_by_seed.append(
            np.mean(np.asarray(seed_stop_gradient_errors), axis=0)
        )

    corrected_errors_array = np.stack(corrected_errors_by_seed)
    stop_gradient_errors_array = np.stack(stop_gradient_errors_by_seed)
    corrected_error = np.mean(corrected_errors_array, axis=0)
    corrected_error_std = np.std(corrected_errors_array, axis=0)
    stop_gradient_error = np.mean(stop_gradient_errors_array, axis=0)
    stop_gradient_error_std = np.std(stop_gradient_errors_array, axis=0)
    losses = _mean_curve(losses_by_seed)
    stop_gradient_losses = _mean_curve(stop_gradient_losses_by_seed)
    grad_norms = _mean_curve(grad_norms_by_seed)
    stop_gradient_grad_norms = _mean_curve(stop_gradient_grad_norms_by_seed)
    update_times = _mean_curve(update_times_by_seed)
    stop_gradient_update_times = _mean_curve(stop_gradient_update_times_by_seed)
    loss_seed_std = (
        np.std(np.stack([curve[: len(losses)] for curve in losses_by_seed]), axis=0)
        if losses.size
        else np.asarray([], dtype=np.float32)
    )
    stop_gradient_loss_seed_std = (
        np.std(
            np.stack(
                [
                    curve[: len(stop_gradient_losses)]
                    for curve in stop_gradient_losses_by_seed
                ]
            ),
            axis=0,
        )
        if stop_gradient_losses.size
        else np.asarray([], dtype=np.float32)
    )
    training_wall = float(np.sum(training_walls))
    stop_gradient_training_wall = float(np.sum(stop_gradient_training_walls))
    total_updates = sum(len(curve) for curve in losses_by_seed)
    stop_gradient_total_updates = sum(
        len(curve) for curve in stop_gradient_losses_by_seed
    )
    steady_update_times = [
        value
        for seed_times in update_times_by_seed
        for value in (seed_times[1:] if len(seed_times) > 1 else seed_times)
    ]
    stop_gradient_steady_update_times = [
        value
        for seed_times in stop_gradient_update_times_by_seed
        for value in (seed_times[1:] if len(seed_times) > 1 else seed_times)
    ]

    final_corrected = float(corrected_error[-1])
    final_stop_gradient = float(stop_gradient_error[-1])
    final_uncorrected = float(uncorrected_error[-1])
    native_final_error = float(np.mean(native_final_errors))
    mean_corrected = float(np.mean(corrected_error[1:]))
    mean_stop_gradient = float(np.mean(stop_gradient_error[1:]))
    mean_uncorrected = float(np.mean(uncorrected_error[1:]))
    rollout_log_gains = [
        _rollout_log_gain(uncorrected_error, seed_error)
        for seed_error in corrected_errors_by_seed
    ]
    stop_gradient_log_gains = [
        _rollout_log_gain(uncorrected_error, seed_error)
        for seed_error in stop_gradient_errors_by_seed
    ]
    solver_vjp_log_lifts = [
        _rollout_log_gain(stopped, full)
        for stopped, full in zip(
            stop_gradient_errors_by_seed,
            corrected_errors_by_seed,
            strict=True,
        )
    ]
    rollout_log_gain = float(np.mean(rollout_log_gains))
    stop_gradient_log_gain = float(np.mean(stop_gradient_log_gains))
    solver_vjp_log_lift = float(np.mean(solver_vjp_log_lifts))
    threshold = float(evaluation.get("stable_error_threshold", 1.0))
    interval_time = float(ctx.phys["dt"]) * frame_steps
    model = models[0]
    parameter_count = sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    median_update_time = (
        float(np.median(steady_update_times)) if steady_update_times else None
    )
    stop_gradient_median_update_time = (
        float(np.median(stop_gradient_steady_update_times))
        if stop_gradient_steady_update_times
        else None
    )

    metrics = {
        "final_rollout_error": final_corrected,
        "final_rollout_error_seed_std": float(corrected_error_std[-1]),
        "stop_gradient_final_rollout_error": final_stop_gradient,
        "stop_gradient_final_rollout_error_seed_std": float(
            stop_gradient_error_std[-1]
        ),
        "uncorrected_rollout_error": final_uncorrected,
        "native_final_rollout_error": native_final_error,
        "state_restart_error_ratio": final_uncorrected / (native_final_error + 1e-12),
        "mean_rollout_error": mean_corrected,
        "stop_gradient_mean_rollout_error": mean_stop_gradient,
        "uncorrected_mean_rollout_error": mean_uncorrected,
        "improvement_pct": 100.0
        * (final_uncorrected - final_corrected)
        / (final_uncorrected + 1e-12),
        "mean_improvement_pct": 100.0
        * (mean_uncorrected - mean_corrected)
        / (mean_uncorrected + 1e-12),
        "rollout_log_gain": rollout_log_gain,
        "rollout_log_gain_seed_std": _std(rollout_log_gains),
        "geometric_error_reduction": float(np.exp(rollout_log_gain)),
        "stop_gradient_rollout_log_gain": stop_gradient_log_gain,
        "stop_gradient_rollout_log_gain_seed_std": _std(stop_gradient_log_gains),
        "stop_gradient_geometric_error_reduction": float(
            np.exp(stop_gradient_log_gain)
        ),
        "solver_vjp_log_lift": solver_vjp_log_lift,
        "solver_vjp_log_lift_seed_std": _std(solver_vjp_log_lifts),
        "solver_vjp_geometric_lift": float(np.exp(solver_vjp_log_lift)),
        "stable_horizon": _first_unstable(corrected_error.tolist(), threshold)
        * interval_time,
        "stop_gradient_stable_horizon": _first_unstable(
            stop_gradient_error.tolist(), threshold
        )
        * interval_time,
        "uncorrected_stable_horizon": _first_unstable(
            uncorrected_error.tolist(), threshold
        )
        * interval_time,
        "initial_train_loss": float(losses[0]) if losses.size else None,
        "final_train_loss": float(losses[-1]) if losses.size else None,
        "best_train_loss": float(np.min(losses)) if losses.size else None,
        "train_loss_log_gain": float(
            np.log((losses[0] + 1e-12) / (np.min(losses) + 1e-12))
        )
        if losses.size
        else None,
        "n_updates": len(losses),
        "total_optimizer_updates": total_updates,
        "training_wall_time_s": training_wall,
        "seconds_per_update": training_wall / max(total_updates, 1),
        "median_update_time_s": median_update_time,
        "stop_gradient_initial_train_loss": float(stop_gradient_losses[0])
        if stop_gradient_losses.size
        else None,
        "stop_gradient_final_train_loss": float(stop_gradient_losses[-1])
        if stop_gradient_losses.size
        else None,
        "stop_gradient_best_train_loss": float(np.min(stop_gradient_losses))
        if stop_gradient_losses.size
        else None,
        "stop_gradient_n_updates": len(stop_gradient_losses),
        "stop_gradient_total_optimizer_updates": stop_gradient_total_updates,
        "stop_gradient_training_wall_time_s": stop_gradient_training_wall,
        "stop_gradient_seconds_per_update": stop_gradient_training_wall
        / max(stop_gradient_total_updates, 1),
        "stop_gradient_median_update_time_s": stop_gradient_median_update_time,
        "solver_vjp_update_overhead_ratio": median_update_time
        / (stop_gradient_median_update_time + 1e-12)
        if median_update_time is not None
        and stop_gradient_median_update_time is not None
        else None,
        "final_grad_norm": float(grad_norms[-1]) if grad_norms.size else None,
        "stop_gradient_final_grad_norm": float(stop_gradient_grad_norms[-1])
        if stop_gradient_grad_norms.size
        else None,
        "end_to_end_fd_rel_error": _mean_optional(fd_errors),
        "final_divergence_rms": divergence_rms(first_corrected[-1], ctx.domain_extent)
        if first_corrected is not None
        else None,
        "stop_gradient_final_divergence_rms": divergence_rms(
            first_stop_gradient[-1], ctx.domain_extent
        )
        if first_stop_gradient is not None
        else None,
        "uncorrected_final_divergence_rms": divergence_rms(
            first_uncorrected[-1], ctx.domain_extent
        )
        if first_uncorrected is not None
        else None,
        "final_centered_divergence_rms": centered_divergence_rms(
            first_corrected[-1], ctx.domain_extent
        )
        if first_corrected is not None
        else None,
        "stop_gradient_final_centered_divergence_rms": centered_divergence_rms(
            first_stop_gradient[-1], ctx.domain_extent
        )
        if first_stop_gradient is not None
        else None,
        "uncorrected_final_centered_divergence_rms": centered_divergence_rms(
            first_uncorrected[-1], ctx.domain_extent
        )
        if first_uncorrected is not None
        else None,
        "final_energy_ratio_to_reference": kinetic_energy(first_corrected[-1])
        / (kinetic_energy(test[0, -1]) + 1e-12)
        if first_corrected is not None
        else None,
        "stop_gradient_final_energy_ratio_to_reference": kinetic_energy(
            first_stop_gradient[-1]
        )
        / (kinetic_energy(test[0, -1]) + 1e-12)
        if first_stop_gradient is not None
        else None,
        "uncorrected_final_energy_ratio_to_reference": kinetic_energy(
            first_uncorrected[-1]
        )
        / (kinetic_energy(test[0, -1]) + 1e-12)
        if first_uncorrected is not None
        else None,
        "final_enstrophy_ratio_to_reference": enstrophy(
            first_corrected[-1], ctx.domain_extent
        )
        / (enstrophy(test[0, -1], ctx.domain_extent) + 1e-12)
        if first_corrected is not None
        else None,
        "uncorrected_final_enstrophy_ratio_to_reference": enstrophy(
            first_uncorrected[-1], ctx.domain_extent
        )
        / (enstrophy(test[0, -1], ctx.domain_extent) + 1e-12)
        if first_uncorrected is not None
        else None,
        "corrector_architecture": model.architecture,
        "corrector_parameter_count": parameter_count,
        "model_seeds": list(model_seeds),
        "n_model_seeds": len(model_seeds),
        "n_test_trajectories": int(test.shape[0]),
        "visualization_model_seed": model_seeds[0],
        "completed": all(completed_by_seed) and all(stop_gradient_completed_by_seed),
        "dataset_hash": dataset_hash,
        "reference_kind": str(
            dataset_cfg.get("reference_kind", "pseudo_spectral_multimode")
        ),
        "correction_intervals": int(test.shape[1] - 1),
        "native_steps": frame_steps * int(test.shape[1] - 1),
        "rollout_final_time": interval_time * int(test.shape[1] - 1),
        "state_restart": True,
    }
    snapshots = {
        "loss": losses,
        "loss_seed_std": np.asarray(loss_seed_std, dtype=np.float32),
        "loss_stop_gradient": stop_gradient_losses,
        "loss_stop_gradient_seed_std": np.asarray(
            stop_gradient_loss_seed_std,
            dtype=np.float32,
        ),
        "grad_norm": grad_norms,
        "grad_norm_stop_gradient": np.asarray(
            stop_gradient_grad_norms,
            dtype=np.float32,
        ),
        "update_time": update_times,
        "update_time_stop_gradient": np.asarray(
            stop_gradient_update_times,
            dtype=np.float32,
        ),
        "error_corrected": np.asarray(corrected_error, dtype=np.float32),
        "error_corrected_seed_std": np.asarray(
            corrected_error_std,
            dtype=np.float32,
        ),
        "error_stop_gradient": np.asarray(
            stop_gradient_error,
            dtype=np.float32,
        ),
        "error_stop_gradient_seed_std": np.asarray(
            stop_gradient_error_std,
            dtype=np.float32,
        ),
        "error_uncorrected": np.asarray(uncorrected_error, dtype=np.float32),
    }
    if (
        first_corrected is not None
        and first_stop_gradient is not None
        and first_uncorrected is not None
    ):
        snapshots["rollout_corrected"] = first_corrected
        snapshots["rollout_stop_gradient"] = first_stop_gradient
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
