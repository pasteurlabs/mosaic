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
import weakref
from functools import lru_cache, partial
from typing import Any, NamedTuple

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
    finite_volume_reference_trajectory,
    init_corrector,
    kinetic_energy,
    reference_trajectory,
    relative_l2,
    spectral_prolong,
    spectral_restrict,
)
from .ics import _multimode, _tgv

_DATASET_LOCK = threading.Lock()
_NATIVE_STATE_SUPPORT_LOCK = threading.Lock()
_NATIVE_STATE_SUPPORT: weakref.WeakKeyDictionary[Any, bool] = (
    weakref.WeakKeyDictionary()
)


class _TrainingStage(NamedTuple):
    """One recurrent-lookahead stage in a corrector curriculum."""

    unroll: int
    updates: int
    lr: float


class _DirectionalFD(NamedTuple):
    """One directional finite-difference comparison."""

    relative_error: float
    finite_difference: float
    autodiff: float


class _TrainingResult(NamedTuple):
    """A trained corrector plus its curriculum checkpoints and traces."""

    model: Any
    losses: list[float]
    grad_norms: list[float]
    update_times: list[float]
    fd_error: float | None
    fd_epsilons: tuple[float, ...]
    fd_errors: tuple[float, ...]
    fd_finite_differences: tuple[float, ...]
    fd_autodiff: tuple[float, ...]
    fd_stage_unrolls: tuple[int, ...]
    fd_stage_errors: tuple[tuple[float, ...], ...]
    fd_stage_finite_differences: tuple[tuple[float, ...], ...]
    fd_stage_autodiff: tuple[tuple[float, ...], ...]
    completed: bool
    stage_models: tuple[Any, ...]
    stage_unrolls: tuple[int, ...]
    stage_boundaries: tuple[int, ...]
    stage_learning_rates: tuple[float, ...]


def _training_stages(training: dict[str, Any]) -> tuple[_TrainingStage, ...]:
    """Normalize a fixed-horizon run or explicit curriculum into stages."""
    default_lr = float(training.get("lr", 1e-4))
    configured = training.get("curriculum")
    if configured is None:
        configured = [
            {
                "unroll": int(training.get("unroll", 4)),
                "updates": int(training.get("max_updates", 100)),
                "lr": default_lr,
            }
        ]
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("training.curriculum must be a non-empty list")

    stages: list[_TrainingStage] = []
    for index, stage in enumerate(configured):
        if not isinstance(stage, dict):
            raise TypeError(f"training.curriculum[{index}] must be a mapping")
        normalized = _TrainingStage(
            unroll=int(stage["unroll"]),
            updates=int(stage["updates"]),
            lr=float(stage.get("lr", default_lr)),
        )
        if normalized.unroll < 1:
            raise ValueError(f"training.curriculum[{index}].unroll must be positive")
        if normalized.updates < 1:
            raise ValueError(f"training.curriculum[{index}].updates must be positive")
        if normalized.lr <= 0:
            raise ValueError(f"training.curriculum[{index}].lr must be positive")
        stages.append(normalized)
    return tuple(stages)


def _maximum_training_unroll(training: dict[str, Any]) -> int:
    """Return the largest look-ahead required by a training configuration."""
    return max(stage.unroll for stage in _training_stages(training))


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
        if reference_kind in {
            "pseudo_spectral_multimode",
            "finite_volume_multimode",
        }:
            fine_initial = _multimode(
                reference_n,
                L=domain_extent,
                seed=seed,
                k0=k0,
                sigma_k=sigma_k,
                amplitude=amplitude,
            )
            trajectory_fn = (
                reference_trajectory
                if reference_kind == "pseudo_spectral_multimode"
                else finite_volume_reference_trajectory
            )
            fine = trajectory_fn(
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


def _reference_convergence_audit(
    *,
    production_trajectories: np.ndarray,
    production_seeds: tuple[int, ...],
    physics: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    domain_extent: float,
) -> dict[str, Any]:
    """Compare the production reference with a finer space-time realization."""
    audit_factor = dataset.get("reference_audit_factor")
    if audit_factor is None:
        return {}

    n = int(physics["N"])
    reference_kind = str(dataset["reference_kind"])
    audit_seeds = tuple(
        int(value)
        for value in dataset.get(
            "reference_audit_seeds",
            [production_seeds[0], production_seeds[-1]],
        )
    )
    missing = sorted(set(audit_seeds) - set(production_seeds))
    if missing:
        raise ValueError(
            f"reference audit seeds are absent from the dataset: {missing}"
        )
    n_frames = int(evaluation.get("rollout_frames", 32))
    production_factor = int(dataset.get("reference_factor", 2))
    production_n = n * production_factor
    audit_n = n * int(audit_factor)
    audit_substeps = int(dataset.get("reference_audit_substeps", audit_factor))
    if audit_n < production_n or audit_n % production_n != 0:
        raise ValueError(
            "reference_audit_factor must be an integer refinement of reference_factor"
        )
    if reference_kind not in {
        "pseudo_spectral_multimode",
        "finite_volume_multimode",
    }:
        raise ValueError(
            f"reference convergence audit is unsupported for {reference_kind!r}"
        )
    trajectory_fn = (
        reference_trajectory
        if reference_kind == "pseudo_spectral_multimode"
        else finite_volume_reference_trajectory
    )
    audit_trajectories = []
    for seed in audit_seeds:
        production_initial = _multimode(
            production_n,
            L=domain_extent,
            seed=seed,
            k0=float(dataset.get("k0", 6.0)),
            sigma_k=float(dataset.get("sigma_k", 1.0)),
            amplitude=float(dataset.get("amplitude", 0.3)),
        )
        audit_initial = spectral_prolong(production_initial, audit_n)
        audit_fine = trajectory_fn(
            audit_initial,
            viscosity=float(physics["nu"]),
            dt=float(physics["dt"]),
            frame_steps=int(physics["steps"]),
            n_frames=n_frames,
            substeps=audit_substeps,
            domain_extent=float(domain_extent),
        )
        audit_trajectories.append(
            np.stack([spectral_restrict(frame, n) for frame in audit_fine])
        )
    audit_trajectories = np.stack(audit_trajectories)

    frames = tuple(
        sorted(
            {
                int(frame)
                for frame in dataset.get(
                    "reference_audit_frames",
                    [1, max(1, n_frames // 3), n_frames],
                )
                if 0 < int(frame) <= n_frames
            }
        )
    )
    if not frames:
        raise ValueError("reference_audit_frames must contain a positive valid frame")
    production_by_seed = {
        seed: production_trajectories[index]
        for index, seed in enumerate(production_seeds)
    }
    errors = np.asarray(
        [
            relative_l2(
                production_by_seed[seed][frame],
                audit_trajectories[seed_index, frame],
            )
            for seed_index, seed in enumerate(audit_seeds)
            for frame in frames
        ],
        dtype=np.float64,
    )
    tolerance = float(dataset.get("reference_convergence_tolerance", 0.05))
    p95 = float(np.percentile(errors, 95))
    return {
        "reference_convergence_audit_applied": True,
        "reference_convergence_scheme": reference_kind,
        "reference_production_grid_size": production_n,
        "reference_audit_grid_size": audit_n,
        "reference_production_dt": float(
            physics["dt"] / int(dataset.get("reference_substeps", 2))
        ),
        "reference_audit_dt": float(physics["dt"] / audit_substeps),
        "reference_convergence_frames": list(frames),
        "reference_convergence_seeds": list(audit_seeds),
        "reference_convergence_errors": errors.tolist(),
        "reference_convergence_error_median": float(np.median(errors)),
        "reference_convergence_error_p95": p95,
        "reference_convergence_error_max": float(np.max(errors)),
        "reference_convergence_tolerance": tolerance,
        "reference_convergence_passed": bool(p95 <= tolerance),
        "eligible_for_corrector_training": bool(p95 <= tolerance),
    }


def _make_reference_datasets(
    *,
    physics: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    training: dict[str, Any],
    domain_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Build train windows plus seen- and held-out-IC evaluation trajectories."""
    n = int(physics["N"])
    train_seeds = tuple(int(v) for v in dataset.get("train_seeds", [0, 1, 2, 3]))
    test_seeds = tuple(int(v) for v in dataset.get("test_seeds", [100, 101]))
    train_frames = int(dataset.get("train_frames", 16))
    eval_frames = int(evaluation.get("rollout_frames", 32))
    unroll = _maximum_training_unroll(training)
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
    train_rollouts = all_trajectories[:n_train, : eval_frames + 1]
    test = all_trajectories[n_train:, : eval_frames + 1]
    audit = _reference_convergence_audit(
        production_trajectories=all_trajectories,
        production_seeds=train_seeds + test_seeds,
        physics=physics,
        dataset=dataset,
        evaluation=evaluation,
        domain_extent=domain_extent,
    )
    return train, train_rollouts, test, _dataset_digest(config), audit


def make_reference_dataset(
    *,
    physics: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    training: dict[str, Any],
    domain_extent: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return training windows and held-out trajectories for public callers."""
    train, _train_rollouts, test, dataset_hash, _audit = _make_reference_datasets(
        physics=physics,
        dataset=dataset,
        evaluation=evaluation,
        training=training,
        domain_extent=domain_extent,
    )
    return train, test, dataset_hash


def _solver_advance(
    t: Any,
    ctx: KernelContext,
    velocity: jax.Array,
    *,
    frame_steps: int,
    native_state: Any | None = None,
) -> tuple[jax.Array, Any | None]:
    """Advance one canonical correction interval through a Tesseract."""
    return _solver_advance_with_physics(
        t,
        ctx,
        velocity,
        dt=float(ctx.phys["dt"]),
        steps=frame_steps,
        native_state=native_state,
    )


def _supports_native_state(t: Any) -> bool:
    """Return whether apply advertises a differentiable recurrent checkpoint."""
    with _NATIVE_STATE_SUPPORT_LOCK:
        try:
            return _NATIVE_STATE_SUPPORT[t]
        except (KeyError, TypeError):
            pass
    try:
        schemas = t.openapi_schema["components"]["schemas"]
        properties = schemas["Apply_InputSchema"]["properties"]
        differentiable_inputs = schemas["ApplyInputSchema"]["differentiable_arrays"]
        differentiable_outputs = schemas["ApplyOutputSchema"]["differentiable_arrays"]
    except (AttributeError, KeyError, TypeError):
        return False
    supports_native_state = (
        "return_state" in properties
        and "state" in properties
        and "state" in differentiable_inputs
        and "state" in differentiable_outputs
    )
    with _NATIVE_STATE_SUPPORT_LOCK:
        try:
            _NATIVE_STATE_SUPPORT[t] = supports_native_state
        except TypeError:
            pass
    return supports_native_state


def _solver_advance_with_physics(
    t: Any,
    ctx: KernelContext,
    velocity: jax.Array | np.ndarray,
    *,
    dt: float,
    steps: int,
    native_state: Any | None = None,
) -> tuple[jax.Array, Any | None]:
    """Advance a state with an explicit grid and temporal compute budget."""
    physics = {
        **ctx.phys,
        "N": int(velocity.shape[0]),
        "dt": float(dt),
        "steps": int(steps),
    }
    inputs = ctx.make_inputs(ctx.name, velocity, **physics)
    supports_native_state = _supports_native_state(t)
    if supports_native_state:
        inputs = {**inputs, "return_state": True}
    if native_state is not None:
        if not supports_native_state:
            raise RuntimeError(
                f"Solver '{ctx.name}' received native state without advertising "
                "'state' and 'return_state' apply inputs"
            )
        inputs = {**inputs, "state": native_state}
    outputs = apply_tesseract(t, inputs)
    if ctx.output_key not in outputs:
        raise RuntimeError(
            f"Solver '{ctx.name}' did not return output {ctx.output_key!r}"
        )
    next_native_state = outputs.get("state") if supports_native_state else None
    if supports_native_state and next_native_state is None:
        raise RuntimeError(
            f"Solver '{ctx.name}' advertised recurrent state but did not return 'state'"
        )
    return outputs[ctx.output_key], next_native_state


def _stop_recurrent_gradient(
    velocity: jax.Array,
    native_state: Any | None,
) -> tuple[jax.Array, Any | None]:
    """Cut solver-VJP paths through canonical and solver-native recurrent state."""
    return (
        jax.lax.stop_gradient(velocity),
        jax.tree_util.tree_map(jax.lax.stop_gradient, native_state),
    )


def _passes_reference_accuracy_gate(
    reference_kind: str,
    *,
    first_interval_error: float,
    first_interval_tolerance: float,
    native_long_error: float,
    native_long_tolerance: float,
) -> bool:
    """Gate only references whose absolute accuracy is shared across solvers."""
    if reference_kind == "solver_self_refined":
        return True
    return bool(
        first_interval_error <= first_interval_tolerance
        and native_long_error <= native_long_tolerance
    )


def _make_solver_self_reference_datasets(
    t: Any,
    ctx: KernelContext,
    *,
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    training: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Build and audit targets from the same solver at a refined compute budget."""
    started = time.perf_counter()
    n = int(ctx.phys["N"])
    reference_factor = int(dataset.get("reference_factor", 2))
    temporal_factor = int(dataset.get("reference_temporal_factor", 2))
    if reference_factor < 1 or temporal_factor < 1:
        raise ValueError("reference refinement factors must be positive")

    train_seeds = tuple(int(v) for v in dataset.get("train_seeds", [0, 1, 2, 3]))
    test_seeds = tuple(int(v) for v in dataset.get("test_seeds", [100, 101]))
    all_seeds = train_seeds + test_seeds
    train_frames = int(dataset.get("train_frames", 16))
    eval_frames = int(evaluation.get("rollout_frames", 32))
    unroll = _maximum_training_unroll(training)
    n_frames = max(train_frames, eval_frames, unroll)

    fine_n = n * reference_factor
    coarse_dt = float(ctx.phys["dt"])
    coarse_steps = int(ctx.phys["steps"])
    fine_dt = coarse_dt / temporal_factor
    fine_steps = coarse_steps * temporal_factor
    k0 = float(dataset.get("k0", 6.0))
    sigma_k = float(dataset.get("sigma_k", 1.0))
    amplitude = float(dataset.get("amplitude", 0.3))

    apply_count = 0

    def advance(
        velocity: np.ndarray,
        *,
        dt: float,
        steps: int,
        native_state: Any | None = None,
    ) -> tuple[np.ndarray, Any | None]:
        nonlocal apply_count
        apply_count += 1
        next_velocity, next_native_state = _solver_advance_with_physics(
            t,
            ctx,
            velocity,
            dt=dt,
            steps=steps,
            native_state=native_state,
        )
        return np.asarray(next_velocity), next_native_state

    trajectories: dict[int, np.ndarray] = {}
    fine_initials: dict[int, np.ndarray] = {}
    for seed in all_seeds:
        fine_state = np.asarray(
            _multimode(
                fine_n,
                L=ctx.domain_extent,
                seed=seed,
                k0=k0,
                sigma_k=sigma_k,
                amplitude=amplitude,
            ),
            dtype=np.float32,
        )
        fine_initials[seed] = fine_state
        frames = [spectral_restrict(fine_state, n)]
        fine_native_state = None
        for _frame in range(n_frames):
            fine_state, fine_native_state = advance(
                fine_state,
                dt=fine_dt,
                steps=fine_steps,
                native_state=fine_native_state,
            )
            frames.append(spectral_restrict(fine_state, n))
        trajectories[seed] = np.asarray(frames, dtype=np.float32)

    requested_audit_seeds = tuple(
        int(v)
        for v in dataset.get(
            "prefix_audit_seeds",
            [train_seeds[0], test_seeds[0]],
        )
    )
    missing_audit_seeds = set(requested_audit_seeds) - set(all_seeds)
    if missing_audit_seeds:
        raise ValueError(
            "dataset.prefix_audit_seeds must belong to the train/test split: "
            f"{sorted(missing_audit_seeds)}"
        )
    audit_frames = tuple(
        sorted(
            {
                int(frame)
                for frame in dataset.get(
                    "prefix_audit_frames",
                    [1, min(train_frames, n_frames), n_frames],
                )
                if 0 < int(frame) <= n_frames
            }
        )
    )
    if not audit_frames:
        raise ValueError("dataset.prefix_audit_frames must select a positive frame")

    coarse_closure_errors: list[float] = []
    fine_closure_errors: list[float] = []
    refinement_signals: list[float] = []
    finite = True
    audit_frame_set = set(audit_frames)
    max_audit_frame = max(audit_frames)
    for seed in requested_audit_seeds:
        target = trajectories[seed]
        coarse_state = np.asarray(target[0])
        coarse_native_state = None
        coarse_repeated: dict[int, np.ndarray] = {}
        for frame in range(1, max_audit_frame + 1):
            coarse_state, coarse_native_state = advance(
                coarse_state,
                dt=coarse_dt,
                steps=coarse_steps,
                native_state=coarse_native_state,
            )
            if frame in audit_frame_set:
                coarse_repeated[frame] = coarse_state

        for frame in audit_frames:
            coarse_native, _ = advance(
                np.asarray(target[0]),
                dt=coarse_dt,
                steps=coarse_steps * frame,
            )
            fine_native, _ = advance(
                fine_initials[seed],
                dt=fine_dt,
                steps=fine_steps * frame,
            )
            fine_native_coarse = spectral_restrict(fine_native, n)
            coarse_closure_errors.append(
                relative_l2(coarse_repeated[frame], coarse_native)
            )
            fine_closure_errors.append(relative_l2(target[frame], fine_native_coarse))
            refinement_signals.append(
                relative_l2(coarse_repeated[frame], target[frame])
            )
            finite = finite and bool(
                np.all(np.isfinite(coarse_repeated[frame]))
                and np.all(np.isfinite(coarse_native))
                and np.all(np.isfinite(target[frame]))
                and np.all(np.isfinite(fine_native_coarse))
            )

    closure_tolerance = float(dataset.get("closure_relative_tolerance", 0.01))
    closure_to_signal_tolerance = float(dataset.get("closure_to_signal_tolerance", 0.1))
    minimum_signal = float(dataset.get("minimum_refinement_signal", 1e-4))
    max_coarse_closure = float(max(coarse_closure_errors))
    max_fine_closure = float(max(fine_closure_errors))
    mean_signal = float(np.mean(refinement_signals))
    coarse_closure_to_signal = [
        closure / (signal + 1e-12)
        for closure, signal in zip(
            coarse_closure_errors,
            refinement_signals,
            strict=True,
        )
    ]
    fine_closure_to_signal = [
        closure / (signal + 1e-12)
        for closure, signal in zip(
            fine_closure_errors,
            refinement_signals,
            strict=True,
        )
    ]
    eligible = bool(
        finite
        and max_coarse_closure <= closure_tolerance
        and max_fine_closure <= closure_tolerance
        and max(coarse_closure_to_signal) <= closure_to_signal_tolerance
        and max(fine_closure_to_signal) <= closure_to_signal_tolerance
        and mean_signal > minimum_signal
    )

    config = {
        "reference_kind": "solver_self_refined",
        "solver": ctx.name,
        "n": n,
        "reference_factor": reference_factor,
        "reference_temporal_factor": temporal_factor,
        "viscosity": float(ctx.phys["nu"]),
        "coarse_dt": coarse_dt,
        "coarse_steps": coarse_steps,
        "fine_dt": fine_dt,
        "fine_steps": fine_steps,
        "n_frames": n_frames,
        "domain_extent": float(ctx.domain_extent),
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "k0": k0,
        "sigma_k": sigma_k,
        "amplitude": amplitude,
        "prefix_audit_seeds": requested_audit_seeds,
        "prefix_audit_frames": audit_frames,
        "closure_relative_tolerance": closure_tolerance,
        "closure_to_signal_tolerance": closure_to_signal_tolerance,
        "minimum_refinement_signal": minimum_signal,
    }
    all_trajectories = np.stack([trajectories[seed] for seed in all_seeds])
    n_train = len(train_seeds)
    train = all_trajectories[:n_train, : train_frames + 1]
    train_rollouts = all_trajectories[:n_train, : eval_frames + 1]
    test = all_trajectories[n_train:, : eval_frames + 1]
    audit = {
        "eligible_for_corrector_training": eligible,
        "reference_generation_wall_time_s": time.perf_counter() - started,
        "reference_generation_apply_count": apply_count,
        "reference_grid_size": fine_n,
        "reference_dt": fine_dt,
        "reference_steps_per_interval": fine_steps,
        "prefix_audit_seeds": list(requested_audit_seeds),
        "prefix_audit_frames": list(audit_frames),
        "coarse_closure_errors": coarse_closure_errors,
        "fine_closure_errors": fine_closure_errors,
        "refinement_signals": refinement_signals,
        "max_coarse_closure_error": max_coarse_closure,
        "max_fine_closure_error": max_fine_closure,
        "max_coarse_closure_to_signal_ratio": float(max(coarse_closure_to_signal)),
        "max_fine_closure_to_signal_ratio": float(max(fine_closure_to_signal)),
        "mean_refinement_signal": mean_signal,
        "closure_relative_tolerance": closure_tolerance,
        "closure_to_signal_tolerance": closure_to_signal_tolerance,
        "minimum_refinement_signal": minimum_signal,
        "reference_states_finite": finite,
    }
    return train, train_rollouts, test, _dataset_digest(config), audit


def _window_loss(
    model: Any,
    targets: jax.Array,
    *,
    t: Any,
    ctx: KernelContext,
    frame_steps: int,
    velocity_scale: float,
    differentiate_solver: bool,
    loss_mode: str,
    solver_loss_weight: float,
    loss_scale: float,
    local_loss_weight: float = 1.0,
    warmup_intervals: int = 0,
) -> jax.Array:
    """Normalized recurrent loss with configurable temporal credit assignment."""
    state = targets[0]
    native_state = None
    losses: list[jax.Array] = []
    solver_mediated_losses: list[jax.Array] = []
    solver_terminal_loss: jax.Array | None = None
    for step, target in enumerate(targets[1:]):
        provisional, native_state = _solver_advance(
            t,
            ctx,
            state,
            frame_steps=frame_steps,
            native_state=native_state,
        )
        if not differentiate_solver or step < warmup_intervals:
            # Keep the identical recurrent forward trajectory but cut the
            # backward graph through both canonical and solver-native recurrent
            # state. Warm-up intervals expose inference-like states without
            # extending the differentiated chain.
            provisional, native_state = _stop_recurrent_gradient(
                provisional,
                native_state,
            )
        provisional_loss = jnp.sum((provisional - target) ** 2) / (
            jnp.sum(target**2) + 1e-12
        )
        if step == targets.shape[0] - 2:
            solver_terminal_loss = provisional_loss
        state = corrected_velocity(
            model,
            provisional,
            velocity_scale=velocity_scale,
            domain_extent=ctx.domain_extent,
        )
        if step < warmup_intervals:
            state, native_state = _stop_recurrent_gradient(state, native_state)
            continue
        losses.append(jnp.sum((state - target) ** 2) / (jnp.sum(target**2) + 1e-12))
        # The first provisional state after warm-up precedes any differentiated
        # correction. Later provisional states measure how earlier corrections
        # survive the intervening numerical solver and therefore require its VJP.
        if step > warmup_intervals:
            solver_mediated_losses.append(provisional_loss)
    if loss_mode == "mean":
        loss = jnp.mean(jnp.stack(losses))
    elif loss_mode == "terminal":
        loss = losses[-1]
    elif loss_mode == "solver_terminal":
        if solver_terminal_loss is None:
            raise RuntimeError("solver-terminal loss requires a non-empty rollout")
        loss = solver_loss_weight * solver_terminal_loss + jnp.mean(jnp.stack(losses))
    elif loss_mode == "solver_terminal_mediated":
        if solver_terminal_loss is None:
            raise RuntimeError("solver-terminal loss requires a non-empty rollout")
        local_loss = jnp.mean(jnp.stack(losses))
        loss = (
            local_loss_weight * local_loss + solver_loss_weight * solver_terminal_loss
        )
    elif loss_mode == "solver_mediated":
        local_loss = jnp.mean(jnp.stack(losses))
        mediated_loss = (
            jnp.mean(jnp.stack(solver_mediated_losses))
            if solver_mediated_losses
            else jnp.zeros_like(local_loss)
        )
        loss = local_loss_weight * local_loss + solver_loss_weight * mediated_loss
    else:
        raise ValueError(f"unknown training.loss_mode: {loss_mode!r}")
    return loss / jnp.asarray(loss_scale, dtype=loss.dtype)


def _directional_fd(
    loss_fn: Any,
    model: Any,
    grads: Any,
    key: jax.Array,
    *,
    epsilon: float,
) -> _DirectionalFD:
    """Compare AD and finite differences along one normalized direction."""
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
    return _DirectionalFD(
        relative_error=float(jnp.abs(fd - ad) / (jnp.abs(fd) + jnp.abs(ad) + 1e-12)),
        finite_difference=float(fd),
        autodiff=float(ad),
    )


def _train_corrector(
    t: Any,
    ctx: KernelContext,
    train: np.ndarray,
    *,
    frame_steps: int,
    training: dict[str, Any],
    velocity_scale: float,
    loss_scale: float,
    differentiate_solver: bool,
    model_seed: int,
) -> _TrainingResult:
    """Train one solver-specific corrector through a fixed or staged look-ahead."""
    stages = _training_stages(training)
    clip_norm = float(training.get("clip_norm", 1.0))
    seed = int(training.get("seed", 2026))
    hidden_channels = int(training.get("hidden_channels", 32))
    kernel_size = int(training.get("kernel_size", 5))
    architecture = str(training.get("architecture", "periodic_residual_cnn"))
    residual_blocks = int(training.get("residual_blocks", 5))
    loss_mode = str(training.get("loss_mode", "mean"))
    solver_loss_weight = float(training.get("solver_loss_weight", 0.1))
    local_loss_weight = float(training.get("local_loss_weight", 1.0))
    warmup_intervals = int(training.get("warmup_intervals", 0))
    fd_epsilon = float(training.get("fd_epsilon", 1e-2))
    configured_fd_epsilons = training.get("fd_epsilons")
    if configured_fd_epsilons is None:
        fd_epsilons = (
            10.0 * fd_epsilon,
            3.0 * fd_epsilon,
            fd_epsilon,
            0.3 * fd_epsilon,
            0.1 * fd_epsilon,
        )
    else:
        fd_epsilons = tuple(float(value) for value in configured_fd_epsilons)
    if not fd_epsilons or any(value <= 0 for value in fd_epsilons):
        raise ValueError("training.fd_epsilons must contain positive values")
    maximum_unroll = max(stage.unroll for stage in stages)
    if train.shape[1] < warmup_intervals + maximum_unroll + 1:
        raise ValueError(
            "the warm-up plus largest look-ahead must fit inside dataset.train_frames"
        )
    if loss_mode not in {
        "mean",
        "terminal",
        "solver_terminal",
        "solver_terminal_mediated",
        "solver_mediated",
    }:
        raise ValueError(f"unknown training.loss_mode: {loss_mode!r}")
    if solver_loss_weight < 0:
        raise ValueError("training.solver_loss_weight must be non-negative")
    if local_loss_weight < 0:
        raise ValueError("training.local_loss_weight must be non-negative")
    if warmup_intervals < 0:
        raise ValueError("training.warmup_intervals must be non-negative")
    if (
        loss_mode in {"solver_mediated", "solver_terminal_mediated"}
        and solver_loss_weight == 0
        and local_loss_weight == 0
    ):
        raise ValueError("solver-mediated training needs a non-zero loss weight")

    model = init_corrector(
        jax.random.PRNGKey(model_seed),
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        architecture=architecture,
        residual_blocks=residual_blocks,
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(stages[0].lr),
    )
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.RandomState(seed)
    losses: list[float] = []
    grad_norms: list[float] = []
    update_times: list[float] = []
    fd_error: float | None = None
    fd_error_curve: tuple[float, ...] = ()
    fd_finite_difference_curve: tuple[float, ...] = ()
    fd_autodiff_curve: tuple[float, ...] = ()
    fd_stage_unrolls: list[int] = []
    fd_stage_error_curves: list[tuple[float, ...]] = []
    fd_stage_finite_difference_curves: list[tuple[float, ...]] = []
    fd_stage_autodiff_curves: list[tuple[float, ...]] = []
    completed = True
    stage_models: list[Any] = []
    stage_boundaries: list[int] = []

    for stage_index, stage in enumerate(stages):
        optimiser = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(stage.lr),
        )
        for stage_update in range(stage.updates):
            update_started = time.perf_counter()
            trajectory_idx = int(rng.randint(train.shape[0]))
            window_intervals = warmup_intervals + stage.unroll
            max_start = train.shape[1] - window_intervals - 1
            start = int(rng.randint(max_start + 1)) if max_start > 0 else 0
            targets = jnp.asarray(
                train[trajectory_idx, start : start + window_intervals + 1]
            )

            loss_fn = partial(
                _window_loss,
                targets=targets,
                t=t,
                ctx=ctx,
                frame_steps=frame_steps,
                velocity_scale=velocity_scale,
                differentiate_solver=differentiate_solver,
                loss_mode=loss_mode,
                solver_loss_weight=solver_loss_weight,
                loss_scale=loss_scale,
                local_loss_weight=local_loss_weight,
                warmup_intervals=warmup_intervals,
            )

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            loss_value = float(loss)
            grad_norm = float(optax.tree.norm(grads))
            if not np.isfinite(loss_value) or not np.isfinite(grad_norm):
                completed = False
                break
            check_this_stage = bool(training.get("check_grad_stages", False)) or (
                stage_index == len(stages) - 1
            )
            if (
                check_this_stage
                and stage_update == 0
                and differentiate_solver
                and bool(training.get("check_grad", True))
            ):
                checks = tuple(
                    _directional_fd(
                        loss_fn,
                        model,
                        grads,
                        jax.random.PRNGKey(model_seed + 1),
                        epsilon=epsilon,
                    )
                    for epsilon in fd_epsilons
                )
                stage_error_curve = tuple(check.relative_error for check in checks)
                stage_finite_difference_curve = tuple(
                    check.finite_difference for check in checks
                )
                stage_autodiff_curve = tuple(check.autodiff for check in checks)
                fd_stage_unrolls.append(stage.unroll)
                fd_stage_error_curves.append(stage_error_curve)
                fd_stage_finite_difference_curves.append(stage_finite_difference_curve)
                fd_stage_autodiff_curves.append(stage_autodiff_curve)
                if stage_index == len(stages) - 1:
                    fd_error_curve = stage_error_curve
                    fd_finite_difference_curve = stage_finite_difference_curve
                    fd_autodiff_curve = stage_autodiff_curve
                    fd_error = min(fd_error_curve)
            updates, opt_state = optimiser.update(
                grads,
                opt_state,
                eqx.filter(model, eqx.is_inexact_array),
            )
            model = eqx.apply_updates(model, updates)
            losses.append(loss_value)
            grad_norms.append(grad_norm)
            update_times.append(time.perf_counter() - update_started)
        stage_models.append(model)
        stage_boundaries.append(len(losses))
        if not completed:
            break
    return _TrainingResult(
        model=model,
        losses=losses,
        grad_norms=grad_norms,
        update_times=update_times,
        fd_error=fd_error,
        fd_epsilons=fd_epsilons if fd_error_curve else (),
        fd_errors=fd_error_curve,
        fd_finite_differences=fd_finite_difference_curve,
        fd_autodiff=fd_autodiff_curve,
        fd_stage_unrolls=tuple(fd_stage_unrolls),
        fd_stage_errors=tuple(fd_stage_error_curves),
        fd_stage_finite_differences=tuple(fd_stage_finite_difference_curves),
        fd_stage_autodiff=tuple(fd_stage_autodiff_curves),
        completed=completed,
        stage_models=tuple(stage_models),
        stage_unrolls=tuple(stage.unroll for stage in stages[: len(stage_models)]),
        stage_boundaries=tuple(stage_boundaries),
        stage_learning_rates=tuple(stage.lr for stage in stages[: len(stage_models)]),
    )


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
    native_state = None
    states = [np.asarray(state)]
    errors = [0.0]
    for target in reference[1:]:
        state, native_state = _solver_advance(
            t,
            ctx,
            state,
            frame_steps=frame_steps,
            native_state=native_state,
        )
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


class _ReferenceEvaluation(NamedTuple):
    first_rollout: np.ndarray | None
    errors: np.ndarray
    correlations: np.ndarray
    final_states: np.ndarray


def _evaluate_reference_set(
    t: Any,
    ctx: KernelContext,
    model: Any,
    references: np.ndarray,
    *,
    frame_steps: int,
    velocity_scale: float,
    corrected: bool,
) -> _ReferenceEvaluation:
    """Evaluate several ICs, retaining the first rollout and every final state."""
    errors: list[list[float]] = []
    correlations: list[list[float]] = []
    final_states: list[np.ndarray] = []
    first_rollout: np.ndarray | None = None
    for reference in references:
        rollout, reference_errors = _evaluate_rollout(
            t,
            ctx,
            model,
            reference,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=corrected,
        )
        errors.append(reference_errors)
        reference_correlations = []
        for predicted, target in zip(rollout, reference, strict=True):
            predicted_flat = np.asarray(predicted, dtype=np.float64).ravel()
            target_flat = np.asarray(target, dtype=np.float64).ravel()
            predicted_flat -= np.mean(predicted_flat)
            target_flat -= np.mean(target_flat)
            reference_correlations.append(
                float(
                    np.vdot(predicted_flat, target_flat).real
                    / (
                        np.linalg.norm(predicted_flat) * np.linalg.norm(target_flat)
                        + 1e-12
                    )
                )
            )
        correlations.append(reference_correlations)
        final_states.append(rollout[-1])
        if first_rollout is None:
            first_rollout = rollout
    return _ReferenceEvaluation(
        first_rollout=first_rollout,
        errors=np.asarray(errors, dtype=np.float64),
        correlations=np.asarray(correlations, dtype=np.float64),
        final_states=np.asarray(final_states),
    )


def _first_unstable(errors: list[float], threshold: float) -> int:
    """Return first bad frame, or the full completed horizon."""
    for idx, error in enumerate(errors):
        if not np.isfinite(error) or error > threshold:
            return idx
    return len(errors) - 1


def _first_below(values: list[float], threshold: float) -> int:
    """Return first frame below a quality threshold, or the completed horizon."""
    for idx, value in enumerate(values):
        if not np.isfinite(value) or value < threshold:
            return idx
    return len(values) - 1


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


def _rollout_log_gain_samples(
    baseline_errors: np.ndarray,
    corrected_errors: np.ndarray,
) -> np.ndarray:
    """Return paired rollout gains with leading model-seed and IC axes."""
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    corrected = np.asarray(corrected_errors, dtype=np.float64)
    if baseline.ndim == 2:
        baseline = baseline[None, ...]
    if baseline.shape[-2:] != corrected.shape[-2:]:
        raise ValueError("rollout ensembles must have matching IC/time axes")
    return np.mean(
        np.log((baseline[..., 1:] + 1e-12) / (corrected[..., 1:] + 1e-12)),
        axis=-1,
    )


def _ensemble_curve_stats(
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve plus independent model-seed and initial-condition spread."""
    return (
        np.mean(errors, axis=(0, 1)),
        np.std(np.mean(errors, axis=1), axis=0),
        np.std(np.mean(errors, axis=0), axis=0),
    )


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


def _stack_curves(curves: list[list[float]]) -> np.ndarray:
    """Stack the common completed prefix of repeated optimization curves."""
    if not curves:
        return np.empty((0, 0), dtype=np.float32)
    common_length = min(len(curve) for curve in curves)
    return np.stack(
        [np.asarray(curve[:common_length]) for curve in curves],
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
        "loss_samples",
        "loss_seed_std",
        "loss_stop_gradient",
        "loss_stop_gradient_samples",
        "loss_stop_gradient_seed_std",
        "loss_one_step",
        "loss_one_step_samples",
        "grad_norm",
        "grad_norm_samples",
        "grad_norm_stop_gradient",
        "grad_norm_stop_gradient_samples",
        "grad_norm_one_step",
        "grad_norm_one_step_samples",
        "update_time",
        "update_time_samples",
        "update_time_stop_gradient",
        "update_time_stop_gradient_samples",
        "update_time_one_step",
        "update_time_one_step_samples",
        "fd_epsilon",
        "fd_rel_error_samples",
        "error_corrected",
        "error_corrected_samples",
        "error_corrected_seed_std",
        "error_corrected_ic_std",
        "error_stop_gradient",
        "error_stop_gradient_samples",
        "error_stop_gradient_seed_std",
        "error_stop_gradient_ic_std",
        "error_uncorrected",
        "error_uncorrected_samples",
        "error_uncorrected_ic_std",
        "error_seen_corrected",
        "error_seen_corrected_samples",
        "error_seen_stop_gradient",
        "error_seen_stop_gradient_samples",
        "error_seen_uncorrected",
        "error_one_step",
        "error_one_step_samples",
        "error_one_step_seed_std",
        "error_one_step_ic_std",
        "error_seen_one_step",
        "error_seen_one_step_samples",
        "correlation_corrected",
        "correlation_corrected_samples",
        "correlation_stop_gradient",
        "correlation_stop_gradient_samples",
        "correlation_one_step",
        "correlation_one_step_samples",
        "correlation_uncorrected",
        "correlation_uncorrected_samples",
        "rollout_log_gain_samples",
        "stop_gradient_log_gain_samples",
        "solver_vjp_log_lift_samples",
        "one_step_rollout_log_gain_samples",
        "unrolling_log_lift_samples",
        "wig_over_one_log_lift_samples",
        "curriculum_stage_unrolls",
        "curriculum_stage_boundaries",
        "curriculum_checkpoint_error_full",
        "curriculum_checkpoint_error_full_samples",
        "curriculum_checkpoint_error_full_seed_std",
        "curriculum_checkpoint_error_nog",
        "curriculum_checkpoint_error_nog_samples",
        "curriculum_checkpoint_error_nog_seed_std",
        "curriculum_checkpoint_error_native",
        "rollout_corrected",
        "rollout_stop_gradient",
        "rollout_one_step",
        "rollout_uncorrected",
        "reference_rollout",
    ),
)
def solver_in_loop(t: Any, ctx: KernelContext) -> dict:
    """Train and evaluate one neural corrector through one candidate solver."""
    training = ctx.run.get("training", {})
    dataset_cfg = ctx.run.get("dataset", {})
    evaluation = ctx.run.get("evaluation", {})
    frame_steps = int(ctx.phys["steps"])
    reference_kind = str(dataset_cfg.get("reference_kind", "pseudo_spectral_multimode"))
    reference_audit: dict[str, Any] = {}
    if reference_kind == "solver_self_refined":
        (
            train,
            train_rollouts,
            test,
            dataset_hash,
            reference_audit,
        ) = _make_solver_self_reference_datasets(
            t,
            ctx,
            dataset=dataset_cfg,
            evaluation=evaluation,
            training=training,
        )
    else:
        (
            train,
            train_rollouts,
            test,
            dataset_hash,
            reference_audit,
        ) = _make_reference_datasets(
            physics=ctx.phys,
            dataset=dataset_cfg,
            evaluation=evaluation,
            training=training,
            domain_extent=ctx.domain_extent,
        )
    interval_time = float(ctx.phys["dt"]) * frame_steps
    if not reference_audit.get("eligible_for_corrector_training", True):
        return {
            "metrics": {
                **reference_audit,
                "completed": True,
                "n_updates": 0,
                "stop_gradient_n_updates": 0,
                "dataset_hash": dataset_hash,
                "reference_kind": reference_kind,
                "correction_intervals": int(test.shape[1] - 1),
                "rollout_final_time": interval_time * int(test.shape[1] - 1),
                "native_state_threading": _supports_native_state(t),
            },
            "snapshots": {"reference_rollout": test[0]},
            "shared": {
                "evaluation_times": np.arange(test.shape[1], dtype=np.float32)
                * interval_time
            },
        }
    velocity_scale = float(np.sqrt(np.mean(train**2)) + 1e-8)
    include_one_step = bool(training.get("include_one_step_baseline", False))
    curriculum_stages = _training_stages(training)
    checkpoint_trajectory_count = min(
        int(evaluation.get("checkpoint_ic_trajectories", 2)),
        test.shape[0],
    )
    checkpoint_rollout_frames = min(
        int(evaluation.get("checkpoint_rollout_frames", test.shape[1] - 1)),
        test.shape[1] - 1,
    )

    configured_seeds = training.get("model_seeds")
    if configured_seeds is None:
        model_seeds = (int(training.get("model_seed", 0)),)
    else:
        model_seeds = tuple(int(seed) for seed in configured_seeds)
    if not model_seeds:
        raise ValueError("training.model_seeds must contain at least one seed")

    # Evaluate the solver itself once: these trajectories do not depend on a
    # neural-model seed. A long native call and a sequence of short canonical
    # intervals expose recurrent-call closure separately from integration accuracy.
    seen_trajectory_count = min(
        int(evaluation.get("seen_ic_trajectories", test.shape[0])),
        train_rollouts.shape[0],
    )
    if seen_trajectory_count < 1:
        raise ValueError("evaluation.seen_ic_trajectories must select at least one IC")
    seen_references = train_rollouts[:seen_trajectory_count]
    semigroup_errors: list[float] = []
    for reference in test:
        one_interval, one_interval_native_state = _solver_advance(
            t,
            ctx,
            jnp.asarray(reference[0]),
            frame_steps=frame_steps,
        )
        repeated, _ = _solver_advance(
            t,
            ctx,
            one_interval,
            frame_steps=frame_steps,
            native_state=one_interval_native_state,
        )
        uninterrupted, _ = _solver_advance(
            t,
            ctx,
            jnp.asarray(reference[0]),
            frame_steps=2 * frame_steps,
        )
        semigroup_errors.append(
            relative_l2(np.asarray(repeated), np.asarray(uninterrupted))
        )
    semigroup_median = float(np.median(semigroup_errors))
    semigroup_p95 = float(np.percentile(semigroup_errors, 95.0))
    semigroup_median_tolerance = float(
        dataset_cfg.get("semigroup_median_tolerance", 0.005)
    )
    semigroup_p95_tolerance = float(dataset_cfg.get("semigroup_p95_tolerance", 0.01))
    valid_for_vjp_ranking = bool(
        semigroup_median <= semigroup_median_tolerance
        and semigroup_p95 <= semigroup_p95_tolerance
    )
    uncorrected_eval = _evaluate_reference_set(
        t,
        ctx,
        None,
        test,
        frame_steps=frame_steps,
        velocity_scale=velocity_scale,
        corrected=False,
    )
    first_uncorrected = uncorrected_eval.first_rollout
    uncorrected_errors_array = uncorrected_eval.errors
    uncorrected_correlations_array = uncorrected_eval.correlations
    recurrent_final_states = uncorrected_eval.final_states
    seen_uncorrected_eval = _evaluate_reference_set(
        t,
        ctx,
        None,
        seen_references,
        frame_steps=frame_steps,
        velocity_scale=velocity_scale,
        corrected=False,
    )
    seen_uncorrected_errors = seen_uncorrected_eval.errors
    native_final_errors: list[float] = []
    long_closure_errors: list[float] = []
    for reference, recurrent_final in zip(test, recurrent_final_states, strict=True):
        native_final, _ = _solver_advance(
            t,
            ctx,
            jnp.asarray(reference[0]),
            frame_steps=frame_steps * (reference.shape[0] - 1),
        )
        native_final_errors.append(relative_l2(np.asarray(native_final), reference[-1]))
        long_closure_errors.append(
            relative_l2(np.asarray(recurrent_final), np.asarray(native_final))
        )

    first_interval_error_p95 = float(
        np.percentile(uncorrected_errors_array[:, 1], 95.0)
    )
    long_closure_median = float(np.median(long_closure_errors))
    long_closure_p95 = float(np.percentile(long_closure_errors, 95.0))
    native_final_error_p95 = float(np.percentile(native_final_errors, 95.0))
    long_closure_tolerance = float(
        dataset_cfg.get("long_closure_tolerance", semigroup_p95_tolerance)
    )
    first_interval_error_tolerance = float(
        evaluation.get(
            "first_interval_error_tolerance",
            evaluation.get("stable_error_threshold", 1.0),
        )
    )
    native_long_error_tolerance = float(
        evaluation.get(
            "native_long_error_tolerance",
            evaluation.get("stable_error_threshold", 1.0),
        )
    )
    valid_for_vjp_ranking = bool(
        valid_for_vjp_ranking
        and long_closure_p95 <= long_closure_tolerance
        and _passes_reference_accuracy_gate(
            reference_kind,
            first_interval_error=first_interval_error_p95,
            first_interval_tolerance=first_interval_error_tolerance,
            native_long_error=native_final_error_p95,
            native_long_tolerance=native_long_error_tolerance,
        )
    )

    uncorrected_error = np.mean(uncorrected_errors_array, axis=0)
    uncorrected_error_ic_std = np.std(uncorrected_errors_array, axis=0)
    seen_uncorrected_error = np.mean(seen_uncorrected_errors, axis=0)
    training_horizon_frame = min(train.shape[1] - 1, seen_uncorrected_error.size - 1)
    loss_normalization = str(training.get("loss_normalization", "target_energy"))
    if loss_normalization == "target_energy":
        training_loss_scale = 1.0
    elif loss_normalization == "solver_baseline":
        loss_scale_floor = float(training.get("loss_scale_floor", 1e-6))
        training_loss_scale = max(
            float(np.mean(seen_uncorrected_error[1 : training_horizon_frame + 1] ** 2)),
            loss_scale_floor,
        )
    else:
        raise ValueError(f"unknown training.loss_normalization: {loss_normalization!r}")

    models: list[Any] = []
    losses_by_seed: list[list[float]] = []
    grad_norms_by_seed: list[list[float]] = []
    update_times_by_seed: list[list[float]] = []
    fd_errors: list[float | None] = []
    fd_model_seeds: list[int] = []
    fd_epsilons_by_seed: list[tuple[float, ...]] = []
    fd_error_curves_by_seed: list[tuple[float, ...]] = []
    fd_finite_difference_curves_by_seed: list[tuple[float, ...]] = []
    fd_autodiff_curves_by_seed: list[tuple[float, ...]] = []
    fd_stage_unrolls_by_seed: list[tuple[int, ...]] = []
    fd_stage_error_curves_by_seed: list[tuple[tuple[float, ...], ...]] = []
    fd_stage_finite_difference_curves_by_seed: list[tuple[tuple[float, ...], ...]] = []
    fd_stage_autodiff_curves_by_seed: list[tuple[tuple[float, ...], ...]] = []
    completed_by_seed: list[bool] = []
    training_walls: list[float] = []
    stop_gradient_losses_by_seed: list[list[float]] = []
    stop_gradient_grad_norms_by_seed: list[list[float]] = []
    stop_gradient_update_times_by_seed: list[list[float]] = []
    stop_gradient_completed_by_seed: list[bool] = []
    stop_gradient_training_walls: list[float] = []
    corrected_errors_by_seed: list[np.ndarray] = []
    stop_gradient_errors_by_seed: list[np.ndarray] = []
    corrected_correlations_by_seed: list[np.ndarray] = []
    stop_gradient_correlations_by_seed: list[np.ndarray] = []
    seen_corrected_errors_by_seed: list[np.ndarray] = []
    seen_stop_gradient_errors_by_seed: list[np.ndarray] = []
    one_step_losses_by_seed: list[list[float]] = []
    one_step_grad_norms_by_seed: list[list[float]] = []
    one_step_update_times_by_seed: list[list[float]] = []
    one_step_completed_by_seed: list[bool] = []
    one_step_training_walls: list[float] = []
    one_step_errors_by_seed: list[np.ndarray] = []
    one_step_correlations_by_seed: list[np.ndarray] = []
    seen_one_step_errors_by_seed: list[np.ndarray] = []
    checkpoint_full_errors_by_seed: list[np.ndarray] = []
    checkpoint_stop_errors_by_seed: list[np.ndarray] = []
    first_corrected: np.ndarray | None = None
    first_stop_gradient: np.ndarray | None = None
    first_one_step: np.ndarray | None = None

    for seed_idx, model_seed in enumerate(model_seeds):
        seed_training = {
            **training,
            "check_grad": bool(training.get("check_grad", True)),
        }
        started = time.perf_counter()
        full_training = _train_corrector(
            t,
            ctx,
            train,
            frame_steps=frame_steps,
            training=seed_training,
            velocity_scale=velocity_scale,
            loss_scale=training_loss_scale,
            differentiate_solver=True,
            model_seed=model_seed,
        )
        model = full_training.model
        losses = full_training.losses
        grad_norms = full_training.grad_norms
        update_times = full_training.update_times
        fd_error = full_training.fd_error
        fd_epsilons = full_training.fd_epsilons
        fd_error_curve = full_training.fd_errors
        fd_finite_difference_curve = full_training.fd_finite_differences
        fd_autodiff_curve = full_training.fd_autodiff
        completed = full_training.completed
        training_walls.append(time.perf_counter() - started)

        stop_gradient_started = time.perf_counter()
        stop_training = _train_corrector(
            t,
            ctx,
            train,
            frame_steps=frame_steps,
            training=seed_training,
            velocity_scale=velocity_scale,
            loss_scale=training_loss_scale,
            differentiate_solver=False,
            model_seed=model_seed,
        )
        stop_gradient_model = stop_training.model
        stop_gradient_losses = stop_training.losses
        stop_gradient_grad_norms = stop_training.grad_norms
        stop_gradient_update_times = stop_training.update_times
        stop_gradient_completed = stop_training.completed
        stop_gradient_training_walls.append(time.perf_counter() - stop_gradient_started)

        one_step_model = None
        one_step_training: _TrainingResult | None = None
        if include_one_step:
            one_step_updates = int(
                training.get(
                    "one_step_updates",
                    sum(stage.updates for stage in curriculum_stages),
                )
            )
            one_step_config = {
                **training,
                "unroll": 1,
                "max_updates": one_step_updates,
                "curriculum": [
                    {
                        "unroll": 1,
                        "updates": one_step_updates,
                        "lr": float(training.get("lr", 1e-4)),
                    }
                ],
                "loss_mode": "mean",
                "solver_loss_weight": 0.0,
                "local_loss_weight": 1.0,
                "warmup_intervals": 0,
                "check_grad": False,
            }
            one_step_started = time.perf_counter()
            one_step_training = _train_corrector(
                t,
                ctx,
                train,
                frame_steps=frame_steps,
                training=one_step_config,
                velocity_scale=velocity_scale,
                loss_scale=training_loss_scale,
                # With a true reference state at the beginning of every
                # one-step sample, the solver output does not depend on model
                # parameters. Stopping it is therefore mathematically exact.
                differentiate_solver=False,
                model_seed=model_seed,
            )
            one_step_training_walls.append(time.perf_counter() - one_step_started)
            one_step_model = one_step_training.model

        corrected_eval = _evaluate_reference_set(
            t,
            ctx,
            model,
            test,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=True,
        )
        corrected_rollout = corrected_eval.first_rollout
        seed_corrected_errors = corrected_eval.errors
        stop_gradient_eval = _evaluate_reference_set(
            t,
            ctx,
            stop_gradient_model,
            test,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=True,
        )
        stop_gradient_rollout = stop_gradient_eval.first_rollout
        seed_stop_gradient_errors = stop_gradient_eval.errors
        seen_corrected_eval = _evaluate_reference_set(
            t,
            ctx,
            model,
            seen_references,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=True,
        )
        seen_seed_corrected_errors = seen_corrected_eval.errors
        seen_stop_gradient_eval = _evaluate_reference_set(
            t,
            ctx,
            stop_gradient_model,
            seen_references,
            frame_steps=frame_steps,
            velocity_scale=velocity_scale,
            corrected=True,
        )
        seen_seed_stop_gradient_errors = seen_stop_gradient_eval.errors
        if one_step_model is not None and one_step_training is not None:
            one_step_eval = _evaluate_reference_set(
                t,
                ctx,
                one_step_model,
                test,
                frame_steps=frame_steps,
                velocity_scale=velocity_scale,
                corrected=True,
            )
            seen_one_step_eval = _evaluate_reference_set(
                t,
                ctx,
                one_step_model,
                seen_references,
                frame_steps=frame_steps,
                velocity_scale=velocity_scale,
                corrected=True,
            )
            one_step_losses_by_seed.append(one_step_training.losses)
            one_step_grad_norms_by_seed.append(one_step_training.grad_norms)
            one_step_update_times_by_seed.append(one_step_training.update_times)
            one_step_completed_by_seed.append(one_step_training.completed)
            one_step_errors_by_seed.append(one_step_eval.errors)
            one_step_correlations_by_seed.append(one_step_eval.correlations)
            seen_one_step_errors_by_seed.append(seen_one_step_eval.errors)
            if seed_idx == 0:
                first_one_step = one_step_eval.first_rollout

        if len(full_training.stage_models) > 1:
            checkpoint_references = test[
                :checkpoint_trajectory_count, : checkpoint_rollout_frames + 1
            ]
            full_checkpoint_errors: list[np.ndarray] = []
            stop_checkpoint_errors: list[np.ndarray] = []
            for full_stage_model, stop_stage_model in zip(
                full_training.stage_models,
                stop_training.stage_models,
                strict=True,
            ):
                full_stage_eval = _evaluate_reference_set(
                    t,
                    ctx,
                    full_stage_model,
                    checkpoint_references,
                    frame_steps=frame_steps,
                    velocity_scale=velocity_scale,
                    corrected=True,
                )
                stop_stage_eval = _evaluate_reference_set(
                    t,
                    ctx,
                    stop_stage_model,
                    checkpoint_references,
                    frame_steps=frame_steps,
                    velocity_scale=velocity_scale,
                    corrected=True,
                )
                full_checkpoint_errors.append(np.mean(full_stage_eval.errors, axis=0))
                stop_checkpoint_errors.append(np.mean(stop_stage_eval.errors, axis=0))
            checkpoint_full_errors_by_seed.append(np.stack(full_checkpoint_errors))
            checkpoint_stop_errors_by_seed.append(np.stack(stop_checkpoint_errors))
        if seed_idx == 0:
            first_corrected = corrected_rollout
            first_stop_gradient = stop_gradient_rollout

        models.append(model)
        losses_by_seed.append(losses)
        grad_norms_by_seed.append(grad_norms)
        update_times_by_seed.append(update_times)
        fd_errors.append(fd_error)
        if fd_error_curve:
            fd_model_seeds.append(model_seed)
            fd_epsilons_by_seed.append(fd_epsilons)
            fd_error_curves_by_seed.append(fd_error_curve)
            fd_finite_difference_curves_by_seed.append(fd_finite_difference_curve)
            fd_autodiff_curves_by_seed.append(fd_autodiff_curve)
            fd_stage_unrolls_by_seed.append(full_training.fd_stage_unrolls)
            fd_stage_error_curves_by_seed.append(full_training.fd_stage_errors)
            fd_stage_finite_difference_curves_by_seed.append(
                full_training.fd_stage_finite_differences
            )
            fd_stage_autodiff_curves_by_seed.append(full_training.fd_stage_autodiff)
        completed_by_seed.append(completed)
        stop_gradient_losses_by_seed.append(stop_gradient_losses)
        stop_gradient_grad_norms_by_seed.append(stop_gradient_grad_norms)
        stop_gradient_update_times_by_seed.append(stop_gradient_update_times)
        stop_gradient_completed_by_seed.append(stop_gradient_completed)
        corrected_errors_by_seed.append(seed_corrected_errors)
        stop_gradient_errors_by_seed.append(seed_stop_gradient_errors)
        corrected_correlations_by_seed.append(corrected_eval.correlations)
        stop_gradient_correlations_by_seed.append(stop_gradient_eval.correlations)
        seen_corrected_errors_by_seed.append(seen_seed_corrected_errors)
        seen_stop_gradient_errors_by_seed.append(seen_seed_stop_gradient_errors)

    corrected_errors_array = np.stack(corrected_errors_by_seed)
    stop_gradient_errors_array = np.stack(stop_gradient_errors_by_seed)
    corrected_correlation = np.mean(
        np.stack(corrected_correlations_by_seed),
        axis=(0, 1),
    )
    stop_gradient_correlation = np.mean(
        np.stack(stop_gradient_correlations_by_seed),
        axis=(0, 1),
    )
    uncorrected_correlation = np.mean(uncorrected_correlations_array, axis=0)
    seen_corrected_errors_array = np.stack(seen_corrected_errors_by_seed)
    seen_stop_gradient_errors_array = np.stack(seen_stop_gradient_errors_by_seed)
    corrected_error, corrected_error_seed_std, corrected_error_ic_std = (
        _ensemble_curve_stats(corrected_errors_array)
    )
    (
        stop_gradient_error,
        stop_gradient_error_seed_std,
        stop_gradient_error_ic_std,
    ) = _ensemble_curve_stats(stop_gradient_errors_array)
    (
        seen_corrected_error,
        _seen_corrected_error_seed_std,
        seen_corrected_error_ic_std,
    ) = _ensemble_curve_stats(seen_corrected_errors_array)
    (
        seen_stop_gradient_error,
        _seen_stop_gradient_error_seed_std,
        _seen_stop_gradient_error_ic_std,
    ) = _ensemble_curve_stats(seen_stop_gradient_errors_array)
    one_step_error: np.ndarray | None = None
    one_step_error_seed_std: np.ndarray | None = None
    one_step_error_ic_std: np.ndarray | None = None
    seen_one_step_error: np.ndarray | None = None
    one_step_errors_array: np.ndarray | None = None
    one_step_correlation: np.ndarray | None = None
    if include_one_step:
        one_step_errors_array = np.stack(one_step_errors_by_seed)
        (
            one_step_error,
            one_step_error_seed_std,
            one_step_error_ic_std,
        ) = _ensemble_curve_stats(one_step_errors_array)
        seen_one_step_error = _ensemble_curve_stats(
            np.stack(seen_one_step_errors_by_seed)
        )[0]
        one_step_correlation = np.mean(
            np.stack(one_step_correlations_by_seed),
            axis=(0, 1),
        )
    losses = _mean_curve(losses_by_seed)
    stop_gradient_losses = _mean_curve(stop_gradient_losses_by_seed)
    one_step_losses = _mean_curve(one_step_losses_by_seed)
    grad_norms = _mean_curve(grad_norms_by_seed)
    stop_gradient_grad_norms = _mean_curve(stop_gradient_grad_norms_by_seed)
    one_step_grad_norms = _mean_curve(one_step_grad_norms_by_seed)
    update_times = _mean_curve(update_times_by_seed)
    stop_gradient_update_times = _mean_curve(stop_gradient_update_times_by_seed)
    one_step_update_times = _mean_curve(one_step_update_times_by_seed)
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
    one_step_training_wall = float(np.sum(one_step_training_walls))
    total_updates = sum(len(curve) for curve in losses_by_seed)
    stop_gradient_total_updates = sum(
        len(curve) for curve in stop_gradient_losses_by_seed
    )
    one_step_total_updates = sum(len(curve) for curve in one_step_losses_by_seed)
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
    one_step_steady_update_times = [
        value
        for seed_times in one_step_update_times_by_seed
        for value in (seed_times[1:] if len(seed_times) > 1 else seed_times)
    ]

    final_corrected = float(corrected_error[-1])
    final_stop_gradient = float(stop_gradient_error[-1])
    final_uncorrected = float(uncorrected_error[-1])
    native_final_error = float(np.mean(native_final_errors))
    mean_corrected = float(np.mean(corrected_error[1:]))
    mean_stop_gradient = float(np.mean(stop_gradient_error[1:]))
    mean_uncorrected = float(np.mean(uncorrected_error[1:]))
    rollout_log_gain_samples = _rollout_log_gain_samples(
        uncorrected_errors_array,
        corrected_errors_array,
    )
    stop_gradient_log_gain_samples = _rollout_log_gain_samples(
        uncorrected_errors_array,
        stop_gradient_errors_array,
    )
    solver_vjp_log_lift_samples = _rollout_log_gain_samples(
        stop_gradient_errors_array,
        corrected_errors_array,
    )
    rollout_log_gains = np.mean(rollout_log_gain_samples, axis=1)
    stop_gradient_log_gains = np.mean(
        stop_gradient_log_gain_samples,
        axis=1,
    )
    solver_vjp_log_lifts = np.mean(
        solver_vjp_log_lift_samples,
        axis=1,
    )
    rollout_log_gain = float(np.mean(rollout_log_gain_samples))
    stop_gradient_log_gain = float(np.mean(stop_gradient_log_gain_samples))
    solver_vjp_log_lift = float(np.mean(solver_vjp_log_lift_samples))
    one_step_rollout_log_gain_samples: np.ndarray | None = None
    unrolling_log_lift_samples: np.ndarray | None = None
    wig_over_one_log_lift_samples: np.ndarray | None = None
    if one_step_errors_array is not None:
        one_step_rollout_log_gain_samples = _rollout_log_gain_samples(
            uncorrected_errors_array,
            one_step_errors_array,
        )
        unrolling_log_lift_samples = _rollout_log_gain_samples(
            one_step_errors_array,
            stop_gradient_errors_array,
        )
        wig_over_one_log_lift_samples = _rollout_log_gain_samples(
            one_step_errors_array,
            corrected_errors_array,
        )
    threshold = float(evaluation.get("stable_error_threshold", 1.0))
    correlation_threshold = float(evaluation.get("correlation_threshold", 0.95))
    matched_horizon_frame = min(train.shape[1] - 1, test.shape[1] - 1)
    long_horizon_frame = test.shape[1] - 1
    seen_matched_error = float(seen_corrected_error[matched_horizon_frame])
    heldout_matched_error = float(corrected_error[matched_horizon_frame])
    seen_long_error = float(seen_corrected_error[long_horizon_frame])
    heldout_long_error = float(corrected_error[long_horizon_frame])
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
    one_step_median_update_time = (
        float(np.median(one_step_steady_update_times))
        if one_step_steady_update_times
        else None
    )
    stage_boundaries = np.cumsum(
        [stage.updates for stage in curriculum_stages],
        dtype=np.int64,
    )
    warmup_intervals = int(training.get("warmup_intervals", 0))
    warmup_solver_intervals_per_seed = warmup_intervals * sum(
        stage.updates for stage in curriculum_stages
    )
    recurrent_solver_intervals_per_seed = int(
        sum(
            (warmup_intervals + stage.unroll) * stage.updates
            for stage in curriculum_stages
        )
    )
    checkpoint_full_error: np.ndarray | None = None
    checkpoint_stop_error: np.ndarray | None = None
    checkpoint_full_seed_std: np.ndarray | None = None
    checkpoint_stop_seed_std: np.ndarray | None = None
    checkpoint_native_error: np.ndarray | None = None
    if checkpoint_full_errors_by_seed:
        full_checkpoints = np.stack(checkpoint_full_errors_by_seed)
        stop_checkpoints = np.stack(checkpoint_stop_errors_by_seed)
        checkpoint_full_error = np.mean(full_checkpoints, axis=0)
        checkpoint_stop_error = np.mean(stop_checkpoints, axis=0)
        checkpoint_full_seed_std = np.std(full_checkpoints, axis=0)
        checkpoint_stop_seed_std = np.std(stop_checkpoints, axis=0)
        checkpoint_native_error = np.mean(
            uncorrected_errors_array[
                :checkpoint_trajectory_count, : checkpoint_rollout_frames + 1
            ],
            axis=0,
        )
    fd_epsilon_array = np.asarray([], dtype=np.float32)
    fd_error_samples = np.empty((0, 0), dtype=np.float32)
    fd_finite_difference_samples = np.empty((0, 0), dtype=np.float32)
    fd_autodiff_samples = np.empty((0, 0), dtype=np.float32)
    fd_stage_unroll_array = np.asarray([], dtype=np.int32)
    fd_stage_error_samples = np.empty((0, 0, 0), dtype=np.float32)
    fd_stage_finite_difference_samples = np.empty((0, 0, 0), dtype=np.float32)
    fd_stage_autodiff_samples = np.empty((0, 0, 0), dtype=np.float32)
    fd_check_summary: list[dict[str, Any]] = []
    fd_horizon_summary: list[dict[str, Any]] = []
    if fd_error_curves_by_seed:
        if len(set(fd_epsilons_by_seed)) != 1:
            raise RuntimeError("FD epsilon grids differ across model seeds")
        if len(set(fd_stage_unrolls_by_seed)) != 1:
            raise RuntimeError("FD horizon grids differ across model seeds")
        fd_epsilon_array = np.asarray(fd_epsilons_by_seed[0], dtype=np.float32)
        fd_error_samples = np.asarray(fd_error_curves_by_seed, dtype=np.float32)
        fd_finite_difference_samples = np.asarray(
            fd_finite_difference_curves_by_seed,
            dtype=np.float32,
        )
        fd_autodiff_samples = np.asarray(
            fd_autodiff_curves_by_seed,
            dtype=np.float32,
        )
        fd_stage_unroll_array = np.asarray(
            fd_stage_unrolls_by_seed[0],
            dtype=np.int32,
        )
        fd_stage_error_samples = np.asarray(
            fd_stage_error_curves_by_seed,
            dtype=np.float32,
        )
        fd_stage_finite_difference_samples = np.asarray(
            fd_stage_finite_difference_curves_by_seed,
            dtype=np.float32,
        )
        fd_stage_autodiff_samples = np.asarray(
            fd_stage_autodiff_curves_by_seed,
            dtype=np.float32,
        )
        fd_check_summary = [
            {
                "model_seed": model_seed,
                "best_epsilon": float(
                    fd_epsilon_array[best_index := int(np.argmin(curve))]
                ),
                "best_relative_error": float(curve[best_index]),
                "max_relative_error": float(np.max(curve)),
                "finite_difference_at_best_epsilon": float(
                    finite_differences[best_index]
                ),
                "autodiff_directional_derivative": float(autodiff_values[best_index]),
                "absolute_directional_error": float(
                    abs(finite_differences[best_index] - autodiff_values[best_index])
                ),
            }
            for model_seed, curve, finite_differences, autodiff_values in zip(
                fd_model_seeds,
                fd_error_samples,
                fd_finite_difference_samples,
                fd_autodiff_samples,
                strict=True,
            )
        ]
        fd_horizon_summary = [
            {
                "model_seed": model_seed,
                "unroll": int(unroll),
                "best_epsilon": float(
                    fd_epsilon_array[best_index := int(np.argmin(curve))]
                ),
                "best_relative_error": float(curve[best_index]),
                "finite_difference_at_best_epsilon": float(
                    finite_differences[best_index]
                ),
                "autodiff_directional_derivative": float(autodiff_values[best_index]),
            }
            for (
                model_seed,
                seed_curves,
                seed_finite_differences,
                seed_autodiff_values,
            ) in zip(
                fd_model_seeds,
                fd_stage_error_samples,
                fd_stage_finite_difference_samples,
                fd_stage_autodiff_samples,
                strict=True,
            )
            for unroll, curve, finite_differences, autodiff_values in zip(
                fd_stage_unroll_array,
                seed_curves,
                seed_finite_differences,
                seed_autodiff_values,
                strict=True,
            )
        ]

    metrics = {
        **reference_audit,
        "eligible_for_corrector_training": True,
        "valid_for_vjp_ranking": valid_for_vjp_ranking,
        "reference_accuracy_gate_applied": reference_kind != "solver_self_refined",
        "semigroup_error_median": semigroup_median,
        "semigroup_error_p95": semigroup_p95,
        "semigroup_error_samples": semigroup_errors,
        "semigroup_median_tolerance": semigroup_median_tolerance,
        "semigroup_p95_tolerance": semigroup_p95_tolerance,
        "first_interval_error_p95": first_interval_error_p95,
        "first_interval_error_tolerance": first_interval_error_tolerance,
        "long_closure_error_median": long_closure_median,
        "long_closure_error_p95": long_closure_p95,
        "long_closure_error_samples": long_closure_errors,
        "long_closure_tolerance": long_closure_tolerance,
        "native_final_rollout_error_p95": native_final_error_p95,
        "native_long_error_tolerance": native_long_error_tolerance,
        "final_rollout_error": final_corrected,
        "final_rollout_error_seed_std": float(corrected_error_seed_std[-1]),
        "final_rollout_error_ic_std": float(corrected_error_ic_std[-1]),
        "stop_gradient_final_rollout_error": final_stop_gradient,
        "stop_gradient_final_rollout_error_seed_std": float(
            stop_gradient_error_seed_std[-1]
        ),
        "stop_gradient_final_rollout_error_ic_std": float(
            stop_gradient_error_ic_std[-1]
        ),
        "uncorrected_rollout_error": final_uncorrected,
        "uncorrected_rollout_error_ic_std": float(uncorrected_error_ic_std[-1]),
        "first_interval_rollout_error": float(uncorrected_error[1]),
        "first_interval_rollout_error_ic_std": float(uncorrected_error_ic_std[1]),
        "native_final_rollout_error": native_final_error,
        "recurrent_to_native_error_ratio": final_uncorrected
        / (native_final_error + 1e-12),
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
        "rollout_log_gain_ic_std": _std(
            np.mean(rollout_log_gain_samples, axis=0).tolist()
        ),
        "geometric_error_reduction": float(np.exp(rollout_log_gain)),
        "stop_gradient_rollout_log_gain": stop_gradient_log_gain,
        "stop_gradient_rollout_log_gain_seed_std": _std(stop_gradient_log_gains),
        "stop_gradient_rollout_log_gain_ic_std": _std(
            np.mean(stop_gradient_log_gain_samples, axis=0).tolist()
        ),
        "stop_gradient_geometric_error_reduction": float(
            np.exp(stop_gradient_log_gain)
        ),
        "solver_vjp_log_lift": solver_vjp_log_lift,
        "solver_vjp_log_lift_seed_std": _std(solver_vjp_log_lifts),
        "solver_vjp_log_lift_ic_std": _std(
            np.mean(solver_vjp_log_lift_samples, axis=0).tolist()
        ),
        "solver_vjp_geometric_lift": float(np.exp(solver_vjp_log_lift)),
        "seen_ic_matched_horizon_error": seen_matched_error,
        "seen_ic_matched_horizon_error_ic_std": float(
            seen_corrected_error_ic_std[matched_horizon_frame]
        ),
        "heldout_ic_matched_horizon_error": heldout_matched_error,
        "heldout_ic_matched_horizon_error_ic_std": float(
            corrected_error_ic_std[matched_horizon_frame]
        ),
        "seen_ic_long_horizon_error": seen_long_error,
        "seen_ic_long_horizon_error_ic_std": float(
            seen_corrected_error_ic_std[long_horizon_frame]
        ),
        "heldout_ic_long_horizon_error": heldout_long_error,
        "heldout_ic_long_horizon_error_ic_std": float(
            corrected_error_ic_std[long_horizon_frame]
        ),
        "ic_generalization_ratio_at_matched_horizon": heldout_matched_error
        / (seen_matched_error + 1e-12),
        "seen_ic_temporal_extrapolation_ratio": seen_long_error
        / (seen_matched_error + 1e-12),
        "heldout_ic_temporal_extrapolation_ratio": heldout_long_error
        / (heldout_matched_error + 1e-12),
        "stop_gradient_seen_ic_matched_horizon_error": float(
            seen_stop_gradient_error[matched_horizon_frame]
        ),
        "stop_gradient_heldout_ic_matched_horizon_error": float(
            stop_gradient_error[matched_horizon_frame]
        ),
        "uncorrected_seen_ic_matched_horizon_error": float(
            seen_uncorrected_error[matched_horizon_frame]
        ),
        "uncorrected_heldout_ic_matched_horizon_error": float(
            uncorrected_error[matched_horizon_frame]
        ),
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
        "final_rollout_correlation": float(corrected_correlation[-1]),
        "stop_gradient_final_rollout_correlation": float(stop_gradient_correlation[-1]),
        "uncorrected_final_rollout_correlation": float(uncorrected_correlation[-1]),
        "correlation_threshold": correlation_threshold,
        "correlation_horizon": _first_below(
            corrected_correlation.tolist(),
            correlation_threshold,
        )
        * interval_time,
        "stop_gradient_correlation_horizon": _first_below(
            stop_gradient_correlation.tolist(),
            correlation_threshold,
        )
        * interval_time,
        "uncorrected_correlation_horizon": _first_below(
            uncorrected_correlation.tolist(),
            correlation_threshold,
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
        "end_to_end_fd_rel_error_max": max(
            (value for value in fd_errors if value is not None),
            default=None,
        ),
        "fd_check_summary": fd_check_summary,
        "fd_horizon_summary": fd_horizon_summary,
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
        "n_train_trajectories": int(train.shape[0]),
        "n_test_trajectories": int(test.shape[0]),
        "n_seen_ic_evaluation_trajectories": seen_trajectory_count,
        "training_frames": int(train.shape[1] - 1),
        "training_unroll": max(stage.unroll for stage in curriculum_stages),
        "training_curriculum": [
            {
                "unroll": stage.unroll,
                "updates": stage.updates,
                "lr": stage.lr,
            }
            for stage in curriculum_stages
        ],
        "training_stage_boundaries": stage_boundaries.tolist(),
        "training_solver_intervals_per_seed": recurrent_solver_intervals_per_seed,
        "training_solver_intervals_total": recurrent_solver_intervals_per_seed
        * len(model_seeds),
        "training_native_steps_per_seed": (
            recurrent_solver_intervals_per_seed * frame_steps
        ),
        "training_native_steps_total": (
            recurrent_solver_intervals_per_seed * frame_steps * len(model_seeds)
        ),
        "training_warmup_solver_intervals_per_seed": (warmup_solver_intervals_per_seed),
        "training_warmup_solver_intervals_total": (
            warmup_solver_intervals_per_seed * len(model_seeds)
        ),
        "training_loss_mode": str(training.get("loss_mode", "mean")),
        "training_solver_loss_weight": float(training.get("solver_loss_weight", 0.1)),
        "training_local_loss_weight": float(training.get("local_loss_weight", 1.0)),
        "training_warmup_intervals": int(training.get("warmup_intervals", 0)),
        "training_warmup_native_steps": warmup_intervals * frame_steps,
        "training_differentiated_native_horizon": (
            max(stage.unroll for stage in curriculum_stages) * frame_steps
        ),
        "training_loss_normalization": loss_normalization,
        "training_loss_scale": training_loss_scale,
        "matched_horizon_time": matched_horizon_frame * interval_time,
        "visualization_model_seed": model_seeds[0],
        "completed": all(completed_by_seed)
        and all(stop_gradient_completed_by_seed)
        and (not include_one_step or all(one_step_completed_by_seed)),
        "dataset_hash": dataset_hash,
        "reference_kind": reference_kind,
        "correction_intervals": int(test.shape[1] - 1),
        "native_steps": frame_steps * int(test.shape[1] - 1),
        "rollout_final_time": interval_time * int(test.shape[1] - 1),
        "native_state_threading": _supports_native_state(t),
    }
    if (
        one_step_error is not None
        and one_step_error_seed_std is not None
        and one_step_error_ic_std is not None
        and one_step_rollout_log_gain_samples is not None
        and unrolling_log_lift_samples is not None
        and wig_over_one_log_lift_samples is not None
    ):
        one_step_mean_error = float(np.mean(one_step_error[1:]))
        one_step_log_gain = float(np.mean(one_step_rollout_log_gain_samples))
        unrolling_log_lift = float(np.mean(unrolling_log_lift_samples))
        wig_over_one_log_lift = float(np.mean(wig_over_one_log_lift_samples))
        metrics.update(
            {
                "one_step_final_rollout_error": float(one_step_error[-1]),
                "one_step_final_rollout_error_seed_std": float(
                    one_step_error_seed_std[-1]
                ),
                "one_step_final_rollout_error_ic_std": float(one_step_error_ic_std[-1]),
                "one_step_mean_rollout_error": one_step_mean_error,
                "one_step_geometric_error_reduction": float(np.exp(one_step_log_gain)),
                "one_step_rollout_log_gain_seed_std": _std(
                    np.mean(one_step_rollout_log_gain_samples, axis=1).tolist()
                ),
                "one_step_rollout_log_gain_ic_std": _std(
                    np.mean(one_step_rollout_log_gain_samples, axis=0).tolist()
                ),
                "unrolling_geometric_lift_nog_over_one": float(
                    np.exp(unrolling_log_lift)
                ),
                "unrolling_log_lift_seed_std": _std(
                    np.mean(unrolling_log_lift_samples, axis=1).tolist()
                ),
                "unrolling_log_lift_ic_std": _std(
                    np.mean(unrolling_log_lift_samples, axis=0).tolist()
                ),
                "wig_geometric_lift_over_one": float(np.exp(wig_over_one_log_lift)),
                "wig_over_one_log_lift_seed_std": _std(
                    np.mean(wig_over_one_log_lift_samples, axis=1).tolist()
                ),
                "wig_over_one_log_lift_ic_std": _std(
                    np.mean(wig_over_one_log_lift_samples, axis=0).tolist()
                ),
                "one_step_seen_ic_matched_horizon_error": float(
                    seen_one_step_error[matched_horizon_frame]
                )
                if seen_one_step_error is not None
                else None,
                "one_step_heldout_ic_matched_horizon_error": float(
                    one_step_error[matched_horizon_frame]
                ),
                "one_step_stable_horizon": _first_unstable(
                    one_step_error.tolist(),
                    threshold,
                )
                * interval_time,
                "one_step_final_rollout_correlation": float(one_step_correlation[-1])
                if one_step_correlation is not None
                else None,
                "one_step_correlation_horizon": _first_below(
                    one_step_correlation.tolist(),
                    correlation_threshold,
                )
                * interval_time
                if one_step_correlation is not None
                else None,
                "one_step_initial_train_loss": float(one_step_losses[0])
                if one_step_losses.size
                else None,
                "one_step_final_train_loss": float(one_step_losses[-1])
                if one_step_losses.size
                else None,
                "one_step_best_train_loss": float(np.min(one_step_losses))
                if one_step_losses.size
                else None,
                "one_step_n_updates": len(one_step_losses),
                "one_step_total_optimizer_updates": one_step_total_updates,
                "one_step_training_wall_time_s": one_step_training_wall,
                "one_step_seconds_per_update": one_step_training_wall
                / max(one_step_total_updates, 1),
                "one_step_median_update_time_s": one_step_median_update_time,
                "one_step_final_grad_norm": float(one_step_grad_norms[-1])
                if one_step_grad_norms.size
                else None,
                "one_step_training_solver_intervals_per_seed": len(one_step_losses),
                "one_step_training_solver_intervals_total": one_step_total_updates,
                "one_step_training_native_steps_per_seed": (
                    len(one_step_losses) * frame_steps
                ),
                "one_step_training_native_steps_total": (
                    one_step_total_updates * frame_steps
                ),
                "one_step_final_divergence_rms": divergence_rms(
                    first_one_step[-1],
                    ctx.domain_extent,
                )
                if first_one_step is not None
                else None,
                "one_step_final_energy_ratio_to_reference": kinetic_energy(
                    first_one_step[-1]
                )
                / (kinetic_energy(test[0, -1]) + 1e-12)
                if first_one_step is not None
                else None,
            }
        )
    if (
        checkpoint_full_error is not None
        and checkpoint_stop_error is not None
        and checkpoint_native_error is not None
    ):
        metrics["curriculum_checkpoint_summary"] = [
            {
                "unroll": stage.unroll,
                "updates_cumulative": int(stage_boundaries[index]),
                "full_geometric_error_reduction": float(
                    np.exp(
                        _rollout_log_gain(
                            checkpoint_native_error,
                            checkpoint_full_error[index],
                        )
                    )
                ),
                "nog_geometric_error_reduction": float(
                    np.exp(
                        _rollout_log_gain(
                            checkpoint_native_error,
                            checkpoint_stop_error[index],
                        )
                    )
                ),
                "wig_geometric_lift_over_nog": float(
                    np.exp(
                        _rollout_log_gain(
                            checkpoint_stop_error[index],
                            checkpoint_full_error[index],
                        )
                    )
                ),
            }
            for index, stage in enumerate(curriculum_stages)
        ]
    snapshots = {
        "loss": losses,
        "loss_samples": _stack_curves(losses_by_seed),
        "loss_seed_std": np.asarray(loss_seed_std, dtype=np.float32),
        "loss_stop_gradient": stop_gradient_losses,
        "loss_stop_gradient_samples": _stack_curves(stop_gradient_losses_by_seed),
        "loss_stop_gradient_seed_std": np.asarray(
            stop_gradient_loss_seed_std,
            dtype=np.float32,
        ),
        "grad_norm": grad_norms,
        "grad_norm_samples": _stack_curves(grad_norms_by_seed),
        "grad_norm_stop_gradient": np.asarray(
            stop_gradient_grad_norms,
            dtype=np.float32,
        ),
        "grad_norm_stop_gradient_samples": _stack_curves(
            stop_gradient_grad_norms_by_seed
        ),
        "update_time": update_times,
        "update_time_samples": _stack_curves(update_times_by_seed),
        "update_time_stop_gradient": np.asarray(
            stop_gradient_update_times,
            dtype=np.float32,
        ),
        "update_time_stop_gradient_samples": _stack_curves(
            stop_gradient_update_times_by_seed
        ),
        "fd_epsilon": fd_epsilon_array,
        "fd_rel_error_samples": fd_error_samples,
        "fd_directional_finite_difference_samples": fd_finite_difference_samples,
        "fd_directional_autodiff_samples": fd_autodiff_samples,
        "fd_stage_unroll": fd_stage_unroll_array,
        "fd_stage_rel_error_samples": fd_stage_error_samples,
        "fd_stage_finite_difference_samples": (fd_stage_finite_difference_samples),
        "fd_stage_autodiff_samples": fd_stage_autodiff_samples,
        "error_corrected": np.asarray(corrected_error, dtype=np.float32),
        "error_corrected_samples": corrected_errors_array.astype(np.float32),
        "error_corrected_seed_std": np.asarray(
            corrected_error_seed_std,
            dtype=np.float32,
        ),
        "error_corrected_ic_std": np.asarray(
            corrected_error_ic_std,
            dtype=np.float32,
        ),
        "error_stop_gradient": np.asarray(
            stop_gradient_error,
            dtype=np.float32,
        ),
        "error_stop_gradient_samples": stop_gradient_errors_array.astype(np.float32),
        "error_stop_gradient_seed_std": np.asarray(
            stop_gradient_error_seed_std,
            dtype=np.float32,
        ),
        "error_stop_gradient_ic_std": np.asarray(
            stop_gradient_error_ic_std,
            dtype=np.float32,
        ),
        "error_uncorrected": np.asarray(uncorrected_error, dtype=np.float32),
        "error_uncorrected_samples": uncorrected_errors_array.astype(np.float32),
        "error_uncorrected_ic_std": np.asarray(
            uncorrected_error_ic_std,
            dtype=np.float32,
        ),
        "error_seen_corrected": np.asarray(
            seen_corrected_error,
            dtype=np.float32,
        ),
        "error_seen_corrected_samples": seen_corrected_errors_array.astype(np.float32),
        "error_seen_stop_gradient": np.asarray(
            seen_stop_gradient_error,
            dtype=np.float32,
        ),
        "error_seen_stop_gradient_samples": (
            seen_stop_gradient_errors_array.astype(np.float32)
        ),
        "error_seen_uncorrected": np.asarray(
            seen_uncorrected_error,
            dtype=np.float32,
        ),
        "correlation_corrected": np.asarray(
            corrected_correlation,
            dtype=np.float32,
        ),
        "correlation_corrected_samples": np.stack(
            corrected_correlations_by_seed
        ).astype(np.float32),
        "correlation_stop_gradient": np.asarray(
            stop_gradient_correlation,
            dtype=np.float32,
        ),
        "correlation_stop_gradient_samples": np.stack(
            stop_gradient_correlations_by_seed
        ).astype(np.float32),
        "correlation_uncorrected": np.asarray(
            uncorrected_correlation,
            dtype=np.float32,
        ),
        "correlation_uncorrected_samples": (
            uncorrected_correlations_array.astype(np.float32)
        ),
        "rollout_log_gain_samples": rollout_log_gain_samples.astype(np.float32),
        "stop_gradient_log_gain_samples": stop_gradient_log_gain_samples.astype(
            np.float32
        ),
        "solver_vjp_log_lift_samples": solver_vjp_log_lift_samples.astype(np.float32),
    }
    if (
        one_step_error is not None
        and one_step_error_seed_std is not None
        and one_step_error_ic_std is not None
        and seen_one_step_error is not None
        and one_step_rollout_log_gain_samples is not None
        and unrolling_log_lift_samples is not None
        and wig_over_one_log_lift_samples is not None
    ):
        snapshots.update(
            {
                "loss_one_step": one_step_losses,
                "loss_one_step_samples": _stack_curves(one_step_losses_by_seed),
                "grad_norm_one_step": one_step_grad_norms,
                "grad_norm_one_step_samples": _stack_curves(
                    one_step_grad_norms_by_seed
                ),
                "update_time_one_step": one_step_update_times,
                "update_time_one_step_samples": _stack_curves(
                    one_step_update_times_by_seed
                ),
                "error_one_step": np.asarray(one_step_error, dtype=np.float32),
                "error_one_step_samples": one_step_errors_array.astype(np.float32),
                "error_one_step_seed_std": np.asarray(
                    one_step_error_seed_std,
                    dtype=np.float32,
                ),
                "error_one_step_ic_std": np.asarray(
                    one_step_error_ic_std,
                    dtype=np.float32,
                ),
                "error_seen_one_step": np.asarray(
                    seen_one_step_error,
                    dtype=np.float32,
                ),
                "error_seen_one_step_samples": np.stack(
                    seen_one_step_errors_by_seed
                ).astype(np.float32),
                "correlation_one_step": np.asarray(
                    one_step_correlation,
                    dtype=np.float32,
                )
                if one_step_correlation is not None
                else np.asarray([], dtype=np.float32),
                "correlation_one_step_samples": np.stack(
                    one_step_correlations_by_seed
                ).astype(np.float32),
                "one_step_rollout_log_gain_samples": (
                    one_step_rollout_log_gain_samples.astype(np.float32)
                ),
                "unrolling_log_lift_samples": unrolling_log_lift_samples.astype(
                    np.float32
                ),
                "wig_over_one_log_lift_samples": (
                    wig_over_one_log_lift_samples.astype(np.float32)
                ),
            }
        )
    if (
        checkpoint_full_error is not None
        and checkpoint_stop_error is not None
        and checkpoint_full_seed_std is not None
        and checkpoint_stop_seed_std is not None
        and checkpoint_native_error is not None
    ):
        snapshots.update(
            {
                "curriculum_stage_unrolls": np.asarray(
                    [stage.unroll for stage in curriculum_stages],
                    dtype=np.int32,
                ),
                "curriculum_stage_boundaries": stage_boundaries.astype(np.int32),
                "curriculum_checkpoint_error_full": checkpoint_full_error.astype(
                    np.float32
                ),
                "curriculum_checkpoint_error_full_samples": np.stack(
                    checkpoint_full_errors_by_seed
                ).astype(np.float32),
                "curriculum_checkpoint_error_full_seed_std": (
                    checkpoint_full_seed_std.astype(np.float32)
                ),
                "curriculum_checkpoint_error_nog": checkpoint_stop_error.astype(
                    np.float32
                ),
                "curriculum_checkpoint_error_nog_samples": np.stack(
                    checkpoint_stop_errors_by_seed
                ).astype(np.float32),
                "curriculum_checkpoint_error_nog_seed_std": (
                    checkpoint_stop_seed_std.astype(np.float32)
                ),
                "curriculum_checkpoint_error_native": checkpoint_native_error.astype(
                    np.float32
                ),
            }
        )
    if (
        first_corrected is not None
        and first_stop_gradient is not None
        and first_uncorrected is not None
    ):
        snapshots["rollout_corrected"] = first_corrected
        snapshots["rollout_stop_gradient"] = first_stop_gradient
        snapshots["rollout_uncorrected"] = first_uncorrected
        if first_one_step is not None:
            snapshots["rollout_one_step"] = first_one_step
    if reference_kind == "solver_self_refined":
        snapshots["reference_rollout"] = test[0]
        shared = {
            "evaluation_times": np.arange(test.shape[1], dtype=np.float32)
            * interval_time
        }
    else:
        shared = {
            "reference_rollout": test[0],
            "evaluation_times": np.arange(test.shape[1], dtype=np.float32)
            * interval_time,
        }
    return {
        "metrics": metrics,
        "snapshots": snapshots,
        "shared": shared,
    }


__all__ = ["make_reference_dataset", "solver_in_loop"]
