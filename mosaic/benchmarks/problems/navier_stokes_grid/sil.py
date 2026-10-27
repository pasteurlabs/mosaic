# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training a Kolmogorov-flow corrector through one solver, four ways.

A *step* advances a canonical (N, N, 1, 2) state by one correction interval:
``step(v, native) -> (v_next, native_next)``. ``native`` is the solver's own
state, carried from one call to the next so that the solver can continue
without a round trip through the canonical grid; solvers without one pass
``None`` along. Steps are differentiable JAX functions, either a Tesseract
call through ``tesseract_jax`` or, in tests, a closed-form map.

The arms differ only in what the corrector is trained on:

- ``uncorrected``: no corrector.
- ``supervised``: fixed pairs (solver output from reference frame k, reference
  frame k + 1), precomputed once; the solver is not in the training graph.
- ``stopped``: the corrected K-interval rollout, with the gradient stopped at
  every solver output and native state.
- ``sil``: the same rollout, differentiated through the solver.

Every trained arm sees the same windows, updates, learning-rate grid and
initialisation. The targets are the solver's own reference, run at a finer
grid and timestep and restricted to the coarse grid.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mosaic.benchmarks.core.experiment import KernelContext, kernel
from mosaic.benchmarks.core.utils import active_differentiable_solvers

from .corrector import apply_corrector
from .kolmogorov import (
    first_unstable_frame,
    high_correlation_time,
    relative_l2,
    resample,
    spectrum_error,
    vorticity_correlation,
)

Step = Callable[[jax.Array, Any], tuple[jax.Array, Any]]

ARMS = ("uncorrected", "supervised", "stopped", "sil")
TRAINED_ARMS = ARMS[1:]

# Frames at the end of an evaluation rollout whose mean energy spectrum is
# compared with the reference's.
SPECTRUM_FRAMES = 10


def rollout(
    step: Step, v0: Any, frames: int, params: Any = None, native: Any = None
) -> np.ndarray:
    """``frames`` intervals from ``v0``, corrected after each when ``params``.

    Returns the stacked states, frame 0 included. Stops early, padding with
    NaN, once a state is no longer finite.
    """
    state = jnp.asarray(v0)
    states = [np.asarray(state)]
    for _ in range(frames):
        state, native = step(state, native)
        if params is not None:
            state = apply_corrector(params, state)
        states.append(np.asarray(state))
        if not np.isfinite(states[-1]).all():
            states += [np.full_like(states[0], np.nan)] * (frames + 1 - len(states))
            break
    return np.stack(states)


def references(
    fine_step: Step, initials: Sequence[Any], frames: int, n: int
) -> np.ndarray:
    """Fine-grid trajectories from ``initials``, restricted to ``n`` cells.

    Shape (len(initials), frames + 1, n, n, 1, 2).
    """
    return np.stack(
        [
            np.stack(
                [np.asarray(resample(s, n)) for s in rollout(fine_step, v, frames)]
            )
            for v in initials
        ]
    )


def mean_relative_error(states: np.ndarray, targets: np.ndarray) -> float:
    """Mean relative L2 error over frames 1 …, the score of a rollout."""
    return float(
        np.mean(
            [
                float(relative_l2(s, t))
                for s, t in zip(states[1:], targets[1:], strict=True)
            ]
        )
    )


def window_loss(
    params: Any,
    targets: jax.Array,
    step: Step,
    *,
    differentiate_solver: bool,
    native: Any = None,
) -> jax.Array:
    """Mean of ‖s − t‖²/‖t‖² over a corrected rollout from ``targets[0]``.

    ``native`` is the solver's state when it reaches ``targets[0]`` (see
    :func:`teacher_forcing`), so a window continues the solver the way a
    rollout does. Without ``differentiate_solver`` the forward pass is
    identical, but the gradient stops at every solver output and native
    state, so only the corrector's own dependence on its input carries
    gradient.
    """
    state = targets[0]
    losses = []
    for target in targets[1:]:
        provisional, native = step(state, native)
        if not differentiate_solver:
            provisional = jax.lax.stop_gradient(provisional)
            native = jax.tree.map(jax.lax.stop_gradient, native)
        state = apply_corrector(params, provisional)
        losses.append(jnp.sum((state - target) ** 2) / jnp.sum(target**2))
    return jnp.mean(jnp.stack(losses))


def teacher_forcing(
    step: Step, trajectories: np.ndarray
) -> tuple[np.ndarray, list[list[Any]]]:
    """The solver driven along each reference trajectory, one interval at a time.

    The native state is carried from frame to frame and assimilates each
    reference frame, as it assimilates a correction in a rollout. Returns the
    solver output from every frame but the last, (S, F, N, N, 1, 2), the
    inputs of the supervised arm; and the native state with which the solver
    reaches each of those frames (``None`` at frame 0), from which the
    recurrent arms start their windows.
    """
    outputs, natives = [], []
    for trajectory in trajectories:
        native = None
        frames, states = [], []
        for v in trajectory[:-1]:
            states.append(native)
            provisional, native = step(jnp.asarray(v), native)
            frames.append(np.asarray(provisional))
        outputs.append(np.stack(frames))
        natives.append(states)
    inputs = np.stack(outputs)
    if not np.isfinite(inputs).all():
        raise FloatingPointError("teacher forcing produced non-finite solver outputs")
    return inputs, natives


def supervised_loss(params: Any, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean of ‖C(input) − t‖²/‖t‖² over independent pairs."""
    losses = [
        jnp.sum((apply_corrector(params, x) - t) ** 2) / jnp.sum(t**2)
        for x, t in zip(inputs, targets, strict=True)
    ]
    return jnp.mean(jnp.stack(losses))


def windows(
    trajectories: int, frames: int, unroll: int, updates: int, seed: int
) -> list[tuple[int, int]]:
    """``updates`` (trajectory, start) draws; each epoch visits every window once."""
    if frames < unroll:
        raise ValueError(f"{frames} training frames cannot hold an unroll of {unroll}")
    pool = [(i, s) for i in range(trajectories) for s in range(frames - unroll + 1)]
    rng = np.random.default_rng(seed)
    draws: list[tuple[int, int]] = []
    while len(draws) < updates:
        draws += [pool[i] for i in rng.permutation(len(pool))]
    return draws[:updates]


def checkpoint_updates(updates: int, interval: int) -> list[int]:
    """Validation points: the identity model, every ``interval``, and the end."""
    return sorted({0, updates, *range(interval, updates + 1, interval)})


def _finite_tree(tree: Any) -> bool:
    return all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(tree))


def global_norm(tree: Any) -> float:
    """Global L2 norm of a pytree."""
    return float(optax.tree.norm(tree))


def fit(
    params: Any,
    loss: Callable[[Any, int], jax.Array],
    validate: Callable[[Any], float],
    *,
    updates: int,
    lr: float,
    validation_interval: int,
    clip_norm: float = 1.0,
    value_and_grad: Callable | None = None,
) -> tuple[Any, dict]:
    """Adam with global-norm clipping over the full budget, keeping the best checkpoint.

    ``loss(params, update)`` is the training loss of one update. The model with
    the lowest validation score wins, the earliest on a tie, so update 0 (the
    uncorrected solver) is kept unless training improves on it. A non-finite
    loss, gradient or model stops training, and the record says where.
    """
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
    state = optimizer.init(params)
    value_and_grad = value_and_grad or jax.value_and_grad(loss)
    scheduled = set(checkpoint_updates(updates, validation_interval))
    record: dict[str, Any] = {
        "learning_rate": lr,
        "loss": [],
        "grad_norm": [],
        "update_seconds": [],
        "checkpoints": [],
        "completed_updates": 0,
        "failure": None,
    }
    best = (float("inf"), 0, params)
    validation_seconds = 0.0

    def evaluate(current: Any, update: int) -> None:
        nonlocal best, validation_seconds
        started = time.perf_counter()
        score = float(validate(current))
        validation_seconds += time.perf_counter() - started
        record["checkpoints"].append({"update": update, "score": score})
        if np.isfinite(score) and score < best[0]:
            best = (score, update, current)

    evaluate(params, 0)
    for update in range(updates):
        started = time.perf_counter()
        value, grads = value_and_grad(params, update)
        value = float(value)
        norm = global_norm(grads)
        if not (np.isfinite(value) and np.isfinite(norm)):
            record["failure"] = f"non-finite loss or gradient at update {update}"
            break
        delta, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, delta)
        if not _finite_tree(params):
            record["failure"] = f"non-finite model after update {update + 1}"
            break
        record["loss"].append(value)
        record["grad_norm"].append(norm)
        record["update_seconds"].append(time.perf_counter() - started)
        record["completed_updates"] = update + 1
        if update + 1 in scheduled:
            evaluate(params, update + 1)
    record["validation_score"] = best[0]
    record["selected_update"] = best[1]
    record["validation_seconds"] = validation_seconds
    record["training_seconds"] = float(sum(record["update_seconds"]))
    return best[2], record


def directional_fd(
    loss: Callable[[Any], jax.Array], params: Any, epsilon: float
) -> dict[str, float]:
    """Central difference of ``loss`` along its own normalised gradient.

    Along the unit gradient direction d the directional derivative is the
    gradient norm, so the check compares ‖∇L‖ with (L(p + εd) − L(p − εd))/2ε.
    """
    grads = jax.grad(loss)(params)
    norm = global_norm(grads)
    if not np.isfinite(norm) or norm == 0.0:
        return {
            "autodiff": norm,
            "finite_difference": float("nan"),
            "relative_error": float("inf"),
            "epsilon": epsilon,
        }
    direction = jax.tree.map(lambda g: g / norm, grads)
    plus = jax.tree.map(lambda p, d: p + epsilon * d, params, direction)
    minus = jax.tree.map(lambda p, d: p - epsilon * d, params, direction)
    fd = float((loss(plus) - loss(minus)) / (2 * epsilon))
    return {
        "autodiff": norm,
        "finite_difference": fd,
        "relative_error": abs(norm - fd) / max(abs(norm), abs(fd), 1e-12),
        "epsilon": epsilon,
    }


def evaluate(
    step: Step, params: Any, targets: np.ndarray, interval: float
) -> dict[str, Any]:
    """Roll out every test trajectory and score it against its reference.

    Returns per-frame relative errors and vorticity correlations, (S, F + 1),
    and per-trajectory high-correlation time, spectrum error over the last
    :data:`SPECTRUM_FRAMES` frames and first unstable frame.
    """
    errors, correlations, hct, spectra, unstable, finals = [], [], [], [], [], []
    for target in targets:
        states = rollout(step, target[0], len(target) - 1, params)
        e = [float(relative_l2(s, t)) for s, t in zip(states, target, strict=True)]
        c = [
            float(vorticity_correlation(s, t))
            for s, t in zip(states, target, strict=True)
        ]
        errors.append(e)
        correlations.append(c)
        hct.append(high_correlation_time(c, interval))
        tail = slice(-SPECTRUM_FRAMES, None)
        spectra.append(
            spectrum_error(states[tail], target[tail])
            if np.isfinite(states[tail]).all()
            else float("inf")
        )
        unstable.append(first_unstable_frame(e))
        finals.append(states[-1])
    return {
        "errors": np.asarray(errors),
        "correlations": np.asarray(correlations),
        "high_correlation_time": np.asarray(hct),
        "spectrum_error": np.asarray(spectra),
        "first_unstable_frame": np.asarray(unstable),
        "final_states": np.asarray(finals),
    }


def gap_closed(arm_errors: np.ndarray, uncorrected_errors: np.ndarray) -> np.ndarray:
    """1 − E_arm/E_uncorrected per trajectory, E the mean error over frames 1 …."""
    e_arm = np.mean(arm_errors[..., 1:], axis=-1)
    e_unc = np.mean(uncorrected_errors[..., 1:], axis=-1)
    return 1.0 - e_arm / e_unc


def value_of_gradient(stopped_errors: np.ndarray, sil_errors: np.ndarray) -> np.ndarray:
    """E_stopped / E_sil per trajectory: above 1 when the solver gradient helps."""
    return np.mean(stopped_errors[..., 1:], axis=-1) / np.mean(
        sil_errors[..., 1:], axis=-1
    )


# ── The benchmark ────────────────────────────────────────────────────────────

# Training configuration, overridden per experiment by its ``optim`` payload.
DEFAULT_TRAINING: dict[str, Any] = {
    "unrolls": [4],
    "updates": 32,
    "model_seeds": [0],
    "learning_rates": [1e-3],
    "validation_interval": 16,
    "seeds": {"train": [0, 1, 2, 3], "validation": [4], "test": [5]},
    "train_frames": 24,
    "eval_frames": 40,
    # Reference audits run over this many intervals of every trajectory.
    "audit_frames": 8,
    "audit_tolerance": 0.25,
    # Closure of chained calls against one call (median, 95th percentile).
    "closure_tolerance": [0.005, 0.01],
    "fd_epsilon": 0.01,
    "fd_tolerance": 0.05,
    "hidden_channels": 8,
    "kernel_size": 5,
    "layers": 3,
}


def supports_native_state(t: Any) -> bool:
    """Whether the solver accepts and returns a differentiable native state."""
    try:
        schema = t.openapi_schema["components"]["schemas"]["ApplyInputSchema"]
        return "state" in schema["differentiable_arrays"]
    except (AttributeError, KeyError, TypeError):
        return False


def solver_step(t: Any, ctx: Any, *, dt: float, steps: int, native_state: bool) -> Step:
    """One correction interval of ``steps`` steps of ``dt``, on the grid of its input."""
    from tesseract_jax import apply_tesseract

    phys = {
        k: v for k, v in ctx.phys.items() if k not in ("N", "dt", "reference_factor")
    }

    def step(v: Any, native: Any) -> tuple[jax.Array, Any]:
        inputs = ctx.make_inputs(ctx.name, jnp.asarray(v), dt=dt, steps=steps, **phys)
        if native_state:
            inputs = {**inputs, "return_state": True}
            if native is not None:
                inputs["state"] = native
        out = apply_tesseract(t, inputs)
        return out[ctx.output_key], (out.get("state") if native_state else None)

    return step


def closure(
    step: Step, joined: Callable[[int], Step], starts: Sequence[Any], frames: int
) -> dict[str, Any]:
    """Relative distance between chained interval calls and one call over the same time.

    ``semigroup`` compares two chained intervals with one call of two,
    ``long`` a chain of ``frames`` intervals with one call of that length.
    The chain carries the native state, so a solver that continues exactly
    closes to round-off, while one restarting from the canonical velocity
    loses its own state at every call.
    """
    two, whole = joined(2), joined(frames)
    semigroup, long = [], []
    for v in starts:
        chain = rollout(step, v, frames)
        semigroup.append(float(relative_l2(chain[2], two(jnp.asarray(v), None)[0])))
        long.append(float(relative_l2(chain[frames], whole(jnp.asarray(v), None)[0])))
    return {"semigroup": semigroup, "long": long}


def _passes_closure(record: dict, tolerance: Sequence[float]) -> bool:
    median, p95 = tolerance
    values = np.asarray(record["semigroup"] + record["long"], dtype=np.float64)
    return bool(
        np.all(np.isfinite(values))
        and np.median(values) <= median
        and np.percentile(values, 95) <= p95
    )


def _audit(
    audit_step: Step,
    coarse_step: Step,
    initials: Sequence[np.ndarray],
    references_: np.ndarray,
    frames: int,
    n: int,
) -> list[dict[str, float]]:
    """Reference error estimate against the coarse gap, per trajectory.

    The audit doubles the reference's resolution in space and time. Its
    distance to the reference, over the first ``frames`` intervals, estimates
    the reference's own error; the coarse configuration's distance is the gap
    a corrector can close.
    """
    audited = references(audit_step, initials, frames, n)
    rows = []
    for a, ref in zip(audited, references_, strict=True):
        target = ref[: frames + 1]
        error = mean_relative_error(a, target)
        gap = mean_relative_error(rollout(coarse_step, target[0], frames), target)
        rows.append({"audit_error": error, "coarse_gap": gap, "ratio": error / gap})
    return rows


def _paired(
    errors: dict, uncorrected: np.ndarray, unrolls: Sequence[int]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for K in unrolls:
        for arm in TRAINED_ARMS:
            out[f"gap_closed/K{K}/{arm}"] = gap_closed(
                errors[(K, arm)], uncorrected[None]
            ).tolist()
        out[f"value_of_gradient/K{K}"] = value_of_gradient(
            errors[(K, "stopped")], errors[(K, "sil")]
        ).tolist()
    return out


def _ineligible(metrics: dict, reason: str) -> dict:
    metrics.update(eligible=False, reason=reason, completed=False)
    return {"metrics": metrics}


def _train_all(
    *,
    coarse: Step,
    references_: dict[str, np.ndarray],
    cfg: dict,
    memory: Callable[[], Any],
) -> tuple[dict, dict, dict]:
    """Every (K, arm, seed, learning rate); returns models, records and costs."""
    from .corrector import init_corrector

    train = jnp.asarray(references_["train"])
    validation = references_["validation"]
    supervised_inputs, natives = teacher_forcing(coarse, references_["train"])
    supervised_inputs = jnp.asarray(supervised_inputs)
    supervised_grad = jax.jit(jax.value_and_grad(supervised_loss))

    def validate(params: Any) -> float:
        return float(
            np.mean(
                [
                    mean_relative_error(rollout(coarse, v[0], len(v) - 1, params), v)
                    for v in validation
                ]
            )
        )

    init = {
        seed: init_corrector(
            seed, cfg["hidden_channels"], cfg["layers"], cfg["kernel_size"]
        )
        for seed in cfg["model_seeds"]
    }
    models, records, costs = {}, {}, {}
    for K in cfg["unrolls"]:
        draws = {
            seed: windows(
                train.shape[0], train.shape[1] - 1, K, cfg["updates"], seed=1000 + seed
            )
            for seed in cfg["model_seeds"]
        }
        for arm in TRAINED_ARMS:

            def value_and_grad(
                params: Any, update: int, *, _arm: str, _draws: list, _k: int
            ) -> tuple:
                i, s = _draws[update]
                if _arm == "supervised":
                    return supervised_grad(
                        params,
                        supervised_inputs[i, s : s + _k],
                        train[i, s + 1 : s + _k + 1],
                    )
                return jax.value_and_grad(window_loss)(
                    params,
                    train[i, s : s + _k + 1],
                    coarse,
                    differentiate_solver=_arm == "sil",
                    native=natives[i][s],
                )

            started = time.perf_counter()
            with memory() as poller:
                for seed in cfg["model_seeds"]:
                    for lr in cfg["learning_rates"]:
                        params, record = fit(
                            init[seed],
                            None,
                            validate,
                            updates=cfg["updates"],
                            lr=lr,
                            validation_interval=cfg["validation_interval"],
                            value_and_grad=functools.partial(
                                value_and_grad, _arm=arm, _draws=draws[seed], _k=K
                            ),
                        )
                        models[(K, arm, seed, lr)] = params
                        records[(K, arm, seed, lr)] = record
            update_times = [
                t
                for (k, a, _, _), r in records.items()
                if (k, a) == (K, arm)
                for t in r["update_seconds"][1:]
            ]
            costs[(K, arm)] = {
                "update_seconds": float(np.median(update_times))
                if update_times
                else None,
                "total_seconds": time.perf_counter() - started,
                **(poller.summary if poller is not None else {}),
            }
    return models, records, costs


def _select(records: dict, K: int, arm: str, cfg: dict) -> float:
    """The learning rate whose models validate best on average over the seeds."""
    rates = cfg["learning_rates"]
    score = {
        lr: float(
            np.mean(
                [
                    records[(K, arm, s, lr)]["validation_score"]
                    for s in cfg["model_seeds"]
                ]
            )
        )
        for lr in rates
    }
    return min(rates, key=lambda lr: (score[lr], rates.index(lr)))


def _finite(values: Any) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def run(t: Any, ctx: Any, *, step_factory: Callable = solver_step) -> dict:
    """One solver's references, gates, training arms and evaluation.

    ``step_factory`` builds the solver's interval steps; tests replace it
    with closed-form transitions.
    """
    from .corrector import init_corrector
    from .kolmogorov import INTERVAL, initial_state, load_ground_truth

    cfg = {**DEFAULT_TRAINING, **ctx.run.get("optim", {})}
    n = int(ctx.phys["N"])
    factor = int(ctx.phys["reference_factor"])
    dt = float(ctx.phys["dt"])
    spi = round(INTERVAL / dt)
    native = supports_native_state(t)

    def make(grid_factor: int, intervals: int = 1) -> Step:
        return step_factory(
            t,
            ctx,
            dt=dt / grid_factor,
            steps=spi * grid_factor * intervals,
            native_state=native,
        )

    coarse, fine, audit_step = make(1), make(factor), make(2 * factor)
    truth = load_ground_truth(ctx.cfg.name)
    seeds = cfg["seeds"]
    frames = {
        "train": cfg["train_frames"],
        "validation": cfg["eval_frames"],
        "test": cfg["eval_frames"],
    }
    metrics: dict[str, Any] = {
        "valid": True,
        "native_state": native,
        "interval": INTERVAL,
        "config": {k: v for k, v in cfg.items()},
    }

    started = time.perf_counter()
    refs, audits = {}, {}
    for split, split_seeds in seeds.items():
        initials = [initial_state(truth, s, n * factor) for s in split_seeds]
        refs[split] = references(fine, initials, frames[split], n)
        audits[split] = _audit(
            audit_step,
            coarse,
            [initial_state(truth, s, 2 * n * factor) for s in split_seeds],
            refs[split],
            min(cfg["audit_frames"], frames[split]),
            n,
        )
    metrics["reference_seconds"] = time.perf_counter() - started
    metrics["audits"] = audits
    ratios = [row["ratio"] for rows in audits.values() for row in rows]
    metrics["audit_passed"] = _finite(ratios) and max(ratios) <= cfg["audit_tolerance"]
    if not metrics["audit_passed"]:
        if not _finite(ratios):
            return _ineligible(
                metrics, "the reference, its audit or the coarse run diverged"
            )
        return _ineligible(
            metrics,
            f"the reference differs from its refinement by {max(ratios):.3g} of "
            f"the coarse gap, above {cfg['audit_tolerance']}",
        )

    record = closure(
        coarse,
        lambda k: make(1, k),
        refs["test"][:, 0],
        cfg["eval_frames"],
    )
    metrics["closure"] = record
    metrics["closure_passed"] = _passes_closure(record, cfg["closure_tolerance"])
    if not metrics["closure_passed"]:
        return _ineligible(metrics, "chained solver calls do not close")

    uncorrected = evaluate(coarse, None, refs["test"], INTERVAL)

    # Gradient checks of the full solver-in-the-loop loss, at the identity
    # model and at a random nonzero readout, on the longest unroll.
    K = max(cfg["unrolls"])
    _, natives = teacher_forcing(coarse, refs["train"][:1])
    start = min(4, refs["train"].shape[1] - 1 - K)
    targets = jnp.asarray(refs["train"][0, start : start + K + 1])

    def loss(params: Any) -> jax.Array:
        return window_loss(
            params, targets, coarse, differentiate_solver=True, native=natives[0][start]
        )

    p0 = init_corrector(
        cfg["model_seeds"][0], cfg["hidden_channels"], cfg["layers"], cfg["kernel_size"]
    )
    readout = 0.01 * jax.random.normal(jax.random.PRNGKey(17), p0[-1]["w"].shape)
    p1 = (*p0[:-1], {"w": readout, "b": p0[-1]["b"]})
    fd = {
        "zero_readout": directional_fd(loss, p0, cfg["fd_epsilon"]),
        "nonzero_readout": directional_fd(loss, p1, cfg["fd_epsilon"]),
    }
    metrics["fd"] = fd
    metrics["fd_passed"] = all(
        r["relative_error"] <= cfg["fd_tolerance"] for r in fd.values()
    )
    metrics["grad_norm_at_identity"] = {
        f"K{k}": {
            arm: global_norm(
                jax.grad(window_loss)(
                    p0,
                    jnp.asarray(refs["train"][0, start : start + k + 1]),
                    coarse,
                    differentiate_solver=arm == "sil",
                    native=natives[0][start],
                )
            )
            for arm in ("stopped", "sil")
        }
        for k in cfg["unrolls"]
        if start + k < refs["train"].shape[1]
    }

    models, records, costs = _train_all(
        coarse=coarse, references_=refs, cfg=cfg, memory=lambda: _memory_poller(t)
    )

    evaluations: dict[str, Any] = {"uncorrected": uncorrected}
    errors: dict[tuple, np.ndarray] = {}
    for K in cfg["unrolls"]:
        for arm in TRAINED_ARMS:
            lr = _select(records, K, arm, cfg)
            per_seed = [
                evaluate(coarse, models[(K, arm, s, lr)], refs["test"], INTERVAL)
                for s in cfg["model_seeds"]
            ]
            errors[(K, arm)] = np.stack([e["errors"] for e in per_seed])
            evaluations[f"K{K}/{arm}"] = {
                "learning_rate": lr,
                "selected_updates": [
                    records[(K, arm, s, lr)]["selected_update"]
                    for s in cfg["model_seeds"]
                ],
                **{
                    key: np.stack([e[key] for e in per_seed]).tolist()
                    for key in (
                        "errors",
                        "correlations",
                        "high_correlation_time",
                        "spectrum_error",
                        "first_unstable_frame",
                    )
                },
            }
    metrics["evaluation"] = {
        k: (
            {
                key: np.asarray(v).tolist()
                for key, v in e.items()
                if key != "final_states"
            }
            if k == "uncorrected"
            else e
        )
        for k, e in evaluations.items()
    }
    metrics["paired"] = _paired(errors, uncorrected["errors"], cfg["unrolls"])
    metrics["training"] = {
        f"K{K}/{arm}/seed{s}/lr{lr:g}": {
            key: r[key]
            for key in (
                "loss",
                "grad_norm",
                "completed_updates",
                "failure",
                "checkpoints",
                "validation_score",
                "selected_update",
                "training_seconds",
                "validation_seconds",
            )
        }
        for (K, arm, s, lr), r in records.items()
    }
    metrics["cost"] = {f"K{K}/{arm}": c for (K, arm), c in costs.items()}
    metrics["completed"] = all(
        r["completed_updates"] == cfg["updates"] and r["failure"] is None
        for r in records.values()
    )
    metrics["finite"] = (
        all(_finite(r["loss"]) and _finite(r["grad_norm"]) for r in records.values())
        and all(_finite(e) for e in errors.values())
        and _finite(uncorrected["errors"])
    )
    metrics["selection_valid"] = all(
        np.isfinite(r["validation_score"])
        and r["selected_update"] in {c["update"] for c in r["checkpoints"]}
        for r in records.values()
    )
    metrics["eligible"] = True
    return {"metrics": metrics}


def _memory_poller(t: Any) -> Any:
    from mosaic.benchmarks.core.memory import MemoryPoller, container_id_from_tesseract
    from mosaic.benchmarks.core.runner import current_worker_context

    # Without --gpus the runner pins no GPU; measure GPU 0, as the harness does.
    gpu_id = current_worker_context().gpu_id
    return MemoryPoller(0 if gpu_id is None else gpu_id, container_id_from_tesseract(t))


@kernel(sweep_mode="none", selector_fn=active_differentiable_solvers)
def solver_in_loop(t: Any, ctx: KernelContext) -> dict:
    """Train the corrector through one solver in all arms; see :func:`run`."""
    return run(t, ctx)
