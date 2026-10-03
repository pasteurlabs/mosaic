# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal input-Jacobian supervision; offline research, not runtime code.

Generate labels inside the XLB image, then fine-tune inside the surrogate image.
Every label uses the same output cotangent for teacher and student. Full 100-step
VJPs avoid treating intermediate XLB velocities as complete lattice states.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import surrogate_model as fno
from jax.example_libraries import optimizers


def relative_mse(
    predicted: jax.Array, target: jax.Array, floor: float = 1e-12
) -> jax.Array:
    """Average per-example relative squared error with an absolute energy floor."""
    axes = tuple(range(1, target.ndim))
    return jnp.mean(
        jnp.mean((predicted - target) ** 2, axis=axes)
        / jnp.maximum(jnp.mean(target**2, axis=axes), floor)
    )


def losses(
    forward: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    target: jax.Array,
    cotangent: jax.Array,
    teacher_vjp: jax.Array,
    field_floor: float,
) -> tuple[jax.Array, jax.Array]:
    """Return value and common-cotangent VJP losses, differentiable in weights."""
    predicted, pullback = jax.vjp(forward, initial)
    student_vjp = pullback(cotangent)[0]
    return (
        relative_mse(predicted, target, field_floor),
        relative_mse(student_vjp, teacher_vjp),
    )


def secant_losses(
    forward: Callable[[jax.Array], jax.Array],
    initial: jax.Array,
    target: jax.Array,
    displacement: jax.Array,
    teacher_secant: jax.Array,
    field_floor: float,
) -> tuple[jax.Array, jax.Array]:
    """Match finite displacements using only first-order parameter gradients.

    Divide by twice the displacement RMS to express the central difference as
    a directional derivative along an RMS-normalized direction.
    """
    axes = tuple(range(1, initial.ndim))
    epsilon = jnp.sqrt(jnp.mean(displacement**2, axis=axes, keepdims=True))
    secant = (forward(initial + displacement) - forward(initial - displacement)) / (
        2 * epsilon
    )
    return relative_mse(forward(initial), target, field_floor), relative_mse(
        secant, teacher_secant
    )


def generate(args: argparse.Namespace) -> None:
    """Label only existing training/validation ICs, retaining their split IDs."""
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location("sobolev_teacher", args.teacher_api)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load teacher API")
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    trajectories = np.load(args.dataset, mmap_mode="r")
    with np.load(args.dataset.with_suffix(".split.npz")) as data:
        split = data["split"]
    rng = np.random.default_rng(args.seed)
    indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(split == part), size=count, replace=False)
            for part, count in ((0, args.train_samples), (1, args.validation_samples))
        ]
    )
    initial = np.asarray(trajectories[indices, 0], dtype=np.float32)
    # Label both ordinary ICs and lower-amplitude states encountered in recovery.
    factors = rng.choice(np.asarray([0.0, 0.1, 0.5, 1.0]), size=len(indices))
    initial = initial * factors[:, None, None, None, None].astype(np.float32)

    def teacher(value: jax.Array) -> jax.Array:
        return api.xlb_fwd(
            value,
            viscosity=0.01,
            dt=0.02,
            steps=100,
            domain_extent=2 * np.pi,
            _use_f64=True,
            _sub_k=1,
            _collision_kind_override="kbc",
        )[0]

    @jax.jit
    def label(value: jax.Array, cotangent: jax.Array):
        result, pullback = jax.vjp(teacher, value)
        return result, pullback(cotangent.astype(result.dtype))[0]

    targets, cotangents, derivatives = [], [], []
    displacements, secants = [], []
    direction_rng = np.random.default_rng(args.seed + 1)
    run_teacher = jax.jit(teacher)
    for value in initial:
        cotangent = rng.normal(size=value.shape).astype(np.float32)
        cotangent /= np.linalg.norm(cotangent)
        target, derivative = label(jnp.asarray(value, dtype=jnp.float64), cotangent)
        targets.append(np.asarray(target, dtype=np.float32))
        derivatives.append(np.asarray(derivative, dtype=np.float32))
        cotangents.append(cotangent)
        if args.secant_step is not None:
            direction = jnp.asarray(
                direction_rng.normal(size=value.shape), dtype=jnp.float32
            )
            direction = np.asarray(fno.helmholtz_project(direction[None])[0])
            direction = direction / np.sqrt(np.mean(direction**2))
            epsilon = args.secant_step * max(float(np.sqrt(np.mean(value**2))), 0.01)
            displacement = direction * epsilon
            plus = run_teacher(jnp.asarray(value, dtype=jnp.float64) + displacement)
            minus = run_teacher(jnp.asarray(value, dtype=jnp.float64) - displacement)
            displacements.append(displacement)
            secants.append(np.asarray((plus - minus) / (2 * epsilon), dtype=np.float32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        initial=initial,
        target=targets,
        cotangent=cotangents,
        teacher_vjp=derivatives,
        split=split[indices],
        indices=indices,
        amplitude_factors=factors,
        **(
            {"displacement": displacements, "teacher_secant": secants}
            if secants
            else {}
        ),
    )
    from train import _sha256

    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset_sha256": _sha256(args.dataset),
                "split_sha256": _sha256(args.dataset.with_suffix(".split.npz")),
                "teacher_api_sha256": _sha256(args.teacher_api),
                "labels_sha256": _sha256(args.output),
                "seed": args.seed,
                "secant_step": args.secant_step,
                "teacher_precision": "float64",
                "saved_precision": "float32",
                "steps": 100,
                "dt": 0.02,
                "viscosity": 0.01,
                "cotangents": "one independent isotropic Gaussian unit vector per IC",
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    """Compare lambda=0 and lambda>0 from identical initial weights and batches."""
    from train import _sha256

    started = time.perf_counter()
    if args.output.resolve() == args.init_weights.resolve():
        raise ValueError("output must not overwrite initial weights")
    if args.weight < 0 or args.updates < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("invalid training hyperparameters")
    with np.load(args.init_weights) as data:
        checkpoint = {key: data[key] for key in data.files}
    width, modes, layers = (
        int(checkpoint[key]) for key in ("width", "modes", "layers")
    )
    parameter_keys = fno.init_params(
        width=width, modes=modes, layers=layers, seed=0
    ).keys()
    params = {key: jnp.asarray(checkpoint[key]) for key in parameter_keys}
    if "w_linear" in checkpoint or args.linear_correction:
        params["w_linear"] = jnp.asarray(
            checkpoint.get(
                "w_linear",
                np.zeros((3 * (fno.N // 2) ** 2 + 1, 3, 3), dtype=np.float32),
            )
        )
    replay_path = getattr(args, "replay_dataset", None)
    replay = None
    if replay_path is not None:
        if args.replay_normalization is None:
            raise ValueError(
                "replay requires the original training normalization metrics"
            )
        replay = np.load(replay_path, mmap_mode="r")
        if replay.shape[1:] != (fno.ROLLOUT_STEPS + 1, fno.N, fno.N, fno.N, 3):
            raise ValueError("invalid replay trajectory shape")
        with np.load(replay_path.with_suffix(".split.npz")) as data:
            replay_split = data["split"]
        replay_train = np.flatnonzero(replay_split == 0)
        replay_val = np.flatnonzero(replay_split == 1)[: args.replay_validation_samples]
        if not len(replay_train) or not len(replay_val):
            raise ValueError("replay requires training and validation trajectories")
        normalization = json.loads(args.replay_normalization.read_text())
        output_scale = np.asarray(normalization["output_scale"], dtype=np.float32)
        if (
            output_scale.shape != (3,)
            or not np.all(np.isfinite(output_scale))
            or np.any(output_scale <= 0)
        ):
            raise ValueError("invalid original output normalization")
        replay_output_scale = jnp.asarray(output_scale).reshape(1, 1, 1, 1, 1, 3)
    method = args.method
    keys = (
        ("initial", "target", "cotangent", "teacher_vjp")
        if method == "vjp"
        else ("initial", "target", "displacement", "teacher_secant")
    )
    with np.load(args.labels) as data:
        arrays = {key: np.asarray(data[key]) for key in (*keys, "split")}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("nonfinite labels")
    if method == "secant":
        axes = tuple(range(1, arrays["displacement"].ndim))
        if np.any(np.sum(arrays["displacement"] ** 2, axis=axes) <= 0):
            raise ValueError("secant displacements must be nonzero")
    train_idx = np.flatnonzero(arrays["split"] == 0)
    validation_idx = np.flatnonzero(arrays["split"] == 1)
    if not len(train_idx) or not len(validation_idx):
        raise ValueError("both training and validation labels are required")
    # Training-only floor prevents zero-state field errors dominating normalization.
    field_floor = max(float(np.mean(arrays["target"][train_idx] ** 2)) * 0.01, 1e-12)

    def replay_objective(current: dict[str, jax.Array], trajectory: jax.Array):
        from train import trajectory_loss

        predicted = fno.rollout(
            current,
            trajectory[:, 0],
            steps=fno.ROLLOUT_STEPS,
            input_scale=jnp.asarray(checkpoint["input_scale"]),
            correction_scale=jnp.asarray(checkpoint["correction_scale"]),
            modes=modes,
            layers=layers,
        )
        loss, _ = trajectory_loss(predicted, trajectory[:, 1:], replay_output_scale)
        loss += 1e-8 * fno.tree_l2(current)
        error = jnp.linalg.norm(
            (predicted[:, -1] - trajectory[:, -1]).reshape(trajectory.shape[0], -1),
            axis=1,
        ) / jnp.maximum(
            jnp.linalg.norm(trajectory[:, -1].reshape(trajectory.shape[0], -1), axis=1),
            1e-8,
        )
        return loss, jnp.mean(error)

    def objective(current: dict[str, jax.Array], batch: tuple[jax.Array, ...]):
        def forward(initial: jax.Array) -> jax.Array:
            return fno.rollout(
                current,
                initial,
                steps=fno.ROLLOUT_STEPS,
                input_scale=jnp.asarray(checkpoint["input_scale"]),
                correction_scale=jnp.asarray(checkpoint["correction_scale"]),
                modes=modes,
                layers=layers,
            )[:, -1]

        loss_function = losses if method == "vjp" else secant_losses
        field, derivative = loss_function(forward, *batch, field_floor)
        return field + args.weight * derivative, (field, derivative)

    init, update, get_params = optimizers.adam(args.lr)
    state = init(params)

    @jax.jit
    def step(
        index: int, state: Any, batch: tuple[jax.Array, ...], trajectory: jax.Array
    ):
        if replay is None:
            (_, _), gradients = jax.value_and_grad(objective, has_aux=True)(
                get_params(state), batch
            )
        else:

            def combined(current: dict[str, jax.Array]):
                replay_loss, _ = replay_objective(current, trajectory)
                # Preserve the original trajectory objective; add derivatives only.
                _, (_, derivative) = objective(current, batch)
                return replay_loss + args.weight * derivative

            gradients = jax.grad(combined)(get_params(state))
        return update(index, gradients, state)

    evaluate = jax.jit(objective)
    evaluate_replay = jax.jit(replay_objective)

    def validation(current: dict[str, jax.Array]):
        values = []
        for index in validation_idx:
            batch = tuple(jnp.asarray(arrays[key][index : index + 1]) for key in keys)
            score, (field, derivative) = evaluate(current, batch)
            values.append([float(score), float(field), float(derivative)])
        return np.mean(values, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    history = []
    rng = np.random.default_rng(args.seed)
    replay_rng = np.random.default_rng(args.seed + 17)
    for index in range(args.updates + 1):
        if index:
            selected = rng.choice(
                train_idx, min(args.batch_size, len(train_idx)), replace=False
            )
            batch = tuple(jnp.asarray(arrays[key][selected]) for key in keys)
            trajectory = jnp.zeros(())
            if replay is not None:
                chosen = replay_rng.choice(
                    replay_train, min(args.batch_size, len(replay_train)), replace=False
                )
                trajectory = jnp.asarray(np.asarray(replay[chosen], dtype=np.float32))
            state = step(index - 1, state, batch, trajectory)
        if index % args.validation_interval and index != args.updates:
            continue
        score, field, derivative = validation(get_params(state))
        replay_metrics = {}
        if replay is not None:
            replay_values = [
                evaluate_replay(
                    get_params(state),
                    jnp.asarray(np.asarray(replay[[i]], dtype=np.float32)),
                )
                for i in replay_val
            ]
            replay_loss, replay_error = np.mean(np.asarray(replay_values), axis=0)
            score = float(replay_loss) + args.weight * derivative
            replay_metrics = {
                "replay_loss": float(replay_loss),
                "replay_final_relative_l2": float(replay_error),
            }
        if not np.isfinite(score):
            raise RuntimeError("nonfinite validation loss")
        record = {
            "step": index,
            "score": float(score),
            "field": float(field),
            method: float(derivative),
            **replay_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score:
            best_score = score
            best_step = index
            updated = {**checkpoint, **jax.tree.map(np.asarray, get_params(state))}
            np.savez(args.output, **updated)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "replay_dataset": str(replay_path) if replay_path is not None else None,
                "replay_dataset_sha256_from_training_metadata": normalization.get(
                    "dataset_sha256"
                )
                if replay is not None
                else None,
                "replay_normalization_sha256": _sha256(args.replay_normalization)
                if replay is not None
                else None,
                "replay_split_sha256": _sha256(replay_path.with_suffix(".split.npz"))
                if replay is not None
                else None,
                "replay_validation_indices": replay_val.tolist()
                if replay is not None
                else None,
                "labels_sha256": _sha256(args.labels),
                "initial_weights_sha256": _sha256(args.init_weights),
                "weights_sha256": _sha256(args.output),
                "elapsed_seconds": time.perf_counter() - started,
                "linear_correction": "w_linear" in params,
                "method": method,
                "weight": args.weight,
                "lr": args.lr,
                "seed": args.seed,
                "updates": args.updates,
                "batch_size": args.batch_size,
                "field_floor": field_floor,
                "best_step": best_step,
                "selection": (
                    "validation "
                    + (
                        "original trajectory loss"
                        if replay is not None
                        else "field relative MSE"
                    )
                    + f" + weight * {method} relative MSE"
                ),
                "history": history,
            },
            indent=2,
        )
    )


def main() -> None:
    """Generate teacher labels or run a controlled fine-tuning pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--teacher-api", type=Path, required=True)
    generate_parser.add_argument("--dataset", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument(
        "--secant-step",
        type=float,
        help="Optional relative RMS perturbation, e.g. 0.01",
    )
    generate_parser.add_argument("--train-samples", type=int, default=128)
    generate_parser.add_argument("--validation-samples", type=int, default=32)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--replay-dataset", type=Path)
    train_parser.add_argument("--replay-normalization", type=Path)
    train_parser.add_argument("--replay-validation-samples", type=int, default=32)
    train_parser.add_argument("--labels", type=Path, required=True)
    train_parser.add_argument("--init-weights", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--method", choices=("vjp", "secant"), default="vjp")
    train_parser.add_argument("--linear-correction", action="store_true")
    train_parser.add_argument("--weight", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--updates", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--validation-interval", type=int, default=50)
    for child in (generate_parser, train_parser):
        child.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.command == "generate":
        if args.secant_step is not None and args.secant_step <= 0:
            parser.error("secant step must be positive")
        generate(args)
    else:
        if args.replay_validation_samples < 1:
            parser.error("replay validation samples must be positive")
        if args.validation_interval < 1:
            parser.error("validation interval must be positive")
        train(args)


if __name__ == "__main__":
    main()
