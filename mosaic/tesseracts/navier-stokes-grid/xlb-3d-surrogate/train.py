# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curriculum-train the autoregressive full-field 3D neural operator.

The input is a memory-mapped trajectory array produced by
``generate_trajectories.py`` and its adjacent ``.split.npz`` file. Cluster
submission and container orchestration intentionally remain outside this
Tesseract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import surrogate_model as fno


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_curriculum(value: str) -> list[tuple[int, int]]:
    phases: list[tuple[int, int]] = []
    for item in value.split(","):
        horizon_text, updates_text = item.split(":", 1)
        horizon = int(horizon_text)
        updates = int(updates_text)
        if horizon < 1 or horizon > fno.ROLLOUT_STEPS or updates < 1:
            raise ValueError(f"invalid curriculum phase {item!r}")
        phases.append((horizon, updates))
    if not phases or phases[-1][0] != fno.ROLLOUT_STEPS:
        raise ValueError(f"curriculum must end at full horizon {fno.ROLLOUT_STEPS}")
    return phases


def _relative_l2(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = tuple(range(2, predicted.ndim))
    return np.sqrt(
        np.sum((predicted - target) ** 2, axis=axes)
        / (np.sum(target**2, axis=axes) + 1e-20)
    )


def _estimate_scales(
    trajectories: np.ndarray,
    train_idx: np.ndarray,
    *,
    scale_samples: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    selected = train_idx[: min(scale_samples, train_idx.size)]
    sum_sq = 0.0
    count = 0
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    channel_count = 0
    for start in range(0, selected.size, 8):
        batch = np.asarray(
            trajectories[selected[start : start + 8]],
            dtype=np.float32,
        )
        sum_sq += float(np.sum(batch.astype(np.float64) ** 2))
        count += batch.size
        channel_sum_sq += np.sum(
            batch.astype(np.float64) ** 2,
            axis=(0, 1, 2, 3, 4),
        )
        channel_count += int(np.prod(batch.shape[:-1]))
    input_scale = float(np.sqrt(sum_sq / count))
    output_scale = np.sqrt(channel_sum_sq / channel_count)

    @jax.jit
    def diffuse(values: jax.Array) -> jax.Array:
        return fno.diffuse_macro(values)

    correction_sum_sq = np.zeros(3, dtype=np.float64)
    correction_count = 0
    for start in range(0, selected.size, 4):
        batch = np.asarray(
            trajectories[selected[start : start + 4]],
            dtype=np.float32,
        )
        current = batch[:, :-1].reshape(-1, fno.N, fno.N, fno.N, 3)
        target = batch[:, 1:].reshape(-1, fno.N, fno.N, fno.N, 3)
        for transition_start in range(0, current.shape[0], 32):
            transition_stop = min(transition_start + 32, current.shape[0])
            linear = np.asarray(
                diffuse(jnp.asarray(current[transition_start:transition_stop]))
            )
            correction = (target[transition_start:transition_stop] - linear).astype(
                np.float64
            )
            correction_sum_sq += np.sum(
                correction**2,
                axis=(0, 1, 2, 3),
            )
            correction_count += int(np.prod(correction.shape[:-1]))
    correction_scale = np.sqrt(correction_sum_sq / correction_count)
    return (
        max(input_scale, 1e-5),
        np.maximum(correction_scale, 1e-6).astype(np.float32),
        np.maximum(output_scale, 1e-5).astype(np.float32),
    )


def main() -> None:
    """Train, select by validation rollout error, and export one checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/surrogate-output/recovery_3d_xlb_trajectories.npy"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/surrogate-output/recovery_3d_autoregressive_weights.npz"),
    )
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--scale-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--validation-interval", type=int, default=400)
    parser.add_argument(
        "--curriculum",
        default="1:7200,2:5600,4:5600,8:4800,12:4000,20:4800",
    )
    args = parser.parse_args()
    phases = _parse_curriculum(args.curriculum)
    started = time.perf_counter()
    print("jax_backend", jax.default_backend(), flush=True)
    print("jax_devices", jax.devices(), flush=True)
    print(f"curriculum={phases}", flush=True)

    trajectories = np.load(args.dataset, mmap_mode="r")
    expected_tail = (
        fno.ROLLOUT_STEPS + 1,
        fno.N,
        fno.N,
        fno.N,
        3,
    )
    if trajectories.shape[1:] != expected_tail:
        raise ValueError(
            f"trajectory shape {trajectories.shape} does not end in {expected_tail}"
        )
    split_path = args.dataset.with_suffix(".split.npz")
    with np.load(split_path, allow_pickle=False) as split_data:
        split = np.asarray(split_data["split"])
        amplitudes = np.asarray(split_data["amplitudes"], dtype=np.float32)
    train_idx = np.flatnonzero(split == 0)
    val_idx = np.flatnonzero(split == 1)
    test_idx = np.flatnonzero(split == 2)
    input_scale, correction_scale, output_scale = _estimate_scales(
        trajectories,
        train_idx,
        scale_samples=args.scale_samples,
    )
    print(
        f"input_scale={input_scale:.7g} "
        f"correction_scale={correction_scale.tolist()} "
        f"output_scale={output_scale.tolist()}",
        flush=True,
    )

    input_scale_jax = jnp.asarray(input_scale, dtype=jnp.float32)
    correction_scale_jax = jnp.asarray(correction_scale)
    output_scale_jax = jnp.asarray(output_scale).reshape(1, 1, 1, 1, 1, 3)
    params = fno.init_params(
        width=args.width,
        modes=args.modes,
        layers=args.layers,
        seed=args.seed,
    )
    if args.init_weights is not None:
        metadata_keys = {
            "input_scale",
            "correction_scale",
            "width",
            "modes",
            "layers",
            "solver_dt",
            "solver_steps",
            "stride",
            "macro_dt",
            "rollout_steps",
            "autoregressive",
        }
        with np.load(args.init_weights, allow_pickle=False) as checkpoint:
            if (
                int(checkpoint["width"]) != args.width
                or int(checkpoint["modes"]) != args.modes
                or int(checkpoint["layers"]) != args.layers
                or int(checkpoint["autoregressive"]) != 1
            ):
                raise ValueError("initial checkpoint architecture mismatch")
            loaded = {
                key: jnp.asarray(checkpoint[key])
                for key in checkpoint.files
                if key not in metadata_keys
            }
        if set(loaded) != set(params):
            raise ValueError(
                "initial checkpoint parameter mismatch: "
                f"missing={sorted(set(params) - set(loaded))}, "
                f"extra={sorted(set(loaded) - set(params))}"
            )
        params = loaded
        print(f"initialized_from={args.init_weights}", flush=True)
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)

    def predict(
        current: dict[str, jax.Array],
        initial: jax.Array,
        horizon: int,
    ) -> jax.Array:
        return fno.rollout(
            current,
            initial,
            steps=horizon,
            input_scale=input_scale_jax,
            correction_scale=correction_scale_jax,
            modes=args.modes,
            layers=args.layers,
        )

    def make_update(horizon: int, phase_updates: int, phase_index: int):
        def loss_fn(
            current: dict[str, jax.Array],
            initial: jax.Array,
            target: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            predicted = predict(current, initial, horizon)
            difference = (predicted - target) / output_scale_jax
            per_time_field = jnp.mean(
                difference**2,
                axis=(0, 2, 3, 4, 5),
            )
            time_weights = jnp.linspace(
                1.0,
                2.0,
                horizon,
                dtype=jnp.float32,
            )
            field_loss = jnp.sum(per_time_field * time_weights) / jnp.sum(time_weights)
            flattened = difference.reshape(-1, fno.N, fno.N, fno.N, 3)
            error_hat = jnp.fft.rfftn(flattened, axes=(1, 2, 3))
            k = jnp.fft.fftfreq(fno.N, d=1.0 / fno.N)
            kz = jnp.fft.rfftfreq(fno.N, d=1.0 / fno.N)
            kx, ky, kz_grid = jnp.meshgrid(k, k, kz, indexing="ij")
            spectral_weight = jnp.sqrt(1.0 + kx**2 + ky**2 + kz_grid**2)
            spectral_loss = jnp.mean(
                jnp.abs(error_hat) ** 2 * spectral_weight[None, ..., None]
            ) / (fno.N**3)
            terminal_loss = jnp.mean(difference[:, -1] ** 2)
            loss = field_loss + 0.02 * spectral_loss + 0.25 * terminal_loss
            loss += 1e-8 * fno.tree_l2(current)
            return loss, (field_loss, spectral_loss, terminal_loss)

        @jax.jit
        def update(
            current: dict[str, jax.Array],
            first: dict[str, jax.Array],
            second: dict[str, jax.Array],
            initial: jax.Array,
            target: jax.Array,
            global_step: jax.Array,
            phase_step: jax.Array,
        ):
            (loss, components), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                current, initial, target
            )
            gradients = jax.tree.map(
                lambda value: jnp.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=1e3,
                    neginf=-1e3,
                ),
                gradients,
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in gradients.values()))
            clip_scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-12))
            gradients = jax.tree.map(
                lambda value: value * clip_scale,
                gradients,
            )
            beta1 = 0.9
            beta2 = 0.999
            first = jax.tree.map(
                lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
                first,
                gradients,
            )
            second = jax.tree.map(
                lambda old, grad: beta2 * old + (1.0 - beta2) * grad**2,
                second,
                gradients,
            )
            global_float = global_step.astype(jnp.float32)
            first_hat = jax.tree.map(
                lambda value: value / (1.0 - beta1**global_float),
                first,
            )
            second_hat = jax.tree.map(
                lambda value: value / (1.0 - beta2**global_float),
                second,
            )
            progress = (phase_step.astype(jnp.float32) - 1.0) / max(
                phase_updates - 1,
                1,
            )
            phase_lr = args.lr * (0.82**phase_index)
            learning_rate = args.min_lr + 0.5 * (phase_lr - args.min_lr) * (
                1.0 + jnp.cos(jnp.pi * progress)
            )
            current = jax.tree.map(
                lambda value, mm, vv: (
                    value - learning_rate * mm / (jnp.sqrt(vv) + 1e-8)
                ),
                current,
                first_hat,
                second_hat,
            )
            return (
                current,
                first,
                second,
                loss,
                components,
                learning_rate,
                grad_norm,
            )

        return update

    predict_full = jax.jit(
        lambda current, initial: predict(
            current,
            initial,
            fno.ROLLOUT_STEPS,
        )
    )

    def evaluate_indices(
        current: dict[str, jax.Array],
        indices: np.ndarray,
        batch_size: int = 4,
    ) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            initial = jnp.asarray(np.asarray(trajectories[batch_indices, 0]))
            predictions.append(np.asarray(predict_full(current, initial)))
        return np.concatenate(predictions)

    validation_subset = val_idx[: min(args.validation_samples, val_idx.size)]
    rng = np.random.default_rng(args.seed + 3)
    best_params: dict[str, Any] = jax.tree.map(
        lambda value: np.asarray(value),
        params,
    )
    best_validation = float("inf")
    best_global_step = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    for phase_index, (horizon, phase_updates) in enumerate(phases):
        print(
            f"phase={phase_index + 1}/{len(phases)} "
            f"horizon={horizon} updates={phase_updates}",
            flush=True,
        )
        update = make_update(horizon, phase_updates, phase_index)
        max_start = fno.ROLLOUT_STEPS - horizon
        for phase_step in range(1, phase_updates + 1):
            global_step += 1
            batch_indices = rng.choice(
                train_idx,
                size=min(args.batch_size, train_idx.size),
                replace=False,
            )
            if max_start:
                starts = rng.integers(0, max_start + 1, size=batch_indices.size)
                starts[0] = 0
            else:
                starts = np.zeros(batch_indices.size, dtype=np.int64)
            initial_np = np.stack(
                [
                    np.asarray(trajectories[index, start], dtype=np.float32)
                    for index, start in zip(batch_indices, starts, strict=True)
                ]
            )
            target_np = np.stack(
                [
                    np.asarray(
                        trajectories[
                            index,
                            start + 1 : start + horizon + 1,
                        ],
                        dtype=np.float32,
                    )
                    for index, start in zip(batch_indices, starts, strict=True)
                ]
            )
            (
                params,
                first_moment,
                second_moment,
                loss,
                components,
                learning_rate,
                grad_norm,
            ) = update(
                params,
                first_moment,
                second_moment,
                jnp.asarray(initial_np),
                jnp.asarray(target_np),
                jnp.asarray(global_step),
                jnp.asarray(phase_step),
            )
            should_validate = (
                phase_step == 1
                or phase_step == phase_updates
                or phase_step % args.validation_interval == 0
            )
            if not should_validate:
                continue
            predicted_val = evaluate_indices(params, validation_subset)
            target_val = np.asarray(
                trajectories[validation_subset, 1:],
                dtype=np.float32,
            )
            validation_errors = _relative_l2(predicted_val, target_val)
            validation_final = float(np.mean(validation_errors[:, -1]))
            validation_mean = float(np.mean(validation_errors))
            if validation_final < best_validation:
                best_validation = validation_final
                best_global_step = global_step
                best_params = jax.tree.map(
                    lambda value: np.asarray(value),
                    params,
                )
            field_loss, spectral_loss, terminal_loss = map(float, components)
            record = {
                "global_step": global_step,
                "phase": phase_index + 1,
                "horizon": horizon,
                "phase_step": phase_step,
                "loss": float(loss),
                "field_loss": field_loss,
                "spectral_loss": spectral_loss,
                "terminal_loss": terminal_loss,
                "validation_rollout_relative_l2_mean": validation_mean,
                "validation_final_relative_l2_mean": validation_final,
                "learning_rate": float(learning_rate),
                "gradient_norm_pre_clip": float(grad_norm),
                "best_global_step": best_global_step,
            }
            history.append(record)
            print(
                " ".join(f"{key}={value}" for key, value in record.items()),
                flush=True,
            )

    best_jax = jax.tree.map(jnp.asarray, best_params)
    predicted_test = evaluate_indices(best_jax, test_idx)
    target_test = np.asarray(trajectories[test_idx, 1:], dtype=np.float32)
    test_error = _relative_l2(predicted_test, target_test)
    test_nonzero = amplitudes[test_idx] >= 0.05
    test_error_nonzero = test_error[test_nonzero]
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        **fno.serialize_params(
            best_params,
            input_scale=input_scale,
            correction_scale=correction_scale,
            width=args.width,
            modes=args.modes,
            layers=args.layers,
        ),
    )
    metrics = {
        "architecture": "autoregressive_fourier_neural_operator",
        "teacher": "xlb_kbc",
        "initialized_from": (
            str(args.init_weights) if args.init_weights is not None else None
        ),
        "dataset_sha256": _sha256(args.dataset),
        "split_sha256": _sha256(split_path),
        "weights_sha256": _sha256(args.weights),
        "samples": int(trajectories.shape[0]),
        "train_samples": int(train_idx.size),
        "validation_samples": int(val_idx.size),
        "test_samples": int(test_idx.size),
        "width": args.width,
        "modes": args.modes,
        "layers": args.layers,
        "solver_steps": fno.SOLVER_STEPS,
        "snapshot_stride_solver_steps": fno.STRIDE,
        "macro_dt": fno.MACRO_DT,
        "rollout_steps": fno.ROLLOUT_STEPS,
        "curriculum": [
            {"unroll_horizon": horizon, "updates": updates}
            for horizon, updates in phases
        ],
        "best_global_step": best_global_step,
        "best_validation_final_relative_l2": best_validation,
        "test_nonzero_samples": int(np.count_nonzero(test_nonzero)),
        "test_rollout_relative_l2_mean": float(np.mean(test_error_nonzero)),
        "test_final_relative_l2_mean": float(np.mean(test_error_nonzero[:, -1])),
        "test_final_relative_l2_median": float(np.median(test_error_nonzero[:, -1])),
        "test_final_relative_l2_p95": float(
            np.quantile(test_error_nonzero[:, -1], 0.95)
        ),
        "test_relative_l2_by_macro_step": np.mean(
            test_error_nonzero,
            axis=0,
        ).tolist(),
        "test_rollout_relative_l2_mean_including_zero_anchors": float(
            np.mean(test_error)
        ),
        "input_scale": input_scale,
        "correction_scale": correction_scale.tolist(),
        "output_scale": output_scale.tolist(),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.weights.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
