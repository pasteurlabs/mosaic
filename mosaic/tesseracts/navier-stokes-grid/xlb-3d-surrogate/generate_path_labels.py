# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache XLB VJPs at surrogate recovery iterates, split by source trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import surrogate_model as fno
from generate_trajectories import _load_api, _sha256
from scipy.optimize import minimize


def main() -> None:
    """Generate the recovery-path pilot labels inside the XLB environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-api", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples) < 1:
        parser.error("both splits need at least one source trajectory")
    jax.config.update("jax_enable_x64", True)
    api = _load_api(args.teacher_api)

    @jax.jit
    def teacher(x: jax.Array) -> jax.Array:
        return api.xlb_fwd(
            x.astype(jnp.float64),
            viscosity=0.01,
            dt=0.02,
            steps=100,
            domain_extent=2 * np.pi,
            _use_f64=True,
            _sub_k=1,
            _collision_kind_override="kbc",
        )[0]

    with np.load(args.init_weights) as data:
        checkpoint = {key: data[key] for key in data.files}
    width, modes, layers = (int(checkpoint[k]) for k in ("width", "modes", "layers"))
    params = {
        key: jnp.asarray(checkpoint[key])
        for key in fno.init_params(width=width, modes=modes, layers=layers, seed=0)
    }
    if "w_linear" in checkpoint:
        params["w_linear"] = jnp.asarray(checkpoint["w_linear"])

    @jax.jit
    def forward(x: jax.Array) -> jax.Array:
        return fno.rollout(
            params,
            x.astype(jnp.float32)[None],
            steps=fno.ROLLOUT_STEPS,
            input_scale=jnp.asarray(checkpoint["input_scale"]),
            correction_scale=jnp.asarray(checkpoint["correction_scale"]),
            modes=modes,
            layers=layers,
        )[0, -1]

    @jax.jit
    def value_grad(z: jax.Array, target: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.value_and_grad(
            lambda value: jnp.mean((forward(value.reshape(target.shape)) - target) ** 2)
        )(z)

    @jax.jit
    def label(x: jax.Array, q: jax.Array) -> tuple[jax.Array, jax.Array]:
        y, back = jax.vjp(teacher, x.astype(jnp.float64))
        return y, back(q.astype(y.dtype))[0]

    with np.load(args.dataset.with_suffix(".split.npz")) as data:
        split, amplitudes = data["split"], data["amplitudes"]
    selected = []
    for part, count in ((0, args.train_samples), (1, args.validation_samples)):
        available = np.flatnonzero((split == part) & (amplitudes > 0.4))
        if len(available) < count:
            raise ValueError(f"not enough amplitude > 0.4 trajectories in split {part}")
        selected.append(available[:count])
    indices = np.concatenate(selected)
    trajectories = np.load(args.dataset, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    records, recoveries = [], []
    for idx in indices:
        truth = np.asarray(trajectories[idx, 0], dtype=np.float32)
        target = np.asarray(teacher(truth), dtype=np.float32)
        states = [np.zeros_like(truth)]
        iteration = 0

        def objective(
            z: np.ndarray, target: np.ndarray = target
        ) -> tuple[float, np.ndarray]:
            value, grad = value_grad(jnp.asarray(z), jnp.asarray(target))
            return float(value), np.asarray(grad, dtype=np.float64)

        def callback(
            z: np.ndarray,
            states: list[np.ndarray] = states,
            shape: tuple[int, ...] = truth.shape,
        ) -> None:
            nonlocal iteration
            iteration += 1
            if iteration in (1, 2, 5, 10, 20, 50, 100):
                states.append(z.reshape(shape).astype(np.float32))

        result = minimize(
            objective,
            np.zeros(truth.size),
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": 100,
                "maxls": 30,
                "gtol": 1e-12,
                "ftol": 1e-15,
                "maxcor": 10,
            },
        )
        # Retain the final state even if it duplicates a scheduled snapshot.
        states.append(result.x.reshape(truth.shape).astype(np.float32))
        recoveries.append(
            {
                "index": int(idx),
                "iterations": int(result.nit),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        for i, x in enumerate(states):
            q = (
                rng.normal(size=x.shape).astype(np.float32)
                if i % 2 == 0
                else np.asarray(teacher(x)) - target
            )
            if np.linalg.norm(q) < 1e-12:
                q = rng.normal(size=x.shape).astype(np.float32)
            q = (q / np.linalg.norm(q)).astype(np.float32)
            y, vjp = label(x, q)
            if not all(np.all(np.isfinite(v)) for v in (x, q, y, vjp)):
                raise RuntimeError("nonfinite path label")
            records.append(
                (
                    x,
                    np.asarray(y, dtype=np.float32),
                    q,
                    np.asarray(vjp, dtype=np.float32),
                    split[idx],
                    idx,
                )
            )
        print(f"Labeled {idx}: split {split[idx]}, {len(states)} states", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = ("initial", "target", "cotangent", "teacher_vjp", "split", "indices")
    np.savez(
        args.output,
        **{key: np.asarray([row[i] for row in records]) for i, key in enumerate(keys)},
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "labels_sha256": _sha256(args.output),
                "initial_weights_sha256": _sha256(args.init_weights),
                "dataset": str(args.dataset),
                "split_sha256": _sha256(args.dataset.with_suffix(".split.npz")),
                "teacher_api_sha256": _sha256(args.teacher_api),
                "source_indices": indices.tolist(),
                "seed": args.seed,
                "train_cases": args.train_samples,
                "validation_cases": args.validation_samples,
                "cotangents": "alternating random and teacher recovery residual",
                "states": len(records),
                "recoveries": recoveries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
