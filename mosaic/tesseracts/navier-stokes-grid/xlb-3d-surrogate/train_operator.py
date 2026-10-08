# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the conditioned operator entirely on node-local data and checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import operator_model as model
from operator_dataset import file_hash
from operator_loader import WindowDataset
from operator_storage import atomic_json, require_local


def trajectory_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Resolution-normalized field and spectral losses with a near-zero floor."""
    error = prediction - target
    energy = jnp.maximum(jnp.mean(target**2, axis=(2, 3, 4, 5)), 1e-4)
    field = jnp.mean(error**2, axis=(2, 3, 4, 5)) / energy
    spectrum = jnp.fft.rfftn(error, axes=(2, 3, 4), norm="forward")
    shape = error.shape[2:5]
    kx, ky, kz = jnp.meshgrid(
        jnp.fft.fftfreq(shape[0]) * shape[0],
        jnp.fft.fftfreq(shape[1]) * shape[1],
        jnp.fft.rfftfreq(shape[2]) * shape[2],
        indexing="ij",
    )
    weights = jnp.sqrt(1 + kx**2 + ky**2 + kz**2)
    multiplicity = jnp.full((shape[2] // 2 + 1,), 2.0).at[0].set(1.0)
    if shape[2] % 2 == 0:
        multiplicity = multiplicity.at[-1].set(1.0)
    spectral = (
        jnp.sum(
            jnp.abs(spectrum) ** 2 * (weights * multiplicity)[None, ..., None],
            axis=(2, 3, 4, 5),
        )
        / 3
        / energy
    )
    return (
        0.8 * jnp.mean(field) + 0.2 * jnp.mean(field[:, -1]) + 0.01 * jnp.mean(spectral)
    )


def make_update(args: argparse.Namespace, updates: int, span: int) -> Any:
    """Compile Adam updates by unroll length; physics values remain dynamic."""

    def loss(
        params: dict,
        initial: jax.Array,
        target: jax.Array,
        nu: jax.Array,
        dt: jax.Array,
        extent: jax.Array,
    ) -> jax.Array:
        predicted = model.rollout(
            params,
            initial,
            nu,
            dt,
            extent,
            updates=updates,
            span=span,
            layers=args.layers,
            project_output=args.project_output,
            conserve_energy=getattr(args, "conserve_energy", False),
        )
        return trajectory_loss(predicted, target)

    @jax.jit
    def update(
        params: dict,
        first: dict,
        second: dict,
        initial: jax.Array,
        target: jax.Array,
        nu: jax.Array,
        dt: jax.Array,
        extent: jax.Array,
        step: jax.Array,
    ):
        value, grads = jax.value_and_grad(loss)(params, initial, target, nu, dt, extent)
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads.values()))
        finite = jnp.isfinite(value) & jnp.isfinite(norm)
        clip = jnp.minimum(1.0, 1 / (norm + 1e-12))
        grads = jax.tree.map(lambda g: g * clip, grads)
        first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grads)
        second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2, second, grads)
        rate = args.lr * (
            0.1
            + 0.9 * 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step / args.updates, 1.0)))
        )
        params = jax.tree.map(
            lambda p, m, v: (
                p
                - rate
                * (m / (1 - 0.9**step))
                / (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8)
            ),
            params,
            first,
            second,
        )
        return params, first, second, value, norm, finite

    return update


def save_checkpoint(
    path: Path, params: dict, first: dict, second: dict, metadata: dict
) -> None:
    """Save complete optimizer/RNG state locally using an atomic replacement."""
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            **{f"param_{k}": np.asarray(v) for k, v in params.items()},
            **{f"first_{k}": np.asarray(v) for k, v in first.items()},
            **{f"second_{k}": np.asarray(v) for k, v in second.items()},
            metadata=np.asarray(json.dumps(metadata, allow_nan=False)),
        )
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple:
    """Restore optimizer arrays and the exact sampling stream."""
    require_local(path)
    with np.load(path, allow_pickle=False) as data:
        groups = [
            {
                k[len(prefix) :]: jnp.asarray(data[k])
                for k in data.files
                if k.startswith(prefix)
            }
            for prefix in ("param_", "first_", "second_")
        ]
        metadata = json.loads(str(data["metadata"]))
    return (*groups, metadata)


def evaluate(
    params: dict,
    dataset: WindowDataset,
    case_ids: list[str],
    args: argparse.Namespace,
    split: str,
    functions: dict,
) -> tuple[float, list]:
    """Evaluate fixed parents at their full saved horizon, streaming aggregates."""
    records = []
    for case_id in case_ids:
        case = dataset.cases[case_id]
        initial, target, parents = dataset.endpoints(
            case_id, split, args.validation_samples
        )
        key = (case["N"], case["steps"], args.eval_stride, len(initial))
        if key not in functions:
            steps = case["steps"]
            functions[key] = jax.jit(
                lambda p, u, nu, dt, L, horizon=steps: model.predict(
                    p,
                    u,
                    nu,
                    dt,
                    L,
                    steps=horizon,
                    stride=args.eval_stride,
                    layers=args.layers,
                    project_output=args.project_output,
                    conserve_energy=getattr(args, "conserve_energy", False),
                )
            )
        predicted = np.asarray(
            functions[key](
                params,
                jnp.asarray(initial),
                *[jnp.float32(case[k]) for k in ("viscosity", "dt", "domain_extent")],
            )
        )
        finite = bool(np.isfinite(predicted).all())
        if finite:
            axes = tuple(range(1, target.ndim))
            rms = np.sqrt(
                np.mean((predicted - target) ** 2, axis=axes, dtype=np.float64)
            )
            target_rms = np.sqrt(np.mean(target**2, axis=axes, dtype=np.float64))
            relative = rms / np.maximum(target_rms, 0.01)
            score = float(np.mean(relative))
        else:
            score = None
        records.append(
            {
                "case_id": case_id,
                "N": case["N"],
                "steps": case["steps"],
                "split": split,
                "parents": parents,
                "finite": finite,
                "normalized_rms_error": score,
                "absolute_rms_error": float(np.mean(rms)) if finite else None,
                "parent_errors": relative.tolist() if finite else None,
            }
        )
    score = (
        float(np.mean([r["normalized_rms_error"] for r in records]))
        if all(r["finite"] for r in records)
        else math.inf
    )
    return score, records


def main() -> None:
    """Run a resumable curriculum with local-only training I/O and finite checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--project-output", action="store_true")
    parser.add_argument("--conserve-energy", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.updates,
            args.batch_size,
            args.validation_interval,
            args.validation_samples,
            args.eval_stride,
        )
        < 1
    ):
        parser.error("positive update/batch/validation settings required")
    storage = [
        require_local(p)
        for p in (Path(__file__), args.dataset, args.output, args.data_cache)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update(
        "jax_compilation_cache_dir", str(args.output / "compilation-cache")
    )
    dataset = WindowDataset(args.dataset, args.data_cache)
    if args.manifest:
        require_local(args.manifest)
        expected_cases = {
            c["id"]: c for c in json.loads(args.manifest.read_text())["cases"]
        }
        if expected_cases != dataset.cases:
            raise ValueError("dataset cases do not match the requested manifest")
    training_cases = [
        key for key, c in dataset.cases.items() if c.get("role", "train") == "train"
    ]
    holdout_cases = [
        key for key, c in dataset.cases.items() if c.get("role") == "resolution_holdout"
    ]
    if not training_cases:
        raise ValueError("no training cases")
    rng = np.random.default_rng(args.seed)
    params = model.init_params(args.width, args.modes, args.layers, args.seed)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    start_step, best_score = 0, math.inf
    history, baseline = [], None
    settings = {
        k: v for k, v in vars(args).items() if not isinstance(v, Path) and k != "resume"
    }
    identity = {
        "model_version": model.VERSION,
        "settings": settings,
        "dataset_identity": dataset.index["identity"],
        "model_sha256": file_hash(Path(model.__file__)),
        "trainer_sha256": file_hash(Path(__file__)),
    }
    if args.resume:
        params, first, second, checkpoint = load_checkpoint(args.resume)
        if checkpoint["identity"] != identity:
            raise ValueError("resume configuration/data/source mismatch")
        start_step = checkpoint["step"]
        if not 0 <= start_step <= args.updates:
            raise ValueError("resume step is outside the training budget")
        best_path = args.output / "best.npz"
        if not best_path.exists():
            raise ValueError("resume requires best.npz; restore the full output bundle")
        with np.load(best_path, allow_pickle=False) as saved:
            best_metadata = json.loads(str(saved["metadata"]))
        if (
            best_metadata["identity"] != identity
            or best_metadata["step"] > start_step
            or best_metadata["best_score"] != checkpoint["best_score"]
        ):
            raise ValueError("resume best checkpoint does not match the saved run")
        best_score = (
            checkpoint["best_score"]
            if checkpoint["best_score"] is not None
            else math.inf
        )
        rng.bit_generator.state = checkpoint["rng"]
        history_path = args.output / "history.json"
        if history_path.exists():
            saved = json.loads(history_path.read_text())
            if saved["identity"] != identity:
                raise ValueError("resume history does not match the saved run")
            history = [r for r in saved["history"] if r["step"] <= start_step]
        report_path = args.output / "report.json"
        if report_path.exists():
            completed = json.loads(report_path.read_text())
            if completed["identity"] != identity:
                raise ValueError("resume report does not match the saved run")
            baseline = completed["diffusion_validation"]
            if (
                start_step == args.updates
                and completed["steps_completed"] == args.updates
                and not completed["interrupted"]
            ):
                print(
                    "Requested training already complete; retained report.", flush=True
                )
                return
    stop_requested = [False]

    def stop_handler(_signal: int, _frame: Any) -> None:
        stop_requested[0] = True

    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    started = time.perf_counter()
    functions, update_functions = {}, {}
    if not args.resume:
        diffusion = dict(params) | {"out": jnp.zeros_like(params["out"])}
        baseline_score, baseline = evaluate(
            diffusion, dataset, training_cases, args, "validation", functions
        )
        best_score, initial_records = evaluate(
            params, dataset, training_cases, args, "validation", functions
        )
        history.append(
            {
                "step": 0,
                "score": best_score if math.isfinite(best_score) else None,
                "cases": initial_records,
                "diffusion_score": baseline_score,
            }
        )
        checkpoint = {
            "identity": identity,
            "step": 0,
            "best_score": best_score if math.isfinite(best_score) else None,
            "rng": rng.bit_generator.state,
        }
        save_checkpoint(args.output / "best.npz", params, first, second, checkpoint)
        print(json.dumps(history[-1]), flush=True)

    # Only fixed-shape update graphs are cached. Sampling/prefetch stays on local
    # mmaps; each batch records the post-sampling RNG for exact checkpoint resume.
    def prepare(step: int) -> tuple:
        fraction = step / args.updates
        updates = 1 if fraction <= 0.4 else 2 if fraction <= 0.7 else 4
        span = 1 if rng.random() < 0.75 else 5
        case_id = training_cases[int(rng.integers(len(training_cases)))]
        initial, target = dataset.sample(
            case_id, "train", args.batch_size, updates, span, rng
        )
        case = dataset.cases[case_id]
        rng_state = json.loads(json.dumps(rng.bit_generator.state))
        return initial, target, case, updates, span, rng_state

    last_step = start_step
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = (
            pool.submit(prepare, start_step + 1) if start_step < args.updates else None
        )
        for step in range(start_step + 1, args.updates + 1):
            initial, target, case, updates, span, rng_state = pending.result()
            pending = pool.submit(prepare, step + 1) if step < args.updates else None
            key = (updates, span)
            if key not in update_functions:
                update_functions[key] = make_update(args, updates, span)
            params, first, second, loss, norm, finite = update_functions[key](
                params,
                first,
                second,
                jnp.asarray(initial),
                jnp.asarray(target),
                *[jnp.float32(case[k]) for k in ("viscosity", "dt", "domain_extent")],
                jnp.float32(step),
            )
            if not bool(finite):
                atomic_json(
                    args.output / "failure.json",
                    {
                        "step": step,
                        "case": case,
                        "error": "nonfinite training loss/gradient",
                    },
                )
                raise FloatingPointError(
                    "nonfinite training loss/gradient; previous checkpoint retained"
                )
            last_step = step
            validate = step % args.validation_interval == 0 or step == args.updates
            if validate or stop_requested[0]:
                score, records = (
                    evaluate(
                        params, dataset, training_cases, args, "validation", functions
                    )
                    if validate
                    else (math.inf, [])
                )
                improved = score < best_score
                if improved:
                    best_score = score
                checkpoint = {
                    "identity": identity,
                    "step": step,
                    "best_score": best_score if math.isfinite(best_score) else None,
                    "rng": rng_state,
                }
                save_checkpoint(
                    args.output / "latest.npz", params, first, second, checkpoint
                )
                if improved:
                    save_checkpoint(
                        args.output / "best.npz", params, first, second, checkpoint
                    )
                record = {
                    "step": step,
                    "loss": float(loss),
                    "gradient_norm": float(norm),
                    "score": score if math.isfinite(score) else None,
                    "cases": records,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(record)
                atomic_json(
                    args.output / "history.json",
                    {"identity": identity, "history": history},
                )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "cases"}),
                    flush=True,
                )
            if stop_requested[0]:
                break
    final = {
        "identity": identity,
        "steps_completed": last_step,
        "interrupted": stop_requested[0],
        "elapsed_seconds": time.perf_counter() - started,
        "storage": storage,
        "device": jax.devices()[0].device_kind,
        "memory": jax.devices()[0].memory_stats(),
        "history": history,
        "diffusion_validation": baseline,
    }
    if not stop_requested[0]:
        best, _, _, checkpoint = load_checkpoint(args.output / "best.npz")
        _, test = evaluate(best, dataset, training_cases, args, "test", functions)
        _, holdout = (
            evaluate(best, dataset, holdout_cases, args, "test", functions)
            if holdout_cases
            else (None, [])
        )
        final.update(
            {
                "selected_step": checkpoint["step"],
                "test": test,
                "resolution_holdout_test": holdout,
            }
        )
    atomic_json(args.output / "report.json", final)


if __name__ == "__main__":
    main()
