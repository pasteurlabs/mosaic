# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate resumable periodic 3D XLB shards and measured throughput metadata.

Use operator_dataset.py to prepare a JSON plan outside the teacher image. This
entry point runs inside the XLB image, directly invoking its native operators.
The legacy fixed-task generator and packaged checkpoint remain reproducible.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from operator_dataset import (
    VERSION,
    file_hash,
    fingerprint,
    make_field,
    snapshot_steps,
    validate_case,
)


def atomic_json(path: Path, payload: dict) -> None:
    """Publish metadata only after the complete file has been flushed."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def completed_shard(path: Path, signature: str) -> dict | None:
    """Verify identity and content on resume; never reuse mismatched data."""
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text())
    if metadata["signature"] != signature:
        raise ValueError(f"resume identity mismatch: {marker}")
    if not path.exists() or file_hash(path) != metadata["sha256"]:
        raise ValueError(f"corrupt or missing completed shard: {path}")
    return metadata


def write_shard(path: Path, arrays: dict, metadata: dict) -> dict:
    """Publish an uncompressed shard, then its hash and completion marker."""
    started = time.perf_counter()
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    metadata = {
        **metadata,
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        "write_seconds": time.perf_counter() - started,
    }
    atomic_json(path.with_suffix(".json"), metadata)
    return metadata


def load_teacher(path: Path) -> Any:
    """Load mounted teacher source before initializing the JAX backend."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("operator_xlb_teacher", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def teacher_policy(api: Any, case: dict, use_f64: bool) -> tuple[int, str]:
    """Reuse the API's static policy and verify the requested operator exists."""
    dummy = np.broadcast_to(np.float32(0), (case["N"],) * 3 + (3,))
    substeps, collision = api._diff_statics(
        {
            "v0": dummy,
            "viscosity": case["viscosity"],
            "dt": case["dt"],
            "domain_extent": case["domain_extent"],
        }
    )
    if (3, use_f64, collision) not in api._OPS:
        raise ValueError(
            f"teacher has no 3D {collision} operators at requested precision"
        )
    return substeps, collision


def lattice_parameters(case: dict, substeps: int, use_f64: bool) -> tuple:
    """Match Python-scalar unit conversion in xlb_fwd before casting constants."""
    dtype = np.float64 if use_f64 else np.float32
    dx = case["domain_extent"] / case["N"]
    dt_eff = case["dt"] / substeps
    scale = dt_eff / dx
    omega = 1.0 / (3.0 * case["viscosity"] * dt_eff / dx**2 + 0.5)
    return dtype(scale), dtype(omega)


def native_runner(api: Any, case: dict, use_f64: bool, snapshots: list[int]) -> Any:
    """Return a batched continuous-population rollout, with dynamic physics."""
    import jax
    import jax.numpy as jnp

    substeps, collision = teacher_policy(api, case, use_f64)
    ops = api._OPS[(3, use_f64, collision)]
    n = case["N"]
    dtype = jnp.float64 if use_f64 else jnp.float32
    intervals = jnp.asarray(np.diff(snapshots) * substeps, dtype=jnp.int32)

    def one(v0: Any, scale: Any, omega: Any):
        velocity = jnp.moveaxis(v0, -1, 0).astype(dtype) * scale
        rho = jnp.ones((1, n, n, n), dtype=dtype)
        initial = ops["eq"](rho, velocity)

        def step(_: int, populations: Any):
            streamed = ops["stream"](populations)
            density, u = ops["macro"](streamed)
            equilibrium = ops["eq"](density, u)
            return ops["bgk"](streamed, equilibrium, density, u, omega)

        def advance(populations: Any, length: Any):
            updated = jax.lax.fori_loop(0, length, step, populations)
            _, u = ops["macro"](updated)
            decoded = (jnp.moveaxis(u, 0, -1) / scale).astype(jnp.float32)
            return updated, decoded

        _, fields = jax.lax.scan(advance, initial, intervals)
        return jnp.concatenate([v0[None], fields], axis=0)

    return jax.jit(jax.vmap(one, in_axes=(0, None, None)))


def compare_fields(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Return finite comparison metrics, or an explicit reference failure."""
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        return {"finite": False, "max_abs": None, "relative_l2": None}
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / max(norm, 1e-12)),
    }


def parity_checks(
    api: Any,
    case: dict,
    use_f64: bool,
    fields: np.ndarray,
    trajectory: np.ndarray,
    snapshots: list[int],
) -> dict:
    """Check interior and final native snapshots against the public apply path."""
    import jax
    import jax.numpy as jnp

    substeps, collision = teacher_policy(api, case, use_f64)
    positions = sorted({1, len(snapshots) // 2, len(snapshots) - 1})
    checks = []
    for position in positions:
        steps = snapshots[position]
        # Same function used by public apply(), with matched precision and policy.
        reference = jax.jit(
            lambda v, horizon=steps: api.xlb_fwd(
                v,
                case["viscosity"],
                case["dt"],
                horizon,
                domain_extent=case["domain_extent"],
                _use_f64=use_f64,
                _sub_k=substeps,
                _collision_kind_override=collision,
            )[0]
        )(jnp.asarray(fields[0]))
        metrics = compare_fields(trajectory[0, position], np.asarray(reference))
        checks.append({"step": steps, **metrics})
        if not metrics["finite"] or metrics["max_abs"] > 2e-5:
            raise RuntimeError(
                f"teacher trajectory parity failed at {steps}: {metrics}"
            )
    # Compare to the higher-precision teacher on multiple IC families. This is
    # evidence, not automatic permission to use float32 for all future data.
    reference64 = jax.jit(
        jax.vmap(
            lambda v: api.xlb_fwd(
                v,
                case["viscosity"],
                case["dt"],
                case["steps"],
                domain_extent=case["domain_extent"],
                _use_f64=True,
                _sub_k=substeps,
                _collision_kind_override=collision,
            )[0]
        )
    )(jnp.asarray(fields))
    precision = [
        compare_fields(a, b)
        for a, b in zip(trajectory[:, -1], np.asarray(reference64), strict=True)
    ]
    return {"native_vs_api": checks, "final_vs_float64": precision}


def run_case(api: Any, case: dict, args: argparse.Namespace, provenance: dict) -> None:
    """Generate one case with bounded buffered writes and verifiable resume."""
    import jax
    import jax.numpy as jnp

    validate_case(case)
    snapshots = case.get("snapshot_steps", snapshot_steps(case["steps"]))
    if (
        not snapshots
        or any(type(step) is not int for step in snapshots)
        or snapshots != sorted(set(snapshots))
        or snapshots[0] != 0
        or snapshots[-1] != case["steps"]
    ):
        raise ValueError("snapshot_steps must increase from zero to steps")
    samples = case["samples"]
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be positive")
    use_f64 = args.precision == "float64"
    substeps, collision = teacher_policy(api, case, use_f64)
    identity = {
        "version": VERSION,
        "case": case,
        "seed": args.seed,
        "precision": args.precision,
        "shard_size": args.shard_size,
        "substeps": substeps,
        "collision": collision,
        "provenance": provenance,
    }
    signature = fingerprint(identity)
    root = args.output / case["id"] / args.precision
    root.mkdir(parents=True, exist_ok=True)
    physics = tuple(jnp.asarray(v) for v in lattice_parameters(case, substeps, use_f64))
    runner = None
    compile_seconds = 0.0
    warmup_seconds = 0.0
    checks = None
    invocation = time.perf_counter()
    pending = None
    reports = []
    with contextlib.ExitStack() as locks, ThreadPoolExecutor(max_workers=1) as writer:
        for shard, start in enumerate(range(0, samples, args.shard_size)):
            if shard % args.workers != args.worker:
                continue
            path = root / f"shard-{shard:05d}.npz"
            # Advisory locks are released on process death; marker publication is
            # atomic. Hold each lock until its asynchronous write is complete.
            lock = locks.enter_context(path.with_suffix(".lock").open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = completed_shard(path, signature)
            if existing is not None:
                reports.append({**existing, "reused": True})
                lock.close()
                continue
            stop = min(start + args.shard_size, samples)
            trajectories, records = [], []
            compute_seconds = transfer_seconds = ic_seconds = 0.0
            for batch_start in range(start, stop, args.batch_size):
                t0 = time.perf_counter()
                count = min(args.batch_size, stop - batch_start)
                batch = [
                    make_field(case["N"], args.seed, i)
                    for i in range(batch_start, batch_start + count)
                ]
                fields = np.stack([b[0] for b in batch])
                records.extend(b[1] for b in batch)
                # Pad the last batch; it is never written or counted as data.
                padded = np.concatenate(
                    [fields, np.repeat(fields[-1:], args.batch_size - count, axis=0)]
                )
                device_fields = jax.device_put(padded)
                device_fields.block_until_ready()
                ic_seconds += time.perf_counter() - t0
                if runner is None:
                    function = native_runner(api, case, use_f64, snapshots)
                    t0 = time.perf_counter()
                    runner = function.lower(device_fields, *physics).compile()
                    compile_seconds = time.perf_counter() - t0
                    print(
                        json.dumps(
                            {
                                "phase": "compiled",
                                "case": case["id"],
                                "seconds": compile_seconds,
                            }
                        ),
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    warm = runner(device_fields, *physics)
                    warm.block_until_ready()
                    warmup_seconds = time.perf_counter() - t0
                    checks = parity_checks(
                        api, case, use_f64, padded, np.asarray(warm), snapshots
                    )
                t0 = time.perf_counter()
                result = runner(device_fields, *physics)
                result.block_until_ready()
                compute_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                trajectories.append(np.asarray(result)[:count])
                transfer_seconds += time.perf_counter() - t0
            data = np.concatenate(trajectories)
            valid = np.isfinite(data).all(axis=tuple(range(1, data.ndim)))
            arrays = {
                "velocity": data,
                "snapshot_steps": np.asarray(snapshots),
                "valid": valid,
                "parent_id": np.asarray([r["parent_id"] for r in records]),
                "split": np.asarray([r["split"] for r in records]),
                "family": np.asarray([r["family"] for r in records]),
                "amplitude": np.asarray([r["amplitude"] for r in records]),
            }
            metadata = {
                **identity,
                "signature": signature,
                "shard": shard,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "valid_count": int(valid.sum()),
                "invalid_count": int((~valid).sum()),
                "invalid_parent_ids": [
                    r["parent_id"]
                    for r, ok in zip(records, valid, strict=True)
                    if not ok
                ],
                "batch_size": args.batch_size,
                "compute_seconds": compute_seconds,
                "transfer_seconds": transfer_seconds,
                "ic_seconds": ic_seconds,
                "native_cell_steps_per_second": (stop - start)
                * case["N"] ** 3
                * case["steps"]
                * substeps
                / compute_seconds,
                "trajectory_checks": checks,
                "device_memory": jax.devices()[0].memory_stats(),
            }
            # Bounded queue: at most one outstanding shard plus the current one.
            if pending is not None:
                reports.append(pending.result())
            pending = writer.submit(write_shard, path, arrays, metadata)
            pending.add_done_callback(lambda _future, handle=lock: handle.close())
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "precision": args.precision,
                        "shard": shard,
                        "valid": int(valid.sum()),
                        "compute_seconds": compute_seconds,
                    }
                ),
                flush=True,
            )
        if pending is not None:
            reports.append(pending.result())
    summary = {
        **identity,
        "signature": signature,
        "worker": args.worker,
        "workers": args.workers,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "invocation_seconds": time.perf_counter() - invocation,
        "shards": [
            {
                k: r[k]
                for k in (
                    "shard",
                    "count",
                    "valid_count",
                    "invalid_count",
                    "compute_seconds",
                    "transfer_seconds",
                    "ic_seconds",
                    "write_seconds",
                    "bytes",
                )
            }
            | {"reused": r.get("reused", False)}
            for r in reports
        ],
    }
    atomic_json(root / f"worker-{args.worker:03d}.json", summary)


def main() -> None:
    """Run selected manifest cases on one GPU; distribute with worker indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-api", type=Path, default=Path("/tesseract/tesseract_api.py")
    )
    parser.add_argument(
        "--case", action="append", help="Case ID; repeat to select multiple"
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    from operator_storage import require_local

    # Fail before imports, hashes or shard scans on a network mount.
    for path in (Path(__file__), args.manifest, args.teacher_api, args.output):
        require_local(path)
    args.cache_dir = args.cache_dir or args.output / "compilation-cache"
    require_local(args.cache_dir)
    if (
        min(args.batch_size, args.shard_size, args.workers) < 1
        or not 0 <= args.worker < args.workers
    ):
        parser.error("positive sizes and 0 <= worker < workers required")
    plan = json.loads(args.manifest.read_text())
    if plan["version"] != VERSION:
        parser.error("unsupported manifest version")
    args.seed = plan["seed"] if args.seed is None else args.seed
    cases = [c for c in plan["cases"] if args.case is None or c["id"] in args.case]
    if not cases or (args.case and set(args.case) - {c["id"] for c in cases}):
        parser.error("unknown/empty case selection")
    api = load_teacher(args.teacher_api)
    import jax

    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(args.cache_dir))
    provenance = {
        "teacher_api_sha256": file_hash(args.teacher_api),
        "generator_sha256": file_hash(Path(__file__)),
        "storage_policy_sha256": file_hash(
            Path(__file__).with_name("operator_storage.py")
        ),
        "distribution_sha256": file_hash(
            Path(__file__).with_name("operator_dataset.py")
        ),
        "jax_version": jax.__version__,
        "xlb_version": importlib.metadata.version("xlb"),
        "image_identity": os.environ.get("MOSAIC_TEACHER_IMAGE_ID", "unrecorded"),
        "substepping_disabled": os.environ.get("XLB_SUB_K_DISABLE", "1"),
        "teacher_vjp_fp32": os.environ.get("XLB_VJP_FP32", "0"),
    }
    for case in cases:
        try:
            run_case(api, case, args, provenance)
        except Exception as exc:
            failure_dir = args.output / case["id"] / args.precision
            failure_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                failure_dir / f"failure-{args.worker}-{time.time_ns()}.json",
                {
                    "case": case,
                    "precision": args.precision,
                    "provenance": provenance,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise


if __name__ == "__main__":
    main()
