# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate native XLB trajectories for autoregressive full-field training.

The teacher populations are evolved continuously for all 100 LBM steps.  We
decode velocity snapshots every ``STRIDE`` steps without reinitialising the
hidden lattice populations from equilibrium between snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from training_data import make_inputs

N = 16
VISCOSITY = 0.01
DT = 0.02
STEPS = 100
STRIDE = 5
ROLLOUT_STEPS = STEPS // STRIDE
DOMAIN_EXTENT = 2.0 * math.pi


def _load_api(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("trajectory_teacher_api", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Generate and persist a deterministic train/validation/test dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--teacher-api",
        type=Path,
        default=Path("/tesseract/tesseract_api.py"),
        help="XLB teacher API; run this script in the XLB solver image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/surrogate-output/recovery_3d_xlb_trajectories.npy"),
    )
    args = parser.parse_args()
    if STEPS % STRIDE:
        raise ValueError(f"steps={STEPS} must be divisible by stride={STRIDE}")

    started = time.perf_counter()
    print("jax_backend", jax.default_backend(), flush=True)
    print("jax_devices", jax.devices(), flush=True)
    api = _load_api(args.teacher_api)
    inputs, amplitudes, families = make_inputs(args.samples, args.seed)

    dx = DOMAIN_EXTENT / N
    scale = DT / dx
    nu_lb = VISCOSITY * DT / dx**2
    omega = 1.0 / (3.0 * nu_lb + 0.5)
    ops = api._OPS[(3, False, "kbc")]
    xlb_eq = ops["eq"]
    xlb_stream = ops["stream"]
    xlb_macro = ops["macro"]
    xlb_collide = ops["bgk"]

    def one_trajectory(v0: jax.Array) -> jax.Array:
        u0 = jnp.moveaxis(v0, -1, 0).astype(jnp.float32) * scale
        rho0 = jnp.ones((1, N, N, N), dtype=jnp.float32)
        f0 = xlb_eq(rho0, u0)

        def lbm_step(_: int, f: jax.Array) -> jax.Array:
            f_streamed = xlb_stream(f)
            rho, velocity = xlb_macro(f_streamed)
            equilibrium = xlb_eq(rho, velocity)
            return xlb_collide(
                f_streamed,
                equilibrium,
                rho,
                velocity,
                omega,
            )

        def macro_step(
            f: jax.Array,
            _: None,
        ) -> tuple[jax.Array, jax.Array]:
            f_next = jax.lax.fori_loop(0, STRIDE, lbm_step, f)
            _, velocity = xlb_macro(f_next)
            decoded = jnp.moveaxis(velocity, 0, -1) / scale
            return f_next, decoded.astype(jnp.float32)

        _, decoded = jax.lax.scan(
            macro_step,
            f0,
            None,
            length=ROLLOUT_STEPS,
        )
        return jnp.concatenate([v0[None], decoded], axis=0)

    run_batch = jax.jit(jax.vmap(one_trajectory))

    # Prove that snapshotting did not alter the exact teacher trajectory.
    check_input = jnp.asarray(inputs[0])
    check_trajectory = run_batch(check_input[None])[0]
    check_final, _ = api.xlb_fwd(
        v0=check_input,
        viscosity=VISCOSITY,
        dt=DT,
        steps=STEPS,
        domain_extent=DOMAIN_EXTENT,
        _use_f64=False,
        _sub_k=1,
        _collision_kind_override="kbc",
    )
    check_max_abs = float(jnp.max(jnp.abs(check_trajectory[-1] - check_final)))
    print(f"native_final_check_max_abs={check_max_abs:.9g}", flush=True)
    if check_max_abs > 2e-5:
        raise RuntimeError(
            "native trajectory final snapshot does not match xlb_fwd: "
            f"max_abs={check_max_abs}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectories = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float32,
        shape=(args.samples, ROLLOUT_STEPS + 1, N, N, N, 3),
    )
    cursor = 0
    while cursor < args.samples:
        stop = min(cursor + args.batch_size, args.samples)
        result = np.asarray(jax.device_get(run_batch(jnp.asarray(inputs[cursor:stop]))))
        expected = trajectories[cursor:stop].shape
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"invalid teacher batch {cursor}:{stop}: "
                f"shape={result.shape}, expected={expected}, "
                f"finite={np.all(np.isfinite(result))}"
            )
        trajectories[cursor:stop] = result
        cursor = stop
        elapsed = time.perf_counter() - started
        print(
            f"generated={cursor}/{args.samples} "
            f"rate={cursor / max(elapsed, 1e-9):.2f}/s",
            flush=True,
        )
    trajectories.flush()
    del trajectories

    permutation = np.random.default_rng(args.seed + 2).permutation(args.samples)
    train_count = math.floor(args.samples * 0.75)
    validation_count = math.floor(args.samples * 0.125)
    split = np.full(args.samples, 2, dtype=np.int8)
    split[permutation[:train_count]] = 0
    split[permutation[train_count : train_count + validation_count]] = 1
    split_path = args.output.with_suffix(".split.npz")
    np.savez(
        split_path,
        amplitudes=amplitudes,
        families=families,
        split=split,
    )
    metadata = {
        "teacher": "xlb",
        "collision": "kbc",
        "samples": args.samples,
        "split": {
            "train": train_count,
            "validation": validation_count,
            "test": args.samples - train_count - validation_count,
        },
        "seed": args.seed,
        "physics": {
            "N": N,
            "viscosity": VISCOSITY,
            "dt": DT,
            "solver_steps": STEPS,
            "domain_extent": DOMAIN_EXTENT,
        },
        "trajectory": {
            "native_continuous_populations": True,
            "snapshot_stride_solver_steps": STRIDE,
            "macro_dt": DT * STRIDE,
            "rollout_steps": ROLLOUT_STEPS,
            "snapshots_including_ic": ROLLOUT_STEPS + 1,
        },
        "benchmark_seeds_excluded": [0, 1, 2],
        "amplitude_min": float(np.min(amplitudes)),
        "amplitude_max": float(np.max(amplitudes)),
        "native_final_check_max_abs": check_max_abs,
        "trajectory_sha256": _sha256(args.output),
        "split_sha256": _sha256(split_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
