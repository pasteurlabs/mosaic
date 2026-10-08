# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate frozen operator weights on registered 3D physics and exact benchmark ICs.

This is a direct API/teacher comparison, not the full Mosaic benchmark harness.
No evaluation fields are added to training. Run from staged node-local source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import operator_api as api
from generate_operator_data import load_teacher, teacher_policy
from operator_dataset import file_hash
from operator_storage import atomic_json, require_local


def energy_fd_sweep(inputs: api.InputSchema) -> dict:
    """Use the registered FD benchmark's energy loss, directions and epsilon grid."""
    with jax.enable_x64(False):
        initial = jnp.asarray(inputs.v0)
        prediction = api.apply(inputs)["result"]
        gradient = api.vector_jacobian_product(
            inputs, {"v0"}, {"result"}, {"result": 2 * prediction}
        )["v0"]
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        directions = [
            jax.random.normal(k, initial.shape, dtype=jnp.float32) for k in keys
        ]
        directions = [v / (jnp.linalg.norm(v) + 1e-30) for v in directions]
        actual = np.asarray([float(jnp.vdot(gradient, v)) for v in directions])
        scale = float(jnp.sqrt(jnp.mean(initial**2) + 1e-30))
        result = {}
        for eps in (5.0, 1.0, 0.1, 0.01, 0.001, 0.0001):
            delta = eps * scale
            fd = []
            for direction in directions:
                plus = api.apply(
                    inputs.model_copy(update={"v0": initial + delta * direction})
                )["result"]
                minus = api.apply(
                    inputs.model_copy(update={"v0": initial - delta * direction})
                )["result"]
                fd.append(float(jnp.sum(plus**2 - minus**2) / (2 * delta)))
            fd = np.asarray(fd)
            error = np.abs(fd - actual) / np.maximum(
                np.maximum(np.abs(fd), np.abs(actual)), 1e-30
            )
            result[str(eps)] = {
                "median_relative_error": float(np.median(error)),
                "cosine": float(
                    np.dot(fd, actual)
                    / (np.linalg.norm(fd) * np.linalg.norm(actual) + 1e-30)
                ),
            }
        return result


def main() -> None:
    """Report every registered case, including nonfinite teacher/model outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--ic-module", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-api", type=Path, default=Path("/tesseract/tesseract_api.py")
    )
    args = parser.parse_args()
    storage = [
        require_local(p)
        for p in (
            Path(__file__),
            args.coverage,
            args.ic_module,
            args.weights,
            args.output,
            args.teacher_api,
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update(
        "jax_compilation_cache_dir", str(args.output.parent / "compilation-cache")
    )
    spec = importlib.util.spec_from_file_location("benchmark_ics", args.ic_module)
    ics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ics)
    api._weights_path = lambda: args.weights
    teacher = load_teacher(args.teacher_api)
    coverage = json.loads(args.coverage.read_text())
    records = []
    report = {
        "weights_sha256": file_hash(args.weights),
        "coverage_sha256": file_hash(args.coverage),
        "storage": storage,
        "device": jax.devices()[0].device_kind,
        "cases": records,
    }
    for case in coverage["cases"]:
        ic = case["ic"]
        maker = {
            "tgv3d": ics._tgv3d,
            "rand_div_free": ics._rand_div_free_3d,
            "abc_flow": ics._abc_flow,
        }[ic["name"]]
        # Loading the XLB API enables global x64. Mosaic constructs ICs in the
        # default float32 host process; random draws change when x64 is enabled.
        with jax.enable_x64(False):
            u = maker(
                case["N"],
                L=case["domain_extent"],
                **{k: v for k, v in ic.items() if k != "name"},
            )
        inputs = api.InputSchema(
            v0=np.asarray(u),
            viscosity=np.array([case["viscosity"]], np.float32),
            dt=np.array([case["dt"]], np.float32),
            steps=case["steps"],
            domain_extent=case["domain_extent"],
        )
        started = time.perf_counter()
        prediction = api.apply(inputs)["result"]
        prediction.block_until_ready()
        cold = time.perf_counter() - started
        started = time.perf_counter()
        prediction = api.apply(inputs)["result"]
        prediction.block_until_ready()
        warm = time.perf_counter() - started
        substeps, collision = teacher_policy(teacher, case, True)
        reference = jax.jit(
            lambda v, case=case, substeps=substeps, collision=collision: (
                teacher.xlb_fwd(
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
        )(u)
        p, target = (
            np.asarray(prediction, dtype=np.float64),
            np.asarray(reference, dtype=np.float64),
        )
        finite, teacher_finite = (
            bool(np.isfinite(p).all()),
            bool(np.isfinite(target).all()),
        )
        row = {
            "id": case["id"],
            "experiment": case["experiment"],
            "N": case["N"],
            "steps": case["steps"],
            "finite": finite,
            "teacher_finite": teacher_finite,
            "cold_seconds": cold,
            "warm_seconds": warm,
            "ic_sha256": hashlib.sha256(np.asarray(u).tobytes()).hexdigest(),
            "ic_factory_x64": False,
        }
        if finite and teacher_finite:
            row["relative_l2"] = float(
                np.linalg.norm(p - target) / max(np.linalg.norm(target), 1e-12)
            )
            row["absolute_rms"] = float(np.sqrt(np.mean((p - target) ** 2)))
            row["energy_ratio"] = float(
                np.sum(p**2) / max(float(np.sum(np.asarray(u) ** 2)), 1e-12)
            )
        if case["experiment"].startswith("gradient/") or case["experiment"].startswith(
            "optimization/"
        ):
            cotangent = np.asarray(u) / max(float(np.linalg.norm(np.asarray(u))), 1e-12)
            started = time.perf_counter()
            gradient = np.asarray(
                api.vector_jacobian_product(
                    inputs, {"v0"}, {"result"}, {"result": cotangent}
                )["v0"]
            )
            row["vjp_seconds"] = time.perf_counter() - started
            row["vjp_finite"] = bool(np.isfinite(gradient).all())
            row["vjp_norm"] = (
                float(np.linalg.norm(gradient.astype(np.float64)))
                if row["vjp_finite"]
                else None
            )
            if case["experiment"] in {
                "gradient/fd_check",
                "optimization/recovery_constant_ic_bfgs_proj",
            }:
                teacher_gradient = np.asarray(
                    teacher.vector_jacobian_product(
                        teacher.InputSchema(**inputs.model_dump()),
                        {"v0"},
                        {"result"},
                        {"result": cotangent},
                    )["v0"],
                    dtype=np.float64,
                )
                row["teacher_vjp_finite"] = bool(np.isfinite(teacher_gradient).all())
                if row["vjp_finite"] and row["teacher_vjp_finite"]:
                    actual = gradient.astype(np.float64)
                    norm = np.linalg.norm(teacher_gradient)
                    row["teacher_vjp_cosine"] = float(
                        np.vdot(actual, teacher_gradient)
                        / max(float(np.linalg.norm(actual) * norm), 1e-30)
                    )
                    row["teacher_vjp_relative_l2"] = float(
                        np.linalg.norm(actual - teacher_gradient)
                        / max(float(norm), 1e-30)
                    )
            if row["vjp_finite"] and case["experiment"] == "gradient/fd_check":
                direction = (
                    np.random.default_rng(902).normal(size=u.shape).astype(np.float32)
                )
                direction /= np.linalg.norm(direction)
                eps = 0.01
                values = []
                for sign in (-1, 1):
                    changed = inputs.model_copy(
                        update={"v0": np.asarray(u) + sign * eps * direction}
                    )
                    values.append(
                        float(
                            np.vdot(
                                np.asarray(
                                    api.apply(changed)["result"], dtype=np.float64
                                ),
                                cotangent.astype(np.float64),
                            )
                        )
                    )
                ad = float(np.vdot(gradient, direction))
                fd = (values[1] - values[0]) / (2 * eps)
                row.update(
                    directional_ad=ad,
                    directional_fd=fd,
                    directional_abs_error=abs(ad - fd),
                )
                row["energy_fd_sweep"] = energy_fd_sweep(inputs)
        records.append(row)
        atomic_json(args.output, report)
        print(json.dumps(row, allow_nan=False), flush=True)
        # Shape/horizon-specialized teacher executables are used once per case.
        # Release them so stress sweeps do not accumulate device memory.
        jax.clear_caches()


if __name__ == "__main__":
    main()
