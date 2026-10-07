"""GPU-only, source-identified forward/adjoint evidence for Warp solver changes.

Run baseline and candidate in separate processes with identical arguments.
Burn traces are descriptive; failures are retained, never silently omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp


def stats(value: np.ndarray, extent: float) -> dict:
    """Measure physical energy and the adapter's centered divergence."""
    u = value.astype(np.float64)
    finite = bool(np.isfinite(u).all())
    if not finite:
        return {"finite": False}
    h = extent / u.shape[0]
    divergence = sum(
        (np.roll(u[..., i], -1, axis=i) - np.roll(u[..., i], 1, axis=i)) / (2 * h)
        for i in range(2)
    )
    return {
        "finite": True,
        "energy": float(np.mean(np.sum(u * u, axis=-1)) / 2),
        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
        "max_absolute": float(np.max(np.abs(u))),
    }


def main() -> None:
    """Write raw checks, timings and optionally a long forced trajectory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--burn-time", type=float, default=75)
    parser.add_argument("--initial", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = "cuda:0"
    if not wp.get_device(device).is_cuda:
        raise RuntimeError("Numerical validation requires allocated GPU")
    spec = importlib.util.spec_from_file_location("stability_api", args.api)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    n, extent, nu = args.n, 2 * np.pi, 0.001
    dt, steps = 0.01 / args.factor, 4 * args.factor
    rng = np.random.default_rng(116)
    x, y = np.meshgrid(
        np.arange(n) * extent / n, np.arange(n) * extent / n, indexing="ij"
    )
    shear = np.zeros((n, n, 1, 2), np.float32)
    shear[:, :, 0, 0] = np.sin(y)
    initial = np.zeros_like(shear)
    for kx, ky in ((1, 0), (0, 1), (1, 1), (2, -1), (3, 2)):
        phase = kx * x + ky * y + rng.uniform(0, 2 * np.pi)
        amplitude = rng.normal() / (kx * kx + ky * ky)
        initial[:, :, 0, 0] += amplitude * ky * np.cos(phase)
        initial[:, :, 0, 1] -= amplitude * kx * np.cos(phase)
    initial *= 0.5 / np.sqrt(np.mean(initial**2))
    if args.initial:
        initial = np.load(args.initial, allow_pickle=False).astype(np.float32)
        if initial.shape != shear.shape:
            raise ValueError("Initial field shape mismatch")
    result = {
        "adapter_sha256": hashlib.sha256(args.api.read_bytes()).hexdigest(),
        "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "device": str(wp.get_device(device)),
        "n": n,
        "dt": dt,
        "steps": steps,
        "burn_requested_time": args.burn_time,
        "viscosity": nu,
        "initial_source": str(args.initial)
        if args.initial
        else "fixed synthetic streamfunction seed116; not original benchmark burn IC",
    }

    def forward(
        u: np.ndarray, viscosity: float = nu, timestep: float = dt, count: int = steps
    ) -> np.ndarray:
        return api.ns2d_solve_forward(u, viscosity, timestep, count, extent, device)

    started = time.perf_counter()
    warm = forward(initial)
    wp.synchronize()
    result["cold_forward_s"] = time.perf_counter() - started
    durations = []
    for _ in range(5):
        started = time.perf_counter()
        forward(initial)
        wp.synchronize()
        durations.append(time.perf_counter() - started)
    result["warm_forward_s"] = durations
    result["initial_stats"] = stats(initial, extent)
    result["one_call_stats"] = stats(warm, extent)
    result["zero_steps_exact"] = bool(
        np.array_equal(forward(initial, count=0), initial)
    )
    analytic = shear * np.exp(-nu * dt * steps)
    actual = forward(shear)
    result["shear_relative_error"] = float(
        np.linalg.norm(actual - analytic) / np.linalg.norm(analytic)
    )
    cotangent = rng.normal(size=initial.shape).astype(np.float32)
    cotangent /= np.linalg.norm(cotangent)
    started = time.perf_counter()
    tape, a, b, ia, ib, viscosity_leaf, dt_leaf = api.ns2d_solve_tape(
        initial, nu, dt, steps, extent, device, track_scalar_grads=True
    )
    taped = np.stack([a.numpy(), b.numpy()], axis=-1)[:, :, None, :]
    gradients = api.ns2d_vjp(
        tape, a, b, ia, ib, cotangent, device, viscosity_leaf, dt_leaf
    )
    wp.synchronize()
    result["cold_forward_and_vjp_s"] = time.perf_counter() - started
    result["taped_forward_max_error"] = float(np.max(np.abs(taped - warm)))
    result["warm_forward_and_vjp_s"] = []
    for _ in range(3):
        started = time.perf_counter()
        current = api.ns2d_solve_tape(
            initial, nu, dt, steps, extent, device, track_scalar_grads=True
        )
        api.ns2d_vjp(*current[:5], cotangent, device, *current[5:])
        wp.synchronize()
        result["warm_forward_and_vjp_s"].append(time.perf_counter() - started)
    split = forward(forward(initial))
    joined = forward(initial, count=2 * steps)
    result["split_join_relative_error"] = float(
        np.linalg.norm(split - joined) / np.linalg.norm(joined)
    )
    divergent = rng.normal(scale=0.1, size=initial.shape).astype(np.float32)
    result["divergent_input_stats"] = stats(divergent, extent)
    result["divergent_output_stats"] = stats(forward(divergent), extent)
    direction = rng.normal(size=initial.shape).astype(np.float32)
    direction /= np.sqrt(np.mean(direction**2))
    checks = []
    for key, epsilon in (("v0", 0.001), ("viscosity", 0.0001), ("dt", 0.0001)):
        if key == "v0":
            plus, minus = (
                forward(initial + epsilon * direction),
                forward(initial - epsilon * direction),
            )
            ad = float(np.sum(gradients[key].astype(np.float64) * direction))
        else:
            kw_plus = {
                "viscosity" if key == "viscosity" else "timestep": (
                    nu if key == "viscosity" else dt
                )
                + epsilon
            }
            kw_minus = {
                "viscosity" if key == "viscosity" else "timestep": (
                    nu if key == "viscosity" else dt
                )
                - epsilon
            }
            plus, minus = forward(initial, **kw_plus), forward(initial, **kw_minus)
            ad = float(gradients[key].item())
        fd = float(
            np.sum((plus.astype(np.float64) - minus) * cotangent) / (2 * epsilon)
        )
        checks.append(
            {
                "input": key,
                "epsilon": epsilon,
                "ad": ad,
                "fd": fd,
                "relative_error": abs(ad - fd) / max(abs(ad) + abs(fd), 1e-12),
            }
        )
    result["gradient_checks"] = checks
    # Identical saved 3-D outputs permit strict comparison across separate processes.
    small = rng.normal(scale=0.05, size=(8, 8, 8, 3)).astype(np.float32)
    three = api.ns3d_solve_forward(small, nu, 0.001, 2, extent, device)
    np.save(args.out / "three_dimensional.npy", three)
    force = np.zeros_like(initial)
    force[:, :, 0, 0] = np.sin(6 * y)
    value = initial.copy()
    trace = [{"time": 0.0, **stats(value, extent)}]
    started = time.perf_counter()
    completed = True
    for frame in range(round(args.burn_time / (dt * steps))):
        try:
            value = forward(value + 0.5 * dt * steps * force) + 0.5 * dt * steps * force
            record = {"time": (frame + 1) * dt * steps, **stats(value, extent)}
            trace.append(record)
            if not record["finite"]:
                completed = False
                break
        except Exception as exc:
            trace.append({"time": (frame + 1) * dt * steps, "exception": repr(exc)})
            completed = False
            break
    result.update(
        burn_completed=completed, burn_wall_s=time.perf_counter() - started, trace=trace
    )
    np.save(args.out / "initial.npy", initial)
    np.save(args.out / "final.npy", value)
    (args.out / "outcome.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps({"out": str(args.out), "burn_completed": completed}), flush=True)


if __name__ == "__main__":
    main()
