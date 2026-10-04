"""Instrument frozen-image forced burn-in; diagnostic controls never train."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


def statistics(value: Any, dt: float, extent: float) -> dict:
    """Canonical CFL and high-frequency energy, in double precision diagnostics."""
    u = np.asarray(value, dtype=np.float64)
    result = {"shape": list(u.shape), "finite": bool(np.isfinite(u).all())}
    if not result["finite"]:
        result["nonfinite_entries"] = int(np.count_nonzero(~np.isfinite(u)))
        return result
    result.update(
        max_absolute=float(np.max(np.abs(u))), rms=float(np.sqrt(np.mean(u * u)))
    )
    if u.ndim == 4 and u.shape[2:] == (1, 2):
        dx = extent / u.shape[0]
        divergence = sum(
            (np.roll(u[..., i], -1, axis=i) - np.roll(u[..., i], 1, axis=i)) / (2 * dx)
            for i in range(2)
        )
        result["divergence_rms"] = float(np.sqrt(np.mean(divergence**2)))
        result["advective_cfl"] = float(dt * np.max(np.sum(np.abs(u), axis=-1)) / dx)
        spectrum = np.abs(np.fft.fftn(u, axes=(0, 1))) ** 2
        fx, fy = np.meshgrid(
            np.fft.fftfreq(u.shape[0]), np.fft.fftfreq(u.shape[1]), indexing="ij"
        )
        high = (np.abs(fx) >= 0.25) | (np.abs(fy) >= 0.25)
        result["high_k_energy_fraction"] = float(
            np.sum(spectrum[high]) / max(float(np.sum(spectrum)), 1e-300)
        )
    return result


def main() -> None:
    """Execute a bounded diagnostic against a frozen solver image."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    if jax.default_backend() != "gpu":
        raise RuntimeError("diagnostic requires allocated GPU")
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == payload["solver"].replace("-", "_"))
    physics = dict(payload["run"]["physics"])
    dataset = payload["run"]["dataset"]
    extent = float(payload.get("domain_extent", 2 * np.pi))
    factor = int(payload["diagnostic_factor"])
    dt = float(physics["dt"]) / factor
    n = int(physics["N"]) * int(dataset["reference_factor"])
    maximum_time = float(payload.get("diagnostic_time", 75))
    maximum_wall = float(payload.get("diagnostic_wall_s", 1800))
    mode = payload.get("diagnostic_mode", "forced_threaded")
    state = np.asarray(
        core._multimode(
            n,
            L=extent,
            seed=0,
            k0=dataset["k0"],
            sigma_k=dataset["sigma_k"],
            amplitude=dataset["amplitude"],
        ),
        dtype=np.float32,
    )
    force = core._kolmogorov_force(
        jnp.asarray(state),
        amplitude=physics["forcing_amplitude"],
        wavenumber=physics["forcing_wavenumber"],
    )
    state = state + np.asarray(force) * 0.1
    initial = state.copy()
    if mode == "unforced_threaded":
        physics["forcing_amplitude"] = 0.0
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys=physics,
        domain_extent=extent,
        run=payload["run"],
        output_key="result",
    )
    native = None
    record_every = max(1, round(0.04 / dt))
    trace = []
    started = time.perf_counter()
    result = {
        "solver": payload["solver"],
        "mode": mode,
        "factor": factor,
        "dt": dt,
        "grid": n,
        "source_sha256": payload["source_sha256"],
        "image": payload["image"],
        "image_sha256": payload["image_sha256"],
        "purpose": "diagnostic only: fixed initial seed0; no training, no admission override",
        "completed": False,
        "client_devices": [str(device) for device in jax.devices()],
        "client_device_kind": [device.device_kind for device in jax.devices()],
    }
    last_finite = initial
    step = 0
    try:
        with (
            Tesseract.from_url(args.url, timeout=(30, 1200)) as solver,
            (args.out / "trace.jsonl").open("w") as stream,
        ):
            result["service_health"] = solver.health()
            result["supports_native_state"] = core._supports_native_state(solver)

            def record(value: Any, index: int):
                point = {
                    "step": index,
                    "time": index * dt,
                    "elapsed_s": time.perf_counter() - started,
                    "canonical": statistics(value, dt, extent),
                    "native_leaves": [
                        statistics(leaf, dt, extent)
                        for leaf in jax.tree_util.tree_leaves(native)
                    ],
                    "native_faces": [
                        statistics(np.asarray(leaf)[..., :2], dt, extent)
                        for leaf in jax.tree_util.tree_leaves(native)
                        if hasattr(leaf, "shape")
                        and len(leaf.shape) == 4
                        and leaf.shape[-1] == 4
                    ],
                }
                trace.append(point)
                stream.write(json.dumps(point) + "\n")
                stream.flush()
                if (
                    index == 0
                    or index % (10 * record_every) == 0
                    or not point["canonical"]["finite"]
                ):
                    print(json.dumps(point), flush=True)

            record(state, 0)
            for step in range(1, round(maximum_time / dt) + 1):
                if time.perf_counter() - started >= maximum_wall:
                    result["stop_reason"] = "diagnostic_wall_budget"
                    break
                if mode == "forced_reset":
                    native = None
                advanced, native = core._solver_advance_with_physics(
                    solver, ctx, jnp.asarray(state), dt=dt, steps=1, native_state=native
                )
                state = np.asarray(advanced)
                finite = bool(np.isfinite(state).all())
                if step <= 10 or step % record_every == 0 or not finite:
                    record(state, step)
                if not finite:
                    result["stop_reason"] = "nonfinite"
                    result["first_nonfinite_time"] = step * dt
                    break
                last_finite = state.copy()
            else:
                result["stop_reason"] = "target_time_reached"
                result["completed"] = True
    except Exception as exc:
        result.update(stop_reason="exception", failure=f"{type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
    result.update(
        steps=step,
        final_time=step * dt,
        wall_time_s=time.perf_counter() - started,
        trace=trace,
    )
    np.savez_compressed(
        args.out / "fields.npz", initial=initial, last_finite=last_finite, final=state
    )
    (args.out / "outcome.json").write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trace"}, default=str
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
