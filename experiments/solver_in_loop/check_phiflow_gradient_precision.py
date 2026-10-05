"""Diagnostic precision/step sweep; never overrides frozen admission outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    """Run fixed precision diagnostics inside an allocated solver container."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", required=True)
    parser.add_argument("--url")
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", type=int, choices=[32, 64], required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", args.precision == 64)
    if args.url:
        import importlib
        from types import SimpleNamespace

        from tesseract_core import Tesseract

        from mosaic.benchmarks.problems import get_config

        core = importlib.import_module(
            "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
        )
        cfg = get_config("ns-grid")
        spec = next(s for s in cfg.solvers if s.key == "phiflow")
        ctx = SimpleNamespace(
            name=spec.name,
            make_inputs=cfg.make_inputs,
            phys={"nu": 0.001, "dt": 0.01, "steps": 4},
            domain_extent=2 * np.pi,
            run={},
            output_key="result",
        )
        solver = Tesseract.from_url(args.url, timeout=(30, 1200))
        adapter_hash = "external RPC: adapter identity supplied by launcher"
    else:
        import tesseract_api as api

        api.math.set_global_precision(args.precision)
        adapter_hash = hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = jnp.float64 if args.precision == 64 else jnp.float32
    with np.load(args.fields) as f:
        initial = jnp.asarray(f["coarse_rollout"][0], dtype=dtype)
    bc = {a + "_" + side: {"type": "periodic"} for a in "xyz" for side in ["lo", "hi"]}

    def step(v: jax.Array, state: jax.Array | None = None):
        if args.url:
            return core._unforced_solver_advance(
                solver, ctx, v, dt=0.01, steps=1, native_state=state
            )
        result, _, native = api.phiflow_fwd(
            v, 0.001, 0.01, 1, float(2 * np.pi), bc, state=state, return_state=True
        )
        return result, native

    force = (
        jnp.zeros_like(initial)
        .at[..., 0]
        .set(jnp.sin(2 * jnp.pi * 6 * jnp.arange(64, dtype=dtype) / 64)[None, :, None])
    )

    def forced(v: jax.Array, count: int):
        v, native = step(v + 0.005 * force)
        v = v + 0.005 * force

        def body(carry: tuple[jax.Array, jax.Array], unused: None):
            x, n = step(carry[0] + 0.005 * force, carry[1])
            return (x + 0.005 * force, n), None

        (v, native), _ = jax.lax.scan(body, (v, native), None, length=count - 1)
        return v

    cot = jax.random.normal(
        jax.random.PRNGKey(118), initial.shape, dtype=jnp.float32
    ).astype(dtype)
    cot = cot / jnp.linalg.norm(cot)
    raw = jax.random.normal(
        jax.random.PRNGKey(117), initial.shape, dtype=jnp.float32
    ).astype(dtype)
    smooth = jnp.sin(2 * jnp.pi * jnp.arange(64, dtype=dtype) / 64)[
        :, None, None, None
    ] * jnp.ones_like(initial)
    directions = {
        "random_rms1": raw / jnp.sqrt(jnp.mean(raw**2)),
        "smooth_rms1": smooth / jnp.sqrt(jnp.mean(smooth**2)),
    }
    cases = {
        "native_one": lambda x: step(x)[0],
        **{f"forced_{n}": (lambda x, n=n: forced(x, n)) for n in [1, 8, 64]},
    }
    report = {
        "precision": args.precision,
        "gpu": [d.device_kind for d in jax.devices()],
        "adapter_sha256": adapter_hash,
        "engine": "rpc" if args.url else "native",
        "fields_sha256": hashlib.sha256(Path(args.fields).read_bytes()).hexdigest(),
        "purpose": "diagnostic only, no admission override",
        "cases": [],
    }
    arrays = {}
    started = time.perf_counter()
    for name, forward in cases.items():
        base = jax.jit(forward)(initial)
        arrays[name + "_forward"] = np.asarray(base)
        for objective in ["centered_energy", "centered_cotangent"]:

            def loss(
                x: jax.Array,
                forward: Callable = forward,
                base: jax.Array = base,
                objective: str = objective,
            ):
                delta = forward(x) - base
                return (
                    jnp.mean(delta * (forward(x) + base))
                    if objective == "centered_energy"
                    else jnp.sum(delta * cot)
                )

            fn = jax.jit(loss)
            grad = jax.jit(jax.grad(loss))(initial)
            arrays[name + "_" + objective + "_gradient"] = np.asarray(grad)
            for direction_name, direction in directions.items():
                ad = float(jnp.vdot(grad, direction))
                entry = {
                    "case": name,
                    "objective": objective,
                    "direction": direction_name,
                    "ad": ad,
                    "gradient_finite": bool(jnp.isfinite(grad).all()),
                    "sweep": [],
                }
                for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
                    plus, minus = (
                        float(fn(initial + eps * direction)),
                        float(fn(initial - eps * direction)),
                    )
                    fd = (plus - minus) / (2 * eps)
                    entry["sweep"].append(
                        {
                            "epsilon_rms": eps,
                            "fd": fd,
                            "relative_error": abs(fd - ad)
                            / (abs(fd) + abs(ad) + 1e-15),
                        }
                    )
                report["cases"].append(entry)
                print(json.dumps(entry), flush=True)
                report["wall_time_s"] = time.perf_counter() - started
                (out / "outcome.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "arrays.npz", **arrays)
    report["completed"] = True
    (out / "outcome.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
