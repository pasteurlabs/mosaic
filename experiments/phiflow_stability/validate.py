"""Allocated-GPU, same-native-state forward stability comparison.

Run baseline and candidate in separate processes on the same GPU. The explicit
adapter path binds the implementation; the container supplies dependencies only.
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


def main() -> None:
    """Measure the frozen adapter on the prescribed forced stress case."""
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--duration", type=float, default=75)
    p.add_argument("--ndim", type=int, choices=[2, 3], default=2)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("tested_adapter", args.adapter)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    jax.config.update("jax_enable_x64", False)
    api.math.set_global_precision(32)
    if jax.default_backend() != "gpu":
        raise RuntimeError("This validation requires an allocated GPU")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.resolution
    dt = 0.01 * 64 / n if args.ndim == 2 else 0.01
    bc = {a + "_" + side: {"type": "periodic"} for a in "xyz" for side in ("lo", "hi")}
    k = jnp.fft.fftfreq(n, d=1 / n)
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    envelope = jnp.exp(-0.5 * ((jnp.sqrt(kx * kx + ky * ky) - 4) / 1) ** 2)
    phases = jax.random.uniform(
        jax.random.PRNGKey(0), (n, n), minval=0, maxval=2 * jnp.pi
    )
    psi = envelope * jnp.exp(1j * phases)
    psi = (0.5 * (psi + jnp.conj(psi[::-1, ::-1]))).at[0, 0].set(0)
    vx = jnp.fft.ifft2(1j * ky * psi).real
    vy = jnp.fft.ifft2(-1j * kx * psi).real
    v = jnp.stack([vx, vy], axis=-1)[:, :, None, :]
    v = 0.05 * v / jnp.sqrt(jnp.sum(v * v, axis=-1)).max()
    force = (
        jnp.zeros_like(v)
        .at[..., 0]
        .set(jnp.sin(2 * jnp.pi * 6 * jnp.arange(n) / n)[None, :, None])
    )
    if args.ndim == 3:
        x = jnp.arange(n) * (2 * jnp.pi / n)
        xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
        v = 0.05 * jnp.stack(
            [
                jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz),
                -jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz),
                jnp.zeros_like(xx),
            ],
            axis=-1,
        )
        force = jnp.zeros_like(v).at[..., 0].set(jnp.sin(6 * yy))
    initial = v + 0.1 * force

    def step(v: jax.Array, state: jax.Array | None):
        result, _, native = api.phiflow_fwd(
            v + 0.5 * dt * force,
            0.001,
            dt,
            1,
            2 * jnp.pi,
            bc,
            state=state,
            return_state=True,
        )
        return result + 0.5 * dt * force, native

    first = jax.jit(lambda v: step(v, None))
    block_steps = round(0.04 / dt)

    @jax.jit
    def block(carry: tuple):
        def body(c: tuple, _: None):
            return step(*c), None

        return jax.lax.scan(body, carry, None, length=block_steps)[0]

    records = []
    fields = {"initial": np.asarray(initial)}
    report = {
        "adapter_sha256": hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest(),
        "resolution": n,
        "ndim": args.ndim,
        "dt": dt,
        "duration": args.duration,
        "nu": 0.001,
        "seed": 0,
        "composition": "native recurrent Strang forcing, amplitude 1, wavenumber 6",
        "device": str(jax.devices()),
        "jax_version": jax.__version__,
        "completed": False,
    }
    started = time.perf_counter()

    def record(value: jax.Array, state: jax.Array, step_count: int):
        a = np.asarray(value)
        native = np.asarray(state)[..., : args.ndim]
        div = sum(
            (native[..., d] - np.roll(native[..., d], 1, axis=d)) / (2 * np.pi / n)
            for d in range(args.ndim)
        )
        row = {
            "time": step_count * dt,
            "finite": bool(np.isfinite(a).all() and np.isfinite(native).all()),
            "energy": float(0.5 * np.mean(np.sum(a * a, axis=-1))),
            "max_speed": float(np.sqrt(np.sum(a * a, axis=-1)).max()),
            "native_divergence_rms": float(np.sqrt(np.mean(div * div))),
            "elapsed_s": time.perf_counter() - started,
        }
        records.append(row)
        (out / "progress.json").write_text(
            json.dumps({"report": report, "trajectory": records}, indent=2)
        )
        return row["finite"]

    try:
        value, state = first(initial)
        jax.block_until_ready((value, state))
        report["first_call_s"] = time.perf_counter() - started
        count = 1
        record(value, state, count)
        target = round(args.duration / dt)
        while count + block_steps <= target:
            value, state = block((value, state))
            jax.block_until_ready((value, state))
            count += block_steps
            if not record(value, state, count):
                raise FloatingPointError("Nonfinite forward trajectory")
            if count % (25 * block_steps) == 1:
                print(json.dumps(records[-1]), flush=True)
        # Exact final physical duration (the initial step leaves a short tail).
        for _ in range(target - count):
            value, state = jax.jit(step)(value, state)
        jax.block_until_ready((value, state))
        record(value, state, target)
        fields["final"] = np.asarray(value)
        fields["native_final"] = np.asarray(state)
        report["completed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        if "value" in locals():
            fields["last_field"] = np.asarray(value)
    report["wall_time_s"] = time.perf_counter() - started
    report["last_finite_time"] = max(
        (r["time"] for r in records if r["finite"]), default=0
    )
    np.savez_compressed(out / "fields.npz", **fields)
    (out / "outcome.json").write_text(
        json.dumps({"report": report, "trajectory": records}, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
