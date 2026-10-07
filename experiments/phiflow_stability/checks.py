"""Focused periodic 2D/3D operator, analytic-flow and derivative checks on GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp


def main() -> None:
    """Exercise the explicit adapter path; save raw numerical measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("tested_adapter", args.adapter)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    jax.config.update("jax_enable_x64", False)
    api.math.set_global_precision(32)
    assert jax.default_backend() == "gpu"
    rows = []
    bc = {a + "_" + side: {"type": "periodic"} for a in "xyz" for side in ("lo", "hi")}
    for ndim in (2, 3):
        shape = (24, 20) if ndim == 2 else (16, 20, 24)
        faces = jax.random.normal(jax.random.PRNGKey(ndim), (ndim, *shape))
        project = jax.jit(api._project_periodic_faces)
        projected = project(faces)
        div = sum(
            (jnp.roll(projected[a], -1, axis=a) - projected[a]) * shape[a]
            for a in range(ndim)
        )
        other = jax.random.normal(jax.random.PRNGKey(ndim + 10), faces.shape)
        operator = {
            "ndim": ndim,
            "divergence_relative": float(jnp.linalg.norm(div) / jnp.linalg.norm(faces)),
            "idempotence_relative": float(
                jnp.linalg.norm(project(projected) - projected)
                / jnp.linalg.norm(projected)
            ),
            "self_adjoint_absolute": float(
                abs(jnp.mean(projected * other) - jnp.mean(faces * project(other)))
            ),
        }
        n = 24 if ndim == 2 else 16
        shape = (n, n, 1) if ndim == 2 else (n, n, n)
        y = jnp.arange(n) * 2 * jnp.pi / n
        initial = (
            jnp.zeros((*shape, ndim)).at[..., 0].set(0.2 * jnp.sin(y)[None, :, None])
        )
        initial = initial.at[..., 1].set(0.15)

        def solve(v: jax.Array, dt: float, steps: int):
            return api.phiflow_fwd(
                v, 0.01, dt, steps, 2 * jnp.pi, bc, return_state=True
            )[0]

        errors = []
        for dt in (0.02, 0.01, 0.005):
            forward = jax.jit(lambda v, dt=dt: solve(v, dt, round(0.2 / dt)))
            actual = forward(initial)
            exact = initial.at[..., 0].set(
                0.2 * jnp.exp(-0.01 * 0.2) * jnp.sin(y - 0.15 * 0.2)[None, :, None]
            )
            errors.append(
                {
                    "dt": dt,
                    "relative_error": float(
                        jnp.linalg.norm(actual - exact) / jnp.linalg.norm(exact)
                    ),
                }
            )
        forward = jax.jit(lambda v: solve(v, 0.01, 8))
        reference = forward(initial)
        direction = (
            jnp.zeros_like(initial).at[..., 0].set(jnp.cos(2 * y)[None, :, None])
        )
        cotangent = direction + 0.3 * initial

        def loss(v: jax.Array):
            return jnp.mean((forward(v) - reference) * cotangent)  # noqa: B023 -- evaluated within this iteration

        start = time.perf_counter()
        grad = jax.jit(jax.grad(loss))(initial)
        jax.block_until_ready(grad)
        ad = float(jnp.sum(grad * direction))
        finite_differences = []
        for eps in (0.001, 0.003, 0.01):
            fd = float(
                (loss(initial + eps * direction) - loss(initial - eps * direction))
                / (2 * eps)
            )
            finite_differences.append(
                {
                    "epsilon": eps,
                    "ad": ad,
                    "fd": fd,
                    "relative_error": abs(fd - ad) / max(abs(fd), abs(ad), 1e-12),
                }
            )
        rows.append(
            {
                "operator": operator,
                "analytic_shear": errors,
                "vjp": finite_differences,
                "first_vjp_s": time.perf_counter() - start,
            }
        )
    report = {
        "adapter_sha256": hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest(),
        "device": str(jax.devices()),
        "checks": rows,
    }
    report["passed"] = all(
        row["operator"]["divergence_relative"] < 2e-5
        and row["operator"]["idempotence_relative"] < 1e-5
        and row["operator"]["self_adjoint_absolute"] < 1e-6
        and row["vjp"][0]["relative_error"] < 0.05
        and max(e["relative_error"] for e in row["analytic_shear"]) < 0.02
        for row in rows
    )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Focused correctness gate failed")


if __name__ == "__main__":
    main()
