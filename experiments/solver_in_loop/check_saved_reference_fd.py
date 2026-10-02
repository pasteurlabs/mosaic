"""Probe the exact saved reference-audit state without regenerating its trajectory."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _directional_fd,
    _solver_advance,
)


def main() -> None:
    """Retain every finite-difference step and the original failed scalar."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    campaign = Path(payload["diagnostic_reference_campaign"])
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == "ins_jl")
    results = []
    with Tesseract.from_url(args.url, timeout=(30, 1200)) as t:
        for cell in payload["diagnostic_cells"]:
            original = json.loads((campaign / "configs" / f"{cell}.json").read_text())
            assert original["image"] == payload["image"], "solver image changed"
            archive_path = campaign / "results" / cell / "results.tar"
            with tarfile.open(archive_path) as archive:
                member = archive.extractfile(
                    "./ns-grid/optimization/solver_in_loop_supervised/corrector_fields.npz"
                )
                with np.load(io.BytesIO(member.read()), allow_pickle=False) as arrays:
                    initial_np = arrays["reference_rollout_0"][0].copy()
            initial = jnp.asarray(initial_np)
            ctx = SimpleNamespace(
                name=spec.name,
                phys=original["run"]["physics"],
                output_key="result",
                domain_extent=2 * np.pi,
                make_inputs=cfg.make_inputs,
            )
            cotangent = jax.random.normal(jax.random.PRNGKey(116), initial.shape)

            def audit_loss(
                value: jax.Array,
                ctx: SimpleNamespace = ctx,
                cotangent: jax.Array = cotangent,
            ) -> jax.Array:
                first, native = _solver_advance(
                    t, ctx, value, frame_steps=ctx.phys["steps"]
                )
                final, _ = _solver_advance(
                    t, ctx, first, frame_steps=ctx.phys["steps"], native_state=native
                )
                return jnp.mean(final * cotangent)

            grads = jax.grad(audit_loss)(initial)
            checks = []
            for direction_seed in payload.get("direction_seeds", [117]):
                for epsilon in payload.get(
                    "epsilons", [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
                ):
                    loss_values = []

                    def recorded_loss(
                        value: jax.Array, loss_values: list = loss_values
                    ) -> jax.Array:
                        loss = audit_loss(value)
                        loss_values.append(float(loss))
                        return loss

                    diagnostic = {}
                    _directional_fd(
                        recorded_loss,
                        initial,
                        grads,
                        jax.random.PRNGKey(direction_seed),
                        epsilon=epsilon,
                        diagnostics=diagnostic,
                    )
                    diagnostic.update(
                        direction_seed=direction_seed,
                        plus_loss=loss_values[0],
                        minus_loss=loss_values[1],
                        absolute_error=abs(
                            diagnostic["finite_difference"] - diagnostic["autodiff"]
                        ),
                    )
                    checks.append(diagnostic)
                    print(json.dumps({"cell": cell, **diagnostic}), flush=True)
            results.append(
                {
                    "original_cell": cell,
                    "original_source_sha256": original["source_sha256"],
                    "image": original["image"],
                    "initial_sha256": hashlib.sha256(initial_np.tobytes()).hexdigest(),
                    "initial_loss": float(audit_loss(initial)),
                    "gradient_l2_norm": float(jnp.linalg.norm(grads)),
                    "checks": checks,
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(
        json.dumps(
            {"source_sha256": payload["source_sha256"], "results": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
