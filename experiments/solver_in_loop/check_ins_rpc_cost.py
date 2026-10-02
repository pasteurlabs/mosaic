"""Measure warmed recurrent loss/gradient/Adam RPC cost for two frozen INS images."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import init_corrector
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _multimode
from mosaic.benchmarks.problems.navier_stokes_grid.physics import make_inputs
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _solver_advance,
    _window_loss,
)


def main() -> None:
    """Use identical inputs and alternating image order on one allocated node."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--jit-candidate", action="store_true")
    parser.add_argument("--prime-model", action="store_true")
    parser.add_argument("--jit-rpc-candidate", action="store_true")
    args = parser.parse_args()
    assert jax.default_backend() == "gpu"
    spec = next(s for s in get_config("ns-grid").solvers if s.key == "ins_jl")
    ctx = SimpleNamespace(
        name=spec.name,
        domain_extent=2 * np.pi,
        output_key="result",
        phys={
            "nu": 0.001,
            "dt": 0.01,
            "steps": 4,
            "domain_extent": 2 * np.pi,
            "forcing_amplitude": 1.0,
            "forcing_wavenumber": 6,
        },
        make_inputs=lambda name, value, **physics: make_inputs(spec, value, **physics),
    )
    initial = _multimode(64, seed=116, k0=4, amplitude=0.05)
    # Synthetic fixed targets exercise the production training graph, not accuracy.
    targets = jnp.stack([initial * (1 + 0.01 * frame) for frame in range(9)])
    model = init_corrector(jax.random.PRNGKey(116))
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(1e-4))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    with contextlib.ExitStack() as stack:
        services = {
            name: stack.enter_context(Tesseract.from_url(url, timeout=(30, 1200)))
            for name, url in [
                ("original", args.original_url),
                ("candidate", args.candidate_url),
            ]
        }
        if args.jit_rpc_candidate:
            solver_module = importlib.import_module(
                "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
            )
            if not hasattr(solver_module, "_compiled_solver_rpc"):
                raise RuntimeError(
                    "RPC JIT requires the optional production stepping implementation"
                )
        contexts = {
            name: SimpleNamespace(
                **vars(ctx),
                run={
                    "execution": {
                        "jit_solver_rpc": args.jit_rpc_candidate and name == "candidate"
                    }
                },
            )
            for name in services
        }
        functions = {
            name: eqx.filter_value_and_grad(
                partial(
                    _window_loss,
                    t=service,
                    ctx=contexts[name],
                    frame_steps=4,
                    velocity_scale=0.05,
                    differentiate_solver=True,
                    loss_mode="mean",
                    solver_loss_weight=0.0,
                    loss_scale=1.0,
                )
            )
            for name, service in services.items()
        }
        if args.jit_candidate:
            functions["candidate"] = eqx.filter_jit(functions["candidate"])
        forward = {}
        for name, service in services.items():
            velocity, native = initial, None
            for _ in range(8):
                velocity, native = _solver_advance(
                    service,
                    contexts[name],
                    velocity,
                    frame_steps=4,
                    native_state=native,
                )
            forward[name] = (np.asarray(velocity), np.asarray(native))
        for original, candidate in zip(
            forward["original"], forward["candidate"], strict=True
        ):
            np.testing.assert_allclose(candidate, original, rtol=1e-6, atol=1e-7)
        if args.prime_model:
            _, gradient = functions["original"](model, targets)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, model
            )
            model = eqx.apply_updates(model, updates)
            jax.block_until_ready((model, optimizer_state))
        timings = {name: [] for name in services}
        outputs = {}
        for sample in range(args.samples + 2):
            order = list(services) if sample % 2 == 0 else list(reversed(services))
            for name in order:
                started = time.perf_counter()
                loss, gradient = functions[name](model, targets)
                updates, state = optimizer.update(gradient, optimizer_state, model)
                updated = eqx.apply_updates(model, updates)
                jax.block_until_ready((loss, gradient, updated, state))
                elapsed = time.perf_counter() - started
                outputs[name] = (loss, gradient, updated)
                if sample >= 2:
                    timings[name].append(elapsed)
                print(
                    json.dumps(
                        {
                            "sample": sample,
                            "image": name,
                            "seconds": elapsed,
                            "loss": float(loss),
                        }
                    ),
                    flush=True,
                )
        diagnostics = {}
        arrays = {}
        for index, label in enumerate(["loss", "gradient", "updated_parameters"]):
            for name in services:
                arrays[f"{name}_{label}"] = np.concatenate(
                    [
                        np.asarray(leaf, dtype=np.float64).ravel()
                        for leaf in jax.tree_util.tree_leaves(outputs[name][index])
                    ]
                )
            original = arrays[f"original_{label}"]
            candidate = arrays[f"candidate_{label}"]
            difference = candidate - original
            diagnostics[label] = {
                "strict_allclose": bool(
                    np.allclose(candidate, original, rtol=1e-5, atol=1e-6)
                ),
                "relative_l2_difference": float(
                    np.linalg.norm(difference) / (np.linalg.norm(original) + 1e-30)
                ),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "mismatched_elements": int(
                    np.count_nonzero(
                        ~np.isclose(candidate, original, rtol=1e-5, atol=1e-6)
                    )
                ),
                "elements": int(original.size),
            }
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
        checks_passed = all(row["strict_allclose"] for row in diagnostics.values())
        result = {
            "checks_passed": checks_passed,
            "primed_model": args.prime_model,
            "comparison_diagnostics": diagnostics,
            "candidate_loss_gradient_jitted": args.jit_candidate,
            "candidate_rpc_jitted": args.jit_rpc_candidate,
            "N": 64,
            "horizon": 8,
            "native_steps_per_interval": 4,
            "forcing_amplitude": 1.0,
            "dt": 0.01,
            "warmup_samples": 2,
            "timed_samples": args.samples,
            "seconds": timings,
            "median_speedup": float(
                np.median(timings["original"]) / np.median(timings["candidate"])
            ),
            "scope": (
                "Production recurrent CNN loss, solver RPC, autodiff, gradient clipping and Adam; "
                "fixed synthetic targets, no reference generation or burn-in; reset model/optimizer each sample."
            ),
        }
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if not checks_passed:
            raise AssertionError(
                "Strict gradient/update comparison failed; diagnostics retained"
            )


if __name__ == "__main__":
    main()
