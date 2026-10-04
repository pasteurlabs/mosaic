"""Small real-RPC optimizer continuation probe; not a scientific accuracy run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.training_continuation import (
    TrainingContinuation,
    TrainingYield,
    array_digest,
)

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


def main() -> None:
    """Compare uninterrupted and resumed updates through the actual allocated solver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == "gpu"
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == "ins_jl")
    physics = {
        "N": 16,
        "nu": 0.001,
        "dt": 0.01,
        "steps": 4,
        "forcing_amplitude": 1.0,
        "forcing_wavenumber": 6,
    }
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys=physics,
        domain_extent=2 * np.pi,
        output_key="result",
    )
    x = np.arange(16, dtype=np.float32) * (2 * np.pi / 16)
    train = np.zeros((2, 6, 16, 16, 1, 2), dtype=np.float32)
    for seed in range(2):
        for frame in range(6):
            train[seed, frame, :, :, 0, 0] = -(0.1 + 0.01 * frame) * np.sin(
                x[None, :] + seed
            )
            train[seed, frame, :, :, 0, 1] = (0.1 + 0.01 * frame) * np.sin(
                x[:, None] + seed
            )
    identity = {
        "source_sha256": args.source_sha256,
        "image_sha256": args.image_sha256,
        "solver": "ins-jl",
        "dataset_sha256": array_digest(train),
    }
    results = {}
    with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
        assert core._supports_native_state(solver)
        pairs = core._make_supervised_inputs(solver, ctx, train, frame_steps=4)
        for arm in ["full", "stopped", "supervised"]:
            kwargs = {
                "frame_steps": 4,
                "training": {
                    "max_updates": 6,
                    "unroll": 4,
                    "lr": 1e-4,
                    "hidden_channels": 4,
                    "kernel_size": 3,
                    "seed": 2026,
                    "check_grad": True,
                },
                "velocity_scale": 1.0,
                "loss_scale": 1.0,
                "differentiate_solver": arm == "full",
                "model_seed": 8,
                "supervised_inputs": pairs if arm == "supervised" else None,
            }
            baseline_checks = []
            baseline = core._train_corrector(
                solver, ctx, train, fd_checks=baseline_checks, **kwargs
            )
            state = TrainingContinuation(
                args.out / f"{arm}.checkpoint",
                identity,
                checkpoint_every=1,
                max_updates_per_allocation=2,
            )
            try:
                core._train_corrector(solver, ctx, train, continuation=state, **kwargs)
                raise AssertionError("expected allocation yield")
            except TrainingYield as stop:
                assert stop.updates == 2
            state.max_updates_per_allocation = None
            resumed_checks = []
            resumed = core._train_corrector(
                solver,
                ctx,
                train,
                continuation=state,
                fd_checks=resumed_checks,
                **kwargs,
            )
            a = [
                np.asarray(v)
                for v in jax.tree_util.tree_leaves(
                    eqx.filter(baseline[0], eqx.is_array)
                )
            ]
            b = [
                np.asarray(v)
                for v in jax.tree_util.tree_leaves(eqx.filter(resumed[0], eqx.is_array))
            ]
            results[arm] = {
                "model_exact": all(
                    np.array_equal(x, y) for x, y in zip(a, b, strict=True)
                ),
                "model_max_absolute": max(
                    float(np.max(np.abs(x - y))) for x, y in zip(a, b, strict=True)
                ),
                "loss_exact": baseline[1] == resumed[1],
                "gradient_exact": baseline[2] == resumed[2],
                "fd_checks_exact": baseline_checks == resumed_checks,
                "completed": baseline[-1] and resumed[-1],
                "updates": len(resumed[1]),
            }
            print(json.dumps({arm: results[arm]}), flush=True)
    passed = all(
        r["completed"]
        and r["model_exact"]
        and r["loss_exact"]
        and r["gradient_exact"]
        and r["fd_checks_exact"]
        for r in results.values()
    )
    (args.out / "probe.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "identity": identity,
                "results": results,
                "probe_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    if not passed:
        raise AssertionError("real solver continuation was not bitwise identical")


if __name__ == "__main__":
    main()
