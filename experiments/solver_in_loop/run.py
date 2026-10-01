"""Run a frozen three-arm protocol against one cluster-hosted Tesseract."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path

import jax
from tesseract_core import Tesseract

from mosaic.benchmarks.core import runner
from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import solver_in_loop


def main() -> None:
    """Execute one solver/seed cell and preserve its complete protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    os.environ["MOSAIC_RESULTS_DIR"] = str(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    print(f"client_devices={jax.devices()}", flush=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("the training client must run on the allocated GPU")

    @contextlib.contextmanager
    def tracked_tesseract(tag: str, gpus: object, docker_args: object):
        with Tesseract.from_url(args.url, timeout=(30, 1200)) as tesseract:
            print(f"solver_health={tesseract.health()}", flush=True)
            yield tesseract

    runner._tracked_tesseract = tracked_tesseract
    base = get_config("ns-grid")
    solver = next(
        s for s in base.solvers if s.key == payload["solver"].replace("-", "_")
    )
    cfg = dataclasses.replace(base, solvers=[solver])
    cfg.add_experiment(
        "optimization/solver_in_loop_supervised", solver_in_loop, runs=[payload["run"]]
    )
    result = cfg.experiments["optimization/solver_in_loop_supervised"].fn(
        cfg, {solver.name: f"url:{args.url}"}, gpu_ids=["0"]
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    metrics = (
        result["results"][0]["metrics"]
        if result["results"]
        else {"completed": False, "solver_failures": result.get("_solver_failures", {})}
    )
    # Keep numerical/admission failures as results, rather than silently dropping them.
    (args.out / "outcome.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
