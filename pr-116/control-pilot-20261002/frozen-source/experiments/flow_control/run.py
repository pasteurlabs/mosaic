"""Run a frozen control gate or development pilot on an allocated GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from tesseract_core import Tesseract

from experiments.flow_control.control import ControlConfig, run_gate
from experiments.flow_control.plots import (
    plot_gate,
    plot_pilot,
    plot_shooting_diagnostic,
)
from mosaic.benchmarks.problems import get_config


def main() -> None:
    """Execute a gate and retain every diagnostic and full field."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    if jax.default_backend() != "gpu":
        raise RuntimeError("Control experiments require the allocated GPU")
    control = ControlConfig(**payload.get("control", {}))
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == "phiflow")
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys={
            "nu": control.nu,
            "dt": control.dt,
            "steps": control.steps_per_slot,
            "domain_extent": control.domain_extent,
        },
        domain_extent=control.domain_extent,
        run={"execution": {"jit_solver_rpc": False}},
        output_key="result",
    )
    phase = payload.get("phase", "gate")
    try:
        with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
            print(f"solver_health={solver.health()}", flush=True)
            if phase == "gate":
                result = run_gate(solver, ctx, control, seeds=payload["task_seeds"])
            elif phase == "shooting_diagnostic":
                from experiments.flow_control.shooting_diagnostic import (
                    run_shooting_diagnostic,
                )

                result = run_shooting_diagnostic(
                    solver, ctx, control, seeds=payload["task_seeds"]
                )
            elif phase == "pilot":
                from experiments.flow_control.pilot import run_pilot

                result = run_pilot(
                    solver,
                    ctx,
                    control,
                    method=payload["method"],
                    model_seed=payload["model_seed"],
                    train_seeds=payload["train_seeds"],
                    validation_seeds=payload["validation_seeds"],
                    out_dir=args.out,
                )
            else:
                raise ValueError(f"Unknown control phase: {phase}")
    except Exception as error:
        (args.out / "outcome.json").write_text(
            json.dumps({"passed": False, "completed": False, "error": repr(error)})
        )
        raise
    np.savez_compressed(args.out / "fields.npz", **result["arrays"])
    (args.out / "outcome.json").write_text(json.dumps(result["metrics"], indent=2))
    if phase == "gate":
        plot_gate(result["arrays"], result["metrics"], args.out / "plots")
    elif phase == "shooting_diagnostic":
        plot_shooting_diagnostic(
            result["arrays"], result["metrics"], args.out / "plots"
        )
    else:
        plot_pilot(result["arrays"], result["metrics"], args.out / "plots")
        if result.get("model_checkpoint") is not None:
            (args.out / "model.eqx").write_bytes(result["model_checkpoint"])
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
