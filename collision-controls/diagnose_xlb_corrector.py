"""Replay failed XLB updates without changing training or solver admission."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_core import Tesseract

from experiments.solver_in_loop.final_run import deserialize_model, load_dataset
from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import init_corrector

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


def main() -> None:
    """Replay exact first/second sampled windows and saved evaluation initial state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.config).read_text())
    (out / "protocol.json").write_text(json.dumps(payload, indent=2))
    arrays, metadata = load_dataset(payload)
    saved = json.loads((Path(args.model_dir) / "outcome.json").read_text())
    trained = deserialize_model(
        str(Path(args.model_dir) / "model.eqx"), saved["model_sha256"], saved
    )
    zero = init_corrector(
        jax.random.PRNGKey(saved["model_seed"]), **saved["model_spec"]
    )
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key == "xlb")
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys=payload["run"]["physics"],
        domain_extent=2 * np.pi,
        run=payload["run"],
        output_key="result",
    )
    train = arrays["train"]
    rng = np.random.RandomState(payload["run"]["training"]["seed"])
    windows = []
    for update in range(2):
        idx = int(rng.randint(len(train)))
        start = int(rng.randint(train.shape[1] - 16))
        windows.append((f"update{update}", train[idx, start : start + 17], idx, start))
    windows.append(("validation", arrays["reference"][0], 0, 0))
    report = {
        "purpose": "diagnostic only; no admission override or production recipe change",
        "backend_environment": payload.get("backend_environment", {}),
        "image_sha256": payload["image_sha256"],
        "dataset_sha256": metadata["dataset_sha256"],
        "model_sha256": saved["model_sha256"],
        "device": [d.device_kind for d in jax.devices()],
        "array_keys": list(arrays),
        "cases": [],
    }
    with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
        for label, target, trajectory, start in windows:
            if label not in payload.get(
                "diagnostic_windows", ["update0", "update1", "validation"]
            ):
                continue
            for alpha in payload.get(
                "diagnostic_update_fractions", [0.0, 1.0, 0.1, 0.01]
            ):
                model = jax.tree.map(
                    lambda a, b, alpha=alpha: a + alpha * (b - a), zero, trained
                )
                value = jnp.asarray(target[0])
                native = None
                rows = []
                case = {
                    "window": label,
                    "trajectory": trajectory,
                    "start": start,
                    "update_fraction": alpha,
                    "steps": rows,
                }
                for frame in range(len(target) - 1):
                    provisional, native = core._solver_advance(
                        solver, ctx, value, frame_steps=4, native_state=native
                    )
                    corrected = core.corrected_velocity(
                        model,
                        provisional,
                        velocity_scale=metadata["velocity_scale"],
                        domain_extent=2 * np.pi,
                    )
                    pv = np.asarray(provisional)
                    cv = np.asarray(corrected)
                    ns = np.asarray(native)
                    populations = ns[:9]
                    density = populations.sum(axis=0)
                    row = {
                        "frame": frame + 1,
                        "finite": bool(np.isfinite(cv).all() and np.isfinite(ns).all()),
                        "velocity_max": float(np.max(np.abs(cv))),
                        "mach_max": float(
                            np.max(np.linalg.norm(cv, axis=-1))
                            * 0.01
                            / (2 * np.pi / 64)
                            * np.sqrt(3)
                        ),
                        "correction_rms": float(np.sqrt(np.mean((cv - pv) ** 2))),
                        "relative_error": float(
                            np.linalg.norm(cv - target[frame + 1])
                            / (np.linalg.norm(target[frame + 1]) + 1e-30)
                        ),
                        "population_min": float(populations.min()),
                        "population_negative_fraction": float(np.mean(populations < 0)),
                        "density_min": float(density.min()),
                        "density_max": float(density.max()),
                    }
                    rows.append(row)
                    value = corrected
                    if not row["finite"]:
                        break
                if label != "validation" and alpha in [0.0, 1.0]:

                    def loss(candidate: Any, target: np.ndarray = target):
                        return core._window_loss(
                            candidate,
                            jnp.asarray(target),
                            t=solver,
                            ctx=ctx,
                            frame_steps=4,
                            velocity_scale=metadata["velocity_scale"],
                            differentiate_solver=True,
                            loss_mode="mean",
                            solver_loss_weight=0.0,
                            loss_scale=metadata["training_loss_scale"],
                        )

                    loss_value, gradient = eqx.filter_value_and_grad(loss)(model)
                    fd_checks = []
                    for epsilon in payload.get("diagnostic_fd_epsilons", []):
                        diagnostic = {}
                        core._directional_fd(
                            loss,
                            model,
                            gradient,
                            jax.random.PRNGKey(saved["model_seed"] + 1),
                            epsilon=float(epsilon),
                            diagnostics=diagnostic,
                        )
                        fd_checks.append(diagnostic)
                    case.update(
                        fd_checks=fd_checks,
                        loss=float(loss_value),
                        gradient_norm=float(optax.tree.norm(gradient)),
                    )
                report["cases"].append(case)
                (out / "outcome.json").write_text(json.dumps(report, indent=2))
                print(
                    json.dumps(
                        {k: v for k, v in case.items() if k != "steps"}
                        | {"final": rows[-1]}
                    ),
                    flush=True,
                )
    report["completed"] = True
    (out / "outcome.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
