"""Post-burn temporal, closure and VJP checks for the explicit Warp candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    relative_l2,
    spectral_restrict,
)

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


def main() -> None:
    """Check one saved post-burn state without fitting or selecting a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.config.read_text())
    (args.out / "protocol.json").write_text(json.dumps(payload, indent=2))
    if jax.default_backend() != "gpu":
        raise RuntimeError("allocated GPU required")
    burn = json.loads(Path(payload["post_burn_outcome"]).read_text())
    if not burn["completed"] or burn["final_time"] != 75.0:
        raise ValueError(
            "candidate must complete the full 75-unit burn before auditing"
        )
    with np.load(payload["post_burn_fields"], allow_pickle=False) as fields:
        initial = fields["final"].copy()
    if not np.isfinite(initial).all():
        raise ValueError("post-burn field must be finite")
    cfg = get_config("ns-grid")
    spec = next(s for s in cfg.solvers if s.key.replace("-", "_") == "warp_ns")
    physics = payload["run"]["physics"]
    ctx = SimpleNamespace(
        name=spec.name,
        make_inputs=cfg.make_inputs,
        phys=physics,
        domain_extent=2 * np.pi,
        run=payload["run"],
        output_key="result",
    )
    started = time.perf_counter()
    result = {
        "variant": payload["variant"],
        "candidate_adapter_sha256": payload["candidate_adapter_sha256"],
        "image_sha256": payload["image_sha256"],
        "client_device_kind": [d.device_kind for d in jax.devices()],
        "post_burn_fields_sha256": hashlib.sha256(
            Path(payload["post_burn_fields"]).read_bytes()
        ).hexdigest(),
        "completed": False,
        "admitted": False,
    }
    arrays = {"initial_fine": initial}
    try:
        with Tesseract.from_url(args.url, timeout=(30, 1200)) as solver:
            result["health"] = solver.health()
            result["native_state_support"] = core._supports_native_state(solver)
            rollouts = {}
            for factor in [3, 6]:
                value = jnp.asarray(initial)
                native = None
                frames = [np.asarray(spectral_restrict(value, 64))]
                for _ in range(48):
                    value, native = core._solver_advance_with_physics(
                        solver,
                        ctx,
                        value,
                        dt=float(physics["dt"]) / factor,
                        steps=int(physics["steps"]) * factor,
                        native_state=native,
                    )
                    frames.append(np.asarray(spectral_restrict(value, 64)))
                rollouts[factor] = np.stack(frames)
                arrays[f"reference_factor{factor}"] = rollouts[factor]
            errors = [
                relative_l2(a, b) for a, b in zip(rollouts[3], rollouts[6], strict=True)
            ]
            result["temporal_errors"] = errors
            result["temporal_max_error"] = float(max(errors))
            coarse_initial = jnp.asarray(spectral_restrict(initial, 64))
            native = None
            value = coarse_initial
            coarse_frames = [np.asarray(value)]
            for _ in range(48):
                value, native = core._solver_advance(
                    solver,
                    ctx,
                    value,
                    frame_steps=int(physics["steps"]),
                    native_state=native,
                )
                coarse_frames.append(np.asarray(value))
            arrays["coarse_rollout"] = np.stack(coarse_frames)
            result["coarse_rollout_finite"] = bool(
                np.isfinite(arrays["coarse_rollout"]).all()
            )
            one, carry = core._unforced_solver_advance(
                solver, ctx, coarse_initial, dt=float(physics["dt"]), steps=4
            )
            split, _ = core._unforced_solver_advance(
                solver, ctx, one, dt=float(physics["dt"]), steps=4, native_state=carry
            )
            joined, _ = core._unforced_solver_advance(
                solver, ctx, coarse_initial, dt=float(physics["dt"]), steps=8
            )
            result["native_closure_error"] = relative_l2(
                np.asarray(split), np.asarray(joined)
            )

            def objective(value: jax.Array):
                first, native = core._solver_advance(solver, ctx, value, frame_steps=4)
                final, _ = core._solver_advance(
                    solver, ctx, first, frame_steps=4, native_state=native
                )
                return jnp.mean(final**2)

            grads = jax.grad(objective)(coarse_initial)
            checks = []
            for epsilon in [1e-3, 3e-3, 1e-2]:
                diagnostic = {}
                core._directional_fd(
                    objective,
                    coarse_initial,
                    grads,
                    jax.random.PRNGKey(117),
                    epsilon=epsilon,
                    diagnostics=diagnostic,
                )
                checks.append(diagnostic | {"epsilon": epsilon})
            result["vjp_checks"] = checks
            result["vjp_finite"] = bool(np.isfinite(np.asarray(grads)).all())
            training = dict(payload["run"]["training"])
            training.update(
                max_updates=1,
                unroll=16,
                lr=1e-4,
                check_grad=True,
                fd_epsilon=1e-3,
                fd_epsilons=[3e-3, 1e-2],
            )
            training_checks = []
            training_data = rollouts[3][None, :25]
            velocity_scale = max(float(np.sqrt(np.mean(training_data**2))), 1e-6)
            trained = core._train_corrector(
                solver,
                ctx,
                training_data,
                frame_steps=4,
                training=training,
                velocity_scale=velocity_scale,
                loss_scale=1.0,
                differentiate_solver=True,
                model_seed=8,
                fd_checks=training_checks,
            )
            result["training_horizon"] = 16
            result["training_gradient_checks"] = training_checks
            result["training_primary_fd_error"] = trained[4]
            result["training_step_completed"] = trained[5]
            result["training_losses"] = trained[1]
            result["training_gradient_norms"] = trained[2]
            result["completed"] = True
            result["admitted"] = bool(
                result["training_step_completed"]
                and result["training_primary_fd_error"] < 0.05
                and result["coarse_rollout_finite"]
                and result["vjp_finite"]
                and np.isfinite(errors).all()
                and max(errors) <= 0.005
                and result["native_closure_error"] <= 0.01
                and checks[0]["relative_error"] < 0.05
            )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result["failure"] = f"{type(exc).__name__}: {exc}"
    result["wall_time_s"] = time.perf_counter() - started
    np.savez_compressed(args.out / "fields.npz", **arrays)
    (args.out / "outcome.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
