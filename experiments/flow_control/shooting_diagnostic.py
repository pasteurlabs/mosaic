"""Separate engineering diagnostic: refine the cheap linear controller with AD.

This neither changes the original zero-start gate nor selects a neural setting.
Every baseline and refined controller is scored using the same fine solver.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import jax.numpy as jnp
import numpy as np

from .baselines import CostLedger
from .control import (
    ControlConfig,
    direct_shooting,
    generate_tasks,
    rollout,
    terminal_loss,
)
from .pilot import _linear_validation, _relative, _task_arrays


def run_shooting_diagnostic(
    t: Any, ctx: Any, config: ControlConfig, seeds: list[int]
) -> dict[str, Any]:
    """Evaluate explicit linear warm starts without altering existing gate results."""
    import time

    ledger = CostLedger()
    prep = ledger.phase("common_data_preparation")
    started = time.perf_counter()
    tasks = []
    try:
        for seed in seeds:
            prep.rollout_count += 1
            tasks.extend(generate_tasks(t, ctx, config, [seed]))
    except Exception:
        prep.failed_rollout_count += 1
        raise
    finally:
        prep.wall_time_s += time.perf_counter() - started
    baseline, arrays = _linear_validation(
        t, ctx, tasks, config, ledger.phase("baseline")
    )
    arrays.update(_task_arrays(tasks, "task"))
    rows, fields, audits, actions, realized = [], [], [], [], []
    work = ledger.phase("warm_start_shooting")
    started = time.perf_counter()
    try:
        for index, task in enumerate(tasks):
            result = direct_shooting(
                t,
                ctx,
                task,
                config,
                initial_controls=arrays["validation_linear_controls"][index],
            )
            work.rollout_count += result["rollout_count"]
            work.gradient_rollout_count += result["gradient_rollout_count"]
            work.failed_rollout_count += result["failed_rollout_count"]
            work.optimizer_updates += result["updates"]
            controls = result["controls"]
            work.rollout_count += 1
            field = np.asarray(
                rollout(
                    t,
                    ctx,
                    task.fine_initial,
                    jnp.asarray(controls),
                    config,
                    temporal_factor=config.reference_temporal_factor,
                    return_states=True,
                )
            )
            work.rollout_count += 1
            audit = np.asarray(
                rollout(
                    t,
                    ctx,
                    task.fine_initial,
                    jnp.asarray(controls),
                    config,
                    temporal_factor=config.audit_temporal_factor,
                )
            )
            loss = float(
                terminal_loss(
                    jnp.asarray(field[-1]),
                    jnp.asarray(task.fine_goal),
                    jnp.asarray(controls),
                    config,
                )
            )
            error = _relative(audit, field[-1])
            finite = bool(
                np.isfinite(field).all()
                and np.isfinite(audit).all()
                and np.isfinite(loss)
            )
            work.failed_rollout_count += int(not finite)
            rows.append(
                {
                    "task_seed": task.seed,
                    "shooting": {
                        k: v
                        for k, v in result.items()
                        if k not in {"controls", "latents", "initial_controls"}
                    },
                    "fine_objective": loss,
                    "linear_fine_objective": baseline["tasks"][index]["objective"],
                    "temporal_relative_error": error,
                    "admitted": bool(
                        finite
                        and result["completed"]
                        and error < config.reference_tolerance
                    ),
                }
            )
            fields.append(field)
            audits.append(audit)
            actions.append(controls)
            realized.append(result["initial_controls"])
    except Exception:
        work.failed_rollout_count += 1
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    arrays.update(
        warm_start_fine_rollouts=np.stack(fields),
        warm_start_audit_terminal=np.stack(audits),
        warm_start_controls=np.stack(actions),
        realized_initial_controls=np.stack(realized),
    )
    return {
        "metrics": {
            "phase": "linear_warm_start_diagnostic",
            "control": asdict(config),
            "tasks": rows,
            "linear_baseline": baseline,
            "costs": ledger.to_dict(),
            "admitted": baseline["admitted"] and all(r["admitted"] for r in rows),
        },
        "arrays": arrays,
    }
