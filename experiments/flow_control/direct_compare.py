"""Paired direct-control development comparison; run only in a Slurm allocation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Any

import jax.numpy as jnp
import numpy as np

from .baselines import CostLedger, linear_action_baseline
from .control import (
    ControlConfig,
    actuator_basis,
    control_effort,
    generate_tasks,
    gradient_check,
    objective,
    rollout,
    terminal_loss,
)
from .direct_optim import optimize_actions


def optimizer_cells() -> list[dict[str, Any]]:
    """Freeze a small balanced optimizer grid before observing development outcomes."""
    cells = [
        {
            "setting_id": f"adam-lr{lr}",
            "method": "adam",
            "optimizer_settings": {"lr": lr},
        }
        for lr in (0.01, 0.05)
    ]
    cells += [
        {
            "setting_id": f"spsa-lr{lr}-eps{eps}-d{directions}",
            "method": "spsa",
            "optimizer_settings": {"lr": lr, "epsilon": eps, "directions": directions},
        }
        for lr in (0.01, 0.05)
        for eps in (0.01, 0.05)
        for directions in (1, 4)
    ]
    cells.append({"setting_id": "powell", "method": "powell", "optimizer_settings": {}})
    return cells


def _hash(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_direct_compare(
    t: Any, ctx: Any, config: ControlConfig, *, payload: dict[str, Any]
) -> dict[str, Any]:
    """Optimize matched inputs first, then audit every eligible fine-grid snapshot."""
    seed = payload["task_seed"]
    comparison_stage = payload.get("comparison_stage", "development")
    cells = payload.get("settings", optimizer_cells())
    if comparison_stage == "development":
        if seed not in range(3000, 3008) or cells != optimizer_cells():
            raise ValueError(
                "development freezes seeds3000–3007 and the full11setting grid"
            )
    elif comparison_stage == "test":
        if seed not in range(5000, 5016) or not payload.get("selection_sha256"):
            raise ValueError(
                "test requires frozen selection provenance and seeds5000–5015"
            )
        allowed = optimizer_cells()
        if (
            len(cells) != 3
            or {c["method"] for c in cells} != {"adam", "spsa", "powell"}
            or any(c not in allowed for c in cells)
        ):
            raise ValueError("test requires one selected frozen setting per method")
    else:
        raise ValueError("unknown comparison stage")
    arrays, metrics = (
        {},
        {
            "phase": "direct_compare",
            "comparison_stage": comparison_stage,
            "selection_sha256": payload.get("selection_sha256"),
            "control": asdict(config),
            "task_seed": seed,
            "held_out_test_used": comparison_stage == "test",
            "completed": False,
            "admitted": False,
            "common_admitted": False,
            "wall_budgets_s": [30, 60, 120],
            "query_budgets": [32, 64, 128],
            "latency_scope": (
                "Shared GPU service warmed by common preparation; optimizer initialization charged. "
                "Gradient admission runs after optimization. Fixed randomized order; not cold-process latency."
            ),
            "selection_rule": (
                "Development mean fine192 objective at120seconds; "
                "failures and missing candidates remain explicit. No test selection."
            ),
            "snapshot_selection": (
                "Only completed coarse objective candidates within each budget; "
                "no fine-field selection."
            ),
        },
    )
    ledger, started = CostLedger(), time.perf_counter()
    stage = "common_preparation"
    try:
        work = ledger.phase(stage)
        prep_started = time.perf_counter()
        try:
            work.rollout_count += 1
            task = generate_tasks(t, ctx, config, [seed])[0]
            for name in (
                "initial",
                "goal",
                "fine_initial",
                "fine_goal",
                "generating_controls",
                "fine_goal_rollout",
            ):
                arrays[name] = np.asarray(getattr(task, name))
            work.rollout_count += 1
            audit = np.asarray(
                rollout(
                    t,
                    ctx,
                    task.fine_initial,
                    jnp.asarray(task.generating_controls),
                    config,
                    temporal_factor=config.audit_temporal_factor,
                )
            )
            arrays["goal_audit_terminal"] = audit
            temporal = float(
                np.linalg.norm(audit - task.fine_goal)
                / max(np.linalg.norm(task.fine_goal), 1e-12)
            )
            metrics["goal_temporal_relative_error"] = temporal
            if (
                not np.isfinite(audit).all()
                or not np.isfinite(temporal)
                or temporal >= config.reference_tolerance
            ):
                raise FloatingPointError("goal failed fine temporal admission")
            work.rollout_count += 1
            zero = np.asarray(
                rollout(
                    t, ctx, task.initial, jnp.zeros((config.control_slots, 8)), config
                )
            )
            linear = linear_action_baseline(
                task.goal,
                zero,
                np.asarray(actuator_basis(config.coarse_n, config.domain_extent)),
                control_slots=config.control_slots,
                control_bound=config.control_bound,
                duration=config.dt * config.steps_per_slot * config.control_slots,
            )
            arrays.update(coarse_zero_terminal=zero, linear_controls=linear)
        except Exception:
            work.failed_rollout_count += 1
            raise
        finally:
            work.wall_time_s += time.perf_counter() - prep_started
        metrics["input_sha256"] = _hash(
            [task.initial, task.goal, task.fine_initial, task.fine_goal, linear]
        )
        metrics["initial_seed"], metrics["goal_seed"] = (
            task.initial_seed,
            task.goal_seed,
        )
        order = np.random.default_rng(
            np.random.SeedSequence([seed, 116, 3000])
        ).permutation(len(cells))
        metrics["optimizer_order"] = [cells[index]["setting_id"] for index in order]
        results = []
        stage = "optimization"

        def loss(actions: Any) -> Any:
            return objective(t, ctx, task.initial, task.goal, actions, config)

        for index in order:
            cell = cells[index]
            arm_started = time.perf_counter()
            try:
                result = optimize_actions(
                    loss,
                    linear.copy(),
                    config,
                    cell["method"],
                    optimizer_settings={
                        **cell["optimizer_settings"],
                        "query_budgets": (32, 64, 128),
                        "max_queries": 100000,
                    },
                    budgets=(30, 60, 120),
                    seed=seed,
                )
            except Exception as error:
                result = {
                    "snapshots": [],
                    "failure": f"{type(error).__name__}: {error}",
                    "query_counts_complete": False,
                    "wall_time_s": time.perf_counter() - arm_started,
                }
            results.append({**cell, "input_sha256": metrics["input_sha256"], **result})
        metrics["settings"] = results
        stage = "gradient_admission"
        gradient_started = time.perf_counter()
        gradwork = ledger.phase("gradient_admission")
        try:
            gradient = gradient_check(t, ctx, task, config)
            gradwork.rollout_count = 1 + 6 * len(config.gradient_epsilons)
            gradwork.gradient_rollout_count = 1
        except Exception as error:
            gradient = {
                "passed": False,
                "failure": f"{type(error).__name__}: {error}",
                "query_counts_complete": False,
            }
            gradwork.failed_rollout_count += 1
        finally:
            gradwork.wall_time_s = time.perf_counter() - gradient_started
        metrics["gradient_check"] = gradient
        # No fine evaluation occurs until every optimizer has finished.
        stage = "fine_evaluation"
        evaluation = ledger.phase(stage)
        cache: dict[str, dict[str, Any]] = {}

        def evaluate(actions: np.ndarray) -> dict[str, Any]:
            actions = np.asarray(actions, dtype=np.float32)
            key = _hash([actions])
            if key in cache:
                return {**cache[key], "cache_hit": True}
            label = f"candidate{len(cache)}"
            arrays[f"{label}_controls"] = actions
            begin = time.perf_counter()
            record = {
                "controls_sha256": key,
                "array_prefix": label,
                "cache_hit": False,
                "admitted": False,
            }
            try:
                if (
                    actions.shape != (config.control_slots, 8)
                    or not np.isfinite(actions).all()
                    or np.max(np.abs(actions)) > config.control_bound + 1e-7
                ):
                    raise ValueError(
                        "candidate violates shared finite bounded action space"
                    )
                evaluation.rollout_count += 1
                field = np.asarray(
                    rollout(
                        t,
                        ctx,
                        task.fine_initial,
                        jnp.asarray(actions),
                        config,
                        temporal_factor=config.reference_temporal_factor,
                        return_states=True,
                    )
                )
                arrays[f"{label}_fine_rollout"] = field
                evaluation.rollout_count += 1
                audit = np.asarray(
                    rollout(
                        t,
                        ctx,
                        task.fine_initial,
                        jnp.asarray(actions),
                        config,
                        temporal_factor=config.audit_temporal_factor,
                    )
                )
                arrays[f"{label}_audit_terminal"] = audit
                objective_value = float(
                    terminal_loss(
                        jnp.asarray(field[-1]),
                        jnp.asarray(task.fine_goal),
                        jnp.asarray(actions),
                        config,
                    )
                )
                error = float(
                    np.linalg.norm(audit - field[-1])
                    / max(np.linalg.norm(field[-1]), 1e-12)
                )
                finite = bool(
                    np.isfinite(field).all()
                    and np.isfinite(audit).all()
                    and np.isfinite(objective_value)
                    and np.isfinite(error)
                )
                evaluation.failed_rollout_count += int(not finite)
                record.update(
                    fine_objective=objective_value,
                    control_effort=float(control_effort(jnp.asarray(actions), config)),
                    terminal_mse_normalized=float(
                        np.mean((field[-1] - task.fine_goal) ** 2)
                        / config.velocity_scale**2
                    ),
                    temporal_relative_error=error,
                    finite=finite,
                    admitted=bool(finite and error < config.reference_tolerance),
                )
            except Exception as error:
                evaluation.failed_rollout_count += 1
                record["failure"] = f"{type(error).__name__}: {error}"
            finally:
                evaluation.wall_time_s += time.perf_counter() - begin
            cache[key] = record
            return record

        metrics["linear_baseline"] = evaluate(linear)
        metrics["zero_baseline"] = evaluate(np.zeros_like(linear))
        zero_prefix = metrics["zero_baseline"]["array_prefix"]
        if f"{zero_prefix}_fine_rollout" in arrays:
            arrays["zero_fine_rollout"] = arrays[f"{zero_prefix}_fine_rollout"]
        metrics["common_admitted"] = bool(
            gradient["passed"]
            and metrics["linear_baseline"]["admitted"]
            and metrics["zero_baseline"]["admitted"]
        )
        baseline_prefix = metrics["linear_baseline"]["array_prefix"]
        if f"{baseline_prefix}_fine_rollout" in arrays:
            arrays["linear_fine_rollout"] = arrays[f"{baseline_prefix}_fine_rollout"]
        for result in results:
            for snapshot_index, snapshot in enumerate(result["snapshots"]):
                actions = snapshot.pop("controls", None)
                if actions is not None:
                    snapshot["fine_evaluation"] = evaluate(actions)
                else:
                    snapshot["fine_evaluation"] = {
                        "admitted": False,
                        "unavailable": True,
                    }
                fine = snapshot["fine_evaluation"]
                snapshot["fine_objective"] = fine.get("fine_objective")
                snapshot["admitted"] = fine["admitted"]
                if "array_prefix" in fine:
                    for suffix in ("controls", "fine_rollout", "audit_terminal"):
                        source = f"{fine['array_prefix']}_{suffix}"
                        if source in arrays:
                            arrays[
                                f"snapshots_{result['setting_id']}_{snapshot_index}_{suffix}"
                            ] = arrays[source]
        metrics.update(
            settings=results,
            completed=True,
            admitted=bool(
                metrics["common_admitted"]
                and all(
                    r.get("failure") is None
                    and r["snapshots"]
                    and all(
                        s["fine_evaluation"]["admitted"]
                        for s in r["snapshots"]
                        if s.get("available")
                    )
                    for r in results
                )
            ),
            unique_fine_candidates=len(cache),
        )
    except Exception as error:
        metrics.update(failure_stage=stage, failure=f"{type(error).__name__}: {error}")
    metrics["common_and_evaluation_costs"] = ledger.to_dict()
    metrics["total_job_wall_time_s"] = time.perf_counter() - started
    return {"metrics": metrics, "arrays": arrays}
