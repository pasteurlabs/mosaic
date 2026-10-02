"""Fixed developmental policy pilot; no held-out test claims or model selection.

Each method pays the same fine-grid task preparation. Only improved imitation
pays for additional action optimization. Checkpoints are evaluated after policy
fitting, so validation work never masquerades as optimizer work.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .baselines import (
    CostLedger,
    Work,
    linear_action_baseline,
    train_imitation_policy,
    train_spsa_policy,
)
from .control import (
    ControlConfig,
    Task,
    actuator_basis,
    control_effort,
    direct_shooting,
    generate_tasks,
    init_policy,
    objective,
    policy_controls,
    rollout,
    terminal_loss,
    train_policy,
)

_METHODS = {"full", "spsa", "demonstration_imitation", "improved_imitation"}


def _serialized(model: Any) -> bytes:
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, model)
    return stream.getvalue()


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
    )


def _task_arrays(tasks: list[Task], prefix: str) -> dict[str, np.ndarray]:
    values = {f"{prefix}_task_seeds": np.asarray([task.seed for task in tasks])}
    for key in (
        "initial",
        "goal",
        "fine_initial",
        "fine_goal",
        "fine_goal_rollout",
        "generating_controls",
        "initial_seed",
        "goal_seed",
    ):
        values[f"{prefix}_{key}"] = np.stack([getattr(task, key) for task in tasks])
    return values


def _validate_splits(
    train_seeds: tuple[int, ...], validation_seeds: tuple[int, ...]
) -> None:
    if not train_seeds or not validation_seeds:
        raise ValueError("both development splits must contain tasks")
    if len(set(train_seeds)) != len(train_seeds) or len(set(validation_seeds)) != len(
        validation_seeds
    ):
        raise ValueError("each split must contain unique task identities")
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("training and validation tasks must be disjoint")
    if min((*train_seeds, *validation_seeds)) < 0:
        raise ValueError("task seeds must be nonnegative")


def _prepare(
    t: Any, ctx: Any, config: ControlConfig, seeds: tuple[int, ...], work: Work
) -> tuple[list[Task], list[dict[str, Any]], np.ndarray]:
    tasks, audits, audit_fields = [], [], []
    started = time.perf_counter()
    try:
        for seed in seeds:
            work.rollout_count += 1
            task = generate_tasks(t, ctx, config, [seed])[0]
            tasks.append(task)
            work.rollout_count += 1
            field = np.asarray(
                rollout(
                    t,
                    ctx,
                    task.fine_initial,
                    jnp.asarray(task.generating_controls),
                    config,
                    temporal_factor=config.audit_temporal_factor,
                )
            )
            error = _relative(field, task.fine_goal)
            passed = bool(
                np.isfinite(field).all()
                and np.isfinite(error)
                and error < config.reference_tolerance
            )
            audits.append(
                {
                    "task_seed": seed,
                    "initial_seed": task.initial_seed,
                    "goal_seed": task.goal_seed,
                    "temporal_relative_error": error,
                    "passed": passed,
                }
            )
            audit_fields.append(field)
    except Exception:
        work.failed_rollout_count += 1
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    return tasks, audits, np.stack(audit_fields)


def _improved_labels(
    t: Any, ctx: Any, tasks: list[Task], config: ControlConfig, ledger: CostLedger
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    work = ledger.phase("label_generation")
    labels, rows = [], []
    started = time.perf_counter()
    try:
        for task in tasks:
            shooting = direct_shooting(t, ctx, task, config)
            work.rollout_count += shooting["rollout_count"]
            work.gradient_rollout_count += shooting["gradient_rollout_count"]
            work.optimizer_updates += shooting["updates"]
            if not shooting["completed"]:
                work.failed_rollout_count += 1
                raise FloatingPointError(f"expert shooting failed for task {task.seed}")
            work.rollout_count += 1
            demo_loss = float(
                objective(
                    t,
                    ctx,
                    task.initial,
                    task.goal,
                    jnp.asarray(task.generating_controls),
                    config,
                )
            )
            if not np.isfinite(demo_loss):
                work.failed_rollout_count += 1
                raise FloatingPointError(
                    f"demonstration objective failed for task {task.seed}"
                )
            use_shooting = shooting["final_loss"] < demo_loss
            labels.append(
                shooting["controls"] if use_shooting else task.generating_controls
            )
            rows.append(
                {
                    "task_seed": task.seed,
                    "demonstration_objective": demo_loss,
                    "shooting_objective": shooting["final_loss"],
                    "label_source": "shooting" if use_shooting else "demonstration",
                    "shooting_trace": shooting["trace"],
                }
            )
            work.label_examples += 1
    except Exception:
        work.failed_rollout_count = max(1, work.failed_rollout_count)
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    return labels, rows


def _evaluate(
    t: Any,
    ctx: Any,
    model: Any,
    tasks: list[Task],
    config: ControlConfig,
    work: Work,
    zero_fields: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rows, fields, actions, audit_fields = [], [], [], []
    started = time.perf_counter()
    task_start_count = work.rollout_count
    try:
        for index, task in enumerate(tasks):
            task_start_count = work.rollout_count
            # Warm policy-only inference once; exclude solver execution from latency.
            jax.block_until_ready(policy_controls(model, task, config))
            latencies = []
            for _ in range(5):
                inference_started = time.perf_counter()
                controls = np.asarray(policy_controls(model, task, config))
                latencies.append(time.perf_counter() - inference_started)
            if not np.isfinite(controls).all():
                raise FloatingPointError(
                    f"nonfinite policy action for task {task.seed}"
                )
            if np.max(np.abs(controls)) > config.control_bound + 1e-7:
                raise FloatingPointError(
                    f"policy exceeds action bounds for task {task.seed}"
                )
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
            effort = float(control_effort(jnp.asarray(controls), config))
            terminal_mse = float(
                np.mean((field[-1] - task.fine_goal) ** 2) / config.velocity_scale**2
            )
            zero_loss = float(
                np.mean((zero_fields[index, -1] - task.fine_goal) ** 2)
                / config.velocity_scale**2
            )
            temporal_error = _relative(audit, field[-1])
            signal = float(np.linalg.norm(task.fine_goal - zero_fields[index, -1]))
            audit_to_signal = float(
                np.linalg.norm(audit - field[-1]) / max(signal, 1e-12)
            )
            finite = bool(
                np.isfinite(field).all()
                and np.isfinite(audit).all()
                and np.isfinite(loss)
                and np.isfinite(zero_fields[index]).all()
                and np.isfinite(zero_loss)
            )
            work.failed_rollout_count += int(not np.isfinite(field).all())
            work.failed_rollout_count += int(not np.isfinite(audit).all())
            passed = finite and temporal_error < config.reference_tolerance
            rows.append(
                {
                    "task_seed": task.seed,
                    "initial_seed": task.initial_seed,
                    "goal_seed": task.goal_seed,
                    "objective": loss,
                    "terminal_mse_normalized": terminal_mse,
                    "terminal_relative_error": _relative(field[-1], task.fine_goal),
                    "control_effort": effort,
                    "weighted_control_effort": config.effort_weight * effort,
                    "zero_objective": zero_loss,
                    "temporal_relative_error": temporal_error,
                    "time_discrepancy_to_control_signal": audit_to_signal,
                    "finite": finite,
                    "admitted": bool(passed),
                    "median_policy_inference_s": float(np.median(latencies)),
                }
            )
            fields.append(field)
            actions.append(controls)
            audit_fields.append(audit)
    except Exception:
        work.failed_rollout_count += int(work.rollout_count > task_start_count)
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    summary = {
        "tasks": rows,
        "mean_objective": float(np.mean([row["objective"] for row in rows])),
        "mean_zero_objective": float(np.mean([row["zero_objective"] for row in rows])),
        "mean_terminal_mse_normalized": float(
            np.mean([row["terminal_mse_normalized"] for row in rows])
        ),
        "mean_control_effort": float(np.mean([row["control_effort"] for row in rows])),
        "finite_fraction": float(np.mean([row["finite"] for row in rows])),
        "admitted": all(row["admitted"] for row in rows),
    }
    return summary, {
        "fine_rollouts": np.stack(fields),
        "controls": np.stack(actions),
        "audit_terminal": np.stack(audit_fields),
    }


def _linear_validation(
    t: Any, ctx: Any, tasks: list[Task], config: ControlConfig, work: Work
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Select controls using coarse fields only, then evaluate and audit at192."""
    started = time.perf_counter()
    rows, fields, actions, audits = [], [], [], []
    try:
        for task in tasks:
            work.rollout_count += 1
            zero = np.asarray(
                rollout(
                    t, ctx, task.initial, jnp.zeros((config.control_slots, 8)), config
                )
            )
            controls = linear_action_baseline(
                task.goal,
                zero,
                np.asarray(actuator_basis(config.coarse_n, config.domain_extent)),
                control_slots=config.control_slots,
                control_bound=config.control_bound,
                duration=config.control_slots * config.steps_per_slot * config.dt,
            )
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
                np.isfinite(zero).all()
                and np.isfinite(field).all()
                and np.isfinite(audit).all()
                and np.isfinite(loss)
                and np.isfinite(error)
            )
            work.failed_rollout_count += int(not finite)
            rows.append(
                {
                    "task_seed": task.seed,
                    "initial_seed": task.initial_seed,
                    "goal_seed": task.goal_seed,
                    "objective": loss,
                    "terminal_mse_normalized": float(
                        np.mean((field[-1] - task.fine_goal) ** 2)
                        / config.velocity_scale**2
                    ),
                    "control_effort": float(
                        control_effort(jnp.asarray(controls), config)
                    ),
                    "temporal_relative_error": error,
                    "finite": finite,
                    "admitted": bool(finite and error < config.reference_tolerance),
                }
            )
            fields.append(field)
            actions.append(controls)
            audits.append(audit)
    except Exception:
        work.failed_rollout_count += 1
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    return {
        "tasks": rows,
        "mean_objective": float(np.mean([r["objective"] for r in rows])),
        "admitted": all(r["admitted"] for r in rows),
    }, {
        "validation_fine_linear_rollouts": np.stack(fields),
        "validation_linear_controls": np.stack(actions),
        "validation_linear_audit_terminal": np.stack(audits),
    }


def run_pilot(
    t: Any,
    ctx: Any,
    config: ControlConfig,
    *,
    method: str,
    model_seed: int,
    train_seeds: tuple[int, ...] = tuple(range(1000, 1016)),
    validation_seeds: tuple[int, ...] = tuple(range(2000, 2008)),
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one fixed development method/seed and preserve all validation checkpoints."""
    if method not in _METHODS:
        raise ValueError(f"unknown pilot method: {method}")
    if model_seed not in (0, 1, 2):
        raise ValueError("development pilot freezes model seeds0,1,2")
    train_seeds, validation_seeds = tuple(train_seeds), tuple(validation_seeds)
    _validate_splits(train_seeds, validation_seeds)
    ledger = CostLedger()
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "phase": "developmental_pilot",
        "method": method,
        "model_seed": model_seed,
        "control": asdict(config),
        "train_seeds": list(train_seeds),
        "validation_seeds": list(validation_seeds),
        "held_out_test_used": False,
        "learning_rate": 1e-3,
        "compute_matched": False,
        "settings_tuned": False,
        "planned_updates": 1000 if "imitation" in method else 300,
        "checkpoint_updates": [250, 500, 1000] if "imitation" in method else [300],
        "common_preparation_charged_to_every_method": True,
        "query_counts_complete": True,
        "label_cost_rule": (
            "Only additional expert shooting is charged to improved imitation; "
            "public demonstrations share task preparation with every method."
        ),
    }
    stage, model = "data_preparation", None
    checkpoint_bytes: dict[int, bytes] = {}
    wall_started = time.perf_counter()
    try:
        prepare = ledger.phase("common_data_preparation")
        train, train_audit, train_audit_fields = _prepare(
            t, ctx, config, train_seeds, prepare
        )
        validation, validation_audit, validation_audit_fields = _prepare(
            t, ctx, config, validation_seeds, prepare
        )
        arrays.update(_task_arrays(train, "train"))
        arrays.update(_task_arrays(validation, "validation"))
        arrays["train_goal_audit_terminal"] = train_audit_fields
        arrays["validation_goal_audit_terminal"] = validation_audit_fields
        arrays["times"] = (
            np.arange(config.control_slots + 1) * config.steps_per_slot * config.dt
        )
        digest = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        metadata["dataset_sha256"] = digest.hexdigest()
        metadata["goal_admission"] = {
            "train": train_audit,
            "validation": validation_audit,
        }
        if not all(row["passed"] for row in (*train_audit, *validation_audit)):
            raise FloatingPointError("new development goals failed temporal admission")
        metadata["initial_model_sha256"] = hashlib.sha256(
            _serialized(init_policy(model_seed, config))
        ).hexdigest()
        stage = "label_generation"
        label_costs = CostLedger()
        labels = [task.generating_controls for task in train]
        if method == "improved_imitation":
            ledger.phases["label_generation"] = label_costs.phase("label_generation")
            labels, label_rows = _improved_labels(t, ctx, train, config, label_costs)
            metadata["expert_labels"] = label_rows
        if "imitation" in method:
            arrays["training_labels"] = np.stack(labels)
        stage = "policy_training"
        fitted = None
        training_started = time.perf_counter()
        try:
            if method == "full":
                fitted = train_policy(
                    t, ctx, train, config, seed=model_seed, updates=300, lr=1e-3
                )
            elif method == "spsa":
                fitted = train_spsa_policy(
                    t,
                    ctx,
                    train,
                    config,
                    seed=model_seed,
                    updates=300,
                    lr=1e-3,
                    perturbation=0.05,
                    directions=1,
                )
            else:
                fitted = train_imitation_policy(
                    train,
                    labels,
                    config,
                    label_costs=label_costs,
                    seed=model_seed,
                    updates=1000,
                    lr=1e-3,
                    checkpoint_updates=(250, 500, 1000),
                )
        finally:
            if fitted is None:
                ledger.phases["policy_training"] = Work(
                    wall_time_s=time.perf_counter() - training_started,
                    failed_rollout_count=1,
                )
                metadata["query_counts_complete"] = False
        model = fitted["model"]
        if "costs" in fitted:
            for name, values in fitted["costs"]["phases"].items():
                ledger.phases[name] = Work(**values)
        else:
            ledger.phases["policy_training"] = Work(
                wall_time_s=fitted["wall_time_s"],
                rollout_count=fitted["rollout_count"],
                gradient_rollout_count=fitted["gradient_rollout_count"],
                optimizer_updates=fitted["updates"],
                failed_rollout_count=fitted.get("failed_rollout_count", 0),
            )
        metadata["training_trace"] = fitted["trace"]
        metadata["updates"] = fitted["updates"]
        if not fitted["completed"] or fitted["updates"] != metadata["planned_updates"]:
            raise FloatingPointError(
                fitted.get("failure") or "incomplete policy training"
            )
        checkpoints = fitted.get("checkpoint_models", {fitted["updates"]: model})
        for update, candidate in checkpoints.items():
            checkpoint_bytes[update] = _serialized(candidate)
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"model-update{update}.eqx").write_bytes(
                    checkpoint_bytes[update]
                )
        stage = "fine_validation"
        evaluate = ledger.phase("validation")
        zero = []
        started = time.perf_counter()
        try:
            for task in validation:
                evaluate.rollout_count += 1
                zero.append(
                    np.asarray(
                        rollout(
                            t,
                            ctx,
                            task.fine_initial,
                            jnp.zeros((config.control_slots, 8)),
                            config,
                            temporal_factor=config.reference_temporal_factor,
                            return_states=True,
                        )
                    )
                )
            if not all(np.isfinite(field).all() for field in zero):
                raise FloatingPointError("nonfinite zero-control validation baseline")
        except Exception:
            evaluate.failed_rollout_count += 1
            raise
        finally:
            evaluate.wall_time_s += time.perf_counter() - started
        zero_fields = np.stack(zero)
        arrays["validation_fine_zero_rollouts"] = zero_fields
        linear_summary, linear_arrays = _linear_validation(
            t, ctx, validation, config, evaluate
        )
        metadata["linear_validation"] = linear_summary
        arrays.update(linear_arrays)
        evaluations = []
        for update, candidate in sorted(checkpoints.items()):
            summary, fields = _evaluate(
                t, ctx, candidate, validation, config, evaluate, zero_fields
            )
            evaluations.append({"update": update, **summary})
            for name, value in fields.items():
                arrays[f"validation_update{update}_{name}"] = value
        metadata["validation_checkpoints"] = evaluations
        metadata["final_validation"] = evaluations[-1]
        metadata["completed"] = True
        metadata["admitted"] = linear_summary["admitted"] and all(
            row["admitted"] for row in evaluations
        )
        metadata["comparison_scope"] = (
            "Validation development only; fixed unequal update budgets with actual "
            "costs reported, no tuned or held-out superiority claim."
        )
    except Exception as error:
        metadata.update(
            completed=False,
            admitted=False,
            failure_stage=stage,
            failure=f"{type(error).__name__}: {error}",
        )
    metadata["costs"] = ledger.to_dict()
    metadata["total_job_wall_time_s"] = time.perf_counter() - wall_started
    return {
        "metrics": metadata,
        "arrays": arrays,
        "model_checkpoint": _serialized(model) if model is not None else None,
        "model_checkpoints": checkpoint_bytes,
    }
