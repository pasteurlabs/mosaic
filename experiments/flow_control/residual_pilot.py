"""Development follow-up: shared data and a controller initialized at linear control."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .baselines import CostLedger, Work, train_imitation_policy, train_spsa_policy
from .control import (
    ControlConfig,
    init_policy,
    objective,
    policy_controls,
    rollout,
    train_policy,
)
from .dataset import load_dataset
from .pilot import _evaluate, _linear_validation, _serialized, _task_arrays
from .residual_policy import attach_features


def run_residual_pilot(
    t: Any, ctx: Any, config: ControlConfig, *, payload: dict, out_dir: Path
) -> dict[str, Any]:
    """Fit only training tasks; score frozen checkpoints after fitting finishes."""
    method, seed = payload["method"], payload["model_seed"]
    if method not in {"full", "spsa", "demonstration_imitation", "improved_imitation"}:
        raise ValueError("unknown residual pilot method")
    train_ids, validation_ids = payload["train_seeds"], payload["validation_seeds"]
    if (
        train_ids != list(range(1000, 1064))
        or validation_ids != list(range(2000, 2016))
        or seed not in (0, 1, 2)
    ):
        raise ValueError(
            "follow-up freezes64training/16validation tasks and three model seeds"
        )
    ledger = CostLedger()
    started = time.perf_counter()
    arrays, models = {}, {}
    metrics = {
        "phase": "residual_development_pilot",
        "method": method,
        "model_seed": seed,
        "control": asdict(config),
        "train_seeds": train_ids,
        "validation_seeds": validation_ids,
        "completed": False,
        "admitted": False,
        "held_out_test_used": False,
        "settings_tuned": False,
        "compute_matched": False,
        "query_counts_complete": True,
        "architecture": "linear_residual",
        "inference_requires_one_coarse_rollout": True,
        "cost_scope": (
            "Job costs exclude separately reported immutable dataset preparation. "
            "Standalone method cost adds common preparation; improved imitation "
            "additionally adds shared expert-label cost."
        ),
    }
    stage = "shared_dataset_loading"
    model = None
    try:
        train, validation, shared, dataset = load_dataset(
            Path(payload["dataset_path"]),
            config,
            train_seeds=train_ids,
            validation_seeds=validation_ids,
            image_sha256=payload["image_sha256"],
        )
        metrics["dataset_sha256"] = dataset["dataset_sha256"]
        metrics["shared_dataset_costs"] = dataset["costs"]
        all_tasks = [
            attach_features(task, zero, config)
            for task, zero in zip(
                train + validation, shared["task_coarse_zero_terminal"], strict=True
            )
        ]
        train, validation = all_tasks[: len(train)], all_tasks[len(train) :]
        arrays.update(_task_arrays(train, "train"))
        arrays.update(_task_arrays(validation, "validation"))
        arrays["train_policy_features"] = np.stack(
            [task.policy_features for task in train]
        )
        arrays["validation_policy_features"] = np.stack(
            [task.policy_features for task in validation]
        )
        arrays["times"] = (
            np.arange(config.control_slots + 1) * config.steps_per_slot * config.dt
        )
        model = init_policy(seed, config, architecture="linear_residual")
        initial_model = model
        metrics["initial_model_sha256"] = hashlib.sha256(_serialized(model)).hexdigest()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model-update0.eqx").write_bytes(_serialized(model))
        stage = "policy_training"
        kwargs = {
            "seed": seed,
            "updates": 300,
            "lr": 1e-3,
            "model": model,
            "checkpoint_updates": (100, 300),
        }
        training_started = time.perf_counter()
        try:
            if method == "full":
                fitted = train_policy(t, ctx, train, config, **kwargs)
            elif method == "spsa":
                fitted = train_spsa_policy(
                    t, ctx, train, config, perturbation=0.05, directions=1, **kwargs
                )
            else:
                labels = (
                    shared["task_improved_controls"][: len(train)]
                    if method == "improved_imitation"
                    else [task.generating_controls for task in train]
                )
                arrays["training_labels"] = np.asarray(labels)
                kwargs.update(updates=1000, checkpoint_updates=(250, 500, 1000))
                fitted = train_imitation_policy(
                    train, labels, config, label_costs=CostLedger(), **kwargs
                )
        except Exception:
            ledger.phases["policy_training"] = Work(
                wall_time_s=time.perf_counter() - training_started,
                failed_rollout_count=1,
            )
            metrics["query_counts_complete"] = False
            raise
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
        metrics.update(
            training_trace=fitted["trace"],
            updates=fitted["updates"],
            planned_updates=kwargs["updates"],
            learning_rate=kwargs["lr"],
        )
        if not fitted["completed"] or fitted["updates"] != kwargs["updates"]:
            raise FloatingPointError(
                fitted.get("failure") or "incomplete residual training"
            )
        if set(fitted["checkpoint_models"]) != set(kwargs["checkpoint_updates"]):
            raise ValueError("training did not retain every frozen checkpoint")
        checkpoints = {0: initial_model, **fitted["checkpoint_models"]}
        for update, candidate in checkpoints.items():
            models[update] = _serialized(candidate)
            (out_dir / f"model-update{update}.eqx").write_bytes(models[update])
        stage = "training_fit_diagnostic"
        fit_work = ledger.phase("training_fit_diagnostic")
        fit_start = time.perf_counter()
        rows = []
        try:
            for task in train:
                fit_work.rollout_count += 1
                loss = float(
                    objective(
                        t,
                        ctx,
                        task.initial,
                        task.goal,
                        policy_controls(model, task, config),
                        config,
                    )
                )
                if not np.isfinite(loss):
                    raise FloatingPointError("nonfinite final training objective")
                rows.append({"task_seed": task.seed, "objective": loss})
        except Exception:
            fit_work.failed_rollout_count += 1
            raise
        finally:
            fit_work.wall_time_s += time.perf_counter() - fit_start
        metrics["training_fit"] = {
            "tasks": rows,
            "mean_objective": float(np.mean([row["objective"] for row in rows])),
            "resolution": 64,
            "scope": "post-training diagnostic; not optimizer work",
        }
        stage = "fine_validation"
        evaluate = ledger.phase("validation")
        zero, zero_start = [], time.perf_counter()
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
                raise FloatingPointError("nonfinite free fine evolution")
        except Exception:
            evaluate.failed_rollout_count += 1
            raise
        finally:
            evaluate.wall_time_s += time.perf_counter() - zero_start
        zero_fields = np.stack(zero)
        arrays["validation_fine_zero_rollouts"] = zero_fields
        linear, fields = _linear_validation(t, ctx, validation, config, evaluate)
        arrays.update(fields)
        metrics["linear_validation"] = linear
        scores = []
        for update, candidate in sorted(checkpoints.items()):
            summary, fields = _evaluate(
                t, ctx, candidate, validation, config, evaluate, zero_fields
            )
            scores.append({"update": update, **summary})
            for name, value in fields.items():
                arrays[f"validation_update{update}_{name}"] = value
        metrics.update(
            validation_checkpoints=scores,
            final_validation=scores[-1],
            completed=True,
            admitted=linear["admitted"] and all(row["admitted"] for row in scores),
        )
    except Exception as error:
        metrics.update(
            completed=False,
            admitted=False,
            failure_stage=stage,
            failure=f"{type(error).__name__}: {error}",
        )
    metrics["costs"] = ledger.to_dict()
    if "shared_dataset_costs" in metrics:
        shared_phases = metrics["shared_dataset_costs"]["phases"]
        charged = ["common_data_preparation"] + (
            ["label_generation"] if method == "improved_imitation" else []
        )
        standalone = CostLedger()
        for name in charged:
            standalone.phases[name] = Work(**shared_phases[name])
        if "policy_training" in ledger.phases:
            standalone.phases["policy_training"] = ledger.phases["policy_training"]
        metrics["standalone_training_costs"] = standalone.to_dict()
    metrics["total_job_wall_time_s"] = time.perf_counter() - started
    return {
        "metrics": metrics,
        "arrays": arrays,
        "model_checkpoint": _serialized(model) if model is not None else None,
        "model_checkpoints": models,
    }
