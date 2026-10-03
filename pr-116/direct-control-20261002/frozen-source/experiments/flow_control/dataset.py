"""Generate once and share immutable control tasks across all training methods."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .baselines import CostLedger
from .control import ControlConfig, Task, direct_shooting, objective, rollout
from .pilot import _prepare, _task_arrays


def file_sha256(path: Path) -> str:
    """Hash the actual shared bytes with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_shard(
    t: Any,
    ctx: Any,
    config: ControlConfig,
    *,
    seeds: list[int],
    training_seeds: list[int],
) -> dict[str, Any]:
    """Audit goals and cache free evolution; improve only training demonstrations."""
    if not set(training_seeds) <= set(seeds):
        raise ValueError("expert labels must belong to this shard")
    ledger = CostLedger()
    tasks, audits, audit_fields = _prepare(
        t, ctx, config, tuple(seeds), ledger.phase("common_data_preparation")
    )
    arrays = _task_arrays(tasks, "task")
    arrays["task_goal_audit_terminal"] = audit_fields
    metadata = {
        "phase": "shared_dataset_shard",
        "control": asdict(config),
        "task_seeds": seeds,
        "training_seeds": training_seeds,
        "goal_admission": audits,
        "completed": False,
        "admitted": False,
    }
    if not all(row["passed"] for row in audits):
        metadata.update(
            failure="goal temporal admission failed", costs=ledger.to_dict()
        )
        return {"metrics": metadata, "arrays": arrays}
    common = ledger.phase("common_data_preparation")
    zero_fields = []
    started = time.perf_counter()
    try:
        for task in tasks:
            common.rollout_count += 1
            zero = np.asarray(
                rollout(
                    t, ctx, task.initial, jnp.zeros((config.control_slots, 8)), config
                )
            )
            if not np.isfinite(zero).all():
                raise FloatingPointError("nonfinite coarse free evolution")
            zero_fields.append(zero)
    except Exception:
        common.failed_rollout_count += 1
        raise
    finally:
        common.wall_time_s += time.perf_counter() - started
    arrays["task_coarse_zero_terminal"] = np.stack(zero_fields)
    labels, rows = [], []
    work = ledger.phase("label_generation")
    teacher = replace(config, shooting_lr=0.01)
    started = time.perf_counter()
    try:
        for task in tasks:
            if task.seed not in training_seeds:
                # NaNs prevent accidental use of validation labels for fitting.
                labels.append(np.full_like(task.generating_controls, np.nan))
                continue
            result = direct_shooting(
                t, ctx, task, teacher, initial_controls=task.generating_controls
            )
            work.rollout_count += result["rollout_count"]
            work.gradient_rollout_count += result["gradient_rollout_count"]
            work.failed_rollout_count += result["failed_rollout_count"]
            work.optimizer_updates += result["updates"]
            if not result["completed"]:
                raise FloatingPointError(
                    f"expert refinement failed for task {task.seed}"
                )
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
                raise FloatingPointError("nonfinite original demonstration objective")
            improved = result["final_loss"] < demo_loss
            labels.append(result["controls"] if improved else task.generating_controls)
            rows.append(
                {
                    "task_seed": task.seed,
                    "demonstration_objective": demo_loss,
                    "selected_objective": min(demo_loss, result["final_loss"]),
                    "label_source": "refined_demonstration"
                    if improved
                    else "demonstration",
                    "trace": result["trace"],
                }
            )
            work.label_examples += 1
    except Exception:
        work.failed_rollout_count += 1
        raise
    finally:
        work.wall_time_s += time.perf_counter() - started
    arrays["task_improved_controls"] = np.stack(labels)
    metadata.update(
        completed=True,
        admitted=True,
        costs=ledger.to_dict(),
        expert_labels=rows,
        teacher_config=asdict(teacher),
    )
    return {"metrics": metadata, "arrays": arrays}


def assemble(
    campaign: Path,
    cells: list[str],
    train_seeds: list[int],
    validation_seeds: list[int],
) -> dict[str, Any]:
    """Join admitted shards on a CPU allocation; reject omissions and duplicates."""
    if not cells or not train_seeds or not validation_seeds:
        raise ValueError("dataset needs shards and nonempty train/validation splits")
    expected = train_seeds + validation_seeds
    if len(set(expected)) != len(expected):
        raise ValueError("dataset splits must be disjoint and unique")
    parts, metrics, protocols = [], [], []
    for cell in cells:
        with tarfile.open(campaign / "results" / cell / "results.tar") as archive:
            row = json.load(archive.extractfile("./outcome.json"))
            protocol = json.load(archive.extractfile("./protocol.json"))
            if not row.get("completed") or not row.get("admitted"):
                raise ValueError(f"{cell} did not pass dataset admission")
            with np.load(
                io.BytesIO(archive.extractfile("./fields.npz").read()),
                allow_pickle=False,
            ) as data:
                parts.append({key: data[key] for key in data.files})
            metrics.append(row)
            protocols.append(protocol)
    first = metrics[0]["control"]
    identity = {key: protocols[0][key] for key in ("image_sha256", "source_sha256")}
    if any(row["control"] != first for row in metrics) or any(
        any(row[key] != value for key, value in identity.items()) for row in protocols
    ):
        raise ValueError(
            "dataset shards have different physics or source/image identities"
        )
    if any(set(part) != set(parts[0]) for part in parts):
        raise ValueError("dataset shard arrays differ")
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    observed = arrays["task_task_seeds"].tolist()
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("dataset task identity mismatch")
    if sorted(seed for row in metrics for seed in row["training_seeds"]) != sorted(
        train_seeds
    ):
        raise ValueError("expert-label split differs from frozen training split")
    order = [observed.index(seed) for seed in expected]
    arrays = {key: value[order] for key, value in arrays.items()}
    label_array = arrays["task_improved_controls"]
    if (
        not np.isfinite(label_array[: len(train_seeds)]).all()
        or not np.isnan(label_array[len(train_seeds) :]).all()
    ):
        raise ValueError(
            "optimized labels must be finite for training and absent for validation"
        )
    if any(
        not np.isfinite(value).all()
        for key, value in arrays.items()
        if key != "task_improved_controls"
    ):
        raise ValueError("nonfinite task arrays cannot enter a shared dataset")
    path = campaign / "dataset.npz"
    temporary = campaign / "dataset.part.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    ledger = CostLedger()
    for row in metrics:
        for name, values in row["costs"]["phases"].items():
            work = ledger.phase(name)
            for key, value in values.items():
                setattr(work, key, getattr(work, key) + value)
    result = {
        "admitted": True,
        "control": first,
        **identity,
        "dataset_sha256": file_sha256(path),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "costs": ledger.to_dict(),
        "shards": dict(zip(cells, metrics, strict=True)),
        "cost_scope": (
            "One shared preparation, reported separately from each training job. "
            "For standalone comparison add common preparation; "
            "only improved imitation adds expert-label generation."
        ),
    }
    (campaign / "dataset.json").write_text(json.dumps(result, indent=2))
    return result


def load_dataset(
    path: Path,
    config: ControlConfig,
    *,
    train_seeds: list[int],
    validation_seeds: list[int],
    image_sha256: str,
) -> tuple[list[Task], list[Task], dict[str, np.ndarray], dict[str, Any]]:
    """Verify shared-file identity, physics and splits before exposing tasks."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if not metadata.get("admitted"):
        raise ValueError("dataset admission failed")
    if metadata["dataset_sha256"] != file_sha256(path):
        raise ValueError("shared dataset bytes do not match their frozen hash")
    if (
        metadata["control"] != json.loads(json.dumps(asdict(config)))
        or metadata["image_sha256"] != image_sha256
    ):
        raise ValueError("shared dataset physics/image mismatch")
    if (
        metadata["train_seeds"] != train_seeds
        or metadata["validation_seeds"] != validation_seeds
    ):
        raise ValueError("shared dataset split mismatch")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if arrays["task_task_seeds"].tolist() != train_seeds + validation_seeds:
        raise ValueError("shared dataset task order mismatch")
    tasks = []
    for index, seed in enumerate(train_seeds + validation_seeds):
        values = {
            key: arrays[f"task_{key}"][index]
            for key in (
                "initial",
                "goal",
                "fine_initial",
                "fine_goal",
                "generating_controls",
                "initial_seed",
                "goal_seed",
                "fine_goal_rollout",
            )
        }
        values["initial_seed"] = int(values["initial_seed"])
        values["goal_seed"] = int(values["goal_seed"])
        tasks.append(Task(seed=seed, **values))
    return tasks[: len(train_seeds)], tasks[len(train_seeds) :], arrays, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "dataset-plan.json").read_text())
    print(json.dumps(assemble(args.campaign, **manifest), indent=2))
