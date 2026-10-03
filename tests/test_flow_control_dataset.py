# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dataset integrity and training-only teacher generation, with tiny fixtures."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import asdict, replace

import numpy as np
import pytest

from experiments.flow_control import dataset
from experiments.flow_control.baselines import CostLedger
from experiments.flow_control.control import ControlConfig, Task
from experiments.flow_control.pilot import _task_arrays


def _task(seed):
    field = np.full((2, 2, 1, 2), seed / 100, dtype=np.float32)
    return Task(
        seed=seed,
        initial=field,
        goal=field + 0.01,
        fine_initial=field.copy(),
        fine_goal=field + 0.01,
        generating_controls=np.full((2, 8), 0.1, dtype=np.float32),
        initial_seed=seed,
        goal_seed=10_000_000 + seed,
        fine_goal_rollout=np.stack([field, field + 0.01, field + 0.01]),
    )


def _shard(
    root, cell, seeds, training, config, *, admitted=True, image="image", corrupt=None
):
    tasks = [_task(seed) for seed in seeds]
    arrays = _task_arrays(tasks, "task")
    arrays["task_goal_audit_terminal"] = np.stack([t.fine_goal for t in tasks])
    arrays["task_coarse_zero_terminal"] = np.stack([t.initial for t in tasks])
    arrays["task_improved_controls"] = np.stack(
        [
            t.generating_controls if t.seed in training else np.full((2, 8), np.nan)
            for t in tasks
        ]
    )
    if corrupt == "heldout_label":
        arrays["task_improved_controls"][:] = 0.1
    elif corrupt == "training_label":
        arrays["task_improved_controls"][:] = np.nan
    elif corrupt == "field":
        arrays["task_initial"][0, 0, 0, 0, 0] = np.nan
    ledger = CostLedger()
    ledger.phase("common_data_preparation").wall_time_s = 3 * len(seeds)
    ledger.phase("label_generation").label_examples = len(training)
    outcome = {
        "completed": admitted,
        "admitted": admitted,
        "control": asdict(config),
        "task_seeds": seeds,
        "training_seeds": training,
        "costs": ledger.to_dict(),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    path = root / "results" / cell / "results.tar"
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in {
            "outcome.json": json.dumps(outcome).encode(),
            "protocol.json": json.dumps(
                {"image_sha256": image, "source_sha256": "source"}
            ).encode(),
            "fields.npz": stream.getvalue(),
        }.items():
            member = tarfile.TarInfo("./" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.fixture
def shared(tmp_path):
    config = ControlConfig(control_slots=2)
    # Deliberately shuffled within and across shards; assembly must order by split.
    _shard(tmp_path, "a", [20, 11], [11], config)
    _shard(tmp_path, "b", [10], [10], config)
    result = dataset.assemble(tmp_path, ["a", "b"], [10, 11], [20])
    return tmp_path, config, result


def _load(root, config, **overrides):
    kwargs = {
        "train_seeds": [10, 11],
        "validation_seeds": [20],
        "image_sha256": "image",
    }
    kwargs.update(overrides)
    return dataset.load_dataset(root / "dataset.npz", config, **kwargs)


def test_assemble_load_preserves_exact_bytes_order_and_costs(shared):
    root, config, metadata = shared
    train, validation, arrays, loaded = _load(root, config)
    assert metadata["dataset_sha256"] == dataset.file_sha256(root / "dataset.npz")
    assert loaded == metadata
    assert [task.seed for task in train] == [10, 11]
    assert [task.seed for task in validation] == [20]
    for task in train + validation:
        np.testing.assert_array_equal(task.initial, _task(task.seed).initial)
        np.testing.assert_array_equal(task.fine_goal, _task(task.seed).fine_goal)
        assert task.goal_seed == 10_000_000 + task.seed
    assert np.isfinite(arrays["task_improved_controls"][:2]).all()
    assert np.isnan(arrays["task_improved_controls"][2]).all()
    assert metadata["costs"]["phases"]["common_data_preparation"]["wall_time_s"] == 9
    assert metadata["costs"]["phases"]["label_generation"]["label_examples"] == 2


def test_load_refuses_changed_archive_bytes(shared):
    root, config, _ = shared
    with (root / "dataset.npz").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="frozen hash"):
        _load(root, config)


@pytest.mark.parametrize(
    "change", ["physics", "image", "train", "validation", "admission"]
)
def test_load_refuses_wrong_contract(shared, change):
    root, config, metadata = shared
    kwargs = {}
    if change == "physics":
        config = replace(config, dt=config.dt * 2)
    elif change == "image":
        kwargs["image_sha256"] = "different"
    elif change == "train":
        kwargs["train_seeds"] = [11, 10]
    elif change == "validation":
        kwargs["validation_seeds"] = [21]
    else:
        metadata["admitted"] = False
        (root / "dataset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        _load(root, config, **kwargs)


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "failed", "physics", "image", "labels", "overlap"],
)
def test_assembly_refuses_invalid_shards(tmp_path, problem):
    config = ControlConfig(control_slots=2)
    _shard(tmp_path, "a", [10], [10], config)
    seeds = [10] if problem == "duplicate" else [20]
    _shard(
        tmp_path,
        "b",
        seeds,
        [20] if problem == "labels" else [],
        replace(config, nu=0.002) if problem == "physics" else config,
        admitted=problem != "failed",
        image="wrong" if problem == "image" else "image",
    )
    cells = ["a"] if problem == "missing" else ["a", "b"]
    with pytest.raises(ValueError):
        dataset.assemble(tmp_path, cells, [10], [10] if problem == "overlap" else [20])
    assert not (tmp_path / "dataset.npz").exists()


def test_prepare_shard_never_optimizes_heldout_labels(monkeypatch):
    config = ControlConfig(control_slots=2)
    tasks = [_task(10), _task(20)]
    called = []
    monkeypatch.setattr(
        dataset,
        "_prepare",
        lambda *_args: (
            tasks,
            [{"passed": True}, {"passed": True}],
            np.stack([task.fine_goal for task in tasks]),
        ),
    )
    monkeypatch.setattr(dataset, "rollout", lambda _t, _ctx, initial, *_args: initial)
    monkeypatch.setattr(dataset, "objective", lambda *_args: 1.0)

    def teacher(_t, _ctx, task, _config, *, initial_controls):
        called.append(task.seed)
        np.testing.assert_array_equal(initial_controls, task.generating_controls)
        return {
            "completed": True,
            "rollout_count": 3,
            "gradient_rollout_count": 1,
            "failed_rollout_count": 0,
            "updates": 1,
            "final_loss": 0.5,
            "controls": task.generating_controls / 2,
            "trace": [],
        }

    monkeypatch.setattr(dataset, "direct_shooting", teacher)
    result = dataset.prepare_shard(
        None, None, config, seeds=[10, 20], training_seeds=[10]
    )
    assert called == [10]
    assert result["metrics"]["completed"]
    assert [row["task_seed"] for row in result["metrics"]["expert_labels"]] == [10]
    labels = result["arrays"]["task_improved_controls"]
    np.testing.assert_array_equal(labels[0], tasks[0].generating_controls / 2)
    assert np.isnan(labels[1]).all()
    with pytest.raises(ValueError, match="belong to this shard"):
        dataset.prepare_shard(None, None, config, seeds=[10], training_seeds=[20])


@pytest.mark.parametrize("corrupt", ["heldout_label", "training_label", "field"])
def test_assembly_refuses_label_leakage_and_nonfinite_fields(tmp_path, corrupt):
    config = ControlConfig(control_slots=2)
    _shard(
        tmp_path,
        "a",
        [10],
        [10],
        config,
        corrupt=corrupt if corrupt != "heldout_label" else None,
    )
    _shard(
        tmp_path,
        "b",
        [20],
        [],
        config,
        corrupt=corrupt if corrupt == "heldout_label" else None,
    )
    with pytest.raises(ValueError):
        dataset.assemble(tmp_path, ["a", "b"], [10], [20])
    assert not (tmp_path / "dataset.npz").exists()


def test_assembly_refuses_empty_shard_list(tmp_path):
    with pytest.raises(ValueError):
        dataset.assemble(tmp_path, [], [10], [20])
