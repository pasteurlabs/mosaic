"""Single-IC assembly preserves full-training normalization and held-out isolation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.solver_in_loop import final_run, transfer_data


def _payload():
    return {
        "solver": "test",
        "source_sha256": "source",
        "image": "image",
        "image_sha256": "sha",
        "run": {
            "physics": {"N": 2, "dt": 0.01, "steps": 4, "nu": 0.001},
            "dataset": {
                "train_seeds": list(range(8)),
                "test_seeds": list(range(20000, 20032)),
            },
            "training": {
                "loss_normalization": "solver_baseline",
                "loss_scale_floor": 1e-6,
            },
            "evaluation": {"seen_ic_trajectories": 4},
        },
    }


@pytest.fixture
def shards(tmp_path):
    payload = _payload()
    directories = []
    for position, seed in enumerate(list(range(8)) + list(range(20000, 20032))):
        directory = tmp_path / str(seed)
        directory.mkdir()
        train = np.full((1, 25, 2, 2, 1, 2), position + 1, dtype=np.float32)
        rollouts = np.full((1, 49, 2, 2, 1, 2), position + 1, dtype=np.float32)
        errors = np.arange(49, dtype=np.float32)[None, :] * (position + 1) / 100
        np.savez_compressed(
            directory / "dataset.npz",
            train=train,
            train_rollouts=rollouts,
            supervised_inputs=train[:, 1:] if seed < 8 else np.empty((0,)),
            train_native_errors=errors,
            train_native_rollouts=rollouts,
            train_seeds=[seed],
            eval_seeds=[],
        )
        metadata = {
            "identity": final_run.identity(payload),
            "dataset_sha256": final_run.file_hash(directory / "dataset.npz"),
            "dataset_role": "validation",
            "single_ic": True,
            "train_seeds": [seed],
            "eval_seeds": [],
            "admitted": True,
            "completed": True,
            "dataset_preparation_wall_time_s": 10,
            "supervised_dataset_wall_time_s": 2 if seed < 8 else 0,
            "native_state_threading": True,
            "reference_audit": {"eligible_for_corrector_training": True},
        }
        (directory / "dataset.json").write_text(json.dumps(metadata))
        directories.append(str(directory))
    payload["shard_dirs"] = directories[::-1]
    return payload, tmp_path


def test_ordered_assembly_matches_unsharded_normalization(shards):
    payload, path = shards
    result = transfer_data.assemble(payload, path / "assembled")
    with np.load(path / "assembled" / "dataset.npz") as data:
        np.testing.assert_array_equal(data["train_seeds"], np.arange(8))
        np.testing.assert_array_equal(data["eval_seeds"], np.arange(20000, 20032))
        np.testing.assert_array_equal(
            data["reference"][:, 0, 0, 0, 0, 0], np.arange(9, 41)
        )
        expected_velocity = float(np.sqrt(np.mean(data["train"] ** 2)) + 1e-8)
        expected_loss = float(
            np.mean(np.mean(data["train_native_errors"][:4], axis=0)[1:25] ** 2)
        )
        assert result["velocity_scale"] == pytest.approx(expected_velocity)
        assert result["training_loss_scale"] == pytest.approx(expected_loss)
        assert data["supervised_inputs"].shape[:2] == (8, 24)
    assert result["dataset_preparation_wall_time_s"] == 400
    assert result["supervised_dataset_wall_time_s"] == 16


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "failed_admission", "heldout_pairs"]
)
def test_no_missing_failed_or_leaking_ic_is_assembled(shards, problem):
    payload, path = shards
    if problem == "missing":
        payload["shard_dirs"].pop()
    elif problem == "duplicate":
        payload["shard_dirs"].append(payload["shard_dirs"][0])
    else:
        directory = path / "20000"
        metadata = json.loads((directory / "dataset.json").read_text())
        if problem == "failed_admission":
            metadata["admitted"] = False
        else:
            with np.load(directory / "dataset.npz") as data:
                arrays = {key: data[key] for key in data.files}
            arrays["supervised_inputs"] = arrays["train"][:, 1:]
            np.savez_compressed(directory / "dataset.npz", **arrays)
            metadata["dataset_sha256"] = final_run.file_hash(directory / "dataset.npz")
        (directory / "dataset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        transfer_data.assemble(payload, path / "assembled")


def test_single_heldout_prepare_never_generates_pairs(monkeypatch, tmp_path):
    payload = copy.deepcopy(_payload())
    payload.update(single_ic=True, generate_supervised_pairs=False)
    payload["run"]["dataset"].update(train_seeds=[20000], test_seeds=[])
    field = np.ones((2, 2, 1, 2), dtype=np.float32)
    train = np.stack([np.stack([field] * 25)])
    rollouts = np.stack([np.stack([field] * 49)])
    monkeypatch.setattr(
        final_run.core,
        "_make_solver_self_reference_datasets",
        lambda *a, **k: (
            train,
            rollouts,
            rollouts[:0],
            "digest",
            {"eligible_for_corrector_training": True},
        ),
    )
    monkeypatch.setattr(
        final_run.core, "_evaluate_rollout", lambda *a, **k: (rollouts[0], np.zeros(49))
    )
    monkeypatch.setattr(
        final_run.core, "_solver_advance", lambda t, c, state, **k: (state, None)
    )
    monkeypatch.setattr(
        final_run.core,
        "_make_supervised_inputs",
        lambda *a, **k: pytest.fail("heldout label generation"),
    )
    monkeypatch.setattr(final_run.core, "_supports_native_state", lambda t: True)
    ctx = SimpleNamespace(phys=payload["run"]["physics"])
    result = final_run.prepare(None, ctx, payload, tmp_path)
    assert result["admitted"] and result["single_ic"]
    assert result["supervised_dataset_wall_time_s"] == 0
    with np.load(tmp_path / "dataset.npz") as data:
        assert data["supervised_inputs"].size == 0
        assert data["train_native_rollouts"].shape == (1, 49, 2, 2, 1, 2)
