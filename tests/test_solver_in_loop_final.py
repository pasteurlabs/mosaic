"""Frozen-data binding, one-arm dispatch and test-only checkpoint evaluation."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from experiments.solver_in_loop import final_run as run


def fixture_data(tmp_path):
    payload = {
        "solver": "ins-jl",
        "source_sha256": "source",
        "image": "image.sqsh",
        "image_sha256": "imagehash",
        "arm": "full",
        "model_seed": 2,
        "run": {
            "physics": {"N": 4, "nu": 0.001, "dt": 0.02, "steps": 1},
            "training": {
                "hidden_channels": 2,
                "kernel_size": 3,
                "lr": 1e-5,
                "unroll": 2,
                "max_updates": 2,
                "seed": 2026,
                "check_grad": True,
            },
        },
    }
    path = tmp_path / "dataset.npz"
    references = np.ones((2, 3, 4, 4, 1, 2), np.float32)
    np.savez(
        path,
        train=references[:1],
        reference=references,
        supervised_inputs=references[:1, 1:],
        native_rollouts=references,
        native_errors=np.array([[0, 0.4, 0.6], [0, 0.5, 0.7]]),
        train_seeds=np.array([0]),
        eval_seeds=np.array([10000, 10001]),
    )
    metadata = {
        "identity": run.identity(payload),
        "dataset_sha256": run.file_hash(path),
        "dataset_role": "validation",
        "admitted": True,
        "velocity_scale": 1.0,
        "training_loss_scale": 0.3,
        "supervised_dataset_wall_time_s": 5.0,
        "dataset_preparation_wall_time_s": 10.0,
        "eval_seeds": [10000, 10001],
    }
    meta_path = tmp_path / "dataset.json"
    meta_path.write_text(json.dumps(metadata))
    payload.update(
        dataset_path=str(path),
        dataset_metadata_path=str(meta_path),
        dataset_sha256=metadata["dataset_sha256"],
    )
    ctx = SimpleNamespace(phys=payload["run"]["physics"], domain_extent=2 * np.pi)
    return payload, ctx, metadata


def mock_evaluate(_solver, _ctx, _model, references, **kwargs):
    return SimpleNamespace(
        errors=np.tile([0.0, 0.2, 0.3], (len(references), 1)),
        first_rollout=references[0],
    )


def test_frozen_dataset_binding_rejects_changed_source_and_bytes(tmp_path):
    payload, _, _ = fixture_data(tmp_path)
    arrays, _ = run.load_dataset(payload)
    assert arrays["reference"].shape[0] == 2
    changed = copy.deepcopy(payload)
    changed["source_sha256"] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        run.load_dataset(changed)
    with open(payload["dataset_path"], "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run.load_dataset(payload)


@pytest.mark.parametrize("arm", ["full", "stopped", "supervised"])
def test_arm_dispatch_preserves_normalization_and_per_ic_evidence(
    tmp_path, monkeypatch, arm
):
    payload, ctx, _ = fixture_data(tmp_path)
    payload["arm"] = arm
    seen = {}

    def trainer(*args, **kwargs):
        seen.update(kwargs)
        model = run.init_corrector(
            jax.random.PRNGKey(2), **run.model_spec(payload["run"]["training"])
        )
        return model, [1.0, 0.5], [2.0, 1.0], [0.1, 0.1], 0.001, True

    monkeypatch.setattr(run.core, "_train_corrector", trainer)
    monkeypatch.setattr(run.core, "_evaluate_reference_set", mock_evaluate)
    result = run.train(None, ctx, payload, tmp_path)
    assert result["admitted"] and result["completed"]
    assert seen["differentiate_solver"] == (arm == "full")
    assert (seen["supervised_inputs"] is not None) == (arm == "supervised")
    assert seen["loss_scale"] == 0.3 and seen["velocity_scale"] == 1.0
    assert result["mean_rollout_error"] == 0.25
    assert result["evaluation_seeds"] == [10000, 10001]
    assert result["model_sha256"] == run.file_hash(tmp_path / "model.eqx")
    with np.load(tmp_path / "fields.npz") as fields:
        assert fields["error_corrected"].shape == (2, 3)
        np.testing.assert_array_equal(
            fields["rollout_uncorrected"], fields["reference_rollout"]
        )
    pair_cost = 5 if arm == "supervised" else 0
    assert (
        result["method_training_including_pretrain_and_pairs_s"]
        == result["training_wall_time_s"] + pair_cost
    )


def test_failed_full_gradient_gate_retains_checkpoint(tmp_path, monkeypatch):
    payload, ctx, _ = fixture_data(tmp_path)
    model = run.init_corrector(
        jax.random.PRNGKey(2), **run.model_spec(payload["run"]["training"])
    )
    monkeypatch.setattr(
        run.core,
        "_train_corrector",
        lambda *a, **kw: (model, [1.0, 0.5], [1.0, 1.0], [0.1, 0.1], 0.2, True),
    )
    monkeypatch.setattr(run.core, "_evaluate_reference_set", mock_evaluate)
    result = run.train(None, ctx, payload, tmp_path)
    assert result["completed"] and not result["admitted"]
    assert not result["gradient_passed"]
    assert (tmp_path / "model.eqx").exists()


def test_warm_start_rejects_unadmitted_or_wrong_recipe(tmp_path):
    payload, ctx, metadata = fixture_data(tmp_path)
    previous = {
        "identity": run.identity(payload),
        "training_dataset_sha256": metadata["dataset_sha256"],
        "model_seed": 2,
        "arm": "supervised",
        "model_spec": run.model_spec(payload["run"]["training"]),
        "completed": True,
        "admitted": False,
        "training": {"lr": 1e-5, "max_updates": 1000, "unroll": 8, "seed": 2026},
    }
    previous_path = tmp_path / "pretrain.json"
    payload.update(
        initial_model_path="irrelevant",
        initial_model_sha256="unused",
        pretrain_outcome_path=str(previous_path),
    )
    for overrides in [
        {"admitted": False},
        {"admitted": True, "training": {**previous["training"], "lr": 1e-4}},
    ]:
        previous_path.write_text(json.dumps(previous | overrides))
        with pytest.raises(ValueError, match="provenance mismatch"):
            run.train(None, ctx, payload, tmp_path)


def test_evaluation_rejects_development_overlap_before_solver_work(tmp_path):
    payload, ctx, metadata = fixture_data(tmp_path)
    previous = {
        "identity": run.identity(payload),
        "admitted": True,
        "evaluation_seeds": metadata["eval_seeds"],
    }
    previous_path = tmp_path / "model-outcome.json"
    previous_path.write_text(json.dumps(previous))
    payload["model_outcome_path"] = str(previous_path)
    with pytest.raises(ValueError, match="overlaps development"):
        run.evaluate(None, ctx, payload, tmp_path)


def test_assembly_orders_ics_and_rejects_changed_shared_training(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    for directory in dirs:
        directory.mkdir()
    payload, _, _ = fixture_data(dirs[0])
    _, _, _ = fixture_data(dirs[1])
    for i, directory in enumerate(dirs):
        path = directory / "dataset.npz"
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["train_rollouts"] = arrays["train"].copy()
        arrays["train_native_errors"] = arrays["native_errors"][:1]
        arrays["eval_seeds"] = np.array([10000 + 2 * i, 10001 + 2 * i])
        np.savez(path, **arrays)
        meta = json.loads((directory / "dataset.json").read_text())
        meta.update(
            dataset_sha256=run.file_hash(path),
            eval_seeds=arrays["eval_seeds"].tolist(),
            reference_audit={},
        )
        for key in [
            "semigroup_errors",
            "long_closure_errors",
            "native_final_errors",
            "native_first_interval_errors",
        ]:
            meta[key] = [0.0]
        (directory / "dataset.json").write_text(json.dumps(meta))
    payload["run"]["dataset"] = {
        "train_seeds": [0],
        "test_seeds": [10003, 10000, 10001, 10002],
    }
    payload["shard_dirs"] = list(map(str, dirs))
    out = tmp_path / "merged"
    out.mkdir()
    metadata = run.assemble(payload, out)
    assert metadata["admitted"] and metadata["shared_training_arrays_identical"]
    with np.load(out / "dataset.npz") as data:
        assert data["eval_seeds"].tolist() == payload["run"]["dataset"]["test_seeds"]
    path = dirs[1] / "dataset.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["train"][0, 0, 0, 0, 0, 0] += 0.01
    np.savez(path, **arrays)
    meta = json.loads((dirs[1] / "dataset.json").read_text())
    meta["dataset_sha256"] = run.file_hash(path)
    (dirs[1] / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="shared training arrays differ"):
        run.assemble(payload, out)


@pytest.mark.parametrize("arm", ["full", "stopped", "supervised"])
def test_actual_training_helper_checkpoint_roundtrip_matches_evaluation(
    tmp_path, monkeypatch, arm
):
    """Exercise real CNN/Adam/core rollout paths with an analytic solver map."""
    import equinox as eqx
    import jax.numpy as jnp

    def advance(_t, _ctx, velocity, *, frame_steps, native_state=None):
        return 0.95**frame_steps * velocity, None

    monkeypatch.setattr(run.core, "_solver_advance", advance)
    rng = np.random.default_rng(42)
    initial = rng.normal(size=(4, 4, 1, 2)).astype(np.float32)
    references = np.stack([initial, 0.9 * initial, 0.81 * initial])[None]
    ctx = SimpleNamespace(phys={"steps": 1, "dt": 0.02}, domain_extent=2 * np.pi)
    training = {
        "max_updates": 1,
        "unroll": 2,
        "lr": 1e-4,
        "hidden_channels": 2,
        "kernel_size": 3,
        "seed": 2026,
        "check_grad": False,
    }
    pairs = run.core._make_supervised_inputs(None, ctx, references, frame_steps=1)
    model, losses, _, _, _, done = run.core._train_corrector(
        None,
        ctx,
        references,
        frame_steps=1,
        training=training,
        velocity_scale=1.0,
        loss_scale=1.0,
        differentiate_solver=arm == "full",
        model_seed=0,
        supervised_inputs=pairs if arm == "supervised" else None,
    )
    assert done and len(losses) == 1 and np.isfinite(losses[0])
    path = tmp_path / "model.eqx"
    eqx.tree_serialise_leaves(path, model)
    metadata = {
        "model_seed": 0,
        "model_spec": run.model_spec(training),
        "model_sha256": run.file_hash(path),
    }
    restored = run.deserialize_model(str(path), metadata["model_sha256"], metadata)
    expected = run.core._evaluate_reference_set(
        None, ctx, model, references, frame_steps=1, velocity_scale=1.0, corrected=True
    )
    actual = run.core._evaluate_reference_set(
        None,
        ctx,
        restored,
        references,
        frame_steps=1,
        velocity_scale=1.0,
        corrected=True,
    )
    np.testing.assert_array_equal(actual.errors, expected.errors)
    np.testing.assert_array_equal(actual.first_rollout, expected.first_rollout)
    assert np.isfinite(np.asarray(jnp.sum(actual.first_rollout)))
