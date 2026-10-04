"""Exact optimizer/RNG continuation and native-window parity, run on Slurm CPU."""

import importlib
import json
import zipfile
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid.training_continuation import (
    TrainingContinuation,
    TrainingYield,
)

core = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)
IDENTITY = {
    "source_sha256": "source",
    "image_sha256": "image",
    "solver": "toy",
    "dataset_sha256": "data",
}


@pytest.fixture
def harness(monkeypatch):
    native_calls = []

    def advance(_t, _ctx, velocity, *, frame_steps, native_state=None):
        native_calls.append(native_state is None)
        memory = 0.0 if native_state is None else native_state
        return 0.8 * velocity + 0.01 * memory, jnp.mean(velocity)

    monkeypatch.setattr(core, "_solver_advance", advance)
    monkeypatch.setattr(
        core, "corrected_velocity", lambda model, velocity, **kwargs: velocity + model
    )
    # Nonconstant trajectories expose changes in either trajectory or window RNG.
    train = (
        np.arange(3 * 7 * 2 * 2 * 2, dtype=np.float32).reshape(3, 7, 2, 2, 1, 2) / 100
        + 0.5
    )

    def run(arm, warm, continuation=None, **changes):
        checks = []
        training = {
            "max_updates": 8,
            "unroll": 3,
            "seed": 123,
            "check_grad": True,
            "fd_epsilon": 0.01,
            "fd_epsilons": [0.001],
            "curriculum": [
                {"updates": 3, "lr": 0.01, "unroll": 1},
                {"updates": 5, "lr": 0.001, "unroll": 3},
            ],
        }
        training.update(changes)
        result = core._train_corrector(
            None,
            SimpleNamespace(domain_extent=2 * np.pi),
            train,
            frame_steps=1,
            training=training,
            velocity_scale=1.0,
            loss_scale=1.0,
            differentiate_solver=arm == "full",
            model_seed=2,
            initial_model=jnp.array(0.2 if warm else 0.1),
            supervised_inputs=train[:, :-1] * 0.8 if arm == "supervised" else None,
            fd_checks=checks,
            continuation=continuation,
        )
        return result, checks

    return run, native_calls


@pytest.mark.parametrize("arm", ["full", "stopped", "supervised"])
@pytest.mark.parametrize("warm", [False, True])
def test_exact_periodic_and_resumed_training_parity(tmp_path, harness, arm, warm):
    run, calls = harness
    baseline, baseline_checks = run(arm, warm)
    baseline_calls = list(calls)
    calls.clear()
    periodic = TrainingContinuation(
        tmp_path / "periodic.zip", IDENTITY, checkpoint_every=2
    )
    result, checks = run(arm, warm, periodic)
    np.testing.assert_array_equal(result[0], baseline[0])
    assert result[1:3] == baseline[1:3]
    assert checks == baseline_checks
    assert calls == baseline_calls
    calls.clear()
    resume = TrainingContinuation(
        tmp_path / "resume.zip",
        IDENTITY,
        checkpoint_every=2,
        max_updates_per_allocation=3,
    )
    with pytest.raises(TrainingYield) as first:
        run(arm, warm, resume)
    assert first.value.updates == 3  # Exactly at the stage boundary.
    resume.max_updates_per_allocation = 2
    with pytest.raises(TrainingYield) as second:
        run(arm, warm, resume)
    assert second.value.updates == 5  # Inside the lower-rate second stage.
    resume.max_updates_per_allocation = None
    result, checks = run(arm, warm, resume)
    np.testing.assert_array_equal(result[0], baseline[0])
    assert result[1:3] == baseline[1:3]
    assert result[4:] == baseline[4:]
    assert checks == baseline_checks
    assert len(result[3]) == 8 and result[-1]
    assert calls == baseline_calls
    assert resume.active_time_s >= sum(result[3])
    with zipfile.ZipFile(resume.path) as archive:
        saved = json.loads(archive.read("metadata.json"))
    assert saved["status"] == "complete" and saved["updates"] == 8
    assert len(saved["fd_checks"]) == (2 if arm == "full" else 0)


@pytest.mark.parametrize("changed", ["source", "recipe", "warm", "corrupt"])
def test_stale_or_corrupt_checkpoint_is_rejected(tmp_path, harness, changed):
    run, _ = harness
    continuation = TrainingContinuation(
        tmp_path / "state.zip", dict(IDENTITY), max_updates_per_allocation=2
    )
    with pytest.raises(TrainingYield):
        run("full", False, continuation)
    kwargs = {}
    if changed == "source":
        continuation.identity["source_sha256"] = "changed"
    elif changed == "recipe":
        kwargs["clip_norm"] = 0.123
    elif changed == "corrupt":
        with zipfile.ZipFile(continuation.path) as original:
            metadata = original.read("metadata.json")
        with zipfile.ZipFile(continuation.path, "w") as altered:
            altered.writestr("metadata.json", metadata)
            altered.writestr("state.eqx", b"corrupt")
    with pytest.raises(ValueError, match=r"binding|checksum"):
        run("full", changed == "warm", continuation, **kwargs)


def test_numerical_failure_is_terminal_not_a_clean_yield(
    tmp_path, harness, monkeypatch
):
    run, _ = harness
    continuation = TrainingContinuation(tmp_path / "failed.zip", IDENTITY)
    monkeypatch.setattr(core, "_window_loss", lambda model, **kwargs: model * jnp.nan)
    result, _ = run("full", False, continuation)
    assert not result[-1]
    with pytest.raises(ValueError, match="failed training cannot resume"):
        run("full", False, continuation)


def test_wrapper_clean_yield_then_deferred_evaluation(tmp_path, monkeypatch):
    import jax
    from test_solver_in_loop_final import fixture_data, mock_evaluate

    from experiments.solver_in_loop import final_run

    payload, ctx, metadata = fixture_data(tmp_path)
    checkpoint = tmp_path / "resume.state"
    payload["continuation"] = {"path": str(checkpoint), "max_updates_per_allocation": 1}
    payload["evaluate_after_training"] = False
    model = final_run.init_corrector(
        jax.random.PRNGKey(2), **final_run.model_spec(payload["run"]["training"])
    )
    calls = []

    def trainer(*args, **kwargs):
        calls.append(kwargs["continuation"])
        if len(calls) == 1:
            checkpoint.write_bytes(b"mock persisted optimizer")
            raise TrainingYield(1, 0.5)
        return model, [1.0, 0.5], [1.0, 0.5], [0.1, 0.1], 0.001, True

    monkeypatch.setattr(final_run.core, "_train_corrector", trainer)
    monkeypatch.setattr(final_run.core, "_evaluate_reference_set", mock_evaluate)
    first = final_run.train(None, ctx, payload, tmp_path)
    assert first["resume_required"] and not first["completed"]
    assert "failure" not in first and not (tmp_path / "model.eqx").exists()
    prior = tmp_path / "previous-outcome.json"
    prior.write_text(json.dumps(first))
    payload["continuation"].update(
        previous_outcome_path=str(prior),
        previous_outcome_sha256=final_run.file_hash(prior),
    )
    result = final_run.train(None, ctx, payload, tmp_path)
    assert result["completed"] and result["admitted"] and result["evaluation_deferred"]
    assert result["evaluation_seeds"] == []
    assert result["prepared_evaluation_seeds"] == metadata["eval_seeds"]
    assert (
        result["training_wall_time_s"]
        == first["training_wall_time_s"] + result["allocation_training_wall_time_s"]
    )
    outcome = tmp_path / "trained.json"
    outcome.write_text(json.dumps(result))
    payload.update(
        model_outcome_path=str(outcome),
        model_path=str(tmp_path / "model.eqx"),
        model_sha256=result["model_sha256"],
        eval_indices=[0, 1],
    )
    evaluated = final_run.evaluate(None, ctx, payload, tmp_path)
    assert (
        evaluated["admitted"]
        and evaluated["evaluation_seeds"] == metadata["eval_seeds"]
    )
    # Truly evaluated development states retain the original overlap refusal.
    result["evaluation_seeds"] = metadata["eval_seeds"]
    outcome.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="overlaps development"):
        final_run.evaluate(None, ctx, payload, tmp_path)
