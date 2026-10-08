"""The public corrector experiment is bounded and uses its own solver references."""

from inspect import signature

from mosaic.benchmarks.problems import get_config


def test_registered_corrector_budget_and_reference_contract():
    cfg = get_config("ns-grid")
    experiment = cfg.experiments["optimization/solver_in_loop"]
    run = signature(experiment.fn).parameters["_kw"].default["runs"][0]
    assert run["physics"] == {"N": 32, "nu": 0.001, "dt": 0.01, "steps": 2}
    dataset = run["dataset"]
    assert dataset["reference_kind"] == "solver_self_refined"
    assert dataset["reference_factor"] == 2
    assert dataset["reference_temporal_factor"] == 4
    assert dataset["reference_audit_temporal_factor"] == 8
    assert set(dataset["train_seeds"]).isdisjoint(dataset["test_seeds"])
    assert run["training"]["max_updates"] == 16
    assert run["training"]["unroll"] == 4
    assert run["training"]["include_supervised_baseline"] is True
    assert run["training"]["check_grad"] is True
    assert run["evaluation"]["rollout_frames"] == 12
    assert experiment.params["status_check"]
