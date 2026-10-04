"""Frozen transfer reports cannot borrow the INS-specific selection contract."""

import json
import sys

from experiments.solver_in_loop import transfer_report as report


def test_incomplete_transfer_remains_inconclusive(tmp_path, monkeypatch):
    plan = {
        "base_payload": {
            "solver": "pict",
            "image_sha256": "image",
            "run": {
                "dataset": {
                    "reference_temporal_factor": 3,
                    "reference_audit_temporal_factor": 6,
                }
            },
        },
        "source_sha256": "source",
        "resources": {"gpus": "gpu:b200:1"},
        "confirmation_recipes": {arm: {} for arm in ("full", "stopped", "supervised")},
    }
    inputs = {
        "expected_model_seeds": list(range(8, 16)),
        "expected_ic_seeds": list(range(20000, 20032)),
        "training": [],
        "test": [],
        "failures": ["reference admission failed"],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "report-input.json").write_text(json.dumps(inputs))
    monkeypatch.setattr(sys, "argv", ["report", "--campaign", str(tmp_path)])
    report.main()
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["report_complete"]
    assert result["conclusion"].startswith("Inconclusive")
    assert "mean_errors" not in result


def test_training_recipe_and_identity_are_bound(tmp_path):
    base = {
        "solver": "pict",
        "source_sha256": "s",
        "image": "i",
        "image_sha256": "h",
        "run": {"physics": {"N": 64}},
    }
    recipe = {"lr": 1e-4, "unroll": 16, "updates": 3000}
    plan = {"base_payload": base, "confirmation_recipes": {"full": recipe}}
    result = {
        "identity": {k: v for k, v in base.items() if k != "run"},
        "arm": "full",
        "model_seed": 8,
        "training": {"lr": 1e-4, "unroll": 16, "max_updates": 3000},
        "optimizer_updates": 3000,
        "evaluation_deferred": True,
        "evaluation_seeds": [],
    }
    result["identity"]["physics"] = {"N": 64}
    path = tmp_path / "outcome.json"
    inputs = {"training": [{"arm": "full", "model_seed": 8, "path": str(tmp_path)}]}
    path.write_text(json.dumps(result))
    assert report.validate_transfer(inputs, plan) == []
    result["optimizer_updates"] = 2999
    path.write_text(json.dumps(result))
    assert "budget" in report.validate_transfer(inputs, plan)[0]
    result["optimizer_updates"] = 3000
    result["identity"]["source_sha256"] = "different"
    path.write_text(json.dumps(result))
    assert "source_sha256" in report.validate_transfer(inputs, plan)[0]


def test_complete_transfer_matrix_and_plots(tmp_path, monkeypatch):
    import numpy as np

    base = {
        "solver": "pict",
        "source_sha256": "source",
        "image": "i",
        "image_sha256": "h",
        "run": {
            "physics": {"N": 64},
            "dataset": {
                "reference_temporal_factor": 3,
                "reference_audit_temporal_factor": 6,
            },
        },
    }
    recipes = {
        arm: {"lr": 1e-4, "unroll": 16, "updates": 3000}
        for arm in ("full", "stopped", "supervised")
    }
    plan = {
        "base_payload": base,
        "source_sha256": "source",
        "resources": {"gpus": "gpu:b200:1"},
        "confirmation_recipes": recipes,
    }
    identity = {
        k: base[k] for k in ("solver", "source_sha256", "image", "image_sha256")
    }
    identity["physics"] = base["run"]["physics"]
    inputs = {
        "expected_model_seeds": list(range(8, 16)),
        "expected_ic_seeds": list(range(20000, 20032)),
        "training": [],
        "test": [],
        "failures": [],
    }
    velocity = np.zeros((49, 64, 64, 1, 2), dtype=np.float32)
    for arm, error in (("full", 0.08), ("stopped", 0.1), ("supervised", 0.12)):
        for seed in range(8, 16):
            folder = tmp_path / f"{arm}-{seed}"
            folder.mkdir()
            checkpoint = f"{arm}-{seed}"
            outcome = {
                "identity": identity,
                "arm": arm,
                "model_seed": seed,
                "model_sha256": checkpoint,
                "completed": True,
                "admitted": True,
                "training": {"lr": 1e-4, "unroll": 16, "max_updates": 3000},
                "optimizer_updates": 3000,
                "evaluation_deferred": True,
                "evaluation_seeds": [],
                "training_wall_time_s": 120.0,
            }
            (folder / "outcome.json").write_text(json.dumps(outcome))
            inputs["training"].append(
                {"arm": arm, "model_seed": seed, "path": str(folder)}
            )
            test = folder / "evaluation"
            test.mkdir()
            evaluated = {
                "identity": identity,
                "arm": arm,
                "model_seed": seed,
                "model_sha256": checkpoint,
                "completed": True,
                "admitted": True,
                "evaluation_seeds": inputs["expected_ic_seeds"],
                "evaluation_dataset_sha256": "shared",
            }
            (test / "outcome.json").write_text(json.dumps(evaluated))
            errors = np.full((32, 49), error)
            errors[:, 0] = 0
            native = np.full((32, 49), 0.2)
            native[:, 0] = 0
            arrays = {"error_corrected": errors, "error_uncorrected": native}
            if seed == 8:
                arrays.update(
                    reference_rollout=velocity,
                    rollout_uncorrected=velocity + 0.2,
                    rollout_corrected=velocity + error,
                )
            np.savez_compressed(test / "fields.npz", **arrays)
            inputs["test"].append(
                {
                    "arm": arm,
                    "model_seed": seed,
                    "ic_seeds": inputs["expected_ic_seeds"],
                    "path": str(test),
                }
            )
    dataset = tmp_path / "results/assembled"
    dataset.mkdir(parents=True)
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "supervised_dataset_wall_time_s": 40.0,
                "dataset_preparation_wall_time_s": 100.0,
            }
        )
    )
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "report-input.json").write_text(json.dumps(inputs))
    monkeypatch.setattr(sys, "argv", ["report", "--campaign", str(tmp_path)])
    report.main()
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["joint_transfer_benefit"] and not result["failures"]
    assert result["costs"]["training_wall_s_all_eight_models"]["full"] == 960
    for name in ("fields", "field-errors", "rollout-and-paired", "training-cost"):
        assert (tmp_path / "report" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
