"""Render the complete registered matrix on a Slurm CPU allocation."""

import json
import sys

import numpy as np
import pytest

from experiments.solver_in_loop import final_publish, final_report


def _json(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("full_error,positive", [(0.08, True), (0.11, False)])
def test_complete_report_renders_and_publishes_classification(
    tmp_path, monkeypatch, full_error, positive
):
    arms = ["full", "stopped", "supervised", "matched_stopped"]
    selected = {
        arm: {
            "candidate_id": arm,
            "arm": "stopped" if arm == "matched_stopped" else arm,
            "lr": 1e-5,
            "unroll": 8,
            "updates": 1000,
            "pretrain": None,
        }
        for arm in arms
    }
    models, ics = list(range(8, 16)), list(range(20000, 20032))
    inputs = {
        "selected": selected,
        "expected_model_seeds": models,
        "expected_ic_seeds": ics,
        "training": [],
        "test": [],
        "validation": [],
        "failures": [],
    }
    x = np.arange(64, dtype=np.float32) * (2 * np.pi / 64)
    velocity = np.zeros((49, 64, 64, 1, 2), dtype=np.float32)
    velocity[..., 0] = np.sin(x)[None, :, None, None]
    rates = {
        "full": full_error,
        "supervised": 0.1,
        "stopped": 0.12,
        "matched_stopped": 0.1,
    }
    for arm in arms:
        for model in models:
            training = tmp_path / f"train-{arm}-{model}"
            training.mkdir()
            checkpoint = f"{arm}-{model}"
            _json(
                training / "outcome.json",
                {
                    "completed": True,
                    "admitted": True,
                    "model_sha256": checkpoint,
                    "model_seed": model,
                    "arm": selected[arm]["arm"],
                    "training": {"lr": 1e-5, "unroll": 8, "max_updates": 1000},
                    "method_training_including_pretrain_and_pairs_s": 120,
                },
            )
            inputs["training"].append(
                {"arm": arm, "model_seed": model, "path": str(training)}
            )
            for offset in range(0, 32, 4):
                path = tmp_path / f"test-{arm}-{model}-{offset}"
                path.mkdir()
                batch = ics[offset : offset + 4]
                _json(
                    path / "outcome.json",
                    {
                        "completed": True,
                        "admitted": True,
                        "model_seed": model,
                        "arm": selected[arm]["arm"],
                        "evaluation_seeds": batch,
                        "evaluation_dataset_sha256": "shared-test",
                        "model_sha256": checkpoint,
                    },
                )
                errors = np.full((4, 49), rates[arm])
                native = np.full((4, 49), 0.2)
                errors[:, 0] = native[:, 0] = 0
                arrays = {"error_corrected": errors, "error_uncorrected": native}
                if model == 8 and offset == 0:
                    arrays.update(
                        reference_rollout=velocity,
                        rollout_uncorrected=velocity + 0.2,
                        rollout_corrected=velocity + rates[arm],
                    )
                np.savez_compressed(path / "fields.npz", **arrays)
                inputs["test"].append(
                    {
                        "arm": arm,
                        "model_seed": model,
                        "ic_seeds": batch,
                        "path": str(path),
                    }
                )
    _json(tmp_path / "report-input.json", inputs)
    monkeypatch.setattr(sys, "argv", ["final_report", "--campaign", str(tmp_path)])
    final_report.main()
    output = tmp_path / "report"
    report = json.loads((output / "report.json").read_text())
    assert report["report_complete"] and not report["failures"]
    assert report["credible_supervised_advantage"] is positive
    assert report["matched_solver_gradient_benefit"] is positive
    assert report["comparisons"]["supervised"]["ratio"] == pytest.approx(
        full_error / 0.1
    )
    assert report["mean_training_seconds_including_pretrain_and_pairs"]["full"] == 120
    assert np.asarray(report["errors_by_model_ic"]["full"]).shape == (8, 32)
    for name in [
        "rollout-and-paired.png",
        "fields.png",
        "field-errors.png",
        "training-cost.png",
    ]:
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (output / name).stat().st_size > 1000
    section = (output / "PRsection.md").read_text()
    assert "≥5%" in section
    assert ("beats tuned supervision" in section) is positive
    replaced = final_publish.merge_section(
        "Earlier negative results remain.\n", section
    )
    assert replaced.startswith("Earlier negative results remain.")
    assert replaced.count(final_publish.START) == 1
    assert final_publish.merge_section(replaced, section) == replaced
