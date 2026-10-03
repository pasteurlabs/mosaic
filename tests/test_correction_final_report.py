import json

import numpy as np

from experiments.solver_in_loop.final_publish import merge_section
from experiments.solver_in_loop.final_report import collect, paired_ratio


def test_ratio_resamples_both_paired_axes():
    baseline = np.arange(1, 33, dtype=float).reshape(4, 8)
    result = paired_ratio(baseline * 0.8, baseline)
    assert abs(result["ratio"] - 0.8) < 1e-12
    np.testing.assert_allclose(result["ci95"], [0.8, 0.8])
    assert result["model_seed_win_count"] == 4
    assert result["ic_win_count"] == 8


def test_complete_matrix_required_and_checkpoint_bound(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    metrics = {
        "completed": True,
        "admitted": True,
        "evaluation_seeds": [20000],
        "model_seed": 8,
        "model_sha256": "model",
        "evaluation_dataset_sha256": "data",
    }
    (trial / "outcome.json").write_text(json.dumps(metrics))
    np.savez(
        trial / "fields.npz",
        error_corrected=np.ones((1, 49)),
        error_uncorrected=np.ones((1, 49)),
    )
    inputs = {
        "expected_model_seeds": [8],
        "expected_ic_seeds": [20000],
        "selected": {"full": {}},
        "training": [{"arm": "full", "model_seed": 8, "path": str(trial)}],
        "test": [
            {"arm": "full", "model_seed": 8, "ic_seeds": [20000], "path": str(trial)}
        ],
    }
    _, _, failures = collect(inputs)
    assert not failures
    inputs["expected_ic_seeds"].append(20001)
    _, _, failures = collect(inputs)
    assert any("missing model/IC pairs" in failure for failure in failures)
    inputs["expected_ic_seeds"].pop()
    inputs["test"].append(dict(inputs["test"][0]))
    _, _, failures = collect(inputs)
    assert any("duplicate" in failure for failure in failures)


def test_publication_preserves_unrelated_pr_content():
    body = "Existing positive direct-control and negative neural-control evidence."
    first = merge_section(body, "Pending.")
    second = merge_section(first, "Final negative outcome.")
    assert body in second
    assert "Pending." not in second
    assert second.count("Final negative outcome.") == 1
