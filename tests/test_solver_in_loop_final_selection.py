"""Bounded INS selection tests use synthetic JSON, never a fluid simulation."""

import json
import math

import pytest

from experiments.solver_in_loop.final_select import select


def _save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    candidates = []
    for arm in ["full", "stopped", "supervised"]:
        for lr in [1e-5, 3e-5, 1e-4]:
            for horizon in [8] if arm == "supervised" else [4, 8, 16]:
                for updates in [1000, 3000] if arm == "supervised" else [1000]:
                    candidates.append(
                        {
                            "candidate_id": f"{arm}-lr{lr}-h{horizon}-u{updates}",
                            "arm": arm,
                            "lr": lr,
                            "unroll": horizon,
                            "updates": updates,
                            "pretrain": None,
                        }
                    )
    for arm in ["full", "stopped"]:
        for lr in [1e-5, 3e-5]:
            candidates.append(
                {
                    "candidate_id": f"{arm}-warm-{lr}",
                    "arm": arm,
                    "lr": lr,
                    "unroll": 8,
                    "updates": 1000,
                    "pretrain": {"lr": 1e-5, "unroll": 8, "updates": 1000},
                }
            )
    base = {
        "solver": "ins",
        "image": "image",
        "source_sha256": "source",
        "image_sha256": "imagehash",
        "run": {"physics": {"nu": 0.001}},
    }
    _save(tmp_path / "plan.json", {"candidates": candidates, "base_payload": base})
    rows = []

    def add(candidate):
        for seed in range(3):
            path = tmp_path / f"{candidate['candidate_id']}-{seed}"
            path.mkdir()
            result = {
                "completed": True,
                "admitted": True,
                "arm": candidate["arm"],
                "model_seed": seed,
                "evaluation_seeds": list(range(10000, 10016)),
                "training": {
                    "lr": candidate["lr"],
                    "unroll": candidate["unroll"],
                    "max_updates": candidate["updates"],
                },
                "identity": {
                    **{
                        k: base[k]
                        for k in ["solver", "image", "source_sha256", "image_sha256"]
                    },
                    "physics": base["run"]["physics"],
                    "domain_extent": 2 * math.pi,
                },
                "initial_model_sha256": "pretrained" if candidate["pretrain"] else None,
                "training_dataset_sha256": "dataset",
                "mean_rollout_error": 0.1,
            }
            _save(path / "outcome.json", result)
            rows.append({"candidate": candidate, "model_seed": seed, "path": str(path)})
        _save(tmp_path / "report-input.json", {"validation": rows})

    for candidate in candidates:
        add(candidate)
    return tmp_path, rows, add


def test_extensions_frozen_grid_tie_break_and_matched_reuse(campaign):
    path, _, add = campaign
    extension = select(path, "select_extension")
    assert extension["full"][0]["lr"] == 1e-5
    assert extension["full"][0]["unroll"] == 4
    assert extension["full"][1]["unroll"] == 8
    assert all(c["updates"] == 3000 for group in extension.values() for c in group)
    for group in extension.values():
        for candidate in group:
            add(candidate)
    chosen = select(path, "select_final")
    assert chosen["matched_stopped"] == chosen["stopped"]
    assert chosen == select(path, "select_final")


@pytest.mark.parametrize(
    "defect",
    [
        "missing_file",
        "missing_seed",
        "duplicate_seed",
        "wrong_ic",
        "wrong_lr",
        "dataset",
    ],
)
def test_invalid_identity_or_infrastructure_blocks_selection(campaign, defect):
    path, rows, _ = campaign
    file = path / (rows[0]["candidate"]["candidate_id"] + "-0") / "outcome.json"
    result = json.loads(file.read_text())
    if defect == "missing_file":
        file.unlink()
    elif defect in ["missing_seed", "duplicate_seed"]:
        modified = rows[1:] if defect == "missing_seed" else [*rows, rows[0]]
        _save(path / "report-input.json", {"validation": modified})
    else:
        if defect == "wrong_ic":
            result["evaluation_seeds"][0] = 20000
        elif defect == "wrong_lr":
            result["training"]["lr"] = 0.5
        else:
            result["training_dataset_sha256"] = "other"
        _save(file, result)
    with pytest.raises((ValueError, FileNotFoundError)):
        select(path, "select_extension")
    assert not (path / "extensions.json").exists()


def test_failed_candidate_is_not_averaged_over_survivors(campaign):
    path, rows, _ = campaign
    file = path / (rows[0]["candidate"]["candidate_id"] + "-0") / "outcome.json"
    result = json.loads(file.read_text())
    result.update(completed=False, admitted=False, failure="nonfinite training")
    _save(file, result)
    chosen = select(path, "select_extension")
    assert all(c["unroll"] != 4 for c in chosen["full"])
    score = json.loads((path / "select_extension-scores.json").read_text())["scores"][0]
    assert not score["eligible"] and score["mean_rollout_error"] is None
    assert score["failures"][0]["model_seed"] == 0
