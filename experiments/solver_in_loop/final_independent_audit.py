"""Independently reproduce final INS selection, paired errors and provenance on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def audit(campaign: Path) -> dict:
    """Read original arrays rather than importing the reporting implementation."""
    inputs = _read(campaign / "report-input.json")
    report = _read(campaign / "report/report.json")
    selected = _read(campaign / "selection.json")
    plan = _read(campaign / "plan.json")
    selection_hash = _hash(campaign / "selection.json")
    assert not inputs["failures"] and not report["failures"]
    assert inputs["selected"] == selected == report["selected"]
    assert _hash(campaign / "source.tar") == plan["base_payload"]["source_sha256"]
    datasets = {}
    for role in ["validation", "test"]:
        folder = campaign / "results" / f"prepare-{role}"
        metadata = _read(folder / "dataset.json")
        assert (
            metadata["admitted"]
            and metadata["closure_passed"]
            and metadata["accuracy_gate_passed"]
        )
        assert metadata["shared_training_arrays_identical"]
        assert _hash(folder / "dataset.npz") == metadata["dataset_sha256"]
        if role == "test":
            assert metadata["selection_sha256"] == selection_hash
        datasets[role] = metadata
    identity = datasets["validation"]["identity"]
    assert datasets["test"]["identity"] == identity
    gate_rows = []
    for file in sorted((campaign / "results").glob("prepare-*-shard*/dataset.json")):
        d = _read(file)
        assert d["admitted"] and d["identity"] == identity
        assert d["reference_audit"]["eligible_for_corrector_training"]
        gate_rows.append(
            {
                "shard": file.parent.name,
                "audit": d["reference_audit"],
                "semigroup_max": max(d["semigroup_errors"]),
                "closure_max": max(d["long_closure_errors"]),
            }
        )
    ranking = {}
    for row in inputs["validation"]:
        result = _read(Path(row["path"]) / "outcome.json")
        candidate = row["candidate"]
        assert result["completed"] and result["admitted"]
        assert result["identity"] == identity
        assert (
            result["training_dataset_sha256"]
            == datasets["validation"]["dataset_sha256"]
        )
        assert result["evaluation_seeds"] == list(range(10000, 10016))
        assert result["model_seed"] == row["model_seed"]
        assert result["arm"] == candidate["arm"]
        assert all(
            result["training"][k] == candidate[v]
            for k, v in [("lr", "lr"), ("unroll", "unroll"), ("max_updates", "updates")]
        )
        with np.load(Path(row["path"]) / "fields.npz", allow_pickle=False) as data:
            value = float(data["error_corrected"][:, 1:].mean())
        assert np.isclose(value, result["mean_rollout_error"], rtol=1e-6)
        ranking.setdefault(
            candidate["candidate_id"],
            {"candidate": candidate, "values": [], "seeds": []},
        )
        ranking[candidate["candidate_id"]]["values"].append(value)
        ranking[candidate["candidate_id"]]["seeds"].append(row["model_seed"])
    assert len(ranking) == 32 and len(inputs["validation"]) == 96
    for r in ranking.values():
        assert sorted(r["seeds"]) == [0, 1, 2]
        r["mean"] = float(np.mean(r["values"]))

    def key(r: dict):
        c = r["candidate"]
        return (
            r["mean"],
            c["updates"] + (c["pretrain"] or {}).get("updates", 0),
            c["lr"],
            c["unroll"],
            c["candidate_id"],
        )

    for arm in ["full", "stopped", "supervised"]:
        best = min(
            [r for r in ranking.values() if r["candidate"]["arm"] == arm], key=key
        )
        assert best["candidate"] == selected[arm]
    extensions = _read(campaign / "extensions.json")
    for arm in ["full", "stopped"]:
        cold = sorted(
            [
                r
                for r in ranking.values()
                if r["candidate"]["arm"] == arm
                and r["candidate"]["pretrain"] is None
                and r["candidate"]["updates"] == 1000
            ],
            key=key,
        )[:2]
        assert extensions[arm] == [
            {
                **r["candidate"],
                "updates": 3000,
                "candidate_id": r["candidate"]["candidate_id"].replace(
                    "u1000", "u3000"
                ),
            }
            for r in cold
        ]
    models, ics = list(range(8, 16)), list(range(20000, 20032))
    assert (
        inputs["expected_model_seeds"] == models and inputs["expected_ic_seeds"] == ics
    )
    training = {}
    gradient_errors = []
    verified_models = {}
    for row in inputs["training"]:
        p = Path(row["path"])
        r = _read(p / "outcome.json")
        assert r["completed"] and r["admitted"] and r["identity"] == identity
        assert r["model_seed"] == row["model_seed"]
        assert r["training_dataset_sha256"] == datasets["validation"]["dataset_sha256"]
        verified_models.setdefault(str(p), _hash(p / "model.eqx"))
        assert verified_models[str(p)] == r["model_sha256"]
        training[row["arm"], row["model_seed"]] = r
        if r["arm"] == "full":
            assert r["gradient_passed"]
            gradient_errors.append(r["gradient_relative_error"])
    values = {arm: np.full((8, 32, 49), np.nan) for arm in selected}
    native = np.full((8, 32, 49), np.nan)
    seen = set()
    for row in inputs["test"]:
        p = Path(row["path"])
        r = _read(p / "outcome.json")
        arm, model = row["arm"], row["model_seed"]
        assert r["completed"] and r["admitted"] and r["identity"] == identity
        assert r["selection_sha256"] == selection_hash
        assert r["model_sha256"] == training[arm, model]["model_sha256"]
        assert r["evaluation_dataset_sha256"] == datasets["test"]["dataset_sha256"]
        assert r["evaluation_seeds"] == row["ic_seeds"] and r["model_seed"] == model
        assert r["arm"] == selected[arm]["arm"]
        with np.load(p / "fields.npz", allow_pickle=False) as data:
            e, b = data["error_corrected"], data["error_uncorrected"]
            assert e.shape == b.shape == (4, 49)
            assert np.isfinite(e).all() and np.isfinite(b).all()
            for j, ic in enumerate(row["ic_seeds"]):
                assert (arm, model, ic) not in seen
                seen.add((arm, model, ic))
                values[arm][model - 8, ic - 20000] = e[j]
                old = native[model - 8, ic - 20000]
                assert np.isnan(old).all() or np.array_equal(old, b[j])
                native[model - 8, ic - 20000] = b[j]
    assert len(seen) == 4 * 8 * 32
    averages = {arm: a[..., 1:].mean(axis=-1) for arm, a in values.items()}
    comparisons = {}
    for arm in ["supervised", "stopped", "matched_stopped"]:
        a, b = averages["full"], averages[arm]
        ratio = float(a.mean() / b.mean())
        rng = np.random.default_rng(11620261003)
        samples = []
        for _ in range(10000):
            m, i = rng.integers(8, size=8), rng.integers(32, size=32)
            samples.append(a[m][:, i].mean() / b[m][:, i].mean())
        ci = np.quantile(samples, [0.025, 0.975]).tolist()
        assert np.isclose(ratio, report["comparisons"][arm]["ratio"], rtol=1e-7)
        assert np.allclose(ci, report["comparisons"][arm]["ci95"], rtol=1e-7)
        comparisons[arm] = {
            "ratio": ratio,
            "ci95": ci,
            "model_wins": int((a.mean(1) < b.mean(1)).sum()),
            "ic_wins": int((a.mean(0) < b.mean(0)).sum()),
        }
    for model in models:
        assert (
            training["stopped", model]["model_sha256"]
            == training["matched_stopped", model]["model_sha256"]
        )
    difference = values["stopped"] - values["matched_stopped"]
    cost = {
        arm: float(
            np.mean(
                [
                    training[arm, m]["method_training_including_pretrain_and_pairs_s"]
                    for m in models
                ]
            )
        )
        for arm in selected
    }
    return {
        "passed": True,
        "selection_sha256": selection_hash,
        "source_sha256": identity["source_sha256"],
        "training_candidates": 32,
        "tuning_runs": 96,
        "paired_test_cells": len(seen),
        "checkpoint_files_verified": len(verified_models),
        "dataset_bytes_rehashed": {k: v["dataset_sha256"] for k, v in datasets.items()},
        "comparisons": comparisons,
        "mean_errors": {k: float(v.mean()) for k, v in averages.items()},
        "max_full_training_primary_gradient_error": max(gradient_errors),
        "training_seconds": cost,
        "full_supervised_training_cost_ratio": cost["full"] / cost["supervised"],
        "stopped_repeat_same_checkpoint": True,
        "stopped_repeat_max_absolute_error_difference": float(np.abs(difference).max()),
        "stopped_repeat_mean_error_difference": float(difference[..., 1:].mean()),
        "reference_gates": gate_rows,
        "resolved_infrastructure_failures": inputs.get(
            "resolved_infrastructure_failures", []
        ),
        "limitations": [
            "Best-found performance under registered search, not matched compute or efficiency.",
            "Full training costs about19 times supervision; tuning costs are additional.",
            "Crossed paired intervals condition on the same eight training ICs.",
            (
                "Matched stopped reuses selected stopped checkpoints; repeated evaluation adds "
                "small floating-point noise, not an independent baseline."
            ),
            "Primary finite differences test initial training points, not every later trajectory.",
            "A single solver/problem/image setting; transfer remains a separate experiment.",
        ],
    }


def main() -> None:
    """Run only through an allocated CPU job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "reference_gates"}))


if __name__ == "__main__":
    main()
