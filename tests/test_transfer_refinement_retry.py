"""Infrastructure retries require verified completed cache and remain bounded."""

import hashlib
import json

import numpy as np
import pytest

from experiments.solver_in_loop.transfer_refinement import (
    RefinementController,
    verified_cache_files,
)


def cache(tmp_path, *, corrupt=False):
    payload = {
        "source_sha256": "source",
        "image": "image",
        "run": {
            "physics": {"dt": 0.01},
            "dataset": {
                "reference_temporal_factor": 192,
                "burn_in_cache_dir": str(tmp_path),
            },
        },
    }
    value = np.ones((2, 2), dtype=np.float32)
    metadata = {
        "binding": "source:image",
        "physics": {"dt": 0.01},
        "dt": 0.01 / 192,
        "shape": [2, 2],
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    np.savez(
        tmp_path / f"{key}.npz",
        state=value,
        metadata=json.dumps(metadata),
        key=key,
        sha256="wrong" if corrupt else hashlib.sha256(value.tobytes()).hexdigest(),
    )
    return payload


def test_completed_cache_identity(tmp_path):
    payload = cache(tmp_path)
    assert len(verified_cache_files(payload)) == 1
    payload["source_sha256"] = "changed"
    assert not verified_cache_files(payload)


def test_corrupt_completed_cache_rejected(tmp_path):
    with pytest.raises(ValueError, match="byte identity"):
        verified_cache_files(cache(tmp_path, corrupt=True))


@pytest.mark.parametrize(
    "state,should_retry",
    [("TIMEOUT", True), ("FAILED", False), ("OUT_OF_MEMORY", False)],
)
def test_only_timeout_retries_once(tmp_path, monkeypatch, state, should_retry):
    payload = cache(tmp_path)
    (tmp_path / "timeout-cache-retry-policy.json").write_text(
        json.dumps({"source_sha256": "source", "max_retries_per_cell": 1})
    )
    controller = object.__new__(RefinementController)
    controller.campaign = tmp_path
    controller.state = {"jobs": {"reference": {"job_id": 12}}}
    submitted = []
    controller.submit = lambda cell, phase, payload: submitted.append(cell)
    controller.save = lambda: None

    def wait(cells):
        if cells == ["reference"]:
            raise RuntimeError("terminal infrastructure failure")

    controller.wait = wait
    monkeypatch.setattr(
        "experiments.solver_in_loop.transfer_refinement.subprocess.check_output",
        lambda *a, **k: f"12|{state}|\n",
    )
    if should_retry:
        assert (
            controller.wait_reference("reference", payload)
            == "reference-timeout-retry1"
        )
        assert submitted == ["reference-timeout-retry1"]
        controller.wait = lambda cells: (_ for _ in ()).throw(
            RuntimeError("retry timed out")
        )
        with pytest.raises(RuntimeError, match="retry timed out"):
            controller.wait_reference("reference", payload)
        assert set(submitted) == {"reference-timeout-retry1"}
    else:
        with pytest.raises(RuntimeError):
            controller.wait_reference("reference", payload)
        assert not submitted
