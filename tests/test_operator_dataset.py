# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for multi-resolution data, coverage, and resumable publication."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1] / "tools/xlb_surrogate"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data = load("operator_dataset")
generator = load("generate_operator_data")


@pytest.mark.parametrize("index", range(6))
def test_parent_field_and_split_agree_across_resolutions(index):
    coarse, a = data.make_field(8, 41, index)
    fine, b = data.make_field(16, 41, index)
    np.testing.assert_allclose(coarse, fine[::2, ::2, ::2], atol=2e-7)
    assert a == b
    assert np.isfinite(coarse).all()


def test_divergence_free_families_and_off_manifold_perturbations():
    n = 16
    wave = np.stack(np.meshgrid(*([np.fft.fftfreq(n) * n] * 3), indexing="ij"), axis=-1)
    for index in range(6):
        field, _ = data.make_field(n, 71, index)
        div = np.fft.ifftn(
            1j * np.sum(np.fft.fftn(field, axes=(0, 1, 2)) * wave, axis=-1)
        ).real
        if index == 5:
            assert np.max(np.abs(div)) > 1e-4
        else:
            assert np.max(np.abs(div)) < 1e-6


def test_splits_are_stable_and_nonempty():
    splits = [data.parent_identity(12, i)[1] for i in range(1000)]
    assert set(splits) == {"train", "validation", "test"}
    assert splits == [data.parent_identity(12, i)[1] for i in range(1000)]


@pytest.mark.parametrize("steps", [1, 5, 10, 37, 100, 10240])
def test_sparse_schedule_retains_dense_windows_and_terminal(steps):
    schedule = data.snapshot_steps(steps)
    assert schedule == sorted(set(schedule))
    assert schedule[0] == 0 and schedule[-1] == steps
    assert set(range(max(0, steps - 10), steps + 1)) <= set(schedule)
    assert set(range(min(steps, 10) + 1)) <= set(schedule)


def test_coverage_uses_registered_physics_and_xlb_scaling():
    manifest = data.benchmark_manifest()
    assert {c["N"] for c in manifest["cases"]} == {8, 16, 20, 32, 48, 64}
    case = next(
        c
        for c in manifest["cases"]
        if c["experiment"] == "cost/spatial_cost" and c["N"] == 64
    )
    assert case["steps"] == 200
    assert case["dt"] == pytest.approx(0.0025)
    assert case["nominal_physics"]["steps"] == 50
    assert max(c["steps"] for c in manifest["cases"]) == 10240
    assert len({c["id"] for c in manifest["cases"]}) == len(manifest["cases"])


def test_resume_requires_matching_source_and_valid_content(tmp_path):
    path = tmp_path / "shard-00000.npz"
    arrays = {"velocity": np.zeros((2, 3, 8, 8, 8, 3), np.float32)}
    assert generator.completed_shard(path, "a") is None
    record = generator.write_shard(path, arrays, {"signature": "a"})
    assert generator.completed_shard(path, "a") == record
    with pytest.raises(ValueError, match="identity mismatch"):
        generator.completed_shard(path, "b")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        generator.completed_shard(path, "a")


def test_native_rollout_preserves_hidden_state_and_snapshot_intervals():
    # A fake teacher with a hidden counter makes restarting from decoded fields
    # observably incorrect: output increments are 1+2+...+steps.
    import jax.numpy as jnp

    api = SimpleNamespace(
        _diff_statics=lambda _: (2, "kbc"),
        _OPS={
            (3, False, "kbc"): {
                "eq": lambda rho, u: (u, jnp.asarray(0)),
                "stream": lambda f: f,
                "macro": lambda f: (jnp.ones_like(f[0][:1]), f[0]),
                "bgk": lambda f, eq, rho, u, omega: (f[0] + f[1] + 1, f[1] + 1),
            }
        },
    )
    case = {"N": 8, "dt": 0.1, "viscosity": 0.01, "domain_extent": 1.0}
    run = generator.native_runner(api, case, False, [0, 1, 3, 4])
    initial = jnp.zeros((2, 8, 8, 8, 3), dtype=jnp.float32)
    actual = np.asarray(run(initial, *generator.lattice_parameters(case, 2, False)))
    scale = 0.1 / 2 * 8
    expected = np.array([0, 3, 21, 36]) / scale
    np.testing.assert_allclose(actual[:, :, 0, 0, 0, 0], np.tile(expected, (2, 1)))


def test_public_lattice_unit_conversion_matches_python_constants():
    case = {
        "N": 8,
        "dt": 0.05000000074505806,
        "viscosity": 0.0010000000474974513,
        "domain_extent": 2 * np.pi,
    }
    scale, omega = generator.lattice_parameters(case, 1, False)
    dx = case["domain_extent"] / case["N"]
    assert scale == np.float32(case["dt"] / dx)
    assert omega == np.float32(1 / (3 * case["viscosity"] * case["dt"] / dx**2 + 0.5))


def test_report_preserves_missing_cases_and_counts_invalid_rows(tmp_path):
    reporting = load("summarize_operator_data")
    case = {
        "id": "case",
        "N": 8,
        "viscosity": 0.01,
        "dt": 0.01,
        "steps": 20,
        "domain_extent": 2 * np.pi,
        "samples": 4,
    }
    manifest = {"cases": [case], "precision": "float32"}
    assert reporting.summarize(tmp_path, manifest)["cases"][0]["status"] == "missing"
    folder = tmp_path / "case" / "float32"
    folder.mkdir(parents=True)
    metadata = {
        "signature": "source",
        "shard_size": 4,
        "start": 0,
        "stop": 4,
        "count": 4,
        "valid_count": 3,
        "compute_seconds": 2.0,
        "write_seconds": 1.0,
        "ic_seconds": 0.2,
        "transfer_seconds": 0.1,
        "bytes": 400,
        "batch_size": 2,
        "trajectory_checks": {
            "native_vs_api": [{"max_abs": 1e-6}],
            "final_vs_float64": [
                {"finite": True, "relative_l2": 0.01},
                {"finite": False, "relative_l2": None},
            ],
        },
    }
    generator.atomic_json(folder / "shard-00000.json", metadata)
    row = reporting.summarize(tmp_path, manifest, target_samples=8)["cases"][0]
    assert row["status"] == "complete"
    assert row["invalid"] == 1
    assert row["float64_comparison_nonfinite"] == 1
    assert row["projection"]["compute_gpu_hours"] == pytest.approx(4 / 3600)
    assert row["projection"]["storage_gib"] == pytest.approx(800 / 2**30)


def test_report_distinguishes_failed_generation_from_missing(tmp_path):
    reporting = load("summarize_operator_data")
    folder = tmp_path / "failed-case" / "float64"
    folder.mkdir(parents=True)
    generator.atomic_json(
        folder / "failure-0-123.json",
        {
            "error_type": "RuntimeError",
            "error": "teacher trajectory parity failed",
        },
    )
    report = reporting.summarize(tmp_path, {"cases": [{"id": "failed-case", "N": 8}]})
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["failures"][0]["error_type"] == "RuntimeError"


def test_expanded_manifest_covers_physics_without_dense_stress_trajectories():
    manifest = data.training_manifest(data.benchmark_manifest())
    assert sum(c["samples"] for c in manifest["cases"]) == 5376
    assert len(manifest["cases"]) == 16
    assert {c["N"] for c in manifest["cases"] if c["role"] == "train"} == {8, 16, 32}
    assert {c["N"] for c in manifest["cases"] if c["role"] == "resolution_holdout"} == {
        20,
        48,
        64,
    }
    assert max(len(c["snapshot_steps"]) for c in manifest["cases"]) <= 64
    for case in manifest["cases"]:
        assert set(range(21)) <= set(case["snapshot_steps"])
        assert case["snapshot_steps"][-1] == case["steps"]
