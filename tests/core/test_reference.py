# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for precomputed reference loading/saving and extraction."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.core.reference import (
    PRECOMPUTED_EXPERIMENTS,
    _reference_dir,
    _reference_filename,
    extract_references_from_fields,
    load_reference,
    reference_exists,
    save_reference,
)


class TestReferenceFilename:
    def test_forward_baseline(self):
        assert _reference_filename("forward/baseline") == "forward_baseline.npz"

    def test_forward_source_linearity(self):
        assert (
            _reference_filename("forward/source_linearity")
            == "forward_source_linearity.npz"
        )


class TestSaveAndLoadReference:
    def test_roundtrip(self, tmp_path, monkeypatch):
        """Save references and load them back; values must match."""
        monkeypatch.setattr("mosaic.benchmarks.core.reference._PROBLEMS_DIR", tmp_path)
        refs = {
            0: np.array(1.5),
            1: np.array(2.7),
            2: np.array(3.14),
        }
        sweep_values = [0.1, 0.5, 1.0]
        save_reference("test-domain", "forward/baseline", refs, sweep_values)

        assert reference_exists("test-domain", "forward/baseline")

        for i, expected in refs.items():
            loaded = load_reference("test-domain", "forward/baseline", i)
            assert loaded is not None
            np.testing.assert_allclose(loaded, expected)

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """load_reference returns None when the NPZ doesn't exist."""
        monkeypatch.setattr("mosaic.benchmarks.core.reference._PROBLEMS_DIR", tmp_path)
        assert load_reference("nonexistent", "forward/baseline", 0) is None

    def test_load_missing_index(self, tmp_path, monkeypatch):
        """load_reference returns None for an index not in the NPZ."""
        monkeypatch.setattr("mosaic.benchmarks.core.reference._PROBLEMS_DIR", tmp_path)
        save_reference("test-domain", "forward/baseline", {0: np.array(1.0)})
        assert load_reference("test-domain", "forward/baseline", 99) is None

    def test_reference_exists_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("mosaic.benchmarks.core.reference._PROBLEMS_DIR", tmp_path)
        assert not reference_exists("test-domain", "forward/baseline")

    def test_array_reference(self, tmp_path, monkeypatch):
        """Reference arrays with shape > scalar round-trip correctly."""
        monkeypatch.setattr("mosaic.benchmarks.core.reference._PROBLEMS_DIR", tmp_path)
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((3, 4, 5)).astype(np.float32)
        save_reference("test-domain", "forward/agreement", {0: ref})
        loaded = load_reference("test-domain", "forward/agreement", 0)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, ref)


class TestExtractReferencesFromFields:
    def test_extract_consensus(self, tmp_path):
        """Extracts consensus_* arrays from a fields.npz."""
        npz_path = tmp_path / "fields.npz"
        arrays = {
            "consensus_0": np.array(1.0),
            "consensus_1": np.array(2.0),
            "consensus_2": np.array(3.0),
            "solver_A_0": np.array(1.1),
            "solver_A_1": np.array(2.1),
            "sweep_values": np.array([10.0, 20.0, 30.0]),
        }
        np.savez(npz_path, **arrays)

        refs = extract_references_from_fields(npz_path, 3)
        assert len(refs) == 3
        np.testing.assert_allclose(refs[0], 1.0)
        np.testing.assert_allclose(refs[1], 2.0)
        np.testing.assert_allclose(refs[2], 3.0)

    def test_missing_file(self, tmp_path):
        refs = extract_references_from_fields(tmp_path / "nope.npz", 3)
        assert refs == {}

    def test_partial_consensus(self, tmp_path):
        """Only consensus_0 present, consensus_1 missing."""
        npz_path = tmp_path / "fields.npz"
        np.savez(npz_path, consensus_0=np.array(42.0))
        refs = extract_references_from_fields(npz_path, 2)
        assert 0 in refs
        assert 1 not in refs


class TestPrecomputedExperiments:
    def test_ns_grid_experiments(self):
        exps = PRECOMPUTED_EXPERIMENTS["ns-grid"]
        assert "forward/cylinder" in exps

    def test_structural_mesh_experiments(self):
        exps = PRECOMPUTED_EXPERIMENTS["structural-mesh"]
        assert "forward/baseline" in exps
        assert "forward/agreement" in exps

    def test_thermal_mesh_experiments(self):
        exps = PRECOMPUTED_EXPERIMENTS["thermal-mesh"]
        assert "forward/baseline" in exps
        assert "forward/agreement" in exps
        assert "forward/source_baseline" in exps
        assert "forward/source_linearity" in exps

    def test_ns_3d_grid_not_listed(self):
        assert "ns-3d-grid" not in PRECOMPUTED_EXPERIMENTS


class TestCheckedInReferences:
    """Verify that the checked-in reference NPZs exist and are loadable."""

    @pytest.mark.parametrize(
        "domain,exp_key",
        [
            (domain, exp_key)
            for domain, exps in PRECOMPUTED_EXPERIMENTS.items()
            for exp_key in exps
        ],
    )
    def test_reference_file_exists(self, domain, exp_key):
        ref_dir = _reference_dir(domain)
        path = ref_dir / _reference_filename(exp_key)
        assert path.exists(), f"Missing reference: {path}"

    @pytest.mark.parametrize(
        "domain,exp_key",
        [
            (domain, exp_key)
            for domain, exps in PRECOMPUTED_EXPERIMENTS.items()
            for exp_key in exps
        ],
    )
    def test_reference_loadable(self, domain, exp_key):
        """Each checked-in reference must have at least one reference_* array."""
        ref_dir = _reference_dir(domain)
        path = ref_dir / _reference_filename(exp_key)
        with np.load(str(path), allow_pickle=False) as data:
            ref_keys = [k for k in data.files if k.startswith("reference_")]
            assert len(ref_keys) > 0, f"No reference_* arrays in {path}"
            assert "sweep_values" in data, f"No sweep_values in {path}"
