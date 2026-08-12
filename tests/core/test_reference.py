# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for precomputed reference loading/saving and extraction."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.core.reference import (
    CONVERGED_REFERENCES,
    PRECOMPUTED_EXPERIMENTS,
    ConvergedSpec,
    _domain_slug_to_package,
    _reference_dir,
    _reference_filename,
    converged_spec,
    extract_references_from_fields,
    is_precomputed_experiment,
    load_reference,
    reference_exists,
    save_reference,
    spectral_downsample,
)


class TestReferenceFilename:
    def test_forward_baseline(self):
        assert _reference_filename("forward/baseline") == "forward_baseline.npz"

    def test_forward_source_linearity(self):
        assert (
            _reference_filename("forward/source_linearity")
            == "forward_source_linearity.npz"
        )


class TestSlugToPackage:
    def test_default_hyphen_to_underscore(self):
        assert _domain_slug_to_package("structural-mesh") == "structural_mesh"
        assert _domain_slug_to_package("ns-grid") == "ns_grid"

    def test_ns_3d_grid_alias(self):
        # The 3D slug does not follow the ``-`` -> ``_`` convention; its
        # package (and reference dir) is navier_stokes_3d_grid.
        assert _domain_slug_to_package("ns-3d-grid") == "navier_stokes_3d_grid"


class TestIsPrecomputedExperiment:
    def test_designated(self):
        assert is_precomputed_experiment("ns-3d-grid", "forward/agreement")

    def test_not_designated(self):
        assert not is_precomputed_experiment("ns-3d-grid", "forward/baseline")

    def test_unknown_domain(self):
        assert not is_precomputed_experiment("nope", "forward/agreement")


class TestConvergedReference:
    def test_ns_3d_grid_agreement_is_converged(self):
        spec = converged_spec("ns-3d-grid", "forward/agreement")
        assert isinstance(spec, ConvergedSpec)
        assert spec.high_n > 16  # above the benchmark grid

    def test_consensus_experiment_has_no_converged_spec(self):
        # structural-mesh agreement is a plain consensus reference.
        assert converged_spec("structural-mesh", "forward/agreement") is None

    def test_every_converged_experiment_is_precomputed(self):
        # A converged strategy only makes sense for a checked-in reference.
        for domain, exp_key in CONVERGED_REFERENCES:
            assert is_precomputed_experiment(domain, exp_key)


class TestSpectralDownsample:
    def test_band_limited_is_exact(self):
        """A field with no energy above the target Nyquist truncates exactly."""
        n, nt = 32, 16
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        X, Y, _ = np.meshgrid(x, x, x, indexing="ij")
        # Modes at |k|<=1, well below the nt=16 cutoff.
        field = np.stack(
            [np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), np.zeros_like(X)],
            axis=-1,
        )
        ds = spectral_downsample(field, nt)

        x2 = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        X2, Y2, _ = np.meshgrid(x2, x2, x2, indexing="ij")
        direct = np.stack(
            [np.sin(X2) * np.cos(Y2), -np.cos(X2) * np.sin(Y2), np.zeros_like(X2)],
            axis=-1,
        )
        assert ds.shape == (nt, nt, nt, 3)
        np.testing.assert_allclose(ds, direct, atol=1e-12)

    def test_upsample_rejected(self):
        field = np.zeros((8, 8, 8, 3))
        with pytest.raises(ValueError):
            spectral_downsample(field, 16)


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

    def test_ns_3d_grid_experiments(self):
        # forward/agreement uses a converged spectral reference, not the
        # linearized analytic decay (issue #123).
        exps = PRECOMPUTED_EXPERIMENTS["ns-3d-grid"]
        assert "forward/agreement" in exps


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
