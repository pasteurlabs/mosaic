# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural nullity must not masquerade as poor non-null conditioning."""

import json
from types import SimpleNamespace

import pytest

from mosaic.benchmarks.problems.buoyancy_grid.spectrum import spectrum_diagnostics
from mosaic.benchmarks.problems.shared.plots.charts.sweeps import (
    jacobian_svd_comparison,
)


def test_structural_zeros_are_separate_from_non_null_conditioning():
    values = [1.0] * 112 + [0.8] + [0.0] * 79
    d = spectrum_diagnostics(values, 8)
    assert d["expected_rank"] == d["numerical_rank"] == 113
    assert d["structural_nullity"] == 79
    assert d["condition_non_null"] == 1.25
    assert d["condition_full"] is None


def test_additional_damping_is_not_assumed_to_be_structural():
    values = [1.0] * 111 + [0.01, 1e-8] + [1e-16] * 79
    d = spectrum_diagnostics(values, 8)
    assert d["expected_rank"] == 113
    assert d["numerical_rank"] == 112
    assert d["condition_non_null"] == 100
    assert d["condition_full"] == 1e16


def test_truncated_spectrum_cannot_claim_full_conditioning():
    with pytest.raises(ValueError, match="complete"):
        spectrum_diagnostics([1.0] * 113, 8)


def test_chart_marks_expected_rank_and_reports_both_ratios(tmp_path):
    directory = tmp_path / "jacobian_svd"
    directory.mkdir()
    (directory / "result.json").write_text("{}")
    data = {
        "params": {"physics": {"N": 8, "nu": 0.001, "dt": 0.01, "steps": 4}},
        "per_solver_spectra": {"PhiFlow": [1.0] * 112 + [0.8] + [1e-16] * 79},
    }
    chart = jacobian_svd_comparison(
        SimpleNamespace(dir=tmp_path, problem="buoyancy-grid", legacy=lambda _: data)
    )
    spec = chart.to_dict()
    d = spec["usermeta"]["spectrum_diagnostics"]["PhiFlow"]
    assert next(iter(d.values()))["condition_non_null"] == 1.25
    assert "Expected rank 113/192" in json.dumps(spec)
    assert "Non-null ratio 1.25" in json.dumps(spec)
    assert "better-conditioned gradient" not in json.dumps(spec)
