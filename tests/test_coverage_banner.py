# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PR-comment coverage banner (``_coverage_banner``).

The banner makes "0 regressions" honest about what actually ran (F3 in the
reporting audit): under ``benchmark:solver`` only changed solvers re-run, so
unchanged cells are baseline copies that can never diff.
"""

from __future__ import annotations

import json

from mosaic.benchmarks.cli._status_helpers import _coverage_banner


def _write(tmp_path, scope: dict) -> str:
    p = tmp_path / "run-scope.json"
    p.write_text(json.dumps(scope))
    return str(p)


def test_no_scope_file_is_empty():
    assert _coverage_banner(None) == ""


def test_unreadable_file_is_empty_not_fatal(tmp_path):
    assert _coverage_banner(str(tmp_path / "missing.json")) == ""


def test_all_label_reports_full_coverage(tmp_path):
    banner = _coverage_banner(_write(tmp_path, {"label": "all"}))
    assert "all solvers" in banner


def test_release_pr_reports_full_coverage(tmp_path):
    banner = _coverage_banner(_write(tmp_path, {"label": "", "is_release_pr": True}))
    assert "all solvers" in banner


def test_solver_label_names_what_ran(tmp_path):
    banner = _coverage_banner(
        _write(
            tmp_path,
            {"label": "solver", "solvers": ["exponax"], "problems": ["ns-grid"]},
        )
    )
    assert "`exponax`" in banner
    assert "`ns-grid`" in banner


def test_solver_label_accepts_csv_strings(tmp_path):
    # CI may emit comma-joined strings rather than JSON arrays.
    banner = _coverage_banner(
        _write(
            tmp_path,
            {"label": "solver", "solvers": "a,b", "problems": "ns-grid,thermal"},
        )
    )
    assert "`a`" in banner and "`b`" in banner


def test_unlabelled_scope_states_partial(tmp_path):
    banner = _coverage_banner(_write(tmp_path, {"label": ""}))
    assert "partial run" in banner
