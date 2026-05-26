# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-(experiment, solver) status filter behind ``--only``."""

from __future__ import annotations

import pytest

from mosaic.benchmarks.core.cell_filter import (
    _cell_matches,
    build_filter,
    filter_solvers,
    set_active,
)
from mosaic.benchmarks.core.status import ANOMALY, EXCLUDED, FAILED, NOT_RUN, OK, Cell


class TestCellMatches:
    """Single-cell match predicate that drives every higher-level filter call."""

    def test_failed_matches_failed_state(self) -> None:
        assert _cell_matches(Cell(FAILED), {"failed"})
        assert not _cell_matches(Cell(OK), {"failed"})

    def test_stale_matches_independently_of_status(self) -> None:
        assert _cell_matches(Cell(OK, stale=True), {"stale"})
        assert _cell_matches(Cell(FAILED, stale=True), {"stale"})
        assert not _cell_matches(Cell(OK, stale=False), {"stale"})

    def test_missing_maps_to_not_run(self) -> None:
        assert _cell_matches(Cell(NOT_RUN), {"missing"})
        assert not _cell_matches(Cell(OK), {"missing"})

    def test_anom_and_excluded(self) -> None:
        assert _cell_matches(Cell(ANOMALY), {"anom"})
        assert _cell_matches(Cell(EXCLUDED), {"excluded"})
        assert not _cell_matches(Cell(ANOMALY), {"excluded"})

    def test_state_set_union_semantics(self) -> None:
        # ``failed,stale`` matches anything that's failed OR stale, in either combo.
        cell = Cell(OK, stale=True)
        assert _cell_matches(cell, {"failed", "stale"})
        cell2 = Cell(FAILED, stale=False)
        assert _cell_matches(cell2, {"failed", "stale"})


class TestActiveFilterLifecycle:
    """``set_active`` / ``get_active`` / ``filter_solvers`` thread-local plumbing."""

    def setup_method(self) -> None:
        set_active(None)

    def teardown_method(self) -> None:
        set_active(None)

    def test_passthrough_when_no_filter_set(self) -> None:
        assert filter_solvers("forward/baseline", ["a", "b", "c"]) == ["a", "b", "c"]

    def test_prunes_to_matching_pairs(self) -> None:
        set_active({("forward/baseline", "a"): True, ("forward/baseline", "c"): True})
        assert filter_solvers("forward/baseline", ["a", "b", "c"]) == ["a", "c"]

    def test_unrelated_experiment_filters_all(self) -> None:
        set_active({("gradient/fd_check", "a"): True})
        assert filter_solvers("forward/baseline", ["a", "b"]) == []


class TestBuildFilter:
    """``build_filter`` validates state names before touching status."""

    def test_rejects_unknown_states(self) -> None:
        from mosaic.benchmarks.core.config import Problem

        cfg = Problem(name="bogus-problem")
        with pytest.raises(ValueError, match="unknown state"):
            build_filter(cfg, None, {"bogus"})

    def test_accepts_each_valid_state(self) -> None:
        from mosaic.benchmarks.core.config import Problem

        cfg = Problem(name="bogus-problem")
        # No results on disk → empty filter, but no validation error.
        for state in ("failed", "anom", "missing", "stale", "excluded"):
            result = build_filter(cfg, None, {state})
            assert result == {}, f"empty cfg should yield empty filter for {state}"
