# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `mosaic timing-report` aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.benchmarks.cli.timing_report import (
    _emit_markdown,
    _format_secs,
    _walk_results,
)


def _write_result(
    root: Path, problem: str, suite: str, experiment: str, wall: dict[str, float]
) -> None:
    """Drop a minimal ``result.json`` containing only ``wall_time_s``."""
    d = root / problem / suite / experiment
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"wall_time_s": wall}))


class TestFormatSecs:
    @pytest.mark.parametrize(
        "secs,expected",
        [
            (0.5, "0.5s"),
            (1.234, "1.2s"),
            (59.9, "59.9s"),
            (60, "1m00s"),
            (75, "1m15s"),
            (3599, "59m59s"),
            (3600, "1h00m"),
            (3661, "1h01m"),
        ],
    )
    def test_human_readable(self, secs: float, expected: str):
        assert _format_secs(secs) == expected


class TestWalkResults:
    def test_empty_root(self, tmp_path: Path):
        assert _walk_results(tmp_path) == []

    def test_missing_root(self, tmp_path: Path):
        assert _walk_results(tmp_path / "does-not-exist") == []

    def test_extracts_wall_times(self, tmp_path: Path):
        _write_result(
            tmp_path, "ns-grid", "forward", "tgv", {"jax_cfd": 1.5, "phiflow": 2.5}
        )
        rows = _walk_results(tmp_path)
        assert len(rows) == 2
        rows_by_solver = {r["solver"]: r for r in rows}
        assert rows_by_solver["jax_cfd"]["seconds"] == pytest.approx(1.5)
        assert rows_by_solver["jax_cfd"]["mode"] == "prod"
        assert rows_by_solver["jax_cfd"]["problem"] == "ns-grid"
        assert rows_by_solver["jax_cfd"]["suite"] == "forward"
        assert rows_by_solver["jax_cfd"]["experiment"] == "tgv"

    def test_debug_suffix_recognised(self, tmp_path: Path):
        """`<experiment>_debug/` should be tagged ``mode=debug`` with the suffix stripped."""
        _write_result(
            tmp_path, "ns-grid", "optimization", "drag_opt_debug", {"phiflow": 1.0}
        )
        rows = _walk_results(tmp_path)
        assert len(rows) == 1
        assert rows[0]["mode"] == "debug"
        assert rows[0]["experiment"] == "drag_opt"

    def test_ignores_non_numeric_entries(self, tmp_path: Path):
        """Strings / None in ``wall_time_s`` are silently skipped, not crashes."""
        _write_result(
            tmp_path,
            "ns-grid",
            "forward",
            "tgv",
            {"jax_cfd": 1.0, "phiflow": "n/a", "xlb": None},  # type: ignore[dict-item]
        )
        rows = _walk_results(tmp_path)
        assert {r["solver"] for r in rows} == {"jax_cfd"}

    def test_ignores_malformed_json(self, tmp_path: Path):
        d = tmp_path / "ns-grid" / "forward" / "tgv"
        d.mkdir(parents=True)
        (d / "result.json").write_text("{ not valid json")
        # Should silently skip rather than raise.
        assert _walk_results(tmp_path) == []


class TestEmitMarkdown:
    def test_empty_rows_produces_placeholder(self):
        out = _emit_markdown([], [])
        assert "No `wall_time_s` entries" in out

    def test_summary_uses_all_rows_not_visible(self):
        """Total/cell-count come from all rows; the table shows only `visible`."""
        all_rows = [
            {
                "problem": "p",
                "suite": "s",
                "experiment": "e",
                "solver": f"slv{i}",
                "mode": "prod",
                "seconds": 10.0,
            }
            for i in range(5)
        ]
        visible = all_rows[:2]
        out = _emit_markdown(all_rows, visible)
        assert "**5** (experiment, solver) cells" in out
        # 5 cells x 10s = 50s
        assert "50.0s" in out
        # Detail table only renders the visible subset.
        assert out.count("| p | s | e |") == 2
