# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for --plots-only.

These tests require example results in mosaic-results/ and are
skipped when no results are present.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MOSAIC = [sys.executable, "-m", "mosaic.benchmarks.cli"]


def _results_root():
    from mosaic.benchmarks.core.io import results_dir

    return results_dir()


def _has_any_results() -> bool:
    d = _results_root()
    return d.is_dir() and any(d.iterdir())


needs_results = pytest.mark.skipif(
    not _has_any_results(),
    reason="No benchmark results on disk",
)


@needs_results
def test_plots_only_gradient():
    """Mosaic run --plots-only must succeed with existing gradient results."""
    result = subprocess.run(
        [*MOSAIC, "run", "--plots-only", "-p", "ns-grid", "--suites", "gradient"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
