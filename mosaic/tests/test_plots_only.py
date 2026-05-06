"""Integration tests for --plots-only and paper-plots commands.

These tests require example results in mosaic-results/ and are
skipped when no results are present.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MOSAIC = [sys.executable, "-m", "mosaic.benchmarks.cli"]


def _has_results() -> bool:
    """Check whether benchmark results exist at the default location."""
    from mosaic.benchmarks.core.utils import results_dir

    d = results_dir()
    return d.is_dir() and any(d.iterdir())


needs_results = pytest.mark.skipif(
    not _has_results(),
    reason="No benchmark results on disk",
)


@needs_results
def test_plots_only_gradient():
    """mosaic run --plots-only must succeed with existing gradient results."""
    result = subprocess.run(
        [*MOSAIC, "run", "--plots-only", "-p", "ns-grid", "--suites", "gradient"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@needs_results
def test_paper_plots_single():
    """mosaic paper-plots --only fd_check must succeed with existing results."""
    result = subprocess.run(
        [*MOSAIC, "paper-plots", "--only", "fd_check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
