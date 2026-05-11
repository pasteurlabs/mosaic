"""Integration tests for --plots-only and paper-plots commands.

These tests require example results in mosaic-results/ and are
skipped when no results are present.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MOSAIC = [sys.executable, "-m", "mosaic.benchmarks.cli"]


def _results_root():
    from mosaic.benchmarks.core.utils import results_dir

    return results_dir()


def _has_any_results() -> bool:
    d = _results_root()
    return d.is_dir() and any(d.iterdir())


def _has_fd_check_all_domains() -> bool:
    """fd_check figure needs result.json for all four benchmark domains."""
    root = _results_root()
    return all(
        (root / subdir / "gradient" / "fd_check" / "result.json").is_file()
        for subdir in ("ns-grid", "ns-3d-grid", "structural-mesh", "thermal-mesh")
    )


needs_results = pytest.mark.skipif(
    not _has_any_results(),
    reason="No benchmark results on disk",
)

needs_fd_check_all_domains = pytest.mark.skipif(
    not _has_fd_check_all_domains(),
    reason="fd_check results missing for one or more domains",
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


@needs_fd_check_all_domains
def test_paper_plots_single():
    """mosaic paper-plots --only fd_check must succeed with existing results."""
    result = subprocess.run(
        [*MOSAIC, "paper-plots", "--only", "fd_check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
