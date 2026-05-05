"""Smoke tests for the CLI: verify all subcommands exist and --help works."""

from __future__ import annotations

import subprocess
import sys

import pytest

MOSAIC = [sys.executable, "-m", "mosaic.benchmarks.cli"]


def _run_help(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*MOSAIC, *args, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_top_level_help():
    result = _run_help([])
    assert result.returncode == 0
    assert "mosaic" in result.stdout.lower() or "usage" in result.stdout.lower()


@pytest.mark.parametrize(
    "subcommand",
    [
        "run-all",
        "forward",
        "cost",
        "gradient",
        "optimization",
        "ics",
        "status",
        "build",
        "clean",
    ],
)
def test_subcommand_help(subcommand: str):
    result = _run_help([subcommand])
    assert result.returncode == 0, f"{subcommand} --help failed:\n{result.stderr}"
