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


SUBCOMMANDS = [
    "run",
    "ics",
    "build",
    "status",
    "compare",
    "tesseracts",
    "paper-plots",
    "validate-domain",
    "new-domain",
    "validate-template",
    "templates",
]


def test_top_level_help():
    """Top-level --help must list every registered subcommand."""
    result = _run_help([])
    assert result.returncode == 0
    out = result.stdout
    assert "Usage:" in out
    missing = [s for s in SUBCOMMANDS if s not in out]
    assert not missing, f"Subcommands missing from --help: {missing}"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help(subcommand: str):
    result = _run_help([subcommand])
    assert result.returncode == 0, f"{subcommand} --help failed:\n{result.stderr}"
    # Each subcommand's --help must show its own Usage line, not the parent's.
    assert f"{subcommand}" in result.stdout
    assert "Usage:" in result.stdout
