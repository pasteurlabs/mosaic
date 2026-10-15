# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for .github/scripts/detect-changed-solvers.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".github" / "scripts" / "detect-changed-solvers.py"


def _detect(*paths: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout
    return [(e["domain"], e["solver"]) for e in json.loads(out)]


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["README.md"], []),
        (
            ["mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py"],
            [("navier-stokes-grid", "xlb")],
        ),
        (
            ["mosaic/mosaic_shared/problems/navier_stokes_grid/ns_solver.jl"],
            [("navier-stokes-grid", "ins-jl")],
        ),
        (
            ["mosaic/mosaic_shared/problems/structural_mesh/topopt_solver.jl"],
            [("structural-mesh", "topopt-jl")],
        ),
        # A shared module every solver imports is not a listed shared source.
        (["mosaic/mosaic_shared/schema_types.py"], []),
    ],
)
def test_detects_solvers_from_changed_paths(paths, expected):
    assert _detect(*paths) == expected


def test_shared_sources_exist():
    """Every ``shared_sources`` entry names a file in the repository."""
    for cfg in sorted((REPO / "mosaic" / "tesseracts").rglob("tesseract_config.yaml")):
        meta = (yaml.safe_load(cfg.read_text()).get("metadata") or {}).get(
            "mosaic"
        ) or {}
        for src in meta.get("shared_sources") or []:
            assert (REPO / src).is_file(), f"{cfg}: shared source {src} does not exist"
