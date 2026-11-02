# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for .github/scripts/detect-changed-solvers.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
            ["mosaic/tesseracts/navier-stokes-grid/ins-jl/solver.jl"],
            [("navier-stokes-grid", "ins-jl")],
        ),
        (
            ["mosaic/tesseracts/structural-mesh/topopt-jl/solver.jl"],
            [("structural-mesh", "topopt-jl")],
        ),
        (["mosaic/mosaic_shared/schema_types.py"], []),
    ],
)
def test_detects_solvers_from_changed_paths(paths, expected):
    assert _detect(*paths) == expected
