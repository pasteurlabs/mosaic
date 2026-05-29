#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand a single changed solver into the full solver list for benchmark:solver mode.

Given the solver matrix JSON (from detect-changed-solvers.py) and the
comma-separated problem list, this script:

1. Extracts the display name of the changed solver.
2. Detects whether it uses CPU or GPU from its tesseract_config.yaml.
3. Adds a same-hardware consensus peer for domains that lack an analytic
   reference (structural-mesh, thermal-mesh), so forward/agreement has
   ≥2 solvers to compare.

Prints a comma-separated solver list to stdout.

Usage (in CI):
    echo "$SOLVER_MATRIX" | python .github/scripts/expand-solver-list.py --problems ns-grid,ns-3d-grid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

TESSERACTS_DIR = Path("mosaic/tesseracts")

# Consensus-based domains need ≥2 solvers for agreement (no analytic
# reference to fall back on). Keyed by (problem, hardware).
# Domains with analytic references (ns-grid, ns-3d-grid) don't need
# this -- forward.py handles single-solver agreement against the
# analytic solution directly.
CONSENSUS_PEERS: dict[tuple[str, str], str] = {
    ("structural-mesh", "cpu"): "FEniCS",
    ("structural-mesh", "gpu"): "JAX-FEM",
    ("thermal-mesh", "cpu"): "FEniCS",
    ("thermal-mesh", "gpu"): "JAX-FEM",
}


def _solver_hardware(entry: dict[str, str]) -> str:
    """Read uses_gpu from the solver's tesseract_config.yaml."""
    cfg_path = (
        TESSERACTS_DIR / entry["domain"] / entry["solver"] / "tesseract_config.yaml"
    )
    if cfg_path.exists():
        meta = (yaml.safe_load(cfg_path.read_text()) or {}).get("metadata", {})
        uses_gpu = meta.get("mosaic", {}).get("uses_gpu", True)
        return "gpu" if uses_gpu else "cpu"
    return "gpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="",
        help="Comma-separated problem names affected by this solver change",
    )
    args = parser.parse_args()

    entries = json.load(sys.stdin)
    if not entries:
        print("")
        return

    # Extract display names of the changed solver(s)
    solvers: list[str] = [e["display_name"] for e in entries if "display_name" in e]

    # Determine hardware of the changed solver (benchmark:solver
    # guarantees exactly one entry)
    hw = _solver_hardware(entries[0])

    # Add consensus peers for domains that need them
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    for problem in problems:
        peer = CONSENSUS_PEERS.get((problem, hw))
        if peer and peer not in solvers:
            solvers.append(peer)

    print(",".join(solvers))


if __name__ == "__main__":
    main()
