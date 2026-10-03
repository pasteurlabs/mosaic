#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect changed benchmark domains from a git diff.

Reads changed file paths from stdin (one per line) and prints a JSON
list of domain names (e.g. ``["thermal-mesh"]``) for every domain whose
problem definition under ``mosaic/benchmarks/problems/<pkg>/`` contains a
changed file.

This keys off the *problem* definitions, not the Tesseract solvers: a
change to a domain's problem/physics/config changes the reference the
whole domain is benchmarked against, so every solver in that domain
should re-run. Solver code changes are ``benchmark:solver``'s job and are
deliberately ignored here.

A single domain can back multiple problem packages (e.g.
``navier_stokes_grid`` and ``navier_stokes_3d_grid`` both map to the
``navier-stokes-grid`` domain), so the package→domain mapping is resolved
through the problem registry rather than by string-munging the path.

Prints ``[]`` if no problem definition changed.

Usage (in CI):
    git diff --name-only BASE HEAD | python .github/scripts/detect-changed-domains.py
"""

from __future__ import annotations

import importlib
import json
import sys

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.problems import _candidate_module_paths

PROBLEMS_PREFIX = "mosaic/benchmarks/problems/"


def _package_to_domain() -> dict[str, str]:
    """Map each problem package dir name to its domain (tesseract dir name).

    Resolved via the registry's own module discovery so a package that
    doesn't hyphenate to its domain (e.g. ``navier_stokes_3d_grid`` →
    ``navier-stokes-grid``) still maps correctly.
    """
    mapping: dict[str, str] = {}
    for module_path in _candidate_module_paths():
        try:
            mod = importlib.import_module(module_path)
        except Exception:  # noqa: S112 — unimportable problems can't be mapped
            continue
        cfg = getattr(mod, "problem", None)
        if not isinstance(cfg, Problem):
            continue
        parts = module_path.split(".")
        pkg = parts[parts.index("problems") + 1]
        mapping[pkg] = cfg.tesseract_dir.name
    return mapping


def main() -> None:
    diff_files = [line.strip() for line in sys.stdin if line.strip()]

    pkg_to_domain = _package_to_domain()

    domains: set[str] = set()
    for f in diff_files:
        if not f.startswith(PROBLEMS_PREFIX):
            continue
        rest = f[len(PROBLEMS_PREFIX) :]
        pkg = rest.split("/", 1)[0]
        domain = pkg_to_domain.get(pkg)
        if domain:
            domains.add(domain)

    json.dump(sorted(domains), sys.stdout)
    print()


if __name__ == "__main__":
    main()
