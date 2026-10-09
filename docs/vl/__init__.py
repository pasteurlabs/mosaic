# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vega-Lite (Altair) versions of the benchmark result plots for the docs.

Each module defines ``BUILDERS``: PNG stem → ``build(ctx) -> chart | None``.
:func:`builders` merges them; docs/export_charts.py writes one spec per PNG.
"""

from __future__ import annotations

import importlib

# Plot-family modules; each defines BUILDERS.
MODULES = ["gradient", "sweeps", "forward", "cost", "optimization", "ics", "animations"]


def builders() -> dict:
    out: dict = {}
    for name in MODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{name}":
                continue  # family not ported yet
            raise
        clash = set(out) & set(mod.BUILDERS)
        if clash:
            raise RuntimeError(f"duplicate builders {clash} in vl.{name}")
        out.update(mod.BUILDERS)
    return out
