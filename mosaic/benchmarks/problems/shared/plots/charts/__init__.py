# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Altair (Vega-Lite) definitions of the benchmark result plots.

Each module defines ``BUILDERS``: plot stem → ``build(ctx) -> chart | None``.
The harness plot functions call :func:`.common.render`, which writes each chart
as ``<stem>.png`` (vl-convert) and ``<stem>.vl.json`` (the interactive version
on the results pages). :func:`builders` merges all ``BUILDERS`` for tests.
"""

from __future__ import annotations

import importlib

# Plot-family modules; each defines BUILDERS.
MODULES = ["gradient", "sweeps", "forward", "cost", "optimization", "ics", "animations"]


def builders() -> dict:
    """Merged ``BUILDERS`` of all chart modules (PNG stem → builder)."""
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
            raise RuntimeError(f"duplicate builders {clash} in charts.{name}")
        out.update(mod.BUILDERS)
    return out
