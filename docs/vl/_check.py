#!/usr/bin/env python3
# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dev helper: build one vl module's charts and write paper SVG/PNG + web specs.

Usage: python docs/vl/_check.py <module> <out_dir>
Writes <out_dir>/<problem>_<path>.{svg,png,vl.json} for every PNG under
mosaic-results/ whose stem the module builds; prints failures. Never writes
into mosaic-results/.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOCS))

from vl.common import Ctx, paper, web_spec  # noqa: E402

RESULTS = DOCS.parent / "mosaic-results"


def main() -> None:
    mod = importlib.import_module(f"vl.{sys.argv[1]}")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for png in sorted(RESULTS.glob("**/*.png")):
        build = mod.BUILDERS.get(png.stem)
        if build is None:
            continue
        rel = png.relative_to(RESULTS)
        problem, suite, *exp = rel.parts[:-1]
        name = "_".join(rel.with_suffix("").parts)
        try:
            chart = build(Ctx(problem, suite, "/".join(exp), png))
        except Exception:
            print(f"FAILED {rel}\n{traceback.format_exc()}")
            continue
        if chart is None:
            print(f"none   {rel} (builder returned None)")
            continue
        (out / f"{name}.vl.json").write_text(
            json.dumps(web_spec(chart)), encoding="utf-8"
        )
        svg = out / f"{name}.svg"
        paper(chart).save(str(svg))
        # Raster preview for reviewing next to the matplotlib PNG.
        subprocess.run(
            ["qlmanage", "-t", "-s", "1400", "-o", str(out), str(svg)],
            capture_output=True,
            check=False,
        )
        print(f"ok     {rel} -> {svg}")


if __name__ == "__main__":
    main()
