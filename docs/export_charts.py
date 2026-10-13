#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the interactive (Vega-Lite) version of each benchmark plot.

Run after ``mosaic run --plots-only``. For every ``<plot>.png`` (and
``<animation>.gif``) under ``mosaic-results/`` with a builder in docs/vl,
writes ``<plot>.vl.json`` next to it; docs/generate_results.py embeds a chart wherever that file exists and
keeps the PNG as the fallback.

Usage:
    python docs/export_charts.py              # specs for the docs
    python docs/export_charts.py --paper DIR  # also SVG paper renders in DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vl import builders
from vl.common import Ctx, paper, web_spec

RESULTS = Path(__file__).resolve().parent.parent / "mosaic-results"


def _ctx(png: Path) -> Ctx:
    rel = png.relative_to(RESULTS).parts
    problem, suite, *exp = rel[:-1]
    return Ctx(problem=problem, suite=suite, experiment="/".join(exp), png=png)


def main() -> None:
    paper_dir = None
    if "--paper" in sys.argv:
        paper_dir = Path(sys.argv[sys.argv.index("--paper") + 1])
        paper_dir.mkdir(parents=True, exist_ok=True)
    table = builders()
    written, failed, missing = 0, [], set()
    plots = sorted([*RESULTS.glob("**/*.png"), *RESULTS.glob("**/*.gif")])
    for png in plots:
        out = png.with_suffix(".vl.json")
        build = table.get(png.stem)
        if build is None:
            missing.add(png.stem)
            out.unlink(missing_ok=True)
            continue
        try:
            chart = build(_ctx(png))
        except Exception as exc:  # never fail the docs build on one chart
            failed.append(f"{png.relative_to(RESULTS)}: {exc!r}")
            out.unlink(missing_ok=True)
            continue
        if chart is None:
            out.unlink(missing_ok=True)
            continue
        # Always SVG: pages scale charts down to the column width, and SVG
        # hover stays accurate when scaled (canvas hit-testing does not).
        spec = web_spec(chart)
        out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
        written += 1
        if paper_dir is not None:
            name = "_".join(png.relative_to(RESULTS).with_suffix("").parts)
            paper(chart).save(str(paper_dir / f"{name}.svg"))
    print(f"Vega-Lite charts written: {written}")
    for f in failed:
        print(f"  failed: {f}")
    if missing:
        print(f"  no builder (PNG only): {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
