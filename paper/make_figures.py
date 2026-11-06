# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.9", "numpy==2.4.4", "pyyaml==6.0.3"]
# ///
"""Regenerate the paper figures from a Mosaic results directory.

Usage:
    uv run paper/make_figures.py [--only NAME,...] [--out-dir DIR] [--extra-results DIR]

Results are read from ``$MOSAIC_RESULTS_DIR`` if set, else ``./mosaic-results``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plots import MODULES, get_generate_fn, results_dir, set_extra_results_dir


def main() -> int:
    """Generate the requested figures and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default=",".join(MODULES),
        help="Comma-separated figure modules to run (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Output directory (default: paper/figures).",
    )
    parser.add_argument(
        "--extra-results",
        type=Path,
        help="Results of the extended mesh cost sweeps (cost/*_large experiments).",
    )
    args = parser.parse_args()
    set_extra_results_dir(args.extra_results)

    names = [n.strip() for n in args.only.split(",")]
    unknown = sorted(set(names) - set(MODULES))
    if unknown:
        parser.error(f"unknown module(s) {unknown}; available: {MODULES}")

    print(f"Reading results from {results_dir()}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for name in names:
        try:
            get_generate_fn(name)(args.out_dir)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    if failed:
        print(f"{len(failed)} module(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
