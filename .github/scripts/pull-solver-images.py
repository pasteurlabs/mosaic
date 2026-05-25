#!/usr/bin/env python3
"""Pull pre-built Tesseract solver images from a container registry.

For each solver matching the requested problems and hardware target,
pulls the image from the registry and retags it to the local short
name expected by ``Tesseract.from_image()``.

Usage (in CI):
    python .github/scripts/pull-solver-images.py \
        --registry ghcr.io/org/mosaic \
        --problems all \
        --hardware gpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from mosaic.benchmarks.problems import PROBLEMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Container registry prefix")
    parser.add_argument(
        "--problems", default="all", help="Comma-separated problems or 'all'"
    )
    parser.add_argument(
        "--hardware", required=True, choices=["gpu", "cpu"], help="Hardware target"
    )
    args = parser.parse_args()

    registry = args.registry.lower().rstrip("/")
    problem_list = (
        PROBLEMS
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",")]
    )

    seen: set[str] = set()
    failed: list[str] = []
    for p in problem_list:
        try:
            cfg = get_config(p)
        except Exception:
            continue
        for spec in cfg.solvers:
            if args.hardware == "gpu" and not getattr(spec, "uses_gpu", True):
                continue
            if args.hardware == "cpu" and getattr(spec, "uses_gpu", True):
                continue
            tag = spec.image_tag or f"{spec.dir}:latest"
            if tag in seen:
                continue
            seen.add(tag)
            remote = f"{registry}/{tag}"
            print(f"Pulling {remote}")
            r = subprocess.run(
                ["docker", "pull", remote], capture_output=True, text=True
            )
            if r.returncode == 0:
                subprocess.run(["docker", "tag", remote, tag])
                print(f"  Tagged as {tag}")
            else:
                print(f"  FAIL: {remote} not found in registry")
                failed.append(remote)

    if failed:
        print(f"\n{len(failed)} image(s) failed to pull:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPulled {len(seen)} image(s) successfully")


if __name__ == "__main__":
    main()
