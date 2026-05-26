#!/usr/bin/env python3
"""Pull pre-built Tesseract solver images from a container registry.

For each solver matching the requested problems and hardware target,
pulls the image from the registry and retags it to the local short
name expected by ``Tesseract.from_image()``.

Usage (in CI):
    python .github/scripts/pull-solver-images.py \
        --registry ghcr.io/org/mosaic \
        --problems all \
        --hardware gpu \
        --tag abc123f
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
    parser.add_argument(
        "--tag",
        default=None,
        help="Registry tag to pull (e.g. a commit SHA). Tries this first, "
        "falls back to 'latest' for images that weren't built at this tag.",
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
    pulled_count = 0
    for p in problem_list:
        try:
            cfg = get_config(p)
        except Exception:
            continue
        for spec in cfg.solvers:
            uses_gpu = getattr(spec, "uses_gpu", True)
            if args.hardware == "gpu" and not uses_gpu:
                continue
            if args.hardware == "cpu" and uses_gpu:
                continue
            tag = spec.image_tag or f"{spec.dir}:latest"
            if tag in seen:
                continue
            seen.add(tag)

            # Local short name is always the :latest form (what the runner expects).
            local_tag = tag
            image_name = tag.rsplit(":", 1)[0]

            # Try --tag first (e.g. :<sha>), fall back to :latest.
            candidates = (
                [f"{registry}/{image_name}:{args.tag}", f"{registry}/{tag}"]
                if args.tag
                else [f"{registry}/{tag}"]
            )
            pulled = False
            for remote in candidates:
                print(f"Pulling {remote}")
                r = subprocess.run(
                    ["docker", "pull", remote], capture_output=True, text=True
                )
                if r.returncode == 0:
                    subprocess.run(["docker", "tag", remote, local_tag])
                    print(f"  Tagged as {local_tag}")
                    pulled = True
                    pulled_count += 1
                    break
                if args.tag:
                    print("  Not found, trying :latest fallback...")
            if not pulled:
                print(f"  FAIL: no image found for {image_name}")
                failed.append(image_name)

    if failed:
        print(f"\n{len(failed)} image(s) failed to pull:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPulled {pulled_count} image(s) successfully")


if __name__ == "__main__":
    main()
