#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull pre-built Tesseract solver images from a container registry.

For each solver matching the requested problems and hardware target,
pulls the image from the registry and retags it to the local short
name expected by ``Tesseract.from_image()``.

Images are pulled by commit-SHA tag, never ``:latest``: an immutable SHA tag
makes ``docker pull`` unambiguous, whereas a runner-cached or not-yet-propagated
``:latest`` can point at the wrong image.

In CI, PR runs build the *changed* solvers and hand them to the benchmark run
as image artifacts (``docker load``, not a registry pull), so this script only
fetches the *unchanged* solvers. Those are pulled at ``--fallback-tag`` (the
PR's base/main commit, always published by the push-to-main build). The
already-loaded solvers are passed via ``--skip-images`` so they are not
re-fetched at a SHA where they were never published. ``--tag`` remains for
direct/manual use (pull a specific SHA first, then fall back).

Usage (in CI):
    python .github/scripts/pull-solver-images.py \
        --registry ghcr.io/org/mosaic \
        --problems all \
        --hardware gpu \
        --fallback-tag <base-sha> \
        --skip-images <name1>,<name2>
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from mosaic.benchmarks.problems import PROBLEMS, get_config


def _pull_and_retag(remote: str, local_tag: str) -> bool:
    """Pull *remote* and retag it to *local_tag*. Returns True on success."""
    print(f"Pulling {remote}")
    if subprocess.run(["docker", "pull", remote]).returncode != 0:
        return False
    subprocess.run(["docker", "tag", remote, local_tag])
    print(f"  Tagged as {local_tag}")
    return True


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
        help="Primary registry tag to pull (this PR's HEAD commit SHA). Built "
        "by the build job for solvers this PR changed.",
    )
    parser.add_argument(
        "--fallback-tag",
        default=None,
        help="Fallback registry tag (the base/main commit SHA) for solvers not "
        "built at --tag. Always an immutable, already-built SHA — never :latest.",
    )
    parser.add_argument(
        "--solvers",
        default=None,
        help="Comma-separated solver display names to restrict pulling to. "
        "When set, only matching solvers are pulled; others are skipped.",
    )
    parser.add_argument(
        "--skip-images",
        default=None,
        help="Comma-separated image names (the tag's repository part, e.g. "
        "'ins_navier_stokes_grid') to skip pulling. Used when those images are "
        "already present locally — e.g. loaded from a build artifact on a PR — "
        "so the registry pull only covers the unchanged solvers.",
    )
    args = parser.parse_args()

    registry = args.registry.lower().rstrip("/")
    problem_list = (
        PROBLEMS
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",")]
    )
    solver_filter: set[str] | None = None
    if args.solvers:
        solver_filter = {s.strip() for s in args.solvers.split(",") if s.strip()}

    skip_images: set[str] = set()
    if args.skip_images:
        skip_images = {s.strip() for s in args.skip_images.split(",") if s.strip()}

    seen: set[str] = set()
    failed: list[str] = []
    pulled_count = 0
    for p in problem_list:
        try:
            cfg = get_config(p)
        except Exception:
            continue
        for spec in cfg.solvers:
            # --solvers filter: skip solvers not in the requested set
            if solver_filter and spec.name not in solver_filter:
                continue
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

            # Already present locally (e.g. docker-loaded from a PR build
            # artifact) → don't pull, it would only fail at a SHA where this
            # changed solver was never published.
            if image_name in skip_images:
                print(f"Skipping {image_name} (already loaded locally)")
                continue

            # Pull by SHA: --tag (HEAD) first, then --fallback-tag (base/main).
            # Both are immutable commit SHAs, so the right image is unambiguous
            # — no :latest, hence no stale/racy pointer (the bug that made a
            # release PR benchmark against pre-fix solver code).
            sha_tags = [t for t in (args.tag, args.fallback_tag) if t]
            candidates = (
                [f"{registry}/{image_name}:{t}" for t in sha_tags]
                if sha_tags
                else [f"{registry}/{tag}"]
            )
            pulled = False
            for i, remote in enumerate(candidates):
                if _pull_and_retag(remote, local_tag):
                    pulled = True
                    pulled_count += 1
                    break
                if i + 1 < len(candidates):
                    print(f"  {remote} not found, trying fallback {candidates[i + 1]}")
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
