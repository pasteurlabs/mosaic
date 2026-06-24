#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
import json
import subprocess
import sys

from mosaic.benchmarks.problems import PROBLEMS, get_config


def _digest_via_buildx(remote: str) -> str | None:
    r = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            remote,
            "--format",
            "{{.Manifest.Digest}}",
        ],
        capture_output=True,
        text=True,
    )
    digest = r.stdout.strip()
    return digest if r.returncode == 0 and digest.startswith("sha256:") else None


def _digest_via_manifest(remote: str) -> str | None:
    r = subprocess.run(
        ["docker", "manifest", "inspect", "-v", remote],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    # A multi-arch tag returns a list of per-platform manifests; a single-arch
    # tag returns one object. The top-level Descriptor.digest is the tag's
    # digest in both shapes — take the first entry for the list case.
    entry = data[0] if isinstance(data, list) else data
    digest = (entry.get("Descriptor") or {}).get("digest", "")
    return digest if digest.startswith("sha256:") else None


def _resolve_digest(remote: str) -> str | None:
    """Resolve a registry reference (``repo:tag``) to its immutable digest.

    Queries the registry directly — not the runner's local image store — so a
    stale ``:latest`` cached on the runner (or a tag that was just pushed)
    can't shadow the authoritative image. Tries ``docker buildx imagetools``
    first, falling back to ``docker manifest inspect`` for runners without
    buildx (the solver jobs run on self-hosted hosts). Returns ``sha256:…`` or
    ``None`` if the reference doesn't exist.
    """
    return _digest_via_buildx(remote) or _digest_via_manifest(remote)


def _pull_pinned(remote: str, local_tag: str) -> bool:
    """Pull *remote* by its registry digest and retag to *local_tag*.

    Pulling by digest (``repo@sha256:…``) is cache-proof: Docker fetches the
    exact image the registry reference points to *now*, never a stale local
    layer. This is what makes a ``:latest`` fallback safe — a release PR that
    falls back to ``:latest`` always gets the current ``main`` build, not
    whatever the runner happened to cache earlier.

    Returns True on success. False means the reference didn't resolve (caller
    can try the next candidate, e.g. the ``:latest`` fallback).
    """
    digest = _resolve_digest(remote)
    if digest is None:
        return False
    repo = remote.rsplit(":", 1)[0]
    pinned = f"{repo}@{digest}"
    print(f"Pulling {remote} -> {pinned}")
    if subprocess.run(["docker", "pull", pinned]).returncode != 0:
        return False
    subprocess.run(["docker", "tag", pinned, local_tag])
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
        help="Registry tag to pull (e.g. a commit SHA). Tries this first, "
        "falls back to 'latest' for images that weren't built at this tag.",
    )
    parser.add_argument(
        "--solvers",
        default=None,
        help="Comma-separated solver display names to restrict pulling to. "
        "When set, only matching solvers are pulled; others are skipped.",
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

            # Try --tag first (e.g. :<sha>), fall back to :latest. Each pull is
            # pinned to the registry digest (see _pull_pinned) so the :latest
            # fallback can't serve a stale runner-cached image — the bug that
            # made release PRs benchmark against pre-fix solver code.
            candidates = (
                [f"{registry}/{image_name}:{args.tag}", f"{registry}/{tag}"]
                if args.tag
                else [f"{registry}/{tag}"]
            )
            pulled = False
            for remote in candidates:
                if _pull_pinned(remote, local_tag):
                    pulled = True
                    pulled_count += 1
                    break
                if args.tag:
                    print(f"  {remote} not found, trying :latest fallback...")
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
