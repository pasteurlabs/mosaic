#!/usr/bin/env python3
"""Pull pre-built Tesseract solver images from a container registry.

For each solver matching the requested problems and hardware target,
pulls the image from the registry and retags it to the local short
name expected by ``Tesseract.from_image()``.

Each missing image is polled until it appears (the build-tesseracts
workflow pushes images as each solver finishes, so a cell whose fast
solvers are ready doesn't have to wait for the whole build). Cap the
total wait per image with ``--per-image-timeout`` (default 90 min); when
the cap is hit, exit non-zero so the benchmark fails loudly rather than
silently running on a partial set.

With ``--tag <sha>`` and ``--build-workflow``, the script queries the GHA
API to find out which solvers ``build-tesseracts`` is actually building
for this SHA. Solvers in that matrix have their ``:<sha>`` image polled
(so PR-built images win over main's ``:latest``); solvers not in the
matrix pull ``:latest`` directly (avoids waiting for a build that will
never push their SHA tag).

Usage (in CI):
    python .github/scripts/pull-solver-images.py \
        --registry ghcr.io/org/mosaic \
        --problems all \
        --hardware gpu \
        --tag "$GITHUB_SHA" \
        --build-workflow build-tesseracts.yml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from mosaic.benchmarks.problems import PROBLEMS, get_config


def _pull(remote: str, *, timeout_s: int = 900) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["docker", "pull", remote],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"docker pull timed out after {timeout_s}s"
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def _pull_with_wait(
    remote: str, *, timeout_s: int, poll_s: int
) -> tuple[bool, float, str]:
    """Pull ``remote``, polling with ``poll_s`` cadence until ``timeout_s``.

    Returns (succeeded, elapsed_s, last_error_line).
    """
    start = time.monotonic()
    last_err = ""
    attempt = 0
    # Heartbeat at most every 60s so GHA's UI shows progress instead of
    # looking frozen during long polls.
    next_heartbeat = 60.0
    while True:
        attempt += 1
        ok, err = _pull(remote)
        elapsed = time.monotonic() - start
        if ok:
            return True, elapsed, ""
        last_err = err.splitlines()[-1] if err else ""
        if elapsed + poll_s > timeout_s:
            return False, elapsed, last_err
        if attempt == 1:
            print(
                f"  pending — polling for {timeout_s}s (last: {last_err[:120]})",
                flush=True,
            )
        elif elapsed >= next_heartbeat:
            remaining = max(0, timeout_s - int(elapsed))
            print(
                f"  still polling — {int(elapsed)}s elapsed, "
                f"{remaining}s left (last: {last_err[:120]})",
                flush=True,
            )
            next_heartbeat = elapsed + 60.0
        time.sleep(poll_s)


# ── GitHub API: discover which solvers are in the build-tesseracts matrix ──


def _gh_api(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _building_solvers_for_sha(
    repo: str, workflow: str, sha: str, token: str | None
) -> set[str] | None:
    """Return the set of (image_name) solvers being built for *sha*.

    Each job in ``build-tesseracts.yml`` is named ``Build <domain>/<solver>``
    (matching the workflow's ``name:`` template). The directory name is what
    becomes the image basename — ``pict`` → ``pict_<domain_underscored>``.
    Returns ``None`` if the workflow run can't be located (treat all solvers
    as candidates for ``:<sha>``, preserving prior behavior).
    """
    list_url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs"
        f"?head_sha={sha}&per_page=1"
    )
    try:
        listing = _gh_api(list_url, token)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [api] could not list build-tesseracts runs: {exc}")
        return None
    runs = listing.get("workflow_runs") or []
    if not runs:
        return None
    run_id = runs[0]["id"]
    jobs_url = (
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    )
    try:
        jobs_resp = _gh_api(jobs_url, token)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [api] could not list jobs for run {run_id}: {exc}")
        return None
    # Job names look like "Build <domain>/<solver>"; capture the solver dir.
    pattern = re.compile(r"^Build\s+([\w.-]+)\s*/\s*([\w.-]+)\s*$")
    solver_dirs: set[str] = set()
    for job in jobs_resp.get("jobs") or []:
        m = pattern.match(job.get("name", ""))
        if m:
            solver_dirs.add(m.group(2))
    return solver_dirs


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
        "--per-image-timeout",
        type=int,
        default=5400,
        help="Max seconds to wait for any single image to appear (default 5400 — "
        "90 min; PICT and a few other CUDA-heavy solvers regularly take 40+ min).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between pull retries (default 15).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="If set, try ``<image>:<tag>`` first and fall back to "
        "``<image>:latest`` when that tag isn't published. Used in CI to "
        "pull PR-specific images by SHA while gracefully degrading to "
        "main's :latest for unchanged solvers.",
    )
    parser.add_argument(
        "--build-workflow",
        default=None,
        help="When set together with --tag, query this workflow's run for "
        "the SHA to discover which solvers are being built. Solvers in "
        "the matrix wait for ``:<sha>``; solvers not in the matrix pull "
        "``:latest`` directly (avoids waiting for an image that will "
        "never be pushed). Requires ``GITHUB_TOKEN`` env var and "
        "``GITHUB_REPOSITORY`` (or the --repo flag).",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (default: $GITHUB_REPOSITORY).",
    )
    args = parser.parse_args()

    registry = args.registry.lower().rstrip("/")
    problem_list = (
        PROBLEMS
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",")]
    )

    # When --tag is set, optionally discover which solver dirs are in
    # build-tesseracts' matrix for this SHA. Solvers IN the matrix
    # legitimately wait for ``:<sha>``; solvers NOT in the matrix go
    # straight to ``:latest`` to avoid waiting for an image that will
    # never be pushed for this commit.
    building: set[str] | None = None
    if args.tag and args.build_workflow:
        if not args.repo:
            print(
                "  [api] --build-workflow needs --repo or $GITHUB_REPOSITORY"
                " — falling back to per-image SHA polling for all solvers."
            )
        else:
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            building = _building_solvers_for_sha(
                args.repo, args.build_workflow, args.tag, token
            )
            if building is not None:
                if building:
                    print(
                        f"  [api] build-tesseracts is building {len(building)} "
                        f"solver(s) for SHA {args.tag[:8]}: "
                        f"{', '.join(sorted(building))}"
                    )
                else:
                    print(
                        f"  [api] no build-tesseracts jobs for SHA {args.tag[:8]} "
                        "— using :latest for all solvers"
                    )

    seen: set[str] = set()
    failed: list[str] = []
    for p in problem_list:
        try:
            cfg = get_config(p)
        except Exception:
            continue
        # cfg.solvers is a list[SolverSpec] on the docs branch (was a dict
        # under the legacy ProblemConfig); iterate the list directly.
        for spec in cfg.solvers:
            if args.hardware == "gpu" and not getattr(spec, "uses_gpu", True):
                continue
            if args.hardware == "cpu" and getattr(spec, "uses_gpu", True):
                continue
            local_tag = spec.image_tag or f"{spec.dir}:latest"
            if local_tag in seen:
                continue
            seen.add(local_tag)
            # local_tag = "<image_name>:latest" — split into name + suffix
            image_name = local_tag.rsplit(":", 1)[0]

            # Build the remote candidate list: prefer ``:<sha>`` for solvers
            # we know are being built for this commit; everything else goes
            # straight to ``:latest``.
            candidates: list[tuple[str, int]] = []
            if args.tag and building is not None and spec.dir in building:
                # Confirmed in the build matrix — wait the full budget for
                # the build to publish its ``:<sha>`` image.
                candidates.append(
                    (f"{registry}/{image_name}:{args.tag}", args.per_image_timeout)
                )
            elif args.tag and building is None:
                # Couldn't determine the matrix (API failure / race with the
                # build workflow). Try ``:<sha>`` briefly so a recently-pushed
                # image still wins, but don't burn 90 min per solver waiting
                # for an image that may never appear.
                candidates.append((f"{registry}/{image_name}:{args.tag}", 60))
            # ``:latest`` fallback: short poll budget — if it's not in the
            # registry now it almost certainly never will be.
            candidates.append((f"{registry}/{image_name}:latest", 60))

            pulled_remote = None
            for remote, budget in candidates:
                print(f"Pulling {remote}", flush=True)
                ok, elapsed, last_err = _pull_with_wait(
                    remote,
                    timeout_s=budget,
                    poll_s=args.poll_interval,
                )
                if ok:
                    subprocess.run(["docker", "tag", remote, local_tag])
                    print(f"  Tagged as {local_tag} ({elapsed:.0f}s)", flush=True)
                    pulled_remote = remote
                    break
                print(
                    f"  miss after {elapsed:.0f}s (last error: {last_err[:160]})",
                    flush=True,
                )

            if pulled_remote is None:
                failed.append(f"{registry}/{image_name}")

    if failed:
        print(f"\n{len(failed)} image(s) failed to pull:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPulled {len(seen)} image(s) successfully")


if __name__ == "__main__":
    main()
