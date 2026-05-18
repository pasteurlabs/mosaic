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
from dataclasses import dataclass, field

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


@dataclass
class _PullTask:
    """One image to pull, with a list of (remote, budget_s) candidates.

    The round-robin scheduler advances through ``candidates`` per task: each
    candidate has its own time budget (e.g. wait the full per-image timeout
    for a known-building ``:<sha>`` image, then fall back to ``:latest`` for
    60s). The scheduler tries every pending task once per pass and sleeps
    between passes, so a fast-finishing build is pulled the moment its
    image lands rather than after every preceding solver is resolved.
    """

    local_tag: str
    image_name: str
    candidates: list[tuple[str, int]]
    idx: int = 0
    candidate_started: float = field(default_factory=time.monotonic)
    last_err: str = ""
    announced: bool = False


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


def _build_candidates(
    *,
    registry: str,
    image_name: str,
    solver_dir: str,
    tag: str | None,
    per_image_timeout: int,
    building: set[str] | None,
) -> list[tuple[str, int]]:
    """Build the ordered list of (remote, budget_s) candidates for an image.

    Prefer ``:<sha>`` for solvers known to be building (full budget) or when
    matrix discovery failed (short budget — don't wait 90 min for an image
    that may never appear). Always fall back to ``:latest`` with a short
    budget.
    """
    candidates: list[tuple[str, int]] = []
    if tag and building is not None and solver_dir in building:
        candidates.append((f"{registry}/{image_name}:{tag}", per_image_timeout))
    elif tag and building is None:
        candidates.append((f"{registry}/{image_name}:{tag}", 60))
    candidates.append((f"{registry}/{image_name}:latest", 60))
    return candidates


def _build_tasks(
    problem_list: list[str],
    hardware: str,
    registry: str,
    tag: str | None,
    per_image_timeout: int,
    building: set[str] | None,
) -> list[_PullTask]:
    seen: set[str] = set()
    tasks: list[_PullTask] = []
    for p in problem_list:
        try:
            cfg = get_config(p)
        except Exception:
            continue
        # cfg.solvers is a list[SolverSpec] on the docs branch (was a dict
        # under the legacy ProblemConfig); iterate the list directly.
        for spec in cfg.solvers:
            uses_gpu = getattr(spec, "uses_gpu", True)
            if hardware == "gpu" and not uses_gpu:
                continue
            if hardware == "cpu" and uses_gpu:
                continue
            local_tag = spec.image_tag or f"{spec.dir}:latest"
            if local_tag in seen:
                continue
            seen.add(local_tag)
            image_name = local_tag.rsplit(":", 1)[0]
            tasks.append(
                _PullTask(
                    local_tag=local_tag,
                    image_name=image_name,
                    candidates=_build_candidates(
                        registry=registry,
                        image_name=image_name,
                        solver_dir=spec.dir,
                        tag=tag,
                        per_image_timeout=per_image_timeout,
                        building=building,
                    ),
                )
            )
    return tasks


def _step_task(task: _PullTask, registry: str) -> str:
    """Attempt one pull for ``task``. Returns "done" | "fail" | "pending"."""
    remote, budget = task.candidates[task.idx]
    if not task.announced:
        print(f"Trying {remote}", flush=True)
        task.announced = True
    ok, err = _pull(remote)
    if ok:
        subprocess.run(["docker", "tag", remote, task.local_tag])
        elapsed = time.monotonic() - task.candidate_started
        print(f"  ✓ {task.local_tag} <- {remote} ({elapsed:.0f}s)", flush=True)
        return "done"
    task.last_err = err.splitlines()[-1] if err else ""
    elapsed = time.monotonic() - task.candidate_started
    if elapsed < budget:
        return "pending"
    print(
        f"  miss {remote} after {elapsed:.0f}s (last error: {task.last_err[:160]})",
        flush=True,
    )
    task.idx += 1
    if task.idx >= len(task.candidates):
        return "fail"
    task.candidate_started = time.monotonic()
    task.announced = False
    return "pending"


def _run_pull_loop(
    tasks: list[_PullTask], registry: str, poll_interval: int
) -> list[str]:
    """Round-robin: try each pending task once per pass, sleep between passes.

    The moment any task's image is published it gets pulled — no head-of-line
    blocking by a slow build. Returns the list of remote names that exhausted
    all candidates.
    """
    failed: list[str] = []
    pending: list[_PullTask] = list(tasks)
    print(f"Pulling {len(pending)} image(s) (round-robin polling)", flush=True)
    last_heartbeat = time.monotonic()
    while pending:
        still_pending: list[_PullTask] = []
        for task in pending:
            status = _step_task(task, registry)
            if status == "pending":
                still_pending.append(task)
            elif status == "fail":
                failed.append(f"{registry}/{task.image_name}")
        pending = still_pending
        now = time.monotonic()
        if pending and now - last_heartbeat >= 60:
            print(
                f"  [heartbeat] {len(pending)} image(s) still pending: "
                f"{', '.join(t.image_name for t in pending[:5])}"
                f"{' …' if len(pending) > 5 else ''}",
                flush=True,
            )
            last_heartbeat = now
        if pending:
            time.sleep(poll_interval)
    return failed


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

    tasks = _build_tasks(
        problem_list,
        args.hardware,
        registry,
        args.tag,
        args.per_image_timeout,
        building,
    )
    failed = _run_pull_loop(tasks, registry, args.poll_interval)

    if failed:
        print(f"\n{len(failed)} image(s) failed to pull:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPulled {len(tasks)} image(s) successfully")


if __name__ == "__main__":
    main()
