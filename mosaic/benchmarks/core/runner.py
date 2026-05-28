# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver invocation helpers."""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Lazy-optional: tesseract_core / tesseract_jax are only needed by code paths
# that actually execute solvers (build_all, solver_sweep, safe_apply, …). The
# CLI imports this module at startup for every subcommand — including plain
# filesystem-only commands like `mosaic status` that must run on lightweight
# CI hosts where tesseract_core isn't installed. Keep the imports optional so
# importing the module doesn't trigger a ModuleNotFoundError.
try:
    import tesseract_core
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract
except ImportError:  # pragma: no cover — CI-only path
    tesseract_core = None  # type: ignore[assignment]
    Tesseract = None  # type: ignore[assignment]
    apply_tesseract = None  # type: ignore[assignment]


# ── Socket-read timeout ─────────────────────────────────────────────────────────
#
# tesseract_core.sdk.tesseract.HTTPClient._request calls self._session.request
# WITHOUT a timeout= argument.  When a tesseract-runtime container dies
# mid-call (OOM kill, segfault, SIGKILL from Azure health probe) the server
# side of the TCP socket enters CLOSE_WAIT/half-closed, but requests' blocking
# read never returns.  /proc/<pid>/wchan shows futex_wait_queue, CPU ~18s over
# 1h wall.  Pattern observed repeatedly with 3-4 concurrent ns-grid daemons.
#
# Fix: monkey-patch HTTPClient._request at runner import time so every HTTP
# call carries a (connect, read) timeout.  If the read times out, requests
# raises ReadTimeout, which bubbles up through apply_tesseract and is caught
# by safe_apply* as a normal failure — the solver cell becomes fail/NaN rather
# than hanging the entire daemon forever.
#
# Timeout is configurable via the MOSAIC_TESSERACT_TIMEOUT env var (seconds).
# Default 1200 s (20 min) — comfortably above the longest observed healthy
# tesseract call (~300 s for fenics_ns at N=64) while fast enough that a dead
# container is detected within one experiment's runtime budget rather than
# accumulating across an overnight sweep.
MOSAIC_TESSERACT_TIMEOUT: float = float(
    os.environ.get("MOSAIC_TESSERACT_TIMEOUT", "1200")
)
# Connect timeout is always short — the tesseract is already serving by the
# time we call apply (Tesseract.from_image waited for /health), so a slow
# connect means the container has gone away.
_MOSAIC_TESSERACT_CONNECT_TIMEOUT: float = 30.0


def _install_tesseract_http_timeout() -> None:
    """Patch tesseract_core HTTPClient so every request carries a (connect, read) timeout.

    Idempotent: repeated calls are no-ops (the patched ``__init__`` carries a
    ``_mosaic_timeout_patched`` attribute).  Safe to call from tests.

    Approach: wrap ``session.request`` per-instance rather than rewriting
    ``HTTPClient._request``. That keeps us decoupled from upstream's
    array-decode logic (which has multiple format-specific branches we don't
    want to re-implement).
    """
    if tesseract_core is None:
        return
    try:
        from tesseract_core.sdk import tesseract as _tcore_tess
    except Exception:
        return
    HTTPClient = getattr(_tcore_tess, "HTTPClient", None)
    if HTTPClient is None:
        return
    orig_init = HTTPClient.__init__
    if getattr(orig_init, "_mosaic_timeout_patched", False):
        return

    def _patched_init(self: object, *args: object, **kwargs: object) -> None:
        orig_init(self, *args, **kwargs)
        session = getattr(self, "_session", None)
        if session is None:
            return
        _orig_request = session.request
        _timeout = (_MOSAIC_TESSERACT_CONNECT_TIMEOUT, MOSAIC_TESSERACT_TIMEOUT)

        def _request_with_timeout(method: str, url: str, **kw: object) -> object:
            kw.setdefault("timeout", _timeout)
            return _orig_request(method, url, **kw)

        session.request = _request_with_timeout  # type: ignore[assignment]

    _patched_init._mosaic_timeout_patched = True  # type: ignore[attr-defined]
    HTTPClient.__init__ = _patched_init  # type: ignore[assignment]


_install_tesseract_http_timeout()

from .config import Problem  # noqa: E402
from .console import (  # noqa: E402
    console,
    make_build_progress,
    make_sweep_progress,
    print_rule,
    print_skip,
    print_warn,
)
from .resources import container_memory_args  # noqa: E402

_tl = threading.local()  # thread-local state (image_tag, gpu_id, last_apply_error)


@dataclass(frozen=True)
class WorkerContext:
    """Per-thread context set by :func:`run_with_gpu_pool`.

    Suites can read this inside their work callback (passed to
    ``run_with_gpu_pool``) to discover which GPU they were assigned and
    which image tag is serving — useful for ResourceSampler / MemoryPoller
    construction.

    Both fields are ``None`` when no GPU pool is active (e.g. serial mode
    or when called outside a runner-managed thread).
    """

    gpu_id: int | str | None
    image_tag: str | None


def current_worker_context() -> WorkerContext:
    """Return the :class:`WorkerContext` for the calling thread.

    Always returns a :class:`WorkerContext`; fields are ``None`` when the
    runner has not set them.
    """
    return WorkerContext(
        gpu_id=getattr(_tl, "gpu_id", None),
        image_tag=getattr(_tl, "image_tag", None),
    )


# Problem name set by run_suite so that run_with_gpu_pool can label containers.
# Docker label scoping prevents concurrent runs on different problems from
# killing each other's containers during cleanup.
_current_problem: str = ""


# ── Container lifecycle tracking ──────────────────────────────────────────────
#
# Tesseract.__exit__ calls .teardown() which docker-removes the container.
# When the process is interrupted (Ctrl-C, SIGTERM, OOM-kill of the Python
# interpreter) the context-manager __exit__ may never fire — especially in
# ThreadPoolExecutor worker threads where KeyboardInterrupt is only delivered
# to the main thread.
#
# We track every container name opened by _tracked_tesseract and remove it from
# the set when the context manager exits normally.  An atexit hook and signal
# handler force-remove any containers still in the set at shutdown.

_live_containers: set[str] = set()
_live_containers_lock = threading.Lock()


def _force_remove_containers() -> None:
    """Force-remove all tracked containers.  Best-effort, never raises."""
    with _live_containers_lock:
        names = list(_live_containers)
        _live_containers.clear()
    for name in names:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass


atexit.register(_force_remove_containers)


@contextlib.contextmanager
def _tracked_tesseract(tag: str, gpus: object, docker_args: object):
    """Open a Tesseract and track its container for cleanup on crash.

    Tags prefixed with ``inprocess:`` use
    :meth:`Tesseract.from_tesseract_api` (no Docker, just imports a
    ``tesseract_api.py``) — meant for end-to-end framework tests
    with the dummy tesseracts in ``tests/dummy_tesseracts/``.
    Every other tag is a Docker image and goes through
    :meth:`Tesseract.from_image`.
    """
    if tag.startswith("inprocess:"):
        with Tesseract.from_tesseract_api(tag[len("inprocess:") :]) as t:
            yield t
        return

    t = Tesseract.from_image(tag, gpus=gpus, docker_args=docker_args, num_workers=1)
    # Entering the context manager starts the container and populates
    # _serve_context, which contains the container name we need to track.
    t.__enter__()
    container_name = (t._serve_context or {}).get("container_name")
    if container_name:
        with _live_containers_lock:
            _live_containers.add(container_name)
    try:
        yield t
    finally:
        try:
            t.__exit__(None, None, None)
        finally:
            if container_name:
                with _live_containers_lock:
                    _live_containers.discard(container_name)


# ── Image management ──────────────────────────────────────────────────────────


def build_all(
    cfg: Problem, tag: str = "latest", max_workers: int = 2
) -> dict[str, str]:
    """Build all solver images in parallel. Returns name → image_tag.

    Shells out to the ``tesseract build`` CLI for each solver and lets
    BuildKit's layer cache decide what to actually re-execute — fully cached
    builds return in seconds, source-changed solvers rebuild only the affected
    layers. Per-problem failures are isolated by the caller in ``cli.build``.

    max_workers limits concurrent Docker builds to avoid overloading the host
    when multiple benchmark runs execute simultaneously.
    """
    import yaml

    def _resolve_tag(name: str, spec: object) -> str:
        """Derive the expected image tag for a solver without building."""
        if spec.image_tag:
            return spec.image_tag
        config_path = cfg.tesseract_dir / spec.dir / "tesseract_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                tconfig = yaml.safe_load(f)
            image_name = tconfig.get("name", spec.dir)
            return f"{image_name}:{tag}"
        return f"{spec.dir}:{tag}"

    def _build(spec: object) -> tuple[str, str]:
        name = spec.name  # type: ignore[attr-defined]
        # Run any adjacent build_base.sh first — lets tesseracts ship a
        # locally-built base-image wrapper (e.g. dealii-root:latest that
        # switches the upstream dealii/dealii image to USER root so the
        # tesseract template's apt-get steps don't fail with EACCES).
        tesseract_path = cfg.tesseract_dir / spec.dir
        base_script = tesseract_path / "build_base.sh"
        if base_script.exists():
            console.print(
                f"  [cyan]{name:<16}[/cyan] → running build_base.sh  [dim](base-image wrapper)[/dim]"
            )
            r = subprocess.run(
                ["bash", str(base_script)],
                cwd=str(tesseract_path),
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"{name}: build_base.sh failed (exit {r.returncode}):\n"
                    f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}"
                )
        t0 = time.monotonic()
        r = subprocess.run(
            ["tesseract", "build", "--tag", tag, str(tesseract_path)],
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - t0
        if r.returncode != 0:
            raise RuntimeError(
                f"{name}: tesseract build failed (exit {r.returncode}):\n"
                f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}"
            )
        # tesseract build prints the built images as a JSON array on stdout —
        # take the last non-empty line so any leading log noise is ignored.
        last_line = next(
            (ln for ln in reversed(r.stdout.splitlines()) if ln.strip()), ""
        )
        try:
            built_tags = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{name}: could not parse tesseract build output: {last_line!r}"
            ) from exc
        # tesseract build always tags the image with both ``:latest`` and
        # ``:<version>``; their order in the JSON output isn't guaranteed,
        # so prefer ``:latest`` deterministically rather than picking
        # ``built_tags[0]`` (which made some solvers log a version-pinned
        # tag while others showed ``:latest``).
        image_tag = next(
            (t for t in built_tags if t.endswith(":latest")),
            built_tags[0],
        )
        console.print(
            f"  [cyan]{name:<16}[/cyan] → {image_tag}  [dim]({elapsed:.1f}s)[/dim]"
        )
        return name, image_tag

    images: dict[str, str] = {}
    failed: list[str] = []
    with make_build_progress() as progress:
        task = progress.add_task("building solver images...", total=len(cfg.solvers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_build, spec): spec.name for spec in cfg.solvers}
            for future in concurrent.futures.as_completed(futures):
                solver_name = futures[future]
                try:
                    name, img_tag = future.result()
                    images[name] = img_tag
                except Exception as exc:
                    failed.append(solver_name)
                    print_warn(f"BUILD FAILED: {solver_name} — {exc}")
                progress.advance(task)
    if failed:
        console.print(
            f"\n[bold yellow]⚠ {len(failed)} solver(s) failed to build and will "
            f"be skipped: {', '.join(sorted(failed))}[/bold yellow]\n"
        )
    return images


def image_tags_no_build(cfg: Problem) -> dict[str, str]:
    """Return name → image_tag without building.

    Uses ``SolverSpec.image_tag`` if set, otherwise derives the image name
    from the ``tesseract_config.yaml``.
    """
    import yaml

    tags = {}
    for spec in cfg.solvers:
        name = spec.name
        if spec.image_tag:
            tags[name] = spec.image_tag
        else:
            config_path = cfg.tesseract_dir / spec.dir / "tesseract_config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    tconfig = yaml.safe_load(f)
                image_name = tconfig.get("name", spec.dir)
                tags[name] = f"{image_name}:latest"
            else:
                tags[name] = f"{spec.dir}:latest"

    registry = os.environ.get("MOSAIC_IMAGE_REGISTRY", "")
    if registry:
        registry = registry.rstrip("/")
        tags = {name: f"{registry}/{tag}" for name, tag in tags.items()}

    return tags


# ── Suite orchestration ───────────────────────────────────────────────────────


def _experiment_already_complete(suite_dir: object, name: str) -> bool:
    """True if a prior run wrote artifacts for *name* under *suite_dir*.

    Matches two on-disk layouts:
      1. ``<suite_dir>/<name>/result.json`` — most experiments.
      2. ``<suite_dir>/<name>/*/result.json`` — multi-IC experiments
         (forward.agreement with several ICs). Considered complete when at
         least one IC subdir has a ``result.json`` and every subdir has one.

    Note that this is a coarse signal: a partially-completed multi-IC
    experiment that didn't get to write any ICs is not detected and will
    re-run from scratch under ``--continue``.
    """
    exp_dir = suite_dir / name
    if not exp_dir.is_dir():
        return False
    if (exp_dir / "result.json").exists():
        return True
    subdirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    return bool(subdirs and all((p / "result.json").exists() for p in subdirs))


def run_suite(
    cfg: Problem,
    tags: dict[str, str],
    experiments: dict,
    to_run: list[str] | None = None,
    plots: bool = True,
    plot_fns: dict | None = None,
    suite_name: str = "",
    verbose_errors: bool = False,
    overrides: dict | None = None,
    skip_completed: bool = False,
) -> dict[str, dict]:
    """Run a set of named experiments and optionally generate plots.

    Args:
        cfg:             Problem instance.
        tags:            solver name → image tag mapping.
        experiments:     {name: callable(cfg, tags) → dict}
        to_run:          subset of names to run; None runs all.
        plots:           if True, call matching entries in plot_fns after experiments.
        plot_fns:        {name: callable(cfg)} for plot generation; None skips
                         plots. Entries whose name starts with ``"_extra/"`` are
                         suite-wide bonus plots — fired unconditionally after the
                         per-experiment plots, regardless of which experiments ran.
        suite_name:      name of the suite (e.g. "forward"), passed for display.
        verbose_errors:  if True, print full traceback on experiment/plot failures.
        overrides:       extra keyword arguments forwarded to each experiment callable.
        skip_completed:  if True, experiments whose result.json already exists
                         on disk are skipped (drives ``mosaic run --continue``).

    Returns:
        {experiment_name: results_dict}
    """
    from .io import results_dir

    global _current_problem
    _current_problem = (
        cfg.name
    )  # label containers so cleanup only kills this problem's containers
    to_run = to_run or list(experiments)
    suffix = "_debug" if (overrides or {}).get("debug") else ""
    suite_dir = results_dir() / cfg.name / suite_name
    n_experiments = len(to_run)
    n_solvers = len(cfg.solvers)
    console.print(
        f"  [dim]{n_experiments} experiment(s) queued, {n_solvers} solver(s) registered[/dim]"
    )
    results: dict[str, dict] = {}
    for ei, name in enumerate(to_run, 1):
        if name not in experiments:
            available = sorted(experiments.keys())
            print_warn(
                f"unknown experiment {name!r}. Available for this suite: {available}"
            )
            continue
        print_rule(f"experiment: {name} [{ei}/{n_experiments}]")
        if skip_completed and _experiment_already_complete(
            suite_dir, f"{name}{suffix}"
        ):
            print_skip(f"{name}: existing results found — re-use under --continue")
            results[name] = {"status": "skipped_continue"}
            continue
        try:
            results[name] = experiments[name](cfg, tags, **(overrides or {}))
        except NotImplementedError as exc:
            print_skip(str(exc))
        except Exception as exc:
            if verbose_errors:
                console.print_exception()
            print_warn(f"{name} failed: {exc}")
        finally:
            # Drop JAX's compiled-program cache between experiments to bound
            # process memory. Heavy optim runs (e.g. L-BFGS) can otherwise
            # leak enough XLA staging memory to make all later JIT compiles
            # fail with ``Cannot allocate memory``.
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass

    if plots and plot_fns:
        print_rule("plots")
        # Use the full (unfiltered) config for plot generation so all solvers
        # appear in plots even when this run targeted only a subset via --solvers.
        try:
            from mosaic.benchmarks.problems import get_config as _get_cfg

            _plot_cfg = _get_cfg(cfg.name)
        except Exception:
            _plot_cfg = cfg
        # Regular per-experiment plots: only fire when the matching experiment ran.
        for name in to_run:
            if name in plot_fns and name in results:
                try:
                    plot_fns[name](_plot_cfg, suffix=suffix)
                except Exception as exc:
                    if verbose_errors:
                        console.print_exception()
                    print_warn(f"plot_{name} failed: {exc}")
        # Suite-wide bonus plots ("_extra/<name>") — fire unconditionally.
        for key, fn in plot_fns.items():
            if not key.startswith("_extra/"):
                continue
            short_name = key[len("_extra/") :]
            try:
                fn(_plot_cfg)
            except Exception as exc:
                if verbose_errors:
                    console.print_exception()
                print_warn(f"{short_name} failed: {exc}")

    return results


# ── Per-solver loop ───────────────────────────────────────────────────────────


def per_solver_loop(
    cfg: Problem,
    tags: dict[str, str],
    solver_names: list[str],
    work_one: Callable,
    *,
    gpu_ids: list[str] | None = None,
    print_done: bool = True,
    catch: bool = False,
    catch_label: str = "work failed",
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, float]:
    """Run ``work_one(name, t)`` for each solver, returning ``{name: wall_seconds}``.

    Centralises the per-solver bookkeeping that every harness used to repeat
    inline:

      * ``t0 = time.perf_counter()`` start / ``elapsed = ... - t0`` end
      * Records ``elapsed`` in the returned ``wall_times`` dict
      * Optionally prints ``  {color}{name}[/] done in {elapsed:.1f}s`` after
        each solver completes (``print_done=True``, default)
      * Optionally swallows worker exceptions and prints a one-line SKIP marker
        in the solver's color (``catch=True`` — used by gradient harnesses
        that want one failing solver to not abort the others)

    The framework runs solver workers either serially (one Tesseract at a time)
    or in parallel across a GPU pool — see :func:`run_with_gpu_pool`. This
    helper sits on top of that loop and adds nothing more than the
    bookkeeping common to every existing harness.
    """
    wall_times: dict[str, float] = {}

    def _wrapper(name: str, t: object) -> None:
        color = cfg.solver(name).color
        t0 = time.perf_counter()
        try:
            work_one(name, t)
        except Exception as exc:
            if not catch:
                raise
            console.print(
                f"  [{color}]{name}[/] [yellow]SKIP ({catch_label}: {exc})[/]"
            )
            if on_error is not None:
                on_error(name, exc)
            return
        elapsed = time.perf_counter() - t0
        wall_times[name] = elapsed
        if print_done:
            console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

    run_with_gpu_pool(solver_names, tags, _wrapper, gpu_ids=gpu_ids, on_error=on_error)
    return wall_times


# ── Parameter sweeps ─────────────────────────────────────────────────────────


def solver_sweep(
    cfg: Problem,
    tags: dict[str, str],
    conditions: list,
    fn: Callable,
    *,
    suite: str = "forward",
    experiment: str | None = None,
    label_fn: Callable | None = None,
    key_fn: Callable | None = None,
    auto_status: bool = True,
    gpu_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Open one Tesseract per solver; call fn(name, t, cond) for each condition.

    fn(name, t, cond) → result
        Return None to signal failure.

    suite          suite name used to filter excluded solvers (default "forward").
    experiment     experiment name (optional) used for most-specific exclusion
                   key matching — e.g. ``suite="forward", experiment="agreement"``
                   will honour a ``"forward/agreement"`` exclusion entry.
    label_fn(cond) → str      progress label for the condition; default str(cond)
    key_fn(cond)   → hashable result dict key extracted from cond; default identity
    auto_status    if True (default), prints "  {label}  ok/FAIL" after each call
                   based on whether the result is None. Set False when fn handles
                   its own per-condition output (e.g. chunk-level progress).
    gpu_ids        if given, run each solver container in parallel, each pinned to
                   a GPU from the list (wraps round-robin if #solvers > #GPUs).

    Returns: ({solver_name: {key: result}}, {solver_name: wall_seconds})
    """
    from .utils import active_solvers

    conditions = list(conditions)
    names = active_solvers(cfg, suite, experiment)
    raw: dict[str, dict] = {name: {} for name in names}
    wall_times: dict[str, float] = {}

    n_solvers = len(names)
    n_conds = len(conditions)
    n_total = n_solvers * n_conds

    if n_total == 0:
        return raw, wall_times

    # Print sweep header so users know the grid size before work begins.
    cond_labels = [label_fn(c) if label_fn else str(c) for c in conditions]
    console.print(
        f"  [dim]sweep: {n_solvers} solver(s) x {n_conds} condition(s)"
        f" = {n_total} calls[/dim]"
    )
    console.print(f"  [dim]solvers: {', '.join(names)}[/dim]")

    # _progress_lock guards the Rich Progress bar advance() calls and the
    # completed-call counter in the multi-GPU parallel path where _per_solver
    # runs on multiple threads.
    _progress_lock = threading.Lock()
    _calls_done = 0

    with make_sweep_progress(total=n_total) as progress:
        task = progress.add_task("running sweep...", total=n_total)

        def _per_solver(name: str, t: object) -> None:
            nonlocal _calls_done
            si = names.index(name) + 1
            color = cfg.solver(name).color
            console.print(f"  [{color}]{name}[/] [dim]solver {si}/{n_solvers}[/dim]")
            t0 = time.perf_counter()
            for ci, cond in enumerate(conditions, 1):
                label = cond_labels[ci - 1]
                key = key_fn(cond) if key_fn else cond
                tc = time.perf_counter()
                result = fn(name, t, cond)
                dt = time.perf_counter() - tc
                raw[name][key] = result
                with _progress_lock:
                    _calls_done += 1
                    done = _calls_done
                if auto_status:
                    step = f"[dim]step {ci}/{n_conds}  [{done}/{n_total} total][/dim]"
                    if result is None:
                        console.print(
                            f"  [{color}]{name:<16}[/] {label:<12} [red]FAIL[/] ({dt:.1f}s)"
                            f"  {step}"
                        )
                    else:
                        console.print(
                            f"  [{color}]{name:<16}[/] {label:<12} [green]ok[/]   ({dt:.1f}s)"
                            f"  {step}"
                        )
                with _progress_lock:
                    progress.advance(task)
            elapsed = time.perf_counter() - t0
            wall_times[name] = elapsed
            console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

        run_with_gpu_pool(names, tags, _per_solver, gpu_ids=gpu_ids)
    return raw, wall_times


# ── GPU pool dispatch ─────────────────────────────────────────────────────────


def run_with_gpu_pool(
    solver_names: list[str],
    tags: dict[str, str],
    fn: Callable,
    gpu_ids: list[str] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> None:
    """Open one Tesseract per solver and call fn(name, t).

    When gpu_ids is None, runs serially with gpus=["all"] (all GPUs available).
    When gpu_ids is [] (empty list), runs serially with gpus=None (CPU-only mode,
    no GPU flags — for --gpus none / CPU-only hosts).
    When gpu_ids is given (non-empty), runs solvers in parallel; each container
    is pinned to a GPU from the pool (wraps round-robin if #solvers > #GPUs).

    Solvers whose image tag is missing from *tags* (e.g. because the build
    failed) are skipped with a warning.

    on_error(name, exc) is invoked when opening the Tesseract container or
    running fn(name, t) raises. Lets callers record the failure so the status
    classifier can mark the solver as FAILED rather than NOT_RUN (which would
    silently hide a broken container).
    """
    # --no-healthcheck prevents Azure's c3-progenitor from killing containers
    # that are mid-computation when the health probe fires (~4 s interval).
    # --label mosaic-problem=<name> scopes cleanup to one problem, preventing
    # concurrent runs from killing each other's containers
    # (use: docker ps -q --filter label=mosaic-problem=<name>).
    _problem_label = _current_problem or "mosaic"
    _NO_HC = ["--no-healthcheck", "--label", f"mosaic-problem={_problem_label}"]
    _NO_HC.extend(container_memory_args())

    # Filter out solvers with no built image.
    missing = [n for n in solver_names if n not in tags]
    if missing:
        print_warn(
            f"skipping {len(missing)} solver(s) with no built image: "
            f"{', '.join(missing)}"
        )
    solver_names = [n for n in solver_names if n in tags]

    if gpu_ids is None or gpu_ids == []:
        # gpu_ids=None  → no --gpus flag, use all GPUs (gpus=["all"])
        # gpu_ids=[]    → --gpus none/cpu, CPU-only host (gpus=None, no GPU flags)
        _gpus = None if gpu_ids == [] else ["all"]
        for name in solver_names:
            _tl.image_tag = tags[name]
            _tl.gpu_id = None  # all GPUs available
            try:
                with _tracked_tesseract(tags[name], _gpus, _NO_HC) as t:
                    fn(name, t)
            except Exception as exc:
                print_warn(f"{name} failed: {exc}")
                if on_error is not None:
                    on_error(name, exc)
        return

    # If gpu_ids=[] was handled above (cpu-only), we never reach here.
    # This branch handles actual GPU IDs.
    gpu_q: queue.Queue = queue.Queue()
    for gid in gpu_ids:
        gpu_q.put(gid)

    def _work(name: str) -> None:
        gid = gpu_q.get()
        try:
            _tl.image_tag = tags[name]
            _tl.gpu_id = gid
            with _tracked_tesseract(tags[name], [gid], _NO_HC) as t:
                fn(name, t)
        except Exception as exc:
            print_warn(f"{name} failed: {exc}")
            if on_error is not None:
                on_error(name, exc)
        finally:
            gpu_q.put(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        list(pool.map(_work, solver_names))


# ── Solver execution ──────────────────────────────────────────────────────────


def safe_apply(t: Tesseract, inputs: dict, output_key: str) -> jax.Array | None:
    """Forward pass with exception handling and finiteness check.

    Returns the output array on success, ``None`` on any failure (exception
    raised, missing output key, non-finite values).

    **Out-of-band error reporting**: on failure the exception message is
    written to thread-local state and must be retrieved via
    :func:`get_last_apply_error`. The pattern is::

        arr = safe_apply(t, inputs, output_key)
        if arr is None:
            err_msg = get_last_apply_error()  # may be None if cleared
            ...

    Each call overwrites the previous thread's last error; callers that
    interleave multiple ``safe_apply`` invocations should read the error
    immediately after each failure. The same convention applies to
    :func:`safe_apply_with_extras`.
    """
    arr, _, _ = safe_apply_with_extras(t, inputs, output_key, [], [])
    return arr


def get_last_apply_error() -> str | None:
    """Return the last :func:`safe_apply` failure message for this thread.

    Returns ``None`` if no :func:`safe_apply` / :func:`safe_apply_with_extras`
    call has failed on this thread, or if a successful call has overwritten
    the slot.
    """
    return getattr(_tl, "last_apply_error", None)


def _apply_tesseract_with_deadline(t: Tesseract, inputs: dict):
    """Call ``apply_tesseract(t, inputs)``.

    Deadline enforcement is handled by the HTTP timeout monkey-patch
    installed at module import (``_install_tesseract_http_timeout``).
    """
    return apply_tesseract(t, inputs)


def safe_apply_with_extras(
    t: Tesseract,
    inputs: dict,
    output_key: str,
    extra_scalar_keys: list[str],
    state_keys: list[str] | None = None,
) -> tuple[jax.Array | None, dict[str, float], dict[str, jax.Array]]:
    """Forward pass returning (primary array, scalar extras, array state).

    extra_scalar_keys: output keys to extract as floats (e.g. potential_energy).
    state_keys:        output keys to return as arrays for state threading
                       (e.g. velocities for chunked MD stability runs).

    Returns (None, {}, {}) on failure.
    """
    if state_keys is None:
        state_keys = []
    try:
        out = _apply_tesseract_with_deadline(t, inputs)
        if output_key not in out:
            available = sorted(out.keys())
            error_msg = (
                f"output_key {output_key!r} not in solver output. "
                f"Available keys: {available}"
            )
            print_warn(f"apply failed: {error_msg}")
            _tl.last_apply_error = error_msg
            return None, {}, {}
        arr = out[output_key]
        if not jnp.all(jnp.isfinite(arr)):
            return None, {}, {}
        extras = {}
        for k in extra_scalar_keys:
            if k in out and out[k] is not None:
                # Tesseracts return scalar outputs as either a Python scalar or a
                # shape-(1,) array (e.g. drag); flatten so float() never sees a
                # multi-element array.
                v = float(jnp.asarray(out[k]).flatten()[0])
                if jnp.isfinite(v):
                    extras[k] = v
        state = {
            k: jnp.array(out[k]) for k in state_keys if k in out and out[k] is not None
        }
        return arr, extras, state
    except Exception as exc:
        _exc_name = type(exc).__name__
        if isinstance(exc, TimeoutError):
            error_msg = (
                f"{_exc_name}: tesseract did not respond within "
                f"{MOSAIC_TESSERACT_TIMEOUT:.0f}s; {exc}"
            )
        else:
            error_msg = f"{_exc_name}: {exc}"
        print_warn(f"apply failed: {error_msg}")
        _tl.last_apply_error = error_msg
        return None, {}, {}
