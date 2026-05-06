"""Solver invocation helpers."""

from __future__ import annotations

import concurrent.futures
import os
import queue
import subprocess
import threading
import time

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
    """Patch tesseract_core.sdk.tesseract.HTTPClient.__init__ so every
    HTTPClient instance carries a requests.Session whose ``request`` method
    always applies a (connect, read) timeout.

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

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        session = getattr(self, "_session", None)
        if session is None:
            return
        _orig_request = session.request
        _timeout = (_MOSAIC_TESSERACT_CONNECT_TIMEOUT, MOSAIC_TESSERACT_TIMEOUT)

        def _request_with_timeout(method, url, **kw):
            kw.setdefault("timeout", _timeout)
            return _orig_request(method, url, **kw)

        session.request = _request_with_timeout  # type: ignore[assignment]

    _patched_init._mosaic_timeout_patched = True  # type: ignore[attr-defined]
    HTTPClient.__init__ = _patched_init  # type: ignore[assignment]


_install_tesseract_http_timeout()

from .config import ProblemConfig  # noqa: E402
from .console import (  # noqa: E402
    console,
    make_build_progress,
    print_rule,
    print_skip,
    print_warn,
)

_tl = threading.local()  # thread-local state (image_tag, gpu_id, last_apply_error)

# Problem name set by run_suite so that run_with_gpu_pool can label containers.
# Docker label scoping prevents concurrent runs on different problems from
# killing each other's containers during cleanup.
_current_problem: str = ""

# ── Image management ──────────────────────────────────────────────────────────


def _image_exists(image_tag: str) -> bool:
    """Return True if the Docker image already exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def build_all(
    cfg: ProblemConfig, tag: str = "latest", max_workers: int = 2
) -> dict[str, str]:
    """Build all solver images in parallel. Returns name → image_tag.

    Skips building any image that already exists locally (checked via
    ``docker image inspect``).  Only images that are genuinely absent are
    built, so a single failing base image cannot abort the whole command
    when the other images are already cached.

    max_workers limits concurrent Docker builds to avoid overloading the host
    when multiple benchmark runs execute simultaneously.
    """
    import yaml

    def _resolve_tag(name: str, spec) -> str:
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

    def _build(item):
        name, spec = item
        expected_tag = _resolve_tag(name, spec)
        if _image_exists(expected_tag):
            console.print(
                f"  [cyan]{name:<16}[/cyan] → {expected_tag}  [dim](cached)[/dim]"
            )
            return name, expected_tag
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
        img = tesseract_core.build_tesseract(tesseract_path, tag)
        # Previously ran `docker system prune -f` after every build. The
        # churn isn't worth it on hosts with plenty of disk. Use
        # `mosaic clean` explicitly when cleanup is needed.
        return name, img.tags[0]

    images: dict[str, str] = {}
    with make_build_progress() as progress:
        task = progress.add_task("building solver images...", total=len(cfg.solvers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for name, img_tag in pool.map(_build, cfg.solvers.items()):
                images[name] = img_tag
                progress.advance(task)
    return images


def image_tags_no_build(cfg: ProblemConfig) -> dict[str, str]:
    """Return name → image_tag without building, using SolverSpec.image_tag
    if set, otherwise deriving the image name from the tesseract_config.yaml."""
    import yaml

    tags = {}
    for name, spec in cfg.solvers.items():
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
    return tags


# ── Suite orchestration ───────────────────────────────────────────────────────


def run_suite(
    cfg: ProblemConfig,
    tags: dict[str, str],
    experiments: dict,
    to_run: list[str] | None = None,
    plots: bool = True,
    plot_fns: dict | None = None,
    suite_name: str = "",
    verbose_errors: bool = False,
    overrides: dict | None = None,
) -> dict[str, dict]:
    """Run a set of named experiments and optionally generate plots.

    Args:
        cfg:             ProblemConfig instance.
        tags:            solver name → image tag mapping.
        experiments:     {name: callable(cfg, tags) → dict}
        to_run:          subset of names to run; None runs all.
        plots:           if True, call matching entries in plot_fns after experiments.
        plot_fns:        {name: callable(cfg)} for plot generation; None skips plots.
        suite_name:      name of the suite (e.g. "calibration") used to look up
                         cfg.extra_plots for problem-specific plot hooks.
        verbose_errors:  if True, print full traceback on experiment/plot failures.

    Returns:
        {experiment_name: results_dict}
    """
    global _current_problem
    _current_problem = (
        cfg.name
    )  # label containers so cleanup only kills this problem's containers
    to_run = to_run or list(experiments)
    results: dict[str, dict] = {}
    for name in to_run:
        print_rule(name)
        try:
            results[name] = experiments[name](cfg, tags, **(overrides or {}))
        except NotImplementedError as exc:
            print_skip(str(exc))
        except Exception as exc:
            if verbose_errors:
                console.print_exception()
            print_warn(f"{name} failed: {exc}")

    if plots and plot_fns:
        print_rule("plots")
        suffix = "_debug" if (overrides or {}).get("debug") else ""
        # Use the full (unfiltered) config for plot generation so all solvers
        # appear in plots even when this run targeted only a subset via --solvers.
        try:
            from mosaic.benchmarks.problems import get_config as _get_cfg

            _plot_cfg = _get_cfg(cfg.name)
        except Exception:
            _plot_cfg = cfg
        for name in to_run:
            if name in plot_fns and name in results:
                try:
                    plot_fns[name](_plot_cfg, suffix=suffix)
                except Exception as exc:
                    if verbose_errors:
                        console.print_exception()
                    print_warn(f"plot_{name} failed: {exc}")
        for fn in _plot_cfg.extra_plots.get(suite_name, []):
            try:
                fn(_plot_cfg)
            except Exception as exc:
                if verbose_errors:
                    console.print_exception()
                print_warn(f"{fn.__name__} failed: {exc}")

    return results


# ── Parameter sweeps ─────────────────────────────────────────────────────────


def solver_sweep(
    cfg: ProblemConfig,
    tags: dict[str, str],
    conditions: list,
    fn,
    *,
    suite: str = "forward",
    experiment: str | None = None,
    label_fn=None,
    key_fn=None,
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

    def _per_solver(name: str, t) -> None:
        color = cfg.solvers[name].color
        t0 = time.perf_counter()
        for cond in conditions:
            label = label_fn(cond) if label_fn else str(cond)
            key = key_fn(cond) if key_fn else cond
            result = fn(name, t, cond)
            raw[name][key] = result
            if auto_status and result is None:
                console.print(f"  {label}  [red]FAIL[/]")
        elapsed = time.perf_counter() - t0
        wall_times[name] = elapsed
        console.print(f"  [{color}]{name}[/] done in {elapsed:.1f}s")

    run_with_gpu_pool(names, tags, _per_solver, gpu_ids=gpu_ids)
    return raw, wall_times


# ── GPU pool dispatch ─────────────────────────────────────────────────────────


def run_with_gpu_pool(
    solver_names: list[str],
    tags: dict[str, str],
    fn,
    gpu_ids: list[str] | None = None,
) -> None:
    """Open one Tesseract per solver and call fn(name, t).

    When gpu_ids is None, runs serially with gpus=["all"] (all GPUs available).
    When gpu_ids is [] (empty list), runs serially with gpus=None (CPU-only mode,
    no GPU flags — for --gpus none / CPU-only hosts).
    When gpu_ids is given (non-empty), runs solvers in parallel; each container
    is pinned to a GPU from the pool (wraps round-robin if #solvers > #GPUs).
    """
    # --no-healthcheck prevents Azure's c3-progenitor from killing containers
    # that are mid-computation when the health probe fires (~4 s interval).
    # --label mosaic-problem=<name> scopes cleanup to one problem, preventing
    # concurrent runs from killing each other's containers
    # (use: docker ps -q --filter label=mosaic-problem=<name>).
    _problem_label = _current_problem or "mosaic"
    _NO_HC = ["--no-healthcheck", "--label", f"mosaic-problem={_problem_label}"]

    # num_workers=2 keeps one uvicorn worker free to answer health probes while
    # the other is blocked in a long JAX/Julia computation.  The Tesseract
    # serve.py wrapper is `async def` but calls endpoint functions synchronously,
    # so a single-worker server can't respond to probes during computation.
    #
    # EXCEPTION: Julia-backed tesseracts (juliacall/PythonCall) crash in a
    # worker-restart loop when uvicorn forks a second worker -- juliacall
    # state isn't fork-safe.  For those we fall back to num_workers=1.
    # --no-healthcheck is already set above so the probe argument doesn't
    # apply during benchmark runs.  Supervisor patch 2026-04-16 (cycle 6).
    def _num_workers_for(tag: str) -> int:
        t = tag.lower()
        # Julia-backed tesseracts must use 1 worker (juliacall is not fork-safe).
        # xlb uses 1 worker: two uvicorn workers JIT-compile the same f64 XLA
        # graph on the same V100 GPU simultaneously, causing CUDA_ERROR_OUT_OF_MEMORY
        # and "CUBIN compiled for different GPU".
        # FEniCS/dolfin_adjoint tesseracts must use 1 worker: DOLFIN's PETSc state
        # and MPI communicators are not fork-safe; a second uvicorn worker causes
        # an immediate SIGKILL / ConnectionResetError (observed with fenics_ns_3d).
        if (
            "_jl_" in t
            or t.endswith("_jl")
            or "topopt_jl" in t
            or "trixi" in t
            or "incompressible_navier_stokes_jl" in t
            or "speedyweather" in t
            or "ins_navier" in t
            or t.startswith("xlb_")
            or "fenics" in t
        ):
            return 1
        return 2

    if gpu_ids is None or gpu_ids == []:
        # gpu_ids=None  → no --gpus flag, use all GPUs (gpus=["all"])
        # gpu_ids=[]    → --gpus none/cpu, CPU-only host (gpus=None, no GPU flags)
        _gpus = None if gpu_ids == [] else ["all"]
        for name in solver_names:
            _tl.image_tag = tags[name]
            _tl.gpu_id = None  # all GPUs available
            _nw = _num_workers_for(tags[name])
            try:
                with Tesseract.from_image(
                    tags[name], gpus=_gpus, docker_args=_NO_HC, num_workers=_nw
                ) as t:
                    fn(name, t)
            except Exception as exc:
                print_warn(f"{name} failed: {exc}")
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
            _nw = _num_workers_for(tags[name])
            with Tesseract.from_image(
                tags[name], gpus=[gid], docker_args=_NO_HC, num_workers=_nw
            ) as t:
                fn(name, t)
        except Exception as exc:
            print_warn(f"{name} failed: {exc}")
        finally:
            gpu_q.put(gid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        list(pool.map(_work, solver_names))


# ── Solver execution ──────────────────────────────────────────────────────────


def safe_apply(t: Tesseract, inputs: dict, output_key: str) -> jax.Array | None:
    """Forward pass with exception handling and finiteness check. Returns None on failure.

    On failure the exception message is stored in _tl.last_apply_error so callers
    can surface it (e.g. forward.py writes it to the JSON 'error' field).
    """
    arr, _, _ = safe_apply_with_extras(t, inputs, output_key, [], [])
    return arr


def get_last_apply_error() -> str | None:
    """Return the error message from the most recent failed safe_apply call on this thread."""
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
    state_keys: list[str] = [],
) -> tuple[jax.Array | None, dict[str, float], dict[str, jax.Array]]:
    """Forward pass returning (primary array, scalar extras, array state).

    extra_scalar_keys: output keys to extract as floats (e.g. potential_energy).
    state_keys:        output keys to return as arrays for state threading
                       (e.g. velocities for chunked MD stability runs).

    Returns (None, {}, {}) on failure.
    """
    try:
        out = _apply_tesseract_with_deadline(t, inputs)
        arr = out[output_key]
        if not jnp.all(jnp.isfinite(arr)):
            return None, {}, {}
        extras = {}
        for k in extra_scalar_keys:
            if k in out:
                v = float(out[k])
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
